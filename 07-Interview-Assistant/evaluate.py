"""
Automated evaluation for the Interview Assistant RAG pipeline.

Run from the project root:

    python evaluate.py

Retrieval-only evaluation:

    python evaluate.py --retrieval-only

Run one test case:

    python evaluate.py --case functional_interface
"""

import argparse
import json
from pathlib import Path

from src.config import TOP_K
from src.rag_chain import create_generation_chain, format_docs
from src.retriever import get_retriever


ABSTENTION_RESPONSE = (
    "The provided documents do not contain this information."
)

DEFAULT_CASES_PATH = Path(__file__).with_name(
    "evaluation_cases.json"
)


def normalize(text):
    """
    Normalize whitespace for stable answer comparison.
    """

    return " ".join(text.strip().split())


def contains_each_concept_group(text, concept_groups):
    """
    Check that the text contains at least one term from
    every expected concept group.

    Example:

        [
            ["anonymous", "nameless"],
            ["function"]
        ]

    The text must contain:
    - either "anonymous" or "nameless"
    - and "function"
    """

    lowered_text = text.lower()
    missing_groups = []

    for alternatives in concept_groups:

        group_found = any(
            term.lower() in lowered_text
            for term in alternatives
        )

        if not group_found:
            missing_groups.append(alternatives)

    return missing_groups


def has_source_and_page(document):
    """
    Verify that a retrieved Document contains source
    and page metadata.

    PyPDFLoader versions may provide either:
    - page_label
    - page
    """

    source = document.metadata.get("source")

    page = document.metadata.get("page_label")

    if page is None:
        page = document.metadata.get("page")

    return bool(source) and page is not None


def load_cases(path):
    """
    Load evaluation cases from the JSON file.
    """

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def print_check(label, passed, details=""):
    """
    Print one evaluation check.
    """

    status = "PASS" if passed else "FAIL"

    suffix = f" - {details}" if details else ""

    print(f"  [{status}] {label}{suffix}")


def evaluate_case(
    case,
    retriever,
    generation_chain,
    retrieval_only=False
):
    """
    Evaluate one question.

    Flow:

        Question
            ↓
        Single Retrieval
            ↓
        Retrieved Documents
          ├── page_content → Context
          └── metadata → Source checks
            ↓
        Context + Question
            ↓
        Generation Chain
            ↓
        Answer checks
    """

    question = case["question"]

    expected_groups = case[
        "expected_concept_groups"
    ]

    # -------------------------------------------------
    # STEP 1: RETRIEVAL
    # -------------------------------------------------
    #
    # Retrieval is performed only once.
    #
    # The same Documents are used for:
    # - context generation
    # - evidence checking
    # - metadata/source checking
    #
    retrieved_docs = retriever.invoke(question)

    # -------------------------------------------------
    # STEP 2: CREATE CONTEXT
    # -------------------------------------------------

    context = format_docs(retrieved_docs)

    checks = {}

    # Verify that the retriever returned TOP_K chunks.
    checks["retrieval_count"] = (
        len(retrieved_docs) == TOP_K
    )

    # Verify source and page metadata.
    checks["metadata"] = (
        bool(retrieved_docs)
        and all(
            has_source_and_page(document)
            for document in retrieved_docs
        )
    )

    missing_context_groups = []

    # Positive questions should retrieve context that
    # contains the expected concepts.
    #
    # We do not expect zero Documents for an unrelated
    # question because top-k similarity search still
    # returns the closest stored chunks.
    if not case["expected_to_abstain"]:

        missing_context_groups = (
            contains_each_concept_group(
                context,
                expected_groups
            )
        )

        checks["retrieved_evidence"] = (
            not missing_context_groups
        )

    answer = None
    missing_answer_groups = []

    # -------------------------------------------------
    # STEP 3: GENERATION
    # -------------------------------------------------

    if not retrieval_only:

        answer = generation_chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # Negative question:
        # Require the exact abstention response.
        if case["expected_to_abstain"]:

            checks["answer"] = (
                normalize(answer)
                == normalize(ABSTENTION_RESPONSE)
            )

        # Positive question:
        # Check expected concepts in the answer.
        else:

            missing_answer_groups = (
                contains_each_concept_group(
                    answer,
                    expected_groups
                )
            )

            checks["answer"] = (
                not missing_answer_groups
            )

    # -------------------------------------------------
    # STEP 4: DISPLAY RESULTS
    # -------------------------------------------------

    print()
    print("========================================")
    print(f"CASE: {case['id']}")
    print("========================================")
    print(f"Question: {question}")
    print()

    print_check(
        f"Retrieved exactly TOP_K={TOP_K} chunks",
        checks["retrieval_count"],
        f"received {len(retrieved_docs)}"
    )

    print_check(
        "Every chunk has source and page metadata",
        checks["metadata"]
    )

    if "retrieved_evidence" in checks:

        details = ""

        if missing_context_groups:
            details = (
                f"missing groups: "
                f"{missing_context_groups}"
            )

        print_check(
            "Retrieved context contains expected concepts",
            checks["retrieved_evidence"],
            details
        )

    if answer is not None:

        if case["expected_to_abstain"]:
            label = (
                "Returned the exact abstention response"
            )
        else:
            label = (
                "Answer contains expected concepts"
            )

        details = ""

        if missing_answer_groups:
            details = (
                f"missing groups: "
                f"{missing_answer_groups}"
            )

        print_check(
            label,
            checks["answer"],
            details
        )

        print()
        print("Answer:")

        for line in answer.strip().splitlines():
            print(f"  {line}")

    print()
    print("Sources:")

    for document in retrieved_docs:

        source = Path(
            document.metadata.get(
                "source",
                "Unknown"
            )
        ).name

        page = document.metadata.get(
            "page_label",
            document.metadata.get(
                "page",
                "Unknown"
            )
        )

        print(f"  - {source} - Page {page}")

    return all(checks.values())


def parse_args():
    """
    Read optional command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval, grounding, "
            "abstention, and metadata."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the JSON evaluation cases."
    )

    parser.add_argument(
        "--case",
        dest="case_id",
        help="Run only the case with this ID."
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Skip Llama generation and evaluate "
            "retrieval and metadata only."
        )
    )

    return parser.parse_args()


def main():

    args = parse_args()

    cases = load_cases(args.cases)

    # Run only one selected case when requested.
    if args.case_id:

        cases = [
            case
            for case in cases
            if case["id"] == args.case_id
        ]

        if not cases:
            raise SystemExit(
                f"Unknown evaluation case: "
                f"{args.case_id}"
            )

    # Create retriever once for the complete evaluation.
    retriever = get_retriever()

    # The generation chain is unnecessary when running
    # retrieval-only evaluation.
    if args.retrieval_only:
        generation_chain = None
    else:
        generation_chain = create_generation_chain()

    results = []

    for case in cases:

        passed = evaluate_case(
            case=case,
            retriever=retriever,
            generation_chain=generation_chain,
            retrieval_only=args.retrieval_only
        )

        results.append(passed)

    passed_count = sum(results)
    total_count = len(results)

    print()
    print("========================================")
    print(
        f"EVALUATION RESULT: "
        f"{passed_count}/{total_count} cases passed"
    )
    print("========================================")

    # Exit code 0 means all cases passed.
    # Exit code 1 means at least one case failed.
    raise SystemExit(
        0 if passed_count == total_count else 1
    )


if __name__ == "__main__":
    main()