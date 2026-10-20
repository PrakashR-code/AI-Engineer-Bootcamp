"""Deterministic checks for the CrewAI banking investigation example.

These checks compare expected source fields and flag a small set of explicit
risk assertions. They do not interpret meaning or prove that a report is safe.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _normalized_amount(value: Any) -> Decimal | None:
    """Parse numeric amounts while allowing presentation commas (200,000/200000)."""
    if value is None:
        return None
    text = str(value).strip()
    # Remove common display separators and currency labels/symbols, but retain
    # decimal point and sign so malformed or genuinely different values fail.
    text = text.replace(",", "").replace(" ", "")
    text = re.sub(r"^[A-Za-z$€£₹]+", "", text)
    text = re.sub(r"[A-Za-z$€£₹]+$", "", text)
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def validate_transaction(transaction_id: str, transactions: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Confirm that the requested transaction exists and has required fields.

    If `transactions` is omitted, the project's `banking_tools.TRANSACTIONS`
    mapping is used. Returns `valid`, `errors`, and `verified_transaction`.
    """
    # Validation flow: load the transaction map -> locate the requested ID ->
    # require the fields used by the investigation -> return a copy of the
    # verified record only when all source checks pass.
    if transactions is None:
        try:
            from banking_tools import TRANSACTIONS
        except ImportError as exc:
            return {"valid": False, "errors": [f"Could not load TRANSACTIONS: {exc}"], "verified_transaction": None}
        transactions = TRANSACTIONS

    record = transactions.get(transaction_id)
    if not isinstance(record, Mapping):
        return {"valid": False, "errors": [f"Transaction {transaction_id!r} was not found."], "verified_transaction": None}

    errors: list[str] = []
    for field in ("transaction_id", "type", "amount", "country", "status"):
        if field not in record or record[field] is None or str(record[field]).strip() == "":
            errors.append(f"Source record is missing required field: {field}.")
    actual_id = record.get("transaction_id")
    if actual_id is not None and str(actual_id).strip() != transaction_id:
        errors.append(f"Source record ID {actual_id!r} does not match requested ID {transaction_id!r}.")

    return {"valid": not errors, "errors": errors, "verified_transaction": dict(record) if not errors else None}


# Deliberately narrow, transparent phrase checks. These catch some direct
# assertions; they cannot reliably distinguish quoting, hypothetical language,
# negation, or nuanced context. Human review remains necessary.
_UNSUPPORTED_RISK_PATTERNS = (
    re.compile(r"\b(?:is|was|constitutes|proves|confirms)\s+(?:a\s+)?(?:fraudulent|fraud|money laundering|money-laundering)\b", re.I),
    re.compile(r"\b(?:money laundering|money-laundering)\s+(?:scheme|activity|transaction)\b", re.I),
    re.compile(r"\b(?:indicates?|demonstrates?|shows?)\s+(?:that\s+)?(?:this\s+)?(?:is\s+)?(?:fraud|money laundering|money-laundering)\b", re.I),
    re.compile(r"\b(?:unusually|exceptionally)\s+high\b", re.I),
)


def _report_value(report: str, value: Any) -> bool:
    """Check whether a source value appears in report; numeric amounts compare numerically."""
    if value is None:
        return False
    # Prefer a literal text match first. This keeps an ID such as TXN102 from
    # being mistaken for the numeric amount 102.
    normalized_value = _normalized_text(value)
    if normalized_value and normalized_value in _normalized_text(report):
        return True
    amount = _normalized_amount(value)
    if amount is not None:
        # Find numeric tokens with optional grouping commas and compare values.
        for token in re.findall(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])", report):
            if _normalized_amount(token) == amount:
                return True
    return False


def validate_agent_report(final_report: str, transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Check for source-field coverage and flag selected risky assertions.

    `valid` means only that these deterministic checks found no issue. It is
    not a semantic or compliance approval of the generated report.
    """
    # Report validation flow: reject an empty report, check that source facts
    # appear (with numeric normalization for amounts), then apply the listed
    # phrase rules. A clean result means only that these rules found no issue.
    report = str(final_report or "")
    errors: list[str] = []
    if not report.strip():
        errors.append("The generated report is empty.")

    for field in ("transaction_id", "type", "amount", "country", "status"):
        value = transaction.get(field)
        if value is None or str(value).strip() == "":
            errors.append(f"Cannot validate report coverage: source field {field!r} is missing.")
        elif not _report_value(report, value):
            errors.append(f"Report does not contain the source {field} value {value!r}.")

    for pattern in _UNSUPPORTED_RISK_PATTERNS:
        match = pattern.search(report)
        if match:
            errors.append(f"Potential unsupported risk assertion matched rule {pattern.pattern!r}: {match.group(0)!r}.")

    return {
        "valid": not errors,
        "errors": errors,
        "limitations": "Rule-based checks only: phrase matches and field presence do not establish semantic accuracy. Review the report manually.",
    }
