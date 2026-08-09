from pydantic import BaseModel


class InterviewResponse(BaseModel):

    topic: str

    definition: str

    banking_example: str

    interview_question: str