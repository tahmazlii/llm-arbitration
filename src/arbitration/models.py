from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CritiqueDimension(str, Enum):
    ACCURACY = "accuracy"
    LOGIC = "logic"
    COMPLETENESS = "completeness"


class Issue(BaseModel):
    quote: str = Field(description="a quote from the original output (the exact span that's problematic)")
    description: str = Field(description="the description of the issue")
    severity: IssueSeverity


class Critique(BaseModel):
    dimension: CritiqueDimension
    score: int = Field(ge=1, le=5, description="1-5, where 5 means no issues found and 1 means severe problems")
    issues: List[Issue] = Field(description="list of issues from the critic")
    confidence: float = Field(ge=0, le=1, description="critic's confidence in its own assessment, 0-1")