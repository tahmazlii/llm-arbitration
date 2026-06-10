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

class ConfirmedIssue(BaseModel):
    quote: str = Field(description="a quote from the original output (the exact span that's problematic)")
    description: str = Field(description="the description of the issue")
    dimensions: List[CritiqueDimension] = Field(description="the critic dimension(s) that raised this issue")
    severity: IssueSeverity = Field(description="the adjudicator's final severity rating for this issue")
    reason: str = Field(description="the reason the adjudicator upheld this issue")

class Critique(BaseModel):
    dimension: CritiqueDimension
    score: int = Field(ge=1, le=5, description="1-5, where 5 means no issues found and 1 means severe problems")
    issues: List[Issue] = Field(description="list of issues from the critic")
    confidence: float = Field(ge=0, le=1, description="critic's confidence in its own assessment, 0-1")

class CriticFlag(BaseModel):
    dimension: CritiqueDimension
    severity: IssueSeverity

class DismissedFlag(BaseModel):
    span: str = Field(description="the text span or issue that a critic flagged but the adjudicator overruled")
    flag: CriticFlag = Field(description="the critic's original flag (dimension and severity) that was dismissed")
    reason: str = Field(description="the reason the adjudicator dismissed this flag")

class SpanOverlap(BaseModel):
    span: str = Field(description="the text span flagged by more than one critic")
    flags: List[CriticFlag] = Field(description="the critics that flagged this span, each with their dimension and severity")


class DisagreementReport(BaseModel):
    score_spread: int = Field(description="the difference between the maximum and minimum scores across the three critiques")
    highest_dimension: CritiqueDimension = Field(description="the dimension that gave the highest score")
    lowest_dimension: CritiqueDimension = Field(description="the dimension that gave the lowest score")
    overlaps: List[SpanOverlap] = Field(description="spans flagged by more than one critic, with each critic's severity")

class Verdict(BaseModel):
    quality: int = Field(ge=1, le=10, description="overall holistic quality score, 1-10, where 10 is flawless and 1 is severely flawed")
    confidence: float = Field(ge=0, le=1, description="how confident the adjudicator is in this verdict, 0-1")
    confirmed_issues: List[ConfirmedIssue] = Field(description="issues the adjudicator upholds as real after weighing the critics, each with the dimensions that raised it and the reasoning for upholding it")
    dismissed_flags: List[DismissedFlag] = Field(description="flags a critic raised that the adjudicator overruled, each with the reason for dismissal")
    summary: str = Field(description="a one-paragraph plain-language assessment of the output's overall quality and the key findings")
