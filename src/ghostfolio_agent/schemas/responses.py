from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ToolCallInfo(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str


class MetricsInfo(BaseModel):
    trace_id: str
    total_duration_s: float
    llm_duration_s: float
    tool_duration_s: float
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    error_type: Optional[str] = None


class GroundingInfo(BaseModel):
    grounded: int
    ungrounded: int
    rate: float


class SourceInfo(BaseModel):
    name: str
    tool: str


class ConstraintViolationInfo(BaseModel):
    tool: str
    rule: str
    severity: str
    detail: str


class VerificationInfo(BaseModel):
    confidence: float
    confidence_label: str
    grounding: GroundingInfo
    sources: list[SourceInfo]
    domain_violations: list[ConstraintViolationInfo] = []
    output_warnings: list[str] = []


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metrics: Optional[MetricsInfo] = None
    verification: Optional[VerificationInfo] = None
    run_id: Optional[str] = None
