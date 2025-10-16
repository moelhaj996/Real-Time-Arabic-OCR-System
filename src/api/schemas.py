"""Pydantic schemas for API."""

from pydantic import BaseModel, Field
from typing import List, Optional


class OCRRequest(BaseModel):
    """Request schema for OCR."""

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    use_beam_search: bool = Field(default=True)
    beam_width: int = Field(default=5, ge=1, le=20)
    return_alternatives: bool = Field(default=False)


class Alternative(BaseModel):
    """Alternative prediction."""

    text: str
    confidence: float


class OCRResponse(BaseModel):
    """Response schema for OCR."""

    text: str
    confidence: Optional[float] = None
    processing_time: float
    alternatives: Optional[List[Alternative]] = None


class BatchOCRResponse(BaseModel):
    """Response schema for batch OCR."""

    results: List[OCRResponse]
    total_processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    vocabulary_size: int


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    parameters: int
    device: str
    vocabulary_size: int
    beam_width: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
