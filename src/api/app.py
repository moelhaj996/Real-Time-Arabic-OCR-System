"""FastAPI application for Arabic OCR."""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference import ArabicOCRPredictor
from src.api.schemas import (
    OCRResponse,
    BatchOCRResponse,
    HealthResponse,
    ModelInfo,
    ErrorResponse,
)

# Initialize app
app = FastAPI(
    title="Arabic OCR API",
    description="Real-time Arabic OCR system with Vision Transformers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded on startup)
predictor: Optional[ArabicOCRPredictor] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor

    try:
        print("Loading OCR model...")
        predictor = ArabicOCRPredictor(
            model_path="models/best_model.pth",  # Update with actual path
            device="auto",
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but /ocr endpoints will not work")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Arabic OCR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr/image",
            "batch": "/ocr/batch",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="unknown",
            vocabulary_size=0,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=str(predictor.device),
        vocabulary_size=predictor.vocabulary.vocab_size,
    )


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.5),
    beam_width: int = Form(default=5),
    return_alternatives: bool = Form(default=False),
):
    """
    Perform OCR on a single image.

    Args:
        file: Image file
        confidence_threshold: Minimum confidence threshold
        beam_width: Beam width for decoding
        return_alternatives: Return alternative predictions

    Returns:
        OCR result
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Predict
        predictor.beam_width = beam_width
        result = predictor.predict(
            image,
            return_confidence=True,
            return_alternatives=return_alternatives,
        )

        # Filter by confidence
        if result.get("confidence", 1.0) < confidence_threshold:
            result["text"] = ""

        return OCRResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(
    files: List[UploadFile] = File(...),
    batch_size: int = Form(default=8),
):
    """
    Perform OCR on multiple images.

    Args:
        files: List of image files
        batch_size: Batch size for processing

    Returns:
        Batch OCR results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per request")

    try:
        start_time = time.time()

        # Read images
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        # Predict batch
        results = predictor.predict_batch(images, batch_size=batch_size)

        # Add processing time to each result
        for result in results:
            result["processing_time"] = 0.0  # Individual times not tracked in batch
            result["confidence"] = None

        total_time = time.time() - start_time

        return BatchOCRResponse(
            results=[OCRResponse(**r) for r in results],
            total_processing_time=total_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        name="Arabic OCR ViT-Base",
        parameters=predictor.model.get_num_parameters(),
        device=str(predictor.device),
        vocabulary_size=predictor.vocabulary.vocab_size,
        beam_width=predictor.beam_width,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
