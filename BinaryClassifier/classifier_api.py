from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from binary_classifier import CBTBinaryClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CBT Binary Classifier API",
    description="API for detecting CBT-triggering conversations",
    version="1.0.0"
)

# Request/Response models
class TextRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    threshold: float = Field(0.7, description="Confidence threshold for CBT trigger detection")

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify")
    threshold: float = Field(0.7, description="Confidence threshold for CBT trigger detection")

class PredictionResponse(BaseModel):
    is_cbt_trigger: bool
    confidence: float
    threshold: float
    text: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Initialize classifier
classifier = None

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global classifier
    try:
        classifier = CBTBinaryClassifier()
        model_path = Path(__file__).parent / "cbt_classifier"
        
        if model_path.exists():
            classifier.load_model(str(model_path))
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.error(f"Model path {model_path} does not exist")
            raise ValueError(f"Model not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "CBT Binary Classifier API",
        "model_loaded": classifier is not None
    }

@app.post("/classify", response_model=PredictionResponse)
async def classify_text(request: TextRequest):
    """Classify a single text"""
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = classifier.predict(request.text, request.threshold)
        
        return PredictionResponse(
            is_cbt_trigger=result['is_cbt_trigger'],
            confidence=result['confidence'],
            threshold=result['threshold'],
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchPredictionResponse)
async def classify_batch(request: BatchTextRequest):
    """Classify multiple texts"""
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = classifier.batch_predict(request.texts, request.threshold)
        
        predictions = []
        for i, result in enumerate(results):
            text_preview = request.texts[i][:100] + "..." if len(request.texts[i]) > 100 else request.texts[i]
            predictions.append(PredictionResponse(
                is_cbt_trigger=result['is_cbt_trigger'],
                confidence=result['confidence'],
                threshold=result['threshold'],
                text=text_preview
            ))
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": classifier.model_name,
        "model_path": str(Path(__file__).parent / "cbt_classifier"),
        "status": "loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)