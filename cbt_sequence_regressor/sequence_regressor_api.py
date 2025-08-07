from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add the sequence model directory to path
sequence_model_path = Path(__file__).parent / "cbt_sequence_model"
sys.path.append(str(sequence_model_path))

from sequence_regressor import CBTSequenceComplianceRegressor

app = FastAPI(
    title="CBT Sequence Regressor API",
    description="API for evaluating CBT conversation compliance and quality",
    version="1.0.0"
)

# Initialize the model
model_path = str(sequence_model_path)
try:
    regressor = CBTSequenceComplianceRegressor(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    regressor = None

# Request/Response models
class PredictionRequest(BaseModel):
    model_question: str = Field(..., description="The question asked by the model")
    user_response: str = Field(..., description="The user's response")
    conversation_context: str = Field(..., description="Previous conversation context")
    trigger_statement: str = Field(..., description="The trigger statement that initiated CBT")
    cbt_step: str = Field(..., description="Current CBT step (e.g., 'thought_identification')")

class PredictionResponse(BaseModel):
    satisfaction_score: float = Field(..., description="User satisfaction score (0-1)")
    ready_for_next_step: bool = Field(..., description="Whether user is ready to progress")
    response_quality: str = Field(..., description="Quality assessment of the response")
    suggested_action: str = Field(..., description="Suggested next action")
    confidence: Dict[str, float] = Field(..., description="Confidence scores for each prediction")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and model are healthy."""
    return HealthResponse(
        status="healthy" if regressor else "model_not_loaded",
        model_loaded=regressor is not None,
        model_info=regressor.model_info if regressor else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the sequence regressor."""
    if not regressor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = regressor.predict(
            model_question=request.model_question,
            user_response=request.user_response,
            conversation_context=request.conversation_context,
            trigger_statement=request.trigger_statement,
            cbt_step=request.cbt_step
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: list[PredictionRequest]):
    """Make predictions for multiple examples."""
    if not regressor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            result = regressor.predict(
                model_question=req.model_question,
                user_response=req.user_response,
                conversation_context=req.conversation_context,
                trigger_statement=req.trigger_statement,
                cbt_step=req.cbt_step
            )
            results.append({"success": True, "result": result})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    
    return {"predictions": results}

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if not regressor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": regressor.model_info,
        "available_cbt_steps": list(regressor.step_encoder.classes_),
        "quality_categories": list(regressor.quality_encoder.classes_),
        "action_categories": list(regressor.action_encoder.classes_)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)