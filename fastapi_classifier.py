from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Initialize FastAPI app
app = FastAPI(
    title="AI vs Human Text Classifier API",
    description="Classifies text as AI-generated feedback or human user description",
    version="1.0.0"
)

# Load model and tokenizer
MODEL_PATH = "distilbert_ai_review_detector"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully on device: {device}")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")
    raise


# Define request/response models
class TextInput(BaseModel):
    text: str
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is great, I loved the quality and fast shipping!"
            }
        }


class ClassificationResponse(BaseModel):
    text: str
    classification: str
    confidence: float
    label: str
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text",
                "classification": "human",
                "confidence": 0.95,
                "label": "User Description"
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Routes
@app.get(
    "/",
    response_model=HealthResponse,
    description="load",
    summary="Health check endpoint"
)
async def root():
    """
    Root endpoint that returns API status and model information.
    """
    return {
        "status": "active",
        "model_loaded": True,
        "device": str(device)
    }


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify text as AI or Human",
    description="Analyzes input text and classifies it as AI-generated or human-written"
)
async def classify_text(input_data: TextInput):
    """
    Classifies whether the provided text is AI-generated feedback or human user description.
    
    - **text**: The text to classify (required)
    
    Returns:
    - **classification**: Either "ai" or "human"
    - **confidence**: Confidence score (0.0 to 1.0)
    - **label**: Human-readable label
    """
    
    try:
        # Validate input
        if not input_data.text or len(input_data.text.strip()) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Text cannot be empty"
            )
        
        if len(input_data.text) > 512:
            raise HTTPException(
                status_code=400,
                detail="Text exceeds maximum length of 512 characters"
            )
        
        # Tokenize and classify
        with torch.no_grad():
            inputs = tokenizer(
                input_data.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][pred_class].item()
        
        # Map prediction to label
        label_map = {0: "human", 1: "ai"}
        label_display = {0: "User Description", 1: "AI Generated Feedback"}
        
        classification = label_map[pred_class]
        label = label_display[pred_class]
        
        return {
            "text": input_data.text,
            "classification": classification,
            "confidence": round(confidence, 4),
            "label": label
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )


@app.get(
    "/model-info",
    summary="Get model information",
    description="Returns details about the loaded classification model"
)
async def model_info():
    """
    Returns information about the loaded model.
    """
    return {
        "model_name": "DistilBERT AI Review Detector",
        "model_path": MODEL_PATH,
        "classes": ["human", "ai"],
        "max_sequence_length": 512,
        "device": str(device),
        "model_type": "DistilBertForSequenceClassification"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
