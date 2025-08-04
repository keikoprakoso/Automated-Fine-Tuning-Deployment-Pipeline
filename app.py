"""
FastAPI Application for LLM Fine-Tuning Pipeline

This module provides a REST API for the fine-tuning pipeline, including
model inference, job management, and usage monitoring endpoints.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules
from preprocess import DataPreprocessor
from fine_tune import FineTuningManager
from inference import InferenceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Fine-Tuning Pipeline API")
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Fine-Tuning Pipeline API")

# Initialize FastAPI app
app = FastAPI(
    title="LLM Fine-Tuning Pipeline API",
    description="Automated Fine-Tuning & Deployment Pipeline for LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers (will be initialized lazily)
fine_tune_manager = None
inference_manager = None
preprocessor = None

def initialize_managers():
    """Initialize managers if not already initialized."""
    global fine_tune_manager, inference_manager, preprocessor
    
    if fine_tune_manager is None:
        try:
            fine_tune_manager = FineTuningManager()
            logger.info("Fine-tuning manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize fine-tuning manager: {str(e)}")
            fine_tune_manager = None
    
    if inference_manager is None:
        try:
            inference_manager = InferenceManager()
            logger.info("Inference manager initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize inference manager: {str(e)}")
            inference_manager = None
    
    if preprocessor is None:
        try:
            preprocessor = DataPreprocessor()
            logger.info("Preprocessor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize preprocessor: {str(e)}")
            preprocessor = None


# Pydantic models for request/response validation
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="User prompt for generation", min_length=1, max_length=2000)
    model_id: str = Field(..., description="Fine-tuned model ID")
    max_tokens: int = Field(default=1000, description="Maximum tokens for response", ge=1, le=4000)
    temperature: float = Field(default=0.7, description="Response randomness", ge=0.0, le=2.0)
    system_message: Optional[str] = Field(default=None, description="System message for context")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    response_time_seconds: float
    model_id: str
    status: str
    error_message: Optional[str] = None


class FineTuneRequest(BaseModel):
    """Request model for fine-tuning."""
    model: str = Field(default="gpt-3.5-turbo", description="Base model to fine-tune")
    validation_file_path: Optional[str] = Field(default=None, description="Path to validation file")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Fine-tuning hyperparameters")
    wait_for_completion: bool = Field(default=False, description="Whether to wait for job completion")


class FineTuneResponse(BaseModel):
    """Response model for fine-tuning."""
    job_id: str
    training_file_id: str
    validation_file_id: Optional[str]
    model: str
    status: str
    final_status: Optional[Dict[str, Any]]
    timestamp: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    model_id: Optional[str]
    created_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]


class UsageSummaryResponse(BaseModel):
    """Response model for usage summary."""
    period_days: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_tokens: int
    total_cost_usd: float
    model_usage: Dict[str, Dict[str, Any]]
    average_tokens_per_request: float
    average_cost_per_request: float


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    # Try to initialize managers if not already done
    initialize_managers()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "fine_tuning": fine_tune_manager is not None,
            "inference": inference_manager is not None,
            "preprocessing": preprocessor is not None
        }
    }


# Data preprocessing endpoints
@app.post("/preprocess", tags=["Data Processing"])
async def preprocess_dataset(
    file: UploadFile = File(..., description="CSV dataset file"),
    output_filename: str = Form(default="processed_training_data.jsonl")
):
    """
    Preprocess uploaded CSV dataset for fine-tuning.
    
    Uploads a CSV file containing Q&A or conversation data and converts it
    to OpenAI fine-tuning format (JSONL).
    """
    try:
        # Initialize preprocessor if needed
        initialize_managers()
        if not preprocessor:
            raise HTTPException(status_code=500, detail="Preprocessor not initialized")
        
        # Save uploaded file
        input_path = f"data/{file.filename}"
        output_path = f"data/{output_filename}"
        
        os.makedirs("data", exist_ok=True)
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the dataset
        summary = preprocessor.process_dataset(input_path, output_path)
        
        return {
            "message": "Dataset processed successfully",
            "summary": summary,
            "output_file": output_path
        }
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


# Fine-tuning endpoints
@app.post("/fine-tune", response_model=FineTuneResponse, tags=["Fine-tuning"])
async def create_fine_tune_job(request: FineTuneRequest):
    """
    Create a fine-tuning job.
    
    Starts a fine-tuning job using the specified training data and parameters.
    """
    try:
        if not fine_tune_manager:
            raise HTTPException(status_code=500, detail="Fine-tuning manager not initialized")
        
        # Use the most recent processed training file
        training_file_path = "data/processed_training_data.jsonl"
        
        if not os.path.exists(training_file_path):
            raise HTTPException(
                status_code=400, 
                detail="No processed training data found. Please run preprocessing first."
            )
        
        # Run fine-tuning pipeline
        summary = fine_tune_manager.run_fine_tuning_pipeline(
            training_file_path=training_file_path,
            model=request.model,
            validation_file_path=request.validation_file_path,
            hyperparameters=request.hyperparameters,
            wait_for_completion=request.wait_for_completion
        )
        
        return FineTuneResponse(**summary)
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")


@app.get("/jobs", tags=["Fine-tuning"])
async def list_jobs(limit: int = 10):
    """
    List recent fine-tuning jobs.
    """
    try:
        if not fine_tune_manager:
            raise HTTPException(status_code=500, detail="Fine-tuning manager not initialized")
        
        jobs = fine_tune_manager.list_jobs(limit=limit)
        return {"jobs": jobs}
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Fine-tuning"])
async def get_job_status(job_id: str):
    """
    Get status of a specific fine-tuning job.
    """
    try:
        if not fine_tune_manager:
            raise HTTPException(status_code=500, detail="Fine-tuning manager not initialized")
        
        status_info = fine_tune_manager.get_job_status(job_id)
        return JobStatusResponse(**status_info)
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.delete("/jobs/{job_id}", tags=["Fine-tuning"])
async def cancel_job(job_id: str):
    """
    Cancel a running fine-tuning job.
    """
    try:
        if not fine_tune_manager:
            raise HTTPException(status_code=500, detail="Fine-tuning manager not initialized")
        
        success = fine_tune_manager.cancel_job(job_id)
        
        if success:
            return {"message": f"Job {job_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel job {job_id}")
        
    except Exception as e:
        logger.error(f"Failed to cancel job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@app.get("/models", tags=["Fine-tuning"])
async def list_active_models():
    """
    List active fine-tuned models.
    """
    try:
        if not fine_tune_manager:
            raise HTTPException(status_code=500, detail="Fine-tuning manager not initialized")
        
        models = fine_tune_manager.get_active_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


# Inference endpoints
@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest):
    """
    Generate text using a fine-tuned model.
    
    Accepts a prompt and returns a generated response with usage statistics.
    """
    try:
        if not inference_manager:
            raise HTTPException(status_code=500, detail="Inference manager not initialized")
        
        # Generate response
        result = inference_manager.generate_response(
            prompt=request.prompt,
            model_id=request.model_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_message=request.system_message
        )
        
        return GenerateResponse(**result)
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/batch", tags=["Inference"])
async def batch_generate(
    prompts: List[str],
    model_id: str,
    max_tokens: int = 1000,
    temperature: float = 0.7
):
    """
    Generate responses for multiple prompts.
    """
    try:
        if not inference_manager:
            raise HTTPException(status_code=500, detail="Inference manager not initialized")
        
        results = inference_manager.batch_generate(
            prompts=prompts,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.post("/test-model", tags=["Inference"])
async def test_model(
    model_id: str,
    test_prompts: Optional[List[str]] = None
):
    """
    Test a fine-tuned model with sample prompts.
    """
    try:
        if not inference_manager:
            raise HTTPException(status_code=500, detail="Inference manager not initialized")
        
        test_results = inference_manager.test_model(model_id, test_prompts)
        return test_results
        
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model testing failed: {str(e)}")


# Monitoring endpoints
@app.get("/usage", response_model=UsageSummaryResponse, tags=["Monitoring"])
async def get_usage_summary(days: int = 30):
    """
    Get usage summary statistics.
    """
    try:
        if not inference_manager:
            raise HTTPException(status_code=500, detail="Inference manager not initialized")
        
        summary = inference_manager.get_usage_summary(days=days)
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return UsageSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Failed to get usage summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage summary: {str(e)}")


@app.get("/logs", tags=["Monitoring"])
async def get_recent_logs(limit: int = 10):
    """
    Get recent inference logs.
    """
    try:
        if not inference_manager:
            raise HTTPException(status_code=500, detail="Inference manager not initialized")
        
        logs = inference_manager.get_recent_logs(limit=limit)
        return {"logs": logs}
        
    except Exception as e:
        logger.error(f"Failed to get logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


# Lifespan context manager for startup/shutdown
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Fine-Tuning Pipeline API")
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Fine-Tuning Pipeline API")




def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main() 