
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional
from contextlib import asynccontextmanager
import evaluation_engine as EvaluationEngine
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ModelConfig, TaskDifficulty
from dataset_loader import DatasetLoader
from evaluation_engine import Leaderboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
dataset_loader: Optional[DatasetLoader] = None
evaluation_engine: Optional[EvaluationEngine] = None
leaderboard: Optional[Leaderboard] = None

# Track running evaluations
running_evaluations: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global dataset_loader, evaluation_engine, leaderboard
    
    logger.info("Starting SWE-rebench API...")
    
    # Initialize components
    dataset_loader = DatasetLoader()
    evaluation_engine = None
    
    leaderboard = Leaderboard()
    
    logger.info("API initialized successfully")
    
    yield
    
    logger.info("Shutting down SWE-rebench API...")


# FastAPI app
app = FastAPI(
    title="SWE-rebench Benchmarking API",
    description="""
    API for the SWE-rebench benchmark platform.
    
    SWE-rebench is a benchmark for evaluating LLM-based software engineering agents
    on real-world GitHub issues. It provides:
    
    - **21,000+ tasks** from 3,468 repositories
    - **Automated evaluation** with Docker-based environments
    - **Decontamination tracking** for fair comparisons
    - **Standardized metrics**: Resolved rate, Pass@5, SEM
    
    ## Quick Start
    
    1. Register your model with `/models/register`
    2. Submit for evaluation with `/evaluation/submit`
    3. Check results on `/leaderboard`
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelRegistration(BaseModel):
    """Request model for registering a new model."""
    name: str = Field(..., description="Display name for the model")
    provider: ModelProvider = Field(..., description="Model provider")
    api_endpoint: Optional[str] = Field(None, description="Custom API endpoint")
    api_key: Optional[str] = Field(None, description="API key (stored securely)")
    model_identifier: str = Field(..., description="Model identifier (e.g., 'gpt-4')")
    max_context_length: int = Field(128000, description="Maximum context length")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    release_date: Optional[datetime] = Field(None, description="Model release date for contamination tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "model_identifier": "gpt-4-turbo",
                "max_context_length": 128000,
                "temperature": 0.0,
                "release_date": "2024-04-09T00:00:00Z"
            }
        }


class ModelResponse(BaseModel):
    """Response model for registered models."""
    model_id: str
    name: str
    provider: str
    model_identifier: str
    registered_at: datetime


class EvaluationRequest(BaseModel):
    """Request model for submitting an evaluation."""
    model_id: str = Field(..., description="ID of registered model")
    num_runs: int = Field(5, ge=1, le=10, description="Number of runs per task")
    max_tasks: Optional[int] = Field(None, description="Limit number of tasks")
    task_filter: Optional[dict] = Field(None, description="Task filtering options")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model-123",
                "num_runs": 5,
                "max_tasks": 10,
                "task_filter": {
                    "max_difficulty": 2,
                    "min_year": 2025
                }
            }
        }


class EvaluationStatusModel(BaseModel):
    """Status of an evaluation."""
    evaluation_id: str
    model_id: str
    status: str
    progress: float
    tasks_completed: int
    total_tasks: int
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    current_task: Optional[str] = None


class TaskInfo(BaseModel):
    """Information about a single task."""
    instance_id: str
    repo: str
    problem_statement: str
    difficulty: int
    created_at: datetime
    num_fail_to_pass_tests: int
    num_pass_to_pass_tests: int


class DatasetStats(BaseModel):
    """Statistics about the dataset."""
    total_tasks: int
    unique_repos: int
    difficulty_distribution: dict[str, int]
    avg_fail_to_pass_tests: float
    avg_pass_to_pass_tests: float


class LeaderboardEntryResponse(BaseModel):
    """Response model for leaderboard entries."""
    rank: int
    model_name: str
    resolved_rate: float
    sem: float
    pass_at_5: float
    num_tasks: int
    evaluation_date: datetime
    is_contaminated: bool


class EvaluationResultResponse(BaseModel):
    """Response model for evaluation results."""
    evaluation_id: str
    model_name: str
    resolved_rate: float
    sem: float
    pass_at_5: float
    total_tasks: int
    num_runs_per_task: int
    started_at: datetime
    completed_at: Optional[datetime]



registered_models: dict[str, dict] = {}


# ==================== API Endpoints ====================

@app.get("/health", tags=["System"])
async def health_check():
    """Check API health status."""
    docker_available = evaluation_engine is not None
    return {
        "status": "healthy",
        "docker_available": docker_available,
        "dataset_loaded": dataset_loader.is_loaded if dataset_loader else False,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/dataset/stats", response_model=DatasetStats, tags=["Dataset"])
async def get_dataset_stats():
    """Get statistics about the SWE-rebench dataset."""
    if not dataset_loader:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    if not dataset_loader.is_loaded:
        # Load a sample for stats
        try:
            dataset_loader.load_from_huggingface()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load dataset: {e}")
    
    stats = dataset_loader.get_statistics()
    return DatasetStats(**stats)


@app.get("/dataset/tasks", tags=["Dataset"])
async def list_tasks(
    limit: int = 20,
    offset: int = 0,
    max_difficulty: Optional[int] = None,
    repo: Optional[str] = None
):
    """List tasks from the dataset with optional filtering."""
    if not dataset_loader or not dataset_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    tasks = dataset_loader.filter_tasks(
        max_difficulty=max_difficulty,
        repos=[repo] if repo else None
    )
    
    # Paginate
    paginated = tasks[offset:offset + limit]
    
    return {
        "total": len(tasks),
        "offset": offset,
        "limit": limit,
        "tasks": [
            {
                "instance_id": t.instance_id,
                "repo": t.repo,
                "problem_statement": t.problem_statement[:500] + "..." if len(t.problem_statement) > 500 else t.problem_statement,
                "difficulty": t.llm_score.difficulty_score,
                "created_at": t.created_at.isoformat(),
                "num_tests": t.total_tests
            }
            for t in paginated
        ]
    }


@app.get("/dataset/tasks/{instance_id}", response_model=TaskInfo, tags=["Dataset"])
async def get_task(instance_id: str):
    """Get details of a specific task."""
    if not dataset_loader:
        raise HTTPException(status_code=503, detail="Dataset not loaded")
    
    task = dataset_loader.get_task(instance_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskInfo(
        instance_id=task.instance_id,
        repo=task.repo,
        problem_statement=task.problem_statement,
        difficulty=task.llm_score.difficulty_score,
        created_at=task.created_at,
        num_fail_to_pass_tests=task.num_fail_to_pass,
        num_pass_to_pass_tests=len(task.pass_to_pass)
    )


@app.post("/models/register", response_model=ModelResponse, tags=["Models"])
async def register_model(registration: ModelRegistration):
    """
    Register a new model for evaluation.
    
    The model will be assigned a unique ID that should be used
    when submitting evaluations.
    """
    import uuid
    
    model_id = str(uuid.uuid4())
    
    model_data = {
        "model_id": model_id,
        "name": registration.name,
        "provider": registration.provider.value,
        "model_identifier": registration.model_identifier,
        "api_endpoint": registration.api_endpoint,
        "max_context_length": registration.max_context_length,
        "temperature": registration.temperature,
        "release_date": registration.release_date,
        "registered_at": datetime.now()
    }
    
    if registration.api_key:
        model_data["api_key_hash"] = hash(registration.api_key)
    
    registered_models[model_id] = model_data
    
    logger.info(f"Registered model: {registration.name} ({model_id})")
    
    return ModelResponse(
        model_id=model_id,
        name=registration.name,
        provider=registration.provider.value,
        model_identifier=registration.model_identifier,
        registered_at=model_data["registered_at"]
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List all registered models."""
    return {
        "models": [
            {
                "model_id": m["model_id"],
                "name": m["name"],
                "provider": m["provider"],
                "registered_at": m["registered_at"].isoformat()
            }
            for m in registered_models.values()
        ]
    }


@app.get("/models/{model_id}", tags=["Models"])
async def get_model(model_id: str):
    """Get details of a registered model."""
    if model_id not in registered_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = registered_models[model_id]
    return {
        "model_id": model["model_id"],
        "name": model["name"],
        "provider": model["provider"],
        "model_identifier": model["model_identifier"],
        "max_context_length": model["max_context_length"],
        "registered_at": model["registered_at"].isoformat()
    }


@app.post("/evaluation/submit", tags=["Evaluation"])
async def submit_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a model for evaluation on SWE-rebench.
    
    The evaluation runs in the background. Use the returned
    evaluation_id to track progress.
    """
    if request.model_id not in registered_models:
        raise HTTPException(status_code=404, detail="Model not registered")
    
    if not evaluation_engine:
        raise HTTPException(
            status_code=503,
            detail="Evaluation engine not available (Docker required)"
        )
    
    import uuid
    evaluation_id = str(uuid.uuid4())
    
    # Initialize evaluation tracking
    running_evaluations[evaluation_id] = {
        "evaluation_id": evaluation_id,
        "model_id": request.model_id,
        "status": "queued",
        "progress": 0.0,
        "tasks_completed": 0,
        "total_tasks": request.max_tasks or 294,
        "started_at": datetime.now(),
        "current_task": None
    }
    
    # Start background evaluation
    background_tasks.add_task(
        run_evaluation_task,
        evaluation_id,
        request
    )
    
    return {
        "evaluation_id": evaluation_id,
        "status": "queued",
        "message": "Evaluation submitted successfully"
    }


async def run_evaluation_task(evaluation_id: str, request: EvaluationRequest):
    """Background task to run evaluation."""
    try:
        running_evaluations[evaluation_id]["status"] = "running"
        
        # Get model config
        model_data = registered_models[request.model_id]
        model_config = ModelConfig(
            model_id=model_data["model_id"],
            name=model_data["name"],
            provider=model_data["provider"],
            max_context_length=model_data["max_context_length"],
            temperature=model_data["temperature"]
        )
        
        # Get tasks
        tasks = dataset_loader.get_benchmark_subset(max_tasks=request.max_tasks or 294)
        running_evaluations[evaluation_id]["total_tasks"] = len(tasks)
        import asyncio
        for i, task in enumerate(tasks[:5]):  # Limited for demo
            running_evaluations[evaluation_id]["current_task"] = task.instance_id
            running_evaluations[evaluation_id]["tasks_completed"] = i
            running_evaluations[evaluation_id]["progress"] = (i / len(tasks)) * 100
            await asyncio.sleep(1)  # Simulated work
        
        running_evaluations[evaluation_id]["status"] = "completed"
        running_evaluations[evaluation_id]["progress"] = 100.0
        
    except Exception as e:
        logger.error(f"Evaluation {evaluation_id} failed: {e}")
        running_evaluations[evaluation_id]["status"] = "failed"
        running_evaluations[evaluation_id]["error"] = str(e)


@app.get("/evaluation/{evaluation_id}", response_model=EvaluationStatusModel, tags=["Evaluation"])
async def get_evaluation_status(evaluation_id: str):
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_data = running_evaluations[evaluation_id]
    return EvaluationStatusModel(**eval_data)


@app.get("/evaluation/{evaluation_id}/results", tags=["Evaluation"])
async def get_evaluation_results(evaluation_id: str):
    if evaluation_id not in running_evaluations:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    eval_data = running_evaluations[evaluation_id]
    
    if eval_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not completed. Current status: {eval_data['status']}"
        )
    
    # Return results (would be stored from actual evaluation)
    return {
        "evaluation_id": evaluation_id,
        "model_id": eval_data["model_id"],
        "status": "completed",
        "summary": {
            "resolved_rate": 0.0,  # Would be actual results
            "pass_at_5": 0.0,
            "sem": 0.0
        },
        "tasks_evaluated": eval_data["tasks_completed"],
        "total_tasks": eval_data["total_tasks"]
    }


# ----- Leaderboard Endpoints -----

@app.get("/leaderboard", tags=["Leaderboard"])
async def get_leaderboard(
    include_contaminated: bool = True,
    top_n: Optional[int] = None
):
    """
    Get the SWE-rebench leaderboard.
    
    Args:
        include_contaminated: Include potentially contaminated results
        top_n: Limit to top N entries
    """
    if not leaderboard:
        raise HTTPException(status_code=503, detail="Leaderboard not available")
    
    entries = leaderboard.get_leaderboard(
        include_contaminated=include_contaminated,
        top_n=top_n
    )
    
    return {
        "leaderboard": [
            LeaderboardEntryResponse(
                rank=e.rank,
                model_name=e.model_name,
                resolved_rate=e.resolved_rate,
                sem=e.sem,
                pass_at_5=e.pass_at_5,
                num_tasks=e.num_tasks,
                evaluation_date=e.evaluation_date,
                is_contaminated=e.is_contaminated
            ).model_dump()
            for e in entries
        ],
        "total_entries": len(entries),
        "updated_at": datetime.now().isoformat()
    }


@app.get("/leaderboard/markdown", tags=["Leaderboard"])
async def get_leaderboard_markdown(include_contaminated: bool = True):
    """Get the leaderboard as a markdown table."""
    if not leaderboard:
        raise HTTPException(status_code=503, detail="Leaderboard not available")
    
    return {"markdown": leaderboard.to_markdown(include_contaminated)}

@app.get("/benchmark/info", tags=["Benchmark"])
async def get_benchmark_info():
    """Get information about the SWE-rebench benchmark."""
    return {
        "name": "SWE-rebench",
        "description": "A benchmark for evaluating LLM-based software engineering agents on real-world GitHub issues",
        "version": "1.0",
        "paper_url": "https://arxiv.org/abs/2505.20411",
        "dataset_url": "https://huggingface.co/datasets/nebius/SWE-rebench",
        "metrics": {
            "resolved_rate": "Percentage of tasks where the generated patch passes all tests",
            "pass_at_5": "Percentage of tasks solved at least once in 5 runs",
            "sem": "Standard Error of Mean for resolution rate"
        },
        "evaluation_protocol": {
            "num_runs": 5,
            "scaffolding": "ReAct-style agent",
            "context_length": 128000,
            "timeout_per_task": 1800
        },
        "decontamination": {
            "description": "Tasks are tracked by creation date to identify potential data leakage",
            "method": "Compare model release date with task creation dates"
        }
    }


# ==================== Main Entry Point ====================

@app.get("/", response_class=HTMLResponse, tags=["System"])
async def root():
    """Root endpoint with welcome page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SWE-rebench Benchmarking API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            a { color: #3498db; }
        </style>
    </head>
    <body>
        <h1>🚀 SWE-rebench Benchmarking API</h1>
        <p>Welcome to the SWE-rebench benchmark platform for evaluating LLM-based software engineering agents.</p>
        
        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">📖 Interactive API Documentation (Swagger)</a></li>
            <li><a href="/redoc">📚 ReDoc Documentation</a></li>
            <li><a href="/health">❤️ Health Check</a></li>
        </ul>
        
        <h2>Key Endpoints</h2>
        <div class="endpoint"><span class="method">GET</span> /dataset/stats - Get dataset statistics</div>
        <div class="endpoint"><span class="method">GET</span> /dataset/tasks - List tasks</div>
        <div class="endpoint"><span class="method">POST</span> /models/register - Register a model</div>
        <div class="endpoint"><span class="method">POST</span> /evaluation/submit - Submit evaluation</div>
        <div class="endpoint"><span class="method">GET</span> /leaderboard - View leaderboard</div>
        
        <h2>Dataset Info</h2>
        <p>21,336 tasks from 3,468 GitHub repositories</p>
    </body>
    </html>
    """


def main():
    """Run the API server."""
    print("\n" + "="*60)
    print("🚀 SWE-rebench Benchmarking API Server")
    print("="*60)
    print("\n📍 Server starting at: http://localhost:8000")
    print("📖 API Docs at: http://localhost:8000/docs")
    print("❤️  Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()