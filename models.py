
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class TaskDifficulty(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class LLMScore:
    difficulty_score: int  # 1-3, lower is easier
    issue_text_score: int  # 1-3, lower is clearer
    test_score: int  # 1-3, lower is better test quality
    
    def is_high_quality(self) -> bool:
        return (self.difficulty_score <= 2 and 
                self.issue_text_score <= 2 and 
                self.test_score <= 2)


@dataclass
class InstallConfig:
    python: str  # Python version (e.g., "3.9")
    install: str  # Installation command
    test_cmd: str  # Test execution command
    packages: Optional[str] = None  # Package source type
    pre_install: list[str] = field(default_factory=list)  # Pre-install commands
    reqs_path: list[str] = field(default_factory=list)  # Requirements file paths
    env_yml_path: list[str] = field(default_factory=list)  # Conda env file paths
    pip_packages: list[str] = field(default_factory=list)  # Additional pip packages
    env_vars: dict[str, str] = field(default_factory=dict)  # Environment variables


@dataclass
class TaskInstance:
    instance_id: str  # Unique identifier (e.g., "owner__repo-123")
    repo: str  # Repository identifier (e.g., "owner/repo")
    base_commit: str  # Commit hash before solution
    version: str  # Project version for grouping
    created_at: datetime  # When the PR was created
    problem_statement: str  # GitHub issue description
    patch: str  # Solution patch (non-test files)
    test_patch: str  # Test file changes
    fail_to_pass: list[str]  # Tests that should go from fail to pass
    pass_to_pass: list[str]  # Tests that should remain passing
    install_config: InstallConfig  # Environment setup
    llm_score: LLMScore  # Quality assessment
    license_name: str  # Repository license
    hints_text: Optional[str] = None  # Issue comments before solution
    requirements: Optional[str] = None  # Frozen pip dependencies
    environment: Optional[str] = None  # Conda environment export
    
    @property
    def total_tests(self) -> int:
        """Total number of tests involved."""
        return len(self.fail_to_pass) + len(self.pass_to_pass)
    
    @property
    def num_fail_to_pass(self) -> int:
        return len(self.fail_to_pass)


@dataclass
class ModelConfig:
    model_id: str  # Unique identifier for the model
    name: str  # Display name
    provider: str  # Model provider (e.g., "openai", "anthropic", "local")
    api_endpoint: Optional[str] = None  # API endpoint URL
    api_key: Optional[str] = None  # API key (should be handled securely)
    max_context_length: int = 128000  # Maximum context window
    temperature: float = 0.0  # Generation temperature
    max_tokens: int = 4096  # Max output tokens
    additional_params: dict = field(default_factory=dict)  # Extra parameters
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = str(uuid.uuid4())


@dataclass
class AgentAction:
    action_type: str  # Type of action (e.g., "bash", "edit", "search")
    content: str  # Action content/command
    timestamp: datetime  # When the action was taken
    observation: Optional[str] = None  # Environment response
    success: bool = True  # Whether action succeeded


@dataclass
class EvaluationRun:
    run_id: str
    task_id: str
    model_id: str
    status: EvaluationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    resolved: bool = False  # Whether task was solved
    actions: list[AgentAction] = field(default_factory=list)
    generated_patch: Optional[str] = None  # Model's solution
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    
    def __post_init__(self):
        if not self.run_id:
            self.run_id = str(uuid.uuid4())


@dataclass
class EvaluationResult:
    task_id: str
    model_id: str
    num_runs: int
    num_resolved: int
    runs: list[EvaluationRun]
    
    @property
    def resolution_rate(self) -> float:
        if self.num_runs == 0:
            return 0.0
        return self.num_resolved / self.num_runs
    
    @property
    def pass_at_k(self) -> bool:
        return self.num_resolved > 0


@dataclass
class BenchmarkResult:
    model_id: str
    model_name: str
    benchmark_name: str
    total_tasks: int
    num_runs_per_task: int
    results: list[EvaluationResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.results if r.resolution_rate > 0.5)
    
    @property
    def resolved_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.resolution_rate for r in self.results) / len(self.results)
    
    @property
    def pass_at_k_count(self) -> int:
        return sum(1 for r in self.results if r.pass_at_k)
    
    @property
    def pass_at_k_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.pass_at_k_count / len(self.results)
    
    def calculate_sem(self) -> float:
        import statistics
        if len(self.results) < 2:
            return 0.0
        rates = [r.resolution_rate for r in self.results]
        return statistics.stdev(rates) / (len(rates) ** 0.5)


@dataclass
class LeaderboardEntry:
    rank: int
    model_id: str
    model_name: str
    resolved_rate: float
    sem: float
    pass_at_5: float
    num_tasks: int
    evaluation_date: datetime
    is_contaminated: bool = False  # True if model may have seen test data
    notes: Optional[str] = None
