
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Optional, List

from models import (
    TaskInstance, ModelConfig, EvaluationRun, EvaluationResult,
    BenchmarkResult, EvaluationStatus, LeaderboardEntry
)
from dataset_loader import DatasetLoader
from execution_env import DockerEnvironment
from agent_interface import BaseAgent, SimpleReActAgent, EnvironmentExecutor, AgentContext

logger = logging.getLogger(__name__)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples (runs)
        c: Number of correct samples
        k: k value for pass@k
        
    Returns:
        Probability that at least one of k samples is correct
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


class EvaluationEngine:
    def __init__(
        self,
        docker_env: Optional[DockerEnvironment] = None,
        dataset_loader: Optional[DatasetLoader] = None,
        results_dir: str = "./results",
        num_workers: int = 1
    ):
        self.docker_env = docker_env or DockerEnvironment()
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
    
    def evaluate_single_task(
        self,
        task: TaskInstance,
        agent: BaseAgent,
        run_id: Optional[str] = None,
        timeout: int = 1800
    ) -> EvaluationRun:
        run_id = run_id or str(uuid.uuid4())[:8]
        started_at = datetime.now()
        
        evaluation_run = EvaluationRun(
            run_id=run_id,
            task_id=task.instance_id,
            model_id=agent.model_config.model_id,
            status=EvaluationStatus.RUNNING,
            started_at=started_at
        )
        
        container_name = f"swe-agent-{run_id}"
        
        try:
            # Setup environment
            logger.info(f"Setting up environment for {task.instance_id}")
            self.docker_env.setup_task_environment(task, container_name)
            
            # Create executor
            executor = EnvironmentExecutor(container_name, self.docker_env)
            context = AgentContext(task=task, repo_path="/workspace/repo")
            
            # Initialize agent
            agent.initialize(context)
            
            # Run agent
            logger.info(f"Running agent on {task.instance_id}")
            actions = []
            generated_patch = None
            start_time = time.time()
            
            try:
                for action in agent.run(executor, max_steps=50):
                    actions.append(action)
                    
                    if time.time() - start_time > timeout:
                        logger.warning(f"Timeout for {task.instance_id}")
                        evaluation_run.status = EvaluationStatus.TIMEOUT
                        break
                    
                    if action.action_type == "submit":
                        # Get the diff from the working container
                        generated_patch = self.docker_env.get_diff_from_container(container_name)
                        logger.info(f"Agent submitted. Patch size: {len(generated_patch) if generated_patch else 0} chars")
                        break
                        
            except Exception as e:
                logger.error(f"Agent error: {e}")
                evaluation_run.error_message = str(e)
            
            evaluation_run.actions = actions
            evaluation_run.execution_time_seconds = time.time() - start_time
            
            # Clean up agent container
            self.docker_env.cleanup_container(container_name)
            
            # Verify if we have a patch
            if generated_patch and generated_patch.strip():
                evaluation_run.generated_patch = generated_patch
                
                # Verify in a FRESH container
                resolved, result = self.docker_env.verify_solution(
                    task, generated_patch
                )
                
                evaluation_run.resolved = resolved
                evaluation_run.status = EvaluationStatus.COMPLETED
                
                status_str = "✅ RESOLVED" if resolved else "❌ FAILED"
                logger.info(f"Task {task.instance_id}: {status_str}")
            else:
                evaluation_run.resolved = False
                evaluation_run.status = EvaluationStatus.COMPLETED
                logger.info(f"Task {task.instance_id}: No patch generated")
            
        except Exception as e:
            logger.error(f"Error on {task.instance_id}: {e}")
            evaluation_run.status = EvaluationStatus.FAILED
            evaluation_run.error_message = str(e)
            # Cleanup on error
            try:
                self.docker_env.cleanup_container(container_name)
            except:
                pass
        
        evaluation_run.completed_at = datetime.now()
        return evaluation_run
    
    def evaluate_task_multiple_runs(
        self,
        task: TaskInstance,
        agent: BaseAgent,
        num_runs: int = 5
    ) -> EvaluationResult:
        runs = []
        num_resolved = 0
        
        for i in range(num_runs):
            logger.info(f"Task {task.instance_id}: Run {i+1}/{num_runs}")
            
            run = self.evaluate_single_task(
                task, agent, run_id=f"{task.instance_id[:10]}-run{i+1}"
            )
            runs.append(run)
            
            if run.resolved:
                num_resolved += 1
        
        resolution_rate = num_resolved / num_runs if num_runs > 0 else 0
        pass_at_k = num_resolved > 0
        
        logger.info(f"Task {task.instance_id}: {num_resolved}/{num_runs} resolved, Pass@{num_runs}: {pass_at_k}")
        
        return EvaluationResult(
            task_id=task.instance_id,
            model_id=agent.model_config.model_id,
            num_runs=num_runs,
            num_resolved=num_resolved,
            runs=runs
        )
    
    def run_benchmark(
        self,
        model_config: ModelConfig,
        llm_client,
        tasks: Optional[List[TaskInstance]] = None,
        num_runs_per_task: int = 5,
        parallel: bool = False
    ) -> BenchmarkResult:
        if tasks is None:
            tasks = self.dataset_loader.get_benchmark_subset()
        
        logger.info(f"Starting benchmark with {len(tasks)} tasks, {num_runs_per_task} runs each")
        
        agent = SimpleReActAgent(model_config=model_config, api_client=llm_client)
        results = []
        started_at = datetime.now()
        
        for i, task in enumerate(tasks):
            logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task.instance_id}")
            
            result = self.evaluate_task_multiple_runs(
                task, agent, num_runs=num_runs_per_task
            )
            results.append(result)
            
            # Log progress
            resolved_so_far = sum(1 for r in results if r.num_resolved > 0)
            logger.info(f"Progress: {resolved_so_far}/{i+1} tasks with any success")
        
        # Create benchmark result
        benchmark = BenchmarkResult(
            model_id=model_config.model_id,
            model_name=model_config.name,
            benchmark_name="SWE-rebench",
            total_tasks=len(tasks),
            num_runs_per_task=num_runs_per_task,
            results=results,
            started_at=started_at,
            completed_at=datetime.now()
        )
        
        # Save results
        self._save_results(benchmark)
        
        return benchmark
    
    def _save_results(self, benchmark: BenchmarkResult) -> str:
        """Save benchmark results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = benchmark.model_name.replace("/", "_").replace(" ", "_")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to dict for JSON serialization
        data = {
            "model_id": benchmark.model_id,
            "model_name": benchmark.model_name,
            "benchmark_name": benchmark.benchmark_name,
            "total_tasks": benchmark.total_tasks,
            "num_runs_per_task": benchmark.num_runs_per_task,
            "resolved_rate": benchmark.resolved_rate,
            "pass_at_k_rate": benchmark.pass_at_k_rate,
            "resolved_count": benchmark.resolved_count,
            "pass_at_k_count": benchmark.pass_at_k_count,
            "sem": benchmark.calculate_sem(),
            "started_at": benchmark.started_at.isoformat(),
            "completed_at": benchmark.completed_at.isoformat() if benchmark.completed_at else None,
            "results": [
                {
                    "task_id": r.task_id,
                    "num_runs": r.num_runs,
                    "num_resolved": r.num_resolved,
                    "resolution_rate": r.resolution_rate,
                    "pass_at_k": r.pass_at_k
                }
                for r in benchmark.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)


class Leaderboard:
    def __init__(self, filepath: str = "./leaderboard.json"):
        self.filepath = Path(filepath)
        self.entries: List[LeaderboardEntry] = []
        self._load()
    
    def _load(self) -> None:
        if self.filepath.exists():
            try:
                with open(self.filepath) as f:
                    data = json.load(f)
                self.entries = [
                    LeaderboardEntry(**e) for e in data.get("entries", [])
                ]
            except Exception as e:
                logger.warning(f"Could not load leaderboard: {e}")
                self.entries = []
    
    def _save(self) -> None:
        # Re-rank before saving
        self._update_ranks()
        
        data = {
            "updated_at": datetime.now().isoformat(),
            "entries": [asdict(e) for e in self.entries]
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _update_ranks(self) -> None:
        self.entries.sort(key=lambda x: x.resolved_rate, reverse=True)
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1
    
    def add_result(
        self,
        benchmark: BenchmarkResult,
        model_release_date: Optional[datetime] = None,
        task_cutoff_date: Optional[datetime] = None
    ) -> LeaderboardEntry:
        # Check contamination
        is_contaminated = False
        if model_release_date and task_cutoff_date:
            is_contaminated = model_release_date > task_cutoff_date
        
        entry = LeaderboardEntry(
            rank=0,
            model_name=benchmark.model_name,
            model_id=benchmark.model_id,
            resolved_rate=benchmark.resolved_rate,
            pass_at_5=benchmark.pass_at_k_rate,
            sem=benchmark.calculate_sem(),
            num_tasks=benchmark.total_tasks,
            num_runs=benchmark.num_runs_per_task,
            evaluation_date=datetime.now(),
            is_contaminated=is_contaminated
        )
        
        # Remove existing entry for same model
        self.entries = [e for e in self.entries if e.model_id != entry.model_id]
        self.entries.append(entry)
        
        self._save()
        return entry
    
    def get_leaderboard(
        self,
        include_contaminated: bool = True,
        top_n: Optional[int] = None
    ) -> List[LeaderboardEntry]:
        entries = self.entries
        
        if not include_contaminated:
            entries = [e for e in entries if not e.is_contaminated]
        
        if top_n:
            entries = entries[:top_n]
        
        return entries
    
    def to_markdown(self, include_contaminated: bool = True) -> str:
        entries = self.get_leaderboard(include_contaminated)
        
        lines = [
            "# SWE-rebench Leaderboard\n",
            "| Rank | Model | Resolved | Pass@5 | SEM | Tasks | Clean |",
            "|------|-------|----------|--------|-----|-------|-------|"
        ]
        
        for e in entries:
            clean = "✓" if not e.is_contaminated else "⚠️"
            lines.append(
                f"| {e.rank} | {e.model_name} | {e.resolved_rate:.1%} | "
                f"{e.pass_at_5:.1%} | ±{e.sem:.2%} | {e.num_tasks} | {clean} |"
            )
        
        return "\n".join(lines)