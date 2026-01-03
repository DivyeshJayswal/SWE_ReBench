"""
SWE-rebench Benchmarking Tool - Dataset Loader
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import asdict

from models import (
    TaskInstance, InstallConfig, LLMScore, TaskDifficulty
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    HUGGINGFACE_DATASET = "nebius/SWE-rebench"
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".swe_rebench_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, TaskInstance] = {}
        self._loaded = False
    
    def load_from_huggingface(self, split: str = "test") -> int:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the 'datasets' package: pip install datasets"
            )
        
        logger.info(f"Loading SWE-rebench dataset from HuggingFace ({split} split)...")
        
        dataset = load_dataset(self.HUGGINGFACE_DATASET, split=split)
        
        for item in dataset:
            task = self._parse_task_item(item)
            if task:
                self._tasks[task.instance_id] = task
        
        self._loaded = True
        logger.info(f"Loaded {len(self._tasks)} tasks from HuggingFace")
        return len(self._tasks)
    
    def load_from_json(self, filepath: str) -> int:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading dataset from {filepath}...")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'tasks' in data:
            items = data['tasks']
        else:
            raise ValueError("Invalid JSON format. Expected list or dict with 'tasks' key.")
        
        for item in items:
            task = self._parse_task_item(item)
            if task:
                self._tasks[task.instance_id] = task
        
        self._loaded = True
        logger.info(f"Loaded {len(self._tasks)} tasks from JSON")
        return len(self._tasks)
    
    def load_from_jsonl(self, filepath: str) -> int:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logger.info(f"Loading dataset from {filepath}...")
        
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    task = self._parse_task_item(item)
                    if task:
                        self._tasks[task.instance_id] = task
        
        self._loaded = True
        logger.info(f"Loaded {len(self._tasks)} tasks from JSONL")
        return len(self._tasks)
    
    def _parse_task_item(self, item: dict) -> Optional[TaskInstance]:
        try:
            # Parse install config
            install_config_data = item.get('install_config', {})
            install_config = InstallConfig(
                python=install_config_data.get('python', '3.9'),
                install=install_config_data.get('install', 'pip install -e .'),
                test_cmd=install_config_data.get('test_cmd', 'pytest'),
                packages=install_config_data.get('packages'),
                pre_install=install_config_data.get('pre_install', []),
                reqs_path=install_config_data.get('reqs_path', []),
                env_yml_path=install_config_data.get('env_yml_path', []),
                pip_packages=install_config_data.get('pip_packages', []),
                env_vars=install_config_data.get('env_vars', {})
            )
            
            # Parse LLM scores from meta
            meta = item.get('meta', {})
            llm_score_data = meta.get('llm_score', {})
            llm_score = LLMScore(
                difficulty_score=llm_score_data.get('difficulty_score', 1),
                issue_text_score=llm_score_data.get('issue_text_score', 1),
                test_score=llm_score_data.get('test_score', 1)
            )
            
            # Parse created_at
            created_at_str = item.get('created_at', '')
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except ValueError:
                    created_at = datetime.now()
            else:
                created_at = datetime.now()
            
            return TaskInstance(
                instance_id=item['instance_id'],
                repo=item['repo'],
                base_commit=item['base_commit'],
                version=item.get('version', '1.0'),
                created_at=created_at,
                problem_statement=item['problem_statement'],
                patch=item['patch'],
                test_patch=item['test_patch'],
                fail_to_pass=item.get('FAIL_TO_PASS', []),
                pass_to_pass=item.get('PASS_TO_PASS', []),
                install_config=install_config,
                llm_score=llm_score,
                license_name=item.get('license_name', 'Unknown'),
                hints_text=item.get('hints_text'),
                requirements=item.get('requirements'),
                environment=item.get('environment')
            )
        except KeyError as e:
            logger.warning(f"Missing required field in task item: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing task item: {e}")
            return None
    
    def get_task(self, instance_id: str) -> Optional[TaskInstance]:
        """Get a specific task by instance ID."""
        return self._tasks.get(instance_id)
    
    def get_all_tasks(self) -> list[TaskInstance]:
        """Get all loaded tasks."""
        return list(self._tasks.values())
    
    def filter_tasks(
        self,
        max_difficulty: Optional[int] = None,
        max_files_edited: Optional[int] = None,
        min_created_date: Optional[datetime] = None,
        max_created_date: Optional[datetime] = None,
        repos: Optional[list[str]] = None,
        max_fail_to_pass_tests: Optional[int] = None,
        require_high_quality: bool = False
    ) -> list[TaskInstance]:
        """
        Filter tasks based on various criteria.
        
        Args:
            max_difficulty: Maximum difficulty score (1-3)
            max_files_edited: Maximum number of files in solution
            min_created_date: Minimum creation date
            max_created_date: Maximum creation date
            repos: List of repository names to include
            max_fail_to_pass_tests: Maximum F2P test count
            require_high_quality: Only include high-quality tasks
            
        Returns:
            List of filtered tasks
        """
        filtered = []
        
        for task in self._tasks.values():
            # Difficulty filter
            if max_difficulty and task.llm_score.difficulty_score > max_difficulty:
                continue
            
            # Date filters
            if min_created_date and task.created_at < min_created_date:
                continue
            if max_created_date and task.created_at > max_created_date:
                continue
            
            # Repository filter
            if repos and task.repo not in repos:
                continue
            
            # Test count filter
            if max_fail_to_pass_tests and task.num_fail_to_pass > max_fail_to_pass_tests:
                continue
            
            # Quality filter
            if require_high_quality and not task.llm_score.is_high_quality():
                continue
            
            filtered.append(task)
        
        return filtered
    
    def get_benchmark_subset(
        self,
        max_tasks: int = 294,
        max_difficulty: int = 2,
        max_files_edited: int = 3,
        max_fail_to_pass_tests: int = 50,
        min_year: int = 2025
    ) -> list[TaskInstance]:
        """
        Get the benchmark subset using SWE-rebench paper criteria.
        
        The paper uses these filters for the 294-task benchmark:
        - Created in 2025
        - Difficulty < 3
        - Files edited <= 3
        - F2P tests <= 50
        - Problem statement 16-1000 words
        - English language
        
        Args:
            max_tasks: Maximum number of tasks to return
            max_difficulty: Maximum difficulty score
            max_files_edited: Maximum files in solution
            max_fail_to_pass_tests: Maximum F2P test count
            min_year: Minimum year for task creation
            
        Returns:
            List of benchmark tasks
        """
        min_date = datetime(min_year, 1, 1)
        
        filtered = self.filter_tasks(
            max_difficulty=max_difficulty,
            min_created_date=min_date,
            max_fail_to_pass_tests=max_fail_to_pass_tests
        )
        
        # Additional filters from paper
        final = []
        for task in filtered:
            # Word count filter (16-1000 words)
            word_count = len(task.problem_statement.split())
            if word_count < 16 or word_count > 1000:
                continue
            
            final.append(task)
            
            if len(final) >= max_tasks:
                break
        
        return final
    
    def iter_tasks(self, batch_size: int = 10) -> Iterator[list[TaskInstance]]:
        tasks = list(self._tasks.values())
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    def get_statistics(self) -> dict:
        """Get statistics about the loaded dataset."""
        if not self._tasks:
            return {"total_tasks": 0}
        
        tasks = list(self._tasks.values())
        
        # Calculate various statistics
        difficulty_counts = {1: 0, 2: 0, 3: 0}
        repos = set()
        total_f2p = 0
        total_p2p = 0
        
        for task in tasks:
            difficulty_counts[task.llm_score.difficulty_score] = \
                difficulty_counts.get(task.llm_score.difficulty_score, 0) + 1
            repos.add(task.repo)
            total_f2p += task.num_fail_to_pass
            total_p2p += len(task.pass_to_pass)
        
        return {
            "total_tasks": len(tasks),
            "unique_repos": len(repos),
            "difficulty_distribution": difficulty_counts,
            "avg_fail_to_pass_tests": total_f2p / len(tasks) if tasks else 0,
            "avg_pass_to_pass_tests": total_p2p / len(tasks) if tasks else 0,
        }
    
    def save_to_json(self, filepath: str) -> None:
        """Save current tasks to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tasks to serializable format
        data = []
        for task in self._tasks.values():
            task_dict = {
                'instance_id': task.instance_id,
                'repo': task.repo,
                'base_commit': task.base_commit,
                'version': task.version,
                'created_at': task.created_at.isoformat(),
                'problem_statement': task.problem_statement,
                'patch': task.patch,
                'test_patch': task.test_patch,
                'FAIL_TO_PASS': task.fail_to_pass,
                'PASS_TO_PASS': task.pass_to_pass,
                'install_config': asdict(task.install_config),
                'meta': {'llm_score': asdict(task.llm_score)},
                'license_name': task.license_name,
                'hints_text': task.hints_text,
                'requirements': task.requirements,
                'environment': task.environment
            }
            data.append(task_dict)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} tasks to {filepath}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset has been loaded."""
        return self._loaded
    
    def __len__(self) -> int:
        return len(self._tasks)
    
    def __iter__(self):
        return iter(self._tasks.values())
