
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from models import (
    TaskInstance, InstallConfig, LLMScore, ModelConfig,
    EvaluationRun, EvaluationResult, BenchmarkResult,
    EvaluationStatus, LeaderboardEntry
)
from dataset_loader import DatasetLoader

class TestModels:
    
    def test_llm_score_high_quality(self):
        high_quality = LLMScore(difficulty_score=1, issue_text_score=1, test_score=1)
        assert high_quality.is_high_quality() is True
        
        low_quality = LLMScore(difficulty_score=3, issue_text_score=2, test_score=3)
        assert low_quality.is_high_quality() is False
    
    def test_install_config_defaults(self):
        config = InstallConfig(
            python="3.9",
            install="pip install -e .",
            test_cmd="pytest"
        )
        
        assert config.python == "3.9"
        assert config.pre_install == []
        assert config.pip_packages == []
    
    def test_task_instance_properties(self):
        task = TaskInstance(
            instance_id="test__repo-1",
            repo="test/repo",
            base_commit="abc123",
            version="1.0",
            created_at=datetime.now(),
            problem_statement="Test issue",
            patch="diff...",
            test_patch="diff...",
            fail_to_pass=["test1", "test2"],
            pass_to_pass=["test3", "test4", "test5"],
            install_config=InstallConfig(python="3.9", install="pip install .", test_cmd="pytest"),
            llm_score=LLMScore(1, 1, 1),
            license_name="MIT"
        )
        
        assert task.total_tests == 5
        assert task.num_fail_to_pass == 2
    
    def test_evaluation_result_metrics(self):
        runs = [
            EvaluationRun(
                run_id=f"run{i}",
                task_id="task1",
                model_id="model1",
                status=EvaluationStatus.COMPLETED,
                started_at=datetime.now(),
                resolved=(i < 3)  # 3 out of 5 resolved
            )
            for i in range(5)
        ]
        
        result = EvaluationResult(
            task_id="task1",
            model_id="model1",
            num_runs=5,
            num_resolved=3,
            runs=runs
        )
        
        assert result.resolution_rate == 0.6
        assert result.pass_at_k is True
    
    def test_benchmark_result_summary(self):
        results = [
            EvaluationResult(
                task_id=f"task{i}",
                model_id="model1",
                num_runs=5,
                num_resolved=3 if i < 7 else 0,  # 7 tasks resolved
                runs=[]
            )
            for i in range(10)
        ]
        
        benchmark = BenchmarkResult(
            model_id="model1",
            model_name="Test Model",
            benchmark_name="SWE-rebench",
            total_tasks=10,
            num_runs_per_task=5,
            results=results,
            started_at=datetime.now()
        )
        
        assert benchmark.resolved_count == 7  # Tasks with >50% resolution
        assert benchmark.pass_at_k_count == 7  # Tasks solved at least once


class TestDatasetLoader:
    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create a sample data file."""
        data = [
            {
                "instance_id": "test__repo-1",
                "repo": "test/repo",
                "base_commit": "abc123",
                "version": "1.0",
                "created_at": "2025-03-15T10:00:00Z",
                "problem_statement": "Test issue description",
                "patch": "diff --git a/file.py...",
                "test_patch": "diff --git a/test.py...",
                "FAIL_TO_PASS": ["test1"],
                "PASS_TO_PASS": ["test2"],
                "meta": {
                    "llm_score": {
                        "difficulty_score": 1,
                        "issue_text_score": 1,
                        "test_score": 1
                    }
                },
                "install_config": {
                    "python": "3.9",
                    "install": "pip install -e .",
                    "test_cmd": "pytest"
                },
                "license_name": "MIT"
            }
        ]
        
        filepath = tmp_path / "test_data.json"
        with open(filepath, "w") as f:
            json.dump(data, f)
        
        return filepath
    
    def test_load_from_json(self, sample_data_path):
        loader = DatasetLoader()
        count = loader.load_from_json(str(sample_data_path))
        
        assert count == 1
        assert loader.is_loaded is True
        assert len(loader) == 1
    
    def test_get_task(self, sample_data_path):
        loader = DatasetLoader()
        loader.load_from_json(str(sample_data_path))
        
        task = loader.get_task("test__repo-1")
        assert task is not None
        assert task.instance_id == "test__repo-1"
        assert task.repo == "test/repo"
    
    def test_filter_tasks_by_difficulty(self, sample_data_path):
        loader = DatasetLoader()
        loader.load_from_json(str(sample_data_path))
        
        # Should include task with difficulty 1
        filtered = loader.filter_tasks(max_difficulty=2)
        assert len(filtered) == 1
        
        # Should exclude task with difficulty 1
        filtered = loader.filter_tasks(max_difficulty=0)
        assert len(filtered) == 0
    
    def test_get_statistics(self, sample_data_path):
        loader = DatasetLoader()
        loader.load_from_json(str(sample_data_path))
        
        stats = loader.get_statistics()
        
        assert stats["total_tasks"] == 1
        assert stats["unique_repos"] == 1
        assert 1 in stats["difficulty_distribution"]


class TestEvaluationEngine:
    def test_calculate_pass_at_k(self):
        from evaluation_engine import calculate_pass_at_k
        
        # All correct - should be 1.0
        assert calculate_pass_at_k(n=5, c=5, k=1) == 1.0
        
        # None correct - should be 0.0
        assert calculate_pass_at_k(n=5, c=0, k=1) == 0.0
        
        # 3 correct out of 5, k=1
        result = calculate_pass_at_k(n=5, c=3, k=1)
        assert 0.5 < result < 0.7


class TestLeaderboard:
    @pytest.fixture
    def temp_leaderboard(self, tmp_path):
        from evaluation_engine import Leaderboard
        
        filepath = tmp_path / "leaderboard.json"
        return Leaderboard(str(filepath))
    
    def test_add_result(self, temp_leaderboard):
        result = BenchmarkResult(
            model_id="model1",
            model_name="Test Model",
            benchmark_name="SWE-rebench",
            total_tasks=10,
            num_runs_per_task=5,
            results=[
                EvaluationResult(
                    task_id=f"task{i}",
                    model_id="model1",
                    num_runs=5,
                    num_resolved=3,
                    runs=[]
                )
                for i in range(10)
            ],
            started_at=datetime.now()
        )
        
        entry = temp_leaderboard.add_result(result)
        
        assert entry.rank == 1
        assert entry.model_name == "Test Model"
    
    def test_leaderboard_ranking(self, temp_leaderboard):
        # Add two results with different resolution rates
        for name, resolved in [("Model A", 3), ("Model B", 5)]:
            result = BenchmarkResult(
                model_id=f"{name.lower().replace(' ', '_')}",
                model_name=name,
                benchmark_name="SWE-rebench",
                total_tasks=10,
                num_runs_per_task=5,
                results=[
                    EvaluationResult(
                        task_id=f"task{i}",
                        model_id=f"{name.lower().replace(' ', '_')}",
                        num_runs=5,
                        num_resolved=resolved,
                        runs=[]
                    )
                    for i in range(10)
                ],
                started_at=datetime.now()
            )
            temp_leaderboard.add_result(result)
        
        entries = temp_leaderboard.get_leaderboard()
        
        # Model B should be ranked first (higher resolution rate)
        assert entries[0].model_name == "Model B"
        assert entries[1].model_name == "Model A"
    
    def test_contamination_filtering(self, temp_leaderboard):
        result = BenchmarkResult(
            model_id="model1",
            model_name="Test Model",
            benchmark_name="SWE-rebench",
            total_tasks=10,
            num_runs_per_task=5,
            results=[
                EvaluationResult(
                    task_id=f"task{i}",
                    model_id="model1",
                    num_runs=5,
                    num_resolved=3,
                    runs=[]
                )
                for i in range(10)
            ],
            started_at=datetime.now()
        )
        
        # Mark as contaminated by setting dates
        temp_leaderboard.add_result(
            result,
            model_release_date=datetime(2025, 6, 1),
            task_cutoff_date=datetime(2025, 1, 1)
        )
        
        all_entries = temp_leaderboard.get_leaderboard(include_contaminated=True)
        clean_entries = temp_leaderboard.get_leaderboard(include_contaminated=False)
        
        assert len(all_entries) == 1
        assert len(clean_entries) == 0


class TestAgentInterface:
    def test_agent_context_problem_description(self):
        from agent_interface import AgentContext
        
        task = TaskInstance(
            instance_id="test__repo-1",
            repo="test/repo",
            base_commit="abc123",
            version="1.0",
            created_at=datetime.now(),
            problem_statement="Fix the bug in calculate_sum",
            patch="diff...",
            test_patch="diff...",
            fail_to_pass=["test_sum"],
            pass_to_pass=[],
            install_config=InstallConfig(python="3.9", install="pip install .", test_cmd="pytest"),
            llm_score=LLMScore(1, 1, 1),
            license_name="MIT"
        )
        
        context = AgentContext(task=task, repo_path="/workspace/repo")
        description = context.get_problem_description()
        
        assert "Fix the bug" in description
        assert "test/repo" in description
        assert "test_sum" in description


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
