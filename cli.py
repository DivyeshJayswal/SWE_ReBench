#!/usr/bin/env python3
"""
SWE-rebench Benchmarking Tool - CLI Interface

Command-line interface for running evaluations and managing the benchmark.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_dataset(args):
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader()
    
    if args.action == "load":
        if args.source == "huggingface":
            count = loader.load_from_huggingface()
            print(f"✓ Loaded {count} tasks from HuggingFace")
        elif args.source.endswith(".json"):
            count = loader.load_from_json(args.source)
            print(f"✓ Loaded {count} tasks from {args.source}")
        elif args.source.endswith(".jsonl"):
            count = loader.load_from_jsonl(args.source)
            print(f"✓ Loaded {count} tasks from {args.source}")
        else:
            print(f"✗ Unknown source format: {args.source}")
            sys.exit(1)
    
    elif args.action == "stats":
        if not loader.is_loaded:
            loader.load_from_huggingface()
        
        stats = loader.get_statistics()
        print("\n📊 Dataset Statistics")
        print("=" * 40)
        print(f"Total Tasks: {stats['total_tasks']:,}")
        print(f"Unique Repositories: {stats['unique_repos']:,}")
        print(f"\nDifficulty Distribution:")
        for level, count in stats['difficulty_distribution'].items():
            print(f"  Level {level}: {count:,}")
        print(f"\nAvg Fail-to-Pass Tests: {stats['avg_fail_to_pass_tests']:.1f}")
        print(f"Avg Pass-to-Pass Tests: {stats['avg_pass_to_pass_tests']:.1f}")
    
    elif args.action == "list":
        if not loader.is_loaded:
            loader.load_from_huggingface()
        
        tasks = loader.filter_tasks(
            max_difficulty=args.max_difficulty,
            repos=[args.repo] if args.repo else None
        )[:args.limit]
        
        print(f"\n📋 Tasks ({len(tasks)} shown)")
        print("=" * 80)
        for task in tasks:
            print(f"\n{task.instance_id}")
            print(f"  Repo: {task.repo}")
            print(f"  Difficulty: {task.llm_score.difficulty_score}/3")
            print(f"  Tests: {task.num_fail_to_pass} F2P, {len(task.pass_to_pass)} P2P")
            print(f"  Problem: {task.problem_statement[:100]}...")


def cmd_evaluate(args):
    from models import ModelConfig
    from dataset_loader import DatasetLoader
    from agent_interface import SimpleReActAgent, OpenAIClient, AnthropicClient, OpenRouterClient
    
    # Initialize components
    loader = DatasetLoader()
    if not loader.is_loaded:
        print("Loading dataset...")
        loader.load_from_huggingface()
    
    # Get tasks
    if args.benchmark:
        tasks = loader.get_benchmark_subset(max_tasks=args.max_tasks)
        print(f"Using benchmark subset: {len(tasks)} tasks")
    else:
        tasks = loader.filter_tasks(max_difficulty=args.max_difficulty)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
        print(f"Using filtered tasks: {len(tasks)} tasks")
    
    # Create model config
    model_config = ModelConfig(
        model_id=args.model_id or args.model,
        name=args.model,
        provider=args.provider,
        temperature=args.temperature,
        max_context_length=args.context_length
    )
    
    # Create LLM client based on provider
    if args.provider == "openai":
        if not args.api_key:
            print("✗ OpenAI API key required (--api-key)")
            sys.exit(1)
        client = OpenAIClient(args.api_key, args.model)
    elif args.provider == "anthropic":
        if not args.api_key:
            print("✗ Anthropic API key required (--api-key)")
            sys.exit(1)
        client = AnthropicClient(args.api_key, args.model)
    elif args.provider == "openrouter":
        if not args.api_key:
            print("✗ OpenRouter API key required (--api-key)")
            sys.exit(1)
        client = OpenRouterClient(args.api_key, args.model)
    else:
        print(f"✗ Unsupported provider: {args.provider}")
        sys.exit(1)
    
    # Initialize evaluation engine
    try:
        from evaluation_engine import EvaluationEngine
        from execution_env import DockerEnvironment
        
        docker_env = DockerEnvironment()
        engine = EvaluationEngine(
            docker_env=docker_env,
            dataset_loader=loader,
            results_dir=args.output_dir,
            num_workers=args.workers
        )
    except Exception as e:
        print(f"✗ Failed to initialize evaluation engine: {e}")
        print("  Make sure Docker is installed and running.")
        sys.exit(1)
    
    print(f"\n🚀 Starting evaluation")
    print(f"  Model: {args.model}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Runs per task: {args.num_runs}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Run evaluation
    result = engine.run_benchmark(
        model_config=model_config,
        llm_client=client,
        tasks=tasks,
        num_runs_per_task=args.num_runs,
        parallel=args.workers > 1
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {result.model_name}")
    print(f"Tasks Evaluated: {result.total_tasks}")
    print(f"Runs per Task: {result.num_runs_per_task}")
    print()
    print(f"Resolved Rate: {result.resolved_rate:.1%}")
    print(f"Pass@{result.num_runs_per_task}: {result.pass_at_k_rate:.1%}")
    print(f"SEM: ±{result.calculate_sem():.2%}")
    print()
    print(f"Tasks Resolved: {result.resolved_count}/{result.total_tasks}")
    print(f"Tasks with Any Success: {result.pass_at_k_count}/{result.total_tasks}")
    
    # Print cost if using OpenRouter
    if args.provider == "openrouter" and hasattr(client, 'estimate_cost'):
        cost = client.estimate_cost()
        print()
        print("💰 Cost Summary:")
        print(f"  Input tokens: {cost['input_tokens']:,}")
        print(f"  Output tokens: {cost['output_tokens']:,}")
        print(f"  Total cost: ${cost['total_cost_usd']:.4f}")


def cmd_estimate(args):
    from agent_interface import OpenRouterClient
    
    estimate = OpenRouterClient.estimate_benchmark_cost(
        model=args.model,
        num_tasks=args.num_tasks,
        num_runs=args.num_runs,
        avg_steps_per_task=args.avg_steps
    )
    
    print("\n💰 Cost Estimation for SWE-rebench Benchmark")
    print("=" * 50)
    print(f"Model: {estimate['model']}")
    print(f"Tasks: {estimate['num_tasks']}")
    print(f"Runs per task: {estimate['num_runs']}")
    print(f"Total agent steps: {estimate['total_steps']:,}")
    print()
    print(f"Estimated input tokens: {estimate['estimated_input_tokens']:,}")
    print(f"Estimated output tokens: {estimate['estimated_output_tokens']:,}")
    print()
    print(f"Input cost: ${estimate['estimated_input_cost_usd']:.2f}")
    print(f"Output cost: ${estimate['estimated_output_cost_usd']:.2f}")
    print(f"TOTAL ESTIMATED COST: ${estimate['estimated_total_cost_usd']:.2f}")
    print()
    print(f"Estimated time: ~{estimate['estimated_time_hours']:.1f} hours")
    print()
    print(f"⚠️  {estimate['note']}")


def cmd_leaderboard(args):
    from evaluation_engine import Leaderboard
    
    leaderboard = Leaderboard(args.file)
    
    if args.action == "show":
        entries = leaderboard.get_leaderboard(
            include_contaminated=not args.clean_only,
            top_n=args.top
        )
        
        if args.format == "markdown":
            print(leaderboard.to_markdown(not args.clean_only))
        else:
            print("\n🏆 SWE-rebench Leaderboard")
            print("=" * 80)
            print(f"{'Rank':<6}{'Model':<30}{'Resolved':<12}{'Pass@5':<10}{'SEM':<10}{'Clean':<8}")
            print("-" * 80)
            
            for e in entries:
                clean = "✓" if not e.is_contaminated else "⚠️"
                print(f"{e.rank:<6}{e.model_name:<30}{e.resolved_rate:>8.1%}{e.pass_at_5:>10.1%}{e.sem:>9.2%}  {clean}")
    
    elif args.action == "export":
        output_path = args.output or "leaderboard.md"
        markdown = leaderboard.to_markdown(not args.clean_only)
        Path(output_path).write_text(markdown)
        print(f"✓ Leaderboard exported to {output_path}")


def cmd_serve(args):
    import uvicorn
    
    print(f"\n🌐 Starting SWE-rebench API server")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="swe-rebench",
        description="SWE-rebench Benchmarking Tool - Evaluate LLM agents on real-world software engineering tasks"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dataset commands
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management")
    dataset_parser.add_argument(
        "action",
        choices=["load", "stats", "list"],
        help="Action to perform"
    )
    dataset_parser.add_argument(
        "--source",
        default="huggingface",
        help="Data source (huggingface or path to JSON/JSONL file)"
    )
    dataset_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of tasks to show (for list action)"
    )
    dataset_parser.add_argument(
        "--max-difficulty",
        type=int,
        choices=[1, 2, 3],
        help="Filter by maximum difficulty"
    )
    dataset_parser.add_argument(
        "--repo",
        help="Filter by repository name"
    )
    dataset_parser.set_defaults(func=cmd_dataset)
    
    # Evaluate commands
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., gpt-4, claude-sonnet-4-20250514, anthropic/claude-3.5-sonnet)"
    )
    eval_parser.add_argument(
        "--model-id",
        help="Custom model ID for tracking"
    )
    eval_parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic", "openrouter", "local"],
        help="Model provider (use 'openrouter' for OpenRouter API)"
    )
    eval_parser.add_argument(
        "--api-key",
        help="API key for the model provider"
    )
    eval_parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per task (default: 5)"
    )
    eval_parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to evaluate"
    )
    eval_parser.add_argument(
        "--max-difficulty",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Maximum task difficulty (default: 2)"
    )
    eval_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use the official benchmark subset (294 tasks)"
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature (default: 0.0)"
    )
    eval_parser.add_argument(
        "--context-length",
        type=int,
        default=128000,
        help="Maximum context length (default: 128000)"
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    eval_parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory for results (default: ./results)"
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Cost estimation command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate benchmark cost")
    estimate_parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model identifier (e.g., anthropic/claude-3.5-sonnet)"
    )
    estimate_parser.add_argument(
        "--num-tasks",
        type=int,
        default=294,
        help="Number of tasks (default: 294)"
    )
    estimate_parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Runs per task (default: 5)"
    )
    estimate_parser.add_argument(
        "--avg-steps",
        type=int,
        default=30,
        help="Average steps per task (default: 30)"
    )
    estimate_parser.set_defaults(func=cmd_estimate)
    
    # Leaderboard commands
    lb_parser = subparsers.add_parser("leaderboard", help="View and manage leaderboard")
    lb_parser.add_argument(
        "action",
        choices=["show", "export"],
        help="Action to perform"
    )
    lb_parser.add_argument(
        "--file",
        default="./leaderboard.json",
        help="Leaderboard data file"
    )
    lb_parser.add_argument(
        "--format",
        choices=["table", "markdown"],
        default="table",
        help="Output format"
    )
    lb_parser.add_argument(
        "--top",
        type=int,
        help="Show only top N entries"
    )
    lb_parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Exclude potentially contaminated results"
    )
    lb_parser.add_argument(
        "--output",
        help="Output file path (for export)"
    )
    lb_parser.set_defaults(func=cmd_leaderboard)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    serve_parser.set_defaults(func=cmd_serve)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()