
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

from models import TaskInstance

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    passed_tests: List[str] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    execution_time: float = 0.0


class DockerEnvironment:
    CONTAINER_WORKDIR = "/workspace"
    
    def __init__(self, timeout: int = 600, use_cache: bool = True):
        self.timeout = timeout
        self.use_cache = use_cache
        self._verify_docker()
    
    def _verify_docker(self) -> None:
        """Verify Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True, encoding='utf-8', errors='replace', timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Docker not running")
        except FileNotFoundError:
            raise RuntimeError("Docker not installed")
    
    def _run_cmd(self, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run command with encoding handling."""
        return subprocess.run(
            args, capture_output=True, encoding='utf-8', errors='replace',
            timeout=timeout or self.timeout
        )
    
    def _exec_in_container(self, container: str, cmd: str, check: bool = True, timeout: Optional[int] = None) -> str:
        """Execute command in container."""
        result = self._run_cmd(
            ["docker", "exec", container, "bash", "-c", cmd],
            timeout=timeout or self.timeout
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr[:500]}")
        return result.stdout
    
    def _copy_to_container(self, container: str, content: str, path: str) -> None:
        """Copy content to file in container."""
        content = content.replace('\r\n', '\n')
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', newline='\n') as f:
            f.write(content)
            temp = f.name
        try:
            self._run_cmd(["docker", "cp", temp, f"{container}:{path}"])
        finally:
            os.unlink(temp)
    
    def _get_image_tag(self, task: TaskInstance) -> str:
        """Generate image tag for task."""
        repo = task.repo.replace("/", "_").replace("-", "_")
        version = (task.version or "latest").replace(".", "_")
        return f"swe-rebench/{repo}:{version}-{task.base_commit[:12]}"
    
    def _generate_dockerfile(self, task: TaskInstance) -> str:
        """Generate Dockerfile for task."""
        python_ver = task.install_config.python or "3.9"
        
        dockerfile = f"""FROM python:{python_ver}-slim

RUN apt-get update && apt-get install -y \\
    git build-essential curl patch \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR {self.CONTAINER_WORKDIR}

RUN git config --global user.email "agent@swe-rebench.com" && \\
    git config --global user.name "SWE Agent"

"""
        if task.install_config.pre_install:
            for cmd in task.install_config.pre_install:
                dockerfile += f"RUN {cmd}\n"
        
        if task.install_config.pip_packages:
            pkgs = " ".join(task.install_config.pip_packages)
            dockerfile += f"RUN pip install --no-cache-dir {pkgs}\n"
        
        dockerfile += "\nCMD [\"bash\"]\n"
        return dockerfile
    
    def build_image(self, task: TaskInstance, force: bool = False) -> str:
        tag = self._get_image_tag(task)
        
        if self.use_cache and not force:
            result = self._run_cmd(["docker", "image", "inspect", tag])
            if result.returncode == 0:
                logger.info(f"Using cached image: {tag}")
                return tag
        
        logger.info(f"Building image: {tag}")
        
        with tempfile.TemporaryDirectory() as build_dir:
            dockerfile = Path(build_dir) / "Dockerfile"
            with open(dockerfile, 'w') as f:
                f.write(self._generate_dockerfile(task))
            
            result = self._run_cmd(
                ["docker", "build", "-t", tag, "-f", str(dockerfile), build_dir],
                timeout=600
            )
            if result.returncode != 0:
                raise RuntimeError(f"Build failed: {result.stderr[:500]}")
        
        logger.info(f"Built: {tag}")
        return tag
    
    def _apply_patch(self, container: str, patch: str, name: str = "patch") -> bool:
        if not patch or not patch.strip():
            return True
        
        patch = patch.replace('\r\n', '\n')
        path = f"/tmp/{name}.patch"
        self._copy_to_container(container, patch, path)
        
        strategies = [
            f"git apply --whitespace=fix {path}",
            f"git apply --ignore-whitespace {path}",
            f"git apply --3way {path}",
            f"patch -p1 --fuzz=3 < {path}",
        ]
        
        for strat in strategies:
            try:
                self._exec_in_container(container, f"cd /workspace/repo && {strat}")
                logger.info(f"{name} patch applied")
                return True
            except RuntimeError:
                try:
                    self._exec_in_container(container, "cd /workspace/repo && git checkout -- .", check=False)
                except:
                    pass
        return False
    
    def setup_task_environment(self, task: TaskInstance, container_name: Optional[str] = None) -> str:
        if not container_name:
            safe_id = task.instance_id.replace('__', '-').replace('/', '-')[:30]
            container_name = f"swe-agent-{safe_id}"
        
        tag = self.build_image(task)
        
        # Remove existing container
        self._run_cmd(["docker", "rm", "-f", container_name])
        
        # Create container
        result = self._run_cmd([
            "docker", "run", "-d", "--name", container_name,
            "-w", self.CONTAINER_WORKDIR, tag,
            "tail", "-f", "/dev/null"
        ])
        if result.returncode != 0:
            raise RuntimeError(f"Container creation failed: {result.stderr}")
        
        # Clone repo - FULL clone, not shallow (to have all commits)
        self._exec_in_container(
            container_name,
            f"git clone https://github.com/{task.repo}.git repo"
        )
        
        # Checkout base commit
        self._exec_in_container(
            container_name,
            f"cd repo && git checkout {task.base_commit}"
        )
        
        # Remove remote to prevent info leakage
        self._exec_in_container(
            container_name,
            "cd repo && git remote remove origin || true",
            check=False
        )
        
        # Install
        if task.install_config.install:
            try:
                self._exec_in_container(
                    container_name,
                    f"cd repo && {task.install_config.install}",
                    timeout=300,
                    check=False
                )
            except Exception as e:
                logger.warning(f"Install issue: {e}")
        
        # Apply test patch
        if task.test_patch:
            self._apply_patch(container_name, task.test_patch, "test")
        
        return container_name
    
    def get_diff_from_container(self, container: str) -> str:
        try:
            self._exec_in_container(container, "cd /workspace/repo && git add -A", check=False)
            diff = self._exec_in_container(container, "cd /workspace/repo && git diff HEAD", check=False)
            if not diff.strip():
                diff = self._exec_in_container(container, "cd /workspace/repo && git diff", check=False)
            return diff
        except Exception as e:
            logger.error(f"Diff error: {e}")
            return ""
    
    def run_tests(self, container: str, task: TaskInstance, timeout: Optional[int] = None) -> ExecutionResult:
        timeout = timeout or self.timeout
        test_cmd = task.install_config.test_cmd or "pytest --no-header -rA --tb=short"
        
        start = time.time()
        try:
            result = self._run_cmd(
                ["docker", "exec", container, "bash", "-c", f"cd /workspace/repo && {test_cmd}"],
                timeout=timeout
            )
            output = result.stdout + result.stderr
            passed, failed = self._parse_tests(output)
            
            return ExecutionResult(
                success=result.returncode == 0,
                passed_tests=passed, failed_tests=failed,
                stdout=result.stdout, stderr=result.stderr,
                execution_time=time.time() - start
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, error_message=f"Timeout {timeout}s")
        except Exception as e:
            return ExecutionResult(success=False, error_message=str(e))
    
    def _parse_tests(self, output: str) -> Tuple[List[str], List[str]]:
        passed, failed = [], []
        for m in re.finditer(r'(\S+::\S+)\s+PASSED', output):
            passed.append(m.group(1))
        for m in re.finditer(r'(\S+::\S+)\s+FAILED', output):
            failed.append(m.group(1))
        return passed, failed
    
    def verify_solution(self, task: TaskInstance, patch: str) -> Tuple[bool, ExecutionResult]:
        if not patch or not patch.strip():
            return False, ExecutionResult(success=False, error_message="No patch")
        
        container = f"swe-verify-{task.instance_id[:20].replace('__', '-')}"
        logger.info(f"Verifying solution ({len(patch)} chars)")
        
        try:
            tag = self.build_image(task)
            self._run_cmd(["docker", "rm", "-f", container])
            
            self._run_cmd([
                "docker", "run", "-d", "--name", container,
                "-w", self.CONTAINER_WORKDIR, tag,
                "tail", "-f", "/dev/null"
            ])
            
            # Full clone
            self._exec_in_container(container, f"git clone https://github.com/{task.repo}.git repo")
            self._exec_in_container(container, f"cd repo && git checkout {task.base_commit}")
            
            # Install
            if task.install_config.install:
                self._exec_in_container(
                    container, f"cd repo && {task.install_config.install}",
                    timeout=300, check=False
                )
            
            # Apply solution first
            logger.info("Applying solution...")
            if not self._apply_patch(container, patch, "solution"):
                return False, ExecutionResult(success=False, error_message="Patch failed")
            
            # Commit solution
            self._exec_in_container(
                container,
                "cd /workspace/repo && git add -A && git commit -m 'solution' --allow-empty",
                check=False
            )
            
            # Apply test patch
            logger.info("Applying test patch...")
            if task.test_patch:
                self._apply_patch(container, task.test_patch, "test")
            
            # Run tests
            logger.info("Running tests...")
            result = self.run_tests(container, task)
            
            # Check criteria
            f2p = set(task.fail_to_pass)
            p2p = set(task.pass_to_pass)
            passed = set(result.passed_tests)
            failed = set(result.failed_tests)
            
            f2p_ok = f2p.issubset(passed) if f2p else True
            p2p_ok = not p2p.intersection(failed) if p2p else True
            
            resolved = f2p_ok and p2p_ok
            logger.info(f"{'✅ RESOLVED' if resolved else '❌ FAILED'}")
            
            return resolved, result
            
        except Exception as e:
            logger.error(f"Verify error: {e}")
            return False, ExecutionResult(success=False, error_message=str(e))
        finally:
            self._run_cmd(["docker", "rm", "-f", container])
    
    def cleanup_container(self, container: str) -> None:
        self._run_cmd(["docker", "rm", "-f", container])