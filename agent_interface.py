
import logging
import re
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Generator, List

from models import TaskInstance, ModelConfig, AgentAction

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent execution."""
    task: TaskInstance
    repo_path: str
    command_history: List[AgentAction] = field(default_factory=list)


# EXACT system prompt from SWE-rebench (text-based mode)
SWEREBENCH_SYSTEM_PROMPT = '''# SETTING

You are an autonomous programming agent. Your goal is to resolve the issue given to you.
You are given access to a terminal environment with some special tools to make your job easier.
You must use the terminal to gain information about the codebase, find or modify the relevant files in order to resolve the issue.
In this environment, all standard unix commands (e.g. grep, sed, echo etc.) will be available to you.
However, the environment does NOT support interactive session commands that expect user input (e.g. vim), so please do not invoke them, it will result in an error.
You can however create python scripts and run them, this is very useful to reproduce errors or test something.
If some packages are missing, you can install them using an appropriate package manager (e.g. pip, apt, etc.).
Do not ask any questions to the environment, it's an automated system that can only execute your commands.
When you are satisfied with the changes you made, you should explicitly submit them using a special command. This will terminate your session.

# SPECIAL TOOLS

In addition to standard unix commands you can use special tools described below.
Please note that some of these commands work with the currently open file, so pay attention to what file is open.

Usage: create [OPTIONS] FILENAME
  Creates and opens a new filename with the given name.

Usage: edit [OPTIONS] LINE_RANGE [REPLACEMENT_TEXT]
  Replaces lines in LINE_RANGE=<start_line>:<end_line> (inclusive) with the
  given text in the currently open or specified file. The REPLACEMENT_TEXT
  will be used as provided including all whitespaces, so make sure your
  indentation is correct.
  To input multiple lines into REPLACEMENT_TEXT, you may use the following
  syntax:
  ```
  edit 1:1 << 'EOF'
  Line1
  Line2
  EOF
  ```
  You can also provide the file to edit via `--file` option.
  ```
  edit --file path/to/file 1:1 "Your Replacement Text Here"
  ```
  Please note that THIS COMMAND REQUIRES PROPER INDENTATION. If you'd like to
  add the line '        print(x)' you must fully write that out, with all
  those spaces before the print statement!
Options:
  --file PATH  The file to edit. (If not provided, edits the currently open
                file)

Usage: goto [OPTIONS] LINE_NUMBER
  Navigates the current window to a given line in the currently open file.

Usage: open [OPTIONS] [FILE] [LINE_NUMBER]
  Opens the file at the given path in the editor. If file is not specified,
  the last open file will be reopened. If line_number is provided, the current
  window will move to show that line.

Usage: replace [OPTIONS] SEARCH REPLACE
  Replaces a given string with another string in the currently open file.
Options:
  --replace-all  Replace all occurrences of the SEARCH text.

Usage: scroll_down [OPTIONS]
  Scroll down the window in the currently open file and output its contents.

Usage: scroll_up [OPTIONS]
  Scroll up the window in the currently open file and output its contents.

Usage: search_file [OPTIONS] SEARCH_TERM [FILE]
  Searches for SEARCH_TERM in file. If FILE is not provided, searches in the currently open file.

Usage: submit [OPTIONS]
  Submits your current code and terminates the session.


# ENVIRONMENT RESPONSE

At the very beginning the environment will provide you with an issue description. In response to every command that you invoke,
the environment will give you the output of the command or an error message followed by a shell prompt.
The shell prompt will be formatted as follows:
```
(Current directory: <current_dir>, current file: <current_file>) bash-$
```
so that you always know what the current directory is and what file is currently open.

# YOUR RESPONSE

Your response should consist of two parts: reasoning (arbitrary text) and command (surrounded by triple ticks and a special 'command' keyword).
Your response should always include A SINGLE reasoning and A SINGLE command as in the following examples:

<response example>
First I'll start by using ls to see what files are in the current directory. I'll look at all files including hidden ones.
```command
ls -a
```
</response example>

<response example>
Let's search the file `models.py` for the UserEntity class definition.
```command
search_file "class UserEntity" models.py
```
</response example>

Everything you include in the reasoning will be made available to you when generating further commands.
If you'd like to issue two command blocks in a single response, PLEASE DO NOT DO THAT! THIS WILL RESULT IN AN ERROR.

# HANDLING TESTS

* You can run existing tests to validate the changes you made or make sure you didn't break anything.
* If missing packages or some environment misconfiguration is preventing you from running the tests, you can install missing packages or fix the environment.
* However UNDER NO CIRCUMSTANCES should you modify existing tests or add new tests to the repository.
  This will lead to an error in the system that evaluates your performance. Instead, you can just create a temporary script, use it to test changes and remove it before submitting.
* If existing tests break because they need to be updated to reflect the changes you made, just ignore it. Evaluation system will not take it into account.
* However if existing tests are broken because your fix is incorrect, you should fix your code and make sure all tests pass before submitting the change.

# USEFUL ADVICE

* As a first step, it might be a good idea to explore the repository to familiarize yourself with its structure.
* You should also come up with a rough plan of how to resolve the issue and put it into your reasoning.
* If the issue description reports some error, create a script to reproduce the error and run it to confirm the error. THIS IS USUALLY A VERY GOOD FIRST STEP!
* Edit the source code of the repo to resolve the issue
* Rerun your reproduce script and confirm that the error is fixed! THIS IS QUITE IMPORTANT!
* Think about edge cases and make sure your fix handles them as well.
* Make sure your solution is general enough and not hardcoded to the specific cases reported in the issue description.
* It might be a good idea to ensure that existing tests in the repository pass before submitting the change. Otherwise it's easy to break existing functionality.
'''


def get_initial_observation(task: TaskInstance) -> str:
    """Get the initial observation shown to the agent."""
    return f'''# ISSUE DESCRIPTION

{task.problem_statement}

# ADDITIONAL ADVICE

Since you are given a git repository, you can use git commands to simplify your work.
For example, if you made a mistake and want to revert your changes, you can use `git checkout <file>` to restore the file to its original state.
You can also reset all changes in the repository using `git reset --hard` command.
Additionally, you can use `git stash` and `git stash pop` to temporarily save your changes and restore them later.
Finally, you can see the changes that you've made (which will be submitted when you call `submit`) using `git status` or `git diff` commands.
However you don't need to use `git add` or `git commit` before submitting your changes. In fact,
`submit` only submits currently unstaged changes because it uses `git diff` to get the changes that need to be submitted.

# CHECKLIST

Before submitting your solution, please go over the following checklist and make sure you've done everything:
- [ ] If an error was reported in the issue description, I have successfully reproduced it.
- [ ] If an error was reported in the issue description, I have confirmed that my fix resolves the error.
- [ ] I have thought about edge cases and made sure my fix handles them as well.
- [ ] I have run existing tests in the repository that might have been affected by the change I made and confirmed that they pass.
I want you to list every bullet from this checklist and write your reasoning for why you think you did it or didn't need to.

Repository has been uploaded and your shell is currently at the repository root. Time to solve the issue!

(Current directory: /workspace/repo, current file: none) bash-$'''


class BaseAgent(ABC):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.context: Optional[AgentContext] = None
        self.current_step = 0
    
    @abstractmethod
    def generate_action(self, observation: str) -> tuple[str, str]:
        pass
    
    def initialize(self, context: AgentContext) -> None:
        self.context = context
        self.current_step = 0
    
    def run(self, executor: 'EnvironmentExecutor', max_steps: int = 50) -> Generator[AgentAction, None, Optional[str]]:
        if not self.context:
            raise RuntimeError("Agent not initialized")
        
        observation = get_initial_observation(self.context.task)
        
        for step in range(max_steps):
            self.current_step = step + 1
            
            try:
                reasoning, command = self.generate_action(observation)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                observation = f"Error: {e}\nPlease provide a command in the format:\n```command\nyour_command_here\n```"
                continue
            
            if not command:
                logger.warning(f"Step {self.current_step}: No command extracted")
                observation = "Error: No command found. Please provide exactly one command in a code block like:\n```command\nls -la\n```"
                continue
            
            logger.debug(f"Step {self.current_step}: {command[:60]}...")
            
            # Check for submit
            if command.strip().lower() == "submit":
                logger.info(f"Agent submitted at step {self.current_step}")
                yield AgentAction(
                    action_type="submit", content="submit",
                    timestamp=datetime.now(), observation="Submitted"
                )
                return executor.get_diff()
            
            # Execute command
            observation = executor.execute(command)
            
            yield AgentAction(
                action_type="bash", content=command,
                timestamp=datetime.now(), observation=observation[:2000]
            )
        
        logger.warning(f"Max steps reached ({max_steps})")
        return None


class EnvironmentExecutor:
    def __init__(self, container: str, docker_env: 'DockerEnvironment'):
        self.container = container
        self.docker_env = docker_env
        self.current_file: Optional[str] = None
        self.file_content: List[str] = []
        self.view_start = 0
        self.view_size = 100
    
    def execute(self, command: str) -> str:
        if not command:
            return "Error: Empty command"
        
        command = command.strip()
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == "open":
                return self._open(args)
            elif cmd == "edit":
                return self._edit(args)
            elif cmd == "create":
                return self._create(args)
            elif cmd == "search_file":
                return self._search(args)
            elif cmd == "goto":
                return self._goto(args)
            elif cmd == "scroll_up":
                return self._scroll_up()
            elif cmd == "scroll_down":
                return self._scroll_down()
            elif cmd == "replace":
                return self._replace(args)
            else:
                return self._bash(command)
        except Exception as e:
            return f"Error: {e}\n{self._prompt()}"
    
    def _bash(self, cmd: str) -> str:
        try:
            output = self.docker_env._exec_in_container(
                self.container, f"cd /workspace/repo && {cmd}", check=False
            )
            return (output or "(no output)") + self._prompt()
        except Exception as e:
            return f"Error: {e}\n{self._prompt()}"
    
    def _open(self, args: str) -> str:
        """Open file."""
        parts = args.strip().split()
        if not parts:
            return f"Usage: open <file> [line]\n{self._prompt()}"
        
        filepath = parts[0]
        line = int(parts[1]) if len(parts) > 1 else 1
        
        try:
            content = self.docker_env._exec_in_container(
                self.container, f"cat /workspace/repo/{filepath}"
            )
            self.current_file = filepath
            self.file_content = content.split('\n')
            self.view_start = max(0, line - 1)
            return self._file_view()
        except Exception as e:
            return f"Error opening file: {e}\n{self._prompt()}"
    
    def _edit(self, args: str) -> str:
        """Edit file."""
        # Handle --file option
        file_match = re.match(r'--file\s+(\S+)\s+(.*)', args, re.DOTALL)
        if file_match:
            filepath = file_match.group(1)
            args = file_match.group(2)
            # Open the file first
            self._open(filepath)
        
        if not self.current_file:
            return f"No file open. Use 'open <file>' first.\n{self._prompt()}"
        
        # Parse line range and replacement
        match = re.match(r'(\d+):(\d+)\s*(.*)', args, re.DOTALL)
        if not match:
            match = re.match(r'(\d+)\s*(.*)', args, re.DOTALL)
            if match:
                start = end = int(match.group(1))
                replacement = match.group(2)
            else:
                return f"Usage: edit <start>:<end> <text>\n{self._prompt()}"
        else:
            start, end = int(match.group(1)), int(match.group(2))
            replacement = match.group(3)
        
        # Handle heredoc
        heredoc = re.search(r"<<\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1", replacement, re.DOTALL)
        if heredoc:
            replacement = heredoc.group(2)
        
        # Apply edit
        new_lines = replacement.split('\n') if replacement else []
        self.file_content = self.file_content[:start-1] + new_lines + self.file_content[end:]
        
        # Write back
        content = '\n'.join(self.file_content)
        self.docker_env._copy_to_container(
            self.container, content, f"/workspace/repo/{self.current_file}"
        )
        
        return f"File updated.\n{self._file_view()}"
    
    def _create(self, args: str) -> str:
        filepath = args.strip()
        if not filepath:
            return f"Usage: create <filepath>\n{self._prompt()}"
        
        self.docker_env._exec_in_container(
            self.container, f"touch /workspace/repo/{filepath}"
        )
        self.current_file = filepath
        self.file_content = []
        return f"[File: {filepath} (0 lines total)]\n{self._prompt()}"
    
    def _search(self, args: str) -> str:
        parts = args.split(maxsplit=1)
        term = parts[0].strip('"\'') if parts else ""
        filepath = parts[1] if len(parts) > 1 else (self.current_file or ".")
        
        if not term:
            return f"Usage: search_file <term> [file]\n{self._prompt()}"
        
        try:
            result = self.docker_env._exec_in_container(
                self.container,
                f"cd /workspace/repo && grep -rn '{term}' {filepath} | head -50",
                check=False
            )
            return (result or "No matches found") + self._prompt()
        except Exception as e:
            return f"Error: {e}\n{self._prompt()}"
    
    def _goto(self, args: str) -> str:
        if not self.current_file:
            return f"No file open.\n{self._prompt()}"
        try:
            line = int(args.strip())
            self.view_start = max(0, line - self.view_size // 2)
            return self._file_view()
        except ValueError:
            return f"Usage: goto <line>\n{self._prompt()}"
    
    def _scroll_up(self) -> str:
        if not self.current_file:
            return f"No file open.\n{self._prompt()}"
        self.view_start = max(0, self.view_start - self.view_size)
        return self._file_view()
    
    def _scroll_down(self) -> str:
        if not self.current_file:
            return f"No file open.\n{self._prompt()}"
        self.view_start = min(max(0, len(self.file_content) - self.view_size), self.view_start + self.view_size)
        return self._file_view()
    
    def _replace(self, args: str) -> str:
        if not self.current_file:
            return f"No file open.\n{self._prompt()}"
        
        # Parse --replace-all flag
        replace_all = "--replace-all" in args
        args = args.replace("--replace-all", "").strip()
        
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return f"Usage: replace <search> <replace>\n{self._prompt()}"
        
        search, replace = parts[0].strip('"\''), parts[1].strip('"\'')
        content = '\n'.join(self.file_content)
        
        if replace_all:
            content = content.replace(search, replace)
        else:
            content = content.replace(search, replace, 1)
        
        self.file_content = content.split('\n')
        self.docker_env._copy_to_container(
            self.container, content, f"/workspace/repo/{self.current_file}"
        )
        
        return f"File updated.\n{self._file_view()}"
    
    def _file_view(self) -> str:
        if not self.current_file:
            return self._prompt()
        
        end = min(self.view_start + self.view_size, len(self.file_content))
        lines = [f"{i+self.view_start+1}:{self.file_content[i]}" for i in range(self.view_start, end)]
        
        return f"[File: /workspace/repo/{self.current_file} ({len(self.file_content)} lines total)]\n" + \
               '\n'.join(lines) + self._prompt()
    
    def _prompt(self) -> str:
        file_str = f"/workspace/repo/{self.current_file}" if self.current_file else "none"
        return f"\n(Current directory: /workspace/repo, current file: {file_str}) bash-$"
    
    def get_diff(self) -> str:
        return self.docker_env.get_diff_from_container(self.container)


class SimpleReActAgent(BaseAgent):
    def __init__(self, model_config: ModelConfig, api_client: Optional['LLMClient'] = None):
        super().__init__(model_config)
        self.api_client = api_client
        self.history: List[dict] = []
    
    def generate_action(self, observation: str) -> tuple[str, str]:
        if not self.api_client:
            raise RuntimeError("No API client")
        
        self.history.append({"role": "user", "content": observation})
        
        response = self.api_client.generate(
            system_prompt=SWEREBENCH_SYSTEM_PROMPT,
            messages=self.history
        )
        
        self.history.append({"role": "assistant", "content": response})
        
        reasoning, command = self._parse(response)
        return reasoning, command
    
    def _parse(self, response: str) -> tuple[str, str]:
        if not response:
            logger.warning("Empty response from LLM")
            return "", ""
        
        # Log full response for debugging
        logger.debug(f"Full response length: {len(response)}")
        
        # Check if response is ONLY a think block with nothing after
        think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
        if think_match:
            logger.debug(f"Found think block ({len(think_match.group(1))} chars)")
        
        # Remove thinking blocks
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        if not clean:
            logger.warning("Response was only a think block with no command")
            logger.warning(f"Think content preview: {response[:500]}...")
            return "", ""
        
        logger.debug(f"Clean response: {clean[:300]}...")
        
        # Look for ```command block (the OFFICIAL format)
        match = re.search(r'```command\s*\n?(.*?)\n?```', clean, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            logger.debug(f"Found ```command block: {cmd[:100]}")
            return "", cmd
        
        # Fallback: ```bash or ```shell block
        match = re.search(r'```(?:bash|shell|sh)\s*\n?(.*?)\n?```', clean, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            logger.debug(f"Found ```bash block: {cmd[:100]}")
            return "", cmd
        
        # Fallback: any code block
        match = re.search(r'```\s*\n?(.*?)\n?```', clean, re.DOTALL)
        if match:
            cmd = match.group(1).strip()
            # Remove language tags
            lines = cmd.split('\n')
            if lines and lines[0].lower() in ['bash', 'shell', 'sh', 'python', 'command', '']:
                cmd = '\n'.join(lines[1:]).strip()
            if cmd:
                logger.debug(f"Found generic code block: {cmd[:100]}")
                return "", cmd
        
        # Last resort: look for common commands on their own line
        cmd_patterns = [
            r'^(ls\b.*?)$',
            r'^(cat\b.*?)$', 
            r'^(cd\b.*?)$',
            r'^(grep\b.*?)$',
            r'^(find\b.*?)$',
            r'^(open\b.*?)$',
            r'^(edit\b.*?)$',
            r'^(create\b.*?)$',
            r'^(search_file\b.*?)$',
            r'^(submit)$',
            r'^(python\b.*?)$',
            r'^(pip\b.*?)$',
            r'^(git\b.*?)$',
        ]
        for pattern in cmd_patterns:
            m = re.search(pattern, clean, re.MULTILINE)
            if m:
                cmd = m.group(1).strip()
                logger.debug(f"Found standalone command: {cmd[:100]}")
                return "", cmd
        
        logger.warning(f"No command found. Clean response: {clean[:300]}...")
        return "", ""
    
    def initialize(self, context: AgentContext) -> None:
        super().initialize(context)
        self.history = []


# LLM Clients

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, messages: list, **kwargs) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, system_prompt: str, messages: list, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        
        msgs = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=self.model, messages=msgs,
            max_tokens=kwargs.get('max_tokens', 4096),
            temperature=kwargs.get('temperature', 0.0)
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
    
    def generate(self, system_prompt: str, messages: list, **kwargs) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        response = client.messages.create(
            model=self.model, system=system_prompt, messages=messages,
            max_tokens=kwargs.get('max_tokens', 4096)
        )
        return response.content[0].text


class OpenRouterClient(LLMClient):
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-r1", max_retries: int = 5):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.input_tokens = 0
        self.output_tokens = 0
    
    def generate(self, system_prompt: str, messages: list, **kwargs) -> str:
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        }
        
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=300.0) as client:
                    response = client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers, json=payload
                    )
                    
                    if response.status_code == 429:
                        delay = 10 * (2 ** attempt) + random.uniform(0, 5)
                        logger.warning(f"Rate limited. Waiting {delay:.0f}s...")
                        time.sleep(delay)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if "usage" in data:
                        self.input_tokens += data["usage"].get("prompt_tokens", 0)
                        self.output_tokens += data["usage"].get("completion_tokens", 0)
                    
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return content or ""
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 5 * (2 ** attempt)
                    logger.warning(f"Error: {e}. Retry in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
        
        return ""
    
    def estimate_cost(self) -> dict:
        costs = {
            "deepseek/deepseek-r1": (0.55, 2.19),
            "deepseek/deepseek-chat": (0.14, 0.28),
        }
        i_rate, o_rate = costs.get(self.model, (1.0, 3.0))
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": round((self.input_tokens / 1e6 * i_rate) + (self.output_tokens / 1e6 * o_rate), 4)
        }