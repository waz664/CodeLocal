from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import AgentConfig, load_config, save_config
from .openai_client import OpenAIClient, OpenAIClientError
from .tools import ToolContext, command_preview, execute_tool, tool_schemas


SYSTEM_PROMPT = """You are CodeLocal, a local coding and systems assistant running inside a terminal.

You are connected to a local model, so be disciplined:
- Prefer small, verifiable steps over large speculative rewrites.
- Use tools to inspect files before editing.
- Use make_dir for directory creation instead of shell mkdir.
- When changing code, make minimal coherent edits and then run focused checks if available.
- For shell commands, explain why the command is needed in one short sentence before calling the tool.
- For Windows tasks, use PowerShell-friendly commands when the environment is Windows.
- For Linux tasks, use POSIX shell commands when possible.
- For Linux Python launcher scripts, detect Python with `command -v python3 >/dev/null 2>&1` first, then `command -v python >/dev/null 2>&1`; store the chosen command in a variable and execute it.
- Use write_python_launcher when asked to create or modify run.sh so it automatically chooses python3 or python.
- Use write_python_requirements when asked to create requirements.txt or add Python dependencies such as Flask.
- When asked for run.sh to install dependencies automatically, call write_python_launcher with install_requirements=true so it uses a local virtual environment instead of system pip.
- Use render_template when asked to create files from a known template, including flask_chat_api and python_gitignore.
- Do not claim you changed or tested something unless a tool result confirms it.
- Do not print file contents unless you read them from a tool result in the current turn.
- If a command fails, inspect the error and adjust once or twice; then explain the blocker.
- Avoid repeated tool calls with the same arguments unless the state has changed.
- When done, summarize changed files, commands run, and any unresolved risks.
- Use the structured tool-calling API when you need tools. Do not write textual tool calls in prose.

Tool rules:
- read_file output includes line numbers; use exact text with replace_text.
- write_file and replace_text modify files.
- make_dir creates directories.
- write_python_launcher creates executable Python run scripts.
- write_python_requirements creates requirements.txt files.
- render_template creates files from reusable templates.
- run_command executes in the current working directory.
- git_status is safe for checking repo state before and after edits.
- change_dir changes only the agent working directory, not the user's shell.
"""


HELP = """Commands:
  /help                         Show this help.
  /exit                         Exit.
  /config                       Show current configuration.
  /set KEY VALUE                Set config: base_url, model, api_key, temperature, top_p,
                                max_tokens, max_turns, command_timeout, approval_mode,
                                allow_outside_root, show_tool_io.
  /save                         Save current configuration.
  /models                       List endpoint models.
  /tools                        List available tools.
  /cd PATH                      Change agent working directory.
  /pwd                          Show agent working directory.
  /clear                        Clear conversation history.
  /system                       Show the system prompt.
  /mode ask|auto                Shortcut for approval_mode.

Approval modes:
  ask                           Ask before file writes, replacements, and commands.
  auto                          Allow tool calls without prompting.
"""


class AgentSession:
    def __init__(self, config: AgentConfig, root: Path):
        self.config = config
        self.client = OpenAIClient(config)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_ctx = ToolContext(root=root, config=config, approval_callback=self._approve)
        self.tools = tool_schemas()

    def _approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        print()
        print(f"Tool approval requested: {tool_name}")
        if tool_name == "run_command":
            print(f"  command: {command_preview(str(args.get('command', '')))}")
            print(f"  cwd:     {self.tool_ctx.cwd}")
        else:
            preview = json.dumps(args, ensure_ascii=False)
            if len(preview) > 1000:
                preview = preview[:1000] + "...[truncated]"
            print(f"  args: {preview}")
        while True:
            answer = input("Allow? [y/N] ").strip().lower()
            if answer in {"y", "yes"}:
                return True
            if answer in {"", "n", "no"}:
                return False

    def ask(self, user_text: str) -> None:
        self.messages.append({"role": "user", "content": user_text})
        empty_responses = 0
        for turn in range(self.config.max_turns):
            try:
                response = self.client.chat(self.messages, self.tools)
            except OpenAIClientError as exc:
                print(f"Endpoint error: {exc}")
                return

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            self.messages.append(message)

            content = message.get("content")
            tool_calls = message.get("tool_calls") or []
            if not tool_calls and content:
                tool_calls = self._tool_calls_from_text(content)

            if not tool_calls and content and self._looks_like_incomplete_tool_call(content):
                if self.config.show_tool_io:
                    print(f"\n[assistant incomplete tool request] {content[:1000]}")
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous tool request was missing required JSON arguments. "
                            "Call the same tool again with complete JSON arguments. "
                            "For example: functions.read_file: {\"path\": \"relative/file\"}"
                        ),
                    }
                )
                continue

            if content and not tool_calls:
                print(content)

            if not tool_calls:
                if content:
                    return
                empty_responses += 1
                if empty_responses > 1:
                    print("The model returned an empty response.")
                    return
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response was empty. Use the available context and answer the request now.",
                    }
                )
                continue

            empty_responses = 0

            if content and self.config.show_tool_io:
                print(f"\n[assistant requested tool via text] {content[:1000]}")

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                raw_args = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if self.config.show_tool_io:
                    print(f"\n[tool] {name}({json.dumps(args, ensure_ascii=False)})")
                result = execute_tool(self.tool_ctx, name, args)
                tool_content = result.content
                if self.config.show_tool_io:
                    status = "ok" if result.ok else "error"
                    print(f"[tool:{status}] {tool_content[:2000]}")
                    if len(tool_content) > 2000:
                        print("[tool] ...truncated in display...")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"local-{turn}"),
                        "content": tool_content,
                    }
                )
        print(f"Stopped after {self.config.max_turns} tool turns. The task may be incomplete.")

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _tool_calls_from_text(self, content: str) -> list[dict[str, Any]]:
        names = {tool["function"]["name"] for tool in self.tools}
        required = {
            tool["function"]["name"]: set(tool["function"].get("parameters", {}).get("required", []))
            for tool in self.tools
        }
        parsed: list[dict[str, Any]] = []
        normalized = content
        for _ in range(8):
            unescaped = html.unescape(normalized)
            if unescaped == normalized:
                break
            normalized = unescaped

        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", normalized, re.DOTALL):
            call = self._parse_text_tool_json(match.group(1), names)
            if call:
                parsed.append(call)

        if parsed:
            return parsed

        json_decoder = json.JSONDecoder(strict=False)
        for match in re.finditer(r"(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*", normalized):
            name = match.group(1)
            if name not in names:
                continue
            remainder = normalized[match.end() :].lstrip()
            args: dict[str, Any] = {}
            parsed_json = False
            if remainder.startswith("{"):
                try:
                    loaded, _ = json_decoder.raw_decode(remainder)
                    if isinstance(loaded, dict):
                        args = loaded
                        parsed_json = True
                except json.JSONDecodeError:
                    parsed_json = False
            if required.get(name) and not parsed_json:
                args = self._infer_missing_args(name)
                if not args:
                    continue
            if not required.get(name).issubset(args):
                continue
            parsed.append(_local_tool_call(name, args))
            return parsed

        for line in normalized.splitlines():
            stripped = line.strip()
            match = re.match(r"^(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", stripped)
            if not match:
                match = re.match(r"^(?:functions\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", stripped)
            if not match:
                continue
            name, raw_args = match.group(1), match.group(2).strip()
            if name not in names:
                continue
            args: dict[str, Any] = {}
            if raw_args:
                try:
                    loaded = json.loads(raw_args, strict=False)
                    if isinstance(loaded, dict):
                        args = loaded
                except json.JSONDecodeError:
                    args = self._parse_loose_arguments(raw_args)
            elif required.get(name):
                args = self._infer_missing_args(name)
            if required.get(name) and not required.get(name).issubset(args):
                continue
            parsed.append(_local_tool_call(name, args))
            break

        return parsed

    def _parse_text_tool_json(self, text: str, names: set[str]) -> dict[str, Any] | None:
        try:
            payload = json.loads(text, strict=False)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        name = str(payload.get("name") or payload.get("function") or "")
        if name.startswith("functions."):
            name = name.split(".", 1)[1]
        if name not in names:
            return None
        args = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return _local_tool_call(name, args)

    def _parse_loose_arguments(self, raw_args: str) -> dict[str, Any]:
        args: dict[str, Any] = {}
        for part in raw_args.split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            args[key.strip()] = value.strip().strip("\"'")
        return args

    def _looks_like_incomplete_tool_call(self, content: str) -> bool:
        normalized = html.unescape(content)
        names = "|".join(re.escape(tool["function"]["name"]) for tool in self.tools)
        return bool(re.search(rf"(?:functions\.)?(?:{names})\s*:\s*$", normalized.strip()))

    def _infer_missing_args(self, name: str) -> dict[str, Any]:
        if name not in {"read_file", "write_file", "write_python_launcher", "write_python_requirements", "render_template"}:
            return {}
        user_text = self._recent_user_text().lower()
        if name == "render_template" and "flask" in user_text and "chat" in user_text:
            return {
                "template": "flask_chat_api",
                "path": "app.py",
                "variables": {"host": "0.0.0.0", "port": 5000, "debug": True},
            }
        if name == "render_template" and ".gitignore" in user_text and "python" in user_text:
            return {"template": "python_gitignore", "path": ".gitignore", "variables": {}}
        if name == "write_file" and ".gitignore" in user_text and "python" in user_text:
            return {}
        if name in {"write_file", "write_python_requirements"} and "requirements" in user_text:
            packages = self._mentioned_python_packages()
            if packages:
                if name == "write_python_requirements":
                    return {"path": "requirements.txt", "packages": packages}
                return {"path": "requirements.txt", "content": "\n".join(packages) + "\n"}
        if name == "write_python_launcher" and "run.sh" in user_text:
            args = {"path": "run.sh", "target": "app.py"}
            if "requirements" in user_text or "dependencies" in user_text or "install" in user_text:
                args["install_requirements"] = True
                args["requirements_path"] = "requirements.txt"
            return args
        mentioned = self._mentioned_path_candidates()
        for candidate in mentioned:
            try:
                path = self.tool_ctx.resolve_path(candidate)
            except Exception:
                continue
            if path.is_file():
                args = {"path": self._tool_path_arg(path)}
                if name == "write_python_launcher":
                    args["target"] = "app.py"
                    if "requirements" in user_text or "dependencies" in user_text or "install" in user_text:
                        args["install_requirements"] = True
                        args["requirements_path"] = "requirements.txt"
                return args
        basenames = [Path(candidate).name for candidate in mentioned if Path(candidate).name]
        matches = []
        for basename in basenames:
            matches.extend(path for path in self.tool_ctx.root.rglob(basename) if path.is_file())
        unique = sorted({path.resolve() for path in matches})
        if len(unique) == 1:
            args = {"path": self._tool_path_arg(unique[0])}
            if name == "write_python_launcher":
                args["target"] = "app.py"
                if "requirements" in user_text or "dependencies" in user_text or "install" in user_text:
                    args["install_requirements"] = True
                    args["requirements_path"] = "requirements.txt"
            return args
        return {}

    def _tool_path_arg(self, path: Path) -> str:
        resolved = path.resolve()
        for base in (self.tool_ctx.cwd, self.tool_ctx.root):
            try:
                return str(resolved.relative_to(base.resolve()))
            except ValueError:
                continue
        return str(resolved)

    def _mentioned_path_candidates(self) -> list[str]:
        text = self._recent_user_text()
        candidates = []
        for match in re.finditer(r"[\w./\\-]+\.[A-Za-z0-9_]+", text):
            candidate = match.group(0).strip(".,:;\"'")
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _recent_user_text(self) -> str:
        text_parts = [
            str(message.get("content") or "")
            for message in self.messages
            if message.get("role") == "user"
        ]
        return "\n".join(text_parts[-3:])

    def _mentioned_python_packages(self) -> list[str]:
        text = self._recent_user_text().lower()
        packages = []
        known = {
            "flask": "flask",
            "fastapi": "fastapi",
            "django": "django",
            "requests": "requests",
            "numpy": "numpy",
            "pandas": "pandas",
            "pytest": "pytest",
        }
        for token, package in known.items():
            if token in text and package not in packages:
                packages.append(package)
        return packages


def _local_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"local-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def parse_value(existing: Any, raw: str) -> Any:
    if isinstance(existing, bool):
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError("expected true/false")
    if isinstance(existing, int):
        return int(raw)
    if isinstance(existing, float):
        return float(raw)
    return raw


def handle_command(session: AgentSession, line: str) -> bool:
    parts = line.strip().split(maxsplit=2)
    command = parts[0].lower()
    if command == "/exit":
        return False
    if command == "/help":
        print(HELP)
        return True
    if command == "/config":
        print(json.dumps(asdict(session.config), indent=2))
        return True
    if command == "/save":
        save_config(session.config)
        print("Configuration saved.")
        return True
    if command == "/models":
        try:
            print(json.dumps(session.client.list_models(), indent=2))
        except OpenAIClientError as exc:
            print(f"Endpoint error: {exc}")
        return True
    if command == "/tools":
        for tool in session.tools:
            fn = tool["function"]
            print(f"{fn['name']}: {fn['description']}")
        return True
    if command == "/pwd":
        print(session.tool_ctx.cwd)
        return True
    if command == "/cd":
        if len(parts) < 2:
            print("Usage: /cd PATH")
            return True
        result = execute_tool(session.tool_ctx, "change_dir", {"path": parts[1]})
        print(result.content)
        return True
    if command == "/clear":
        session.clear()
        print("Conversation cleared.")
        return True
    if command == "/system":
        print(SYSTEM_PROMPT)
        return True
    if command == "/mode":
        if len(parts) < 2 or parts[1] not in {"ask", "auto"}:
            print("Usage: /mode ask|auto")
            return True
        session.config.approval_mode = parts[1]
        print(f"approval_mode = {session.config.approval_mode}")
        return True
    if command == "/set":
        if len(parts) < 3:
            print("Usage: /set KEY VALUE")
            return True
        key, raw = parts[1], parts[2]
        if not hasattr(session.config, key):
            print(f"Unknown config key: {key}")
            return True
        try:
            current = getattr(session.config, key)
            value = parse_value(current, raw)
            if key == "approval_mode" and value not in {"ask", "auto"}:
                raise ValueError("approval_mode must be ask or auto")
            setattr(session.config, key, value)
            print(f"{key} = {value}")
        except Exception as exc:
            print(f"Could not set {key}: {exc}")
        return True
    print(f"Unknown command: {command}. Try /help.")
    return True


def banner(session: AgentSession) -> str:
    return textwrap.dedent(
        f"""
        CodeLocal
        endpoint: {session.config.base_url}
        model:    {session.config.model}
        root:     {session.tool_ctx.root}
        cwd:      {session.tool_ctx.cwd}
        mode:     {session.config.approval_mode}

        Type /help for commands, /exit to quit.
        """
    ).strip()


def repl(session: AgentSession) -> int:
    print(banner(session))
    while True:
        try:
            line = input("\nCodeLocal> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line.strip():
            continue
        if line.startswith("/"):
            if not handle_command(session, line):
                return 0
            continue
        session.ask(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeLocal local coding assistant")
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt")
    parser.add_argument("--root", default=os.getcwd(), help="Workspace root")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--auto", action="store_true", help="Auto-approve commands and file edits")
    args = parser.parse_args(argv)

    config = load_config()
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model
    if args.auto:
        config.approval_mode = "auto"

    session = AgentSession(config=config, root=Path(args.root))
    if args.prompt:
        prompt = " ".join(args.prompt)
        if prompt.startswith("/"):
            handle_command(session, prompt)
        else:
            session.ask(prompt)
        return 0
    return repl(session)


if __name__ == "__main__":
    raise SystemExit(main())
