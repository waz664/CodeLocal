from __future__ import annotations

import fnmatch
import json
import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import AgentConfig


MAX_TOOL_OUTPUT = 12000
TEXT_EXTENSIONS = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".cmd",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".md",
    ".ps1",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass
class ToolResult:
    ok: bool
    content: str


class ToolError(RuntimeError):
    pass


class ToolContext:
    def __init__(
        self,
        root: Path,
        config: AgentConfig,
        approval_callback: Callable[[str, dict[str, Any]], bool],
    ):
        self.root = root.resolve()
        self.cwd = self.root
        self.config = config
        self.approval_callback = approval_callback

    def resolve_path(self, raw_path: str | None) -> Path:
        if not raw_path:
            return self.cwd
        path = Path(os.path.expanduser(raw_path))
        if not path.is_absolute():
            path = self.cwd / path
        resolved = path.resolve()
        if not self.config.allow_outside_root:
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise ToolError(
                    f"Path is outside the workspace root: {resolved}. "
                    "Use /set allow_outside_root true if you want to allow this."
                ) from exc
        return resolved

    def require_approval(self, tool_name: str, args: dict[str, Any]) -> None:
        if self.config.approval_mode == "auto":
            return
        if not self.approval_callback(tool_name, args):
            raise ToolError("User declined tool call.")


def trim_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n\n...[truncated]...\n\n" + text[-limit // 2 :]


def as_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_environment",
                "description": "Get OS, shell, Python, current directory, workspace root, and agent configuration.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files and directories at a path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_entries": {"type": "integer", "default": 200},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file, optionally with line offsets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "default": 1},
                        "max_lines": {"type": "integer", "default": 240},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_info",
                "description": "Get existence, type, size, and modification time for a path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a UTF-8 text file. Requires approval unless approval mode is auto.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_text",
                "description": "Replace text in a UTF-8 file. Fails if the old text is absent unless replace_all is true and count can be zero.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": "Search text files under a path for a literal string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "glob": {"type": "string", "default": "*"},
                        "max_matches": {"type": "integer", "default": 80},
                    },
                    "required": ["path", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "Find files below a path using a glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string", "default": "*"},
                        "max_results": {"type": "integer", "default": 200},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the current working directory. Requires approval unless approval mode is auto.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Return concise git status for the current workspace without modifying files.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "change_dir",
                "description": "Change the agent working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]


def execute_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> ToolResult:
    try:
        if name == "get_environment":
            return ToolResult(True, _get_environment(ctx))
        if name == "list_dir":
            return ToolResult(True, _list_dir(ctx, args))
        if name == "read_file":
            return ToolResult(True, _read_file(ctx, args))
        if name == "file_info":
            return ToolResult(True, _file_info(ctx, args))
        if name == "write_file":
            ctx.require_approval(name, args)
            return ToolResult(True, _write_file(ctx, args))
        if name == "replace_text":
            ctx.require_approval(name, args)
            return ToolResult(True, _replace_text(ctx, args))
        if name == "search_text":
            return ToolResult(True, _search_text(ctx, args))
        if name == "find_files":
            return ToolResult(True, _find_files(ctx, args))
        if name == "run_command":
            ctx.require_approval(name, args)
            return ToolResult(True, _run_command(ctx, args))
        if name == "git_status":
            return ToolResult(True, _git_status(ctx))
        if name == "change_dir":
            return ToolResult(True, _change_dir(ctx, args))
        return ToolResult(False, f"Unknown tool: {name}")
    except Exception as exc:
        return ToolResult(False, f"{type(exc).__name__}: {exc}")


def _get_environment(ctx: ToolContext) -> str:
    return as_json(
        {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "shell": os.environ.get("SHELL") or os.environ.get("COMSPEC") or "",
            "root": str(ctx.root),
            "cwd": str(ctx.cwd),
            "approval_mode": ctx.config.approval_mode,
            "allow_outside_root": ctx.config.allow_outside_root,
        }
    )


def _list_dir(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args.get("path"))
    max_entries = int(args.get("max_entries") or 200)
    entries = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_entries]:
        entries.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return as_json({"path": str(path), "entries": entries})


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args["path"])
    start_line = max(1, int(args.get("start_line") or 1))
    max_lines = max(1, int(args.get("max_lines") or 240))
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    selected = lines[start_line - 1 : start_line - 1 + max_lines]
    numbered = [f"{start_line + i}: {line}" for i, line in enumerate(selected)]
    return trim_output("\n".join(numbered))


def _file_info(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args["path"])
    if not path.exists():
        return as_json({"path": str(path), "exists": False})
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return as_json(
        {
            "path": str(path),
            "exists": True,
            "type": "dir" if path.is_dir() else "file",
            "size": stat.st_size if path.is_file() else None,
            "modified_epoch": stat.st_mtime,
            "modified_utc": modified,
        }
    )


def _write_file(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return as_json({"written": str(path), "bytes": len(args["content"].encode("utf-8"))})


def _replace_text(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args["path"])
    old = args["old"]
    new = args["new"]
    replace_all = bool(args.get("replace_all", False))
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        raise ToolError("old text was not found")
    if not replace_all and count > 1:
        raise ToolError(f"old text occurs {count} times; set replace_all=true or make old text more specific")
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")
    return as_json({"path": str(path), "replacements": count if replace_all else 1})


def _is_probably_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.suffix == ""


def _search_text(ctx: ToolContext, args: dict[str, Any]) -> str:
    root = ctx.resolve_path(args.get("path"))
    query = args["query"]
    glob = args.get("glob") or "*"
    max_matches = int(args.get("max_matches") or 80)
    matches = []
    for path in root.rglob("*") if root.is_dir() else [root]:
        if len(matches) >= max_matches:
            break
        if not path.is_file() or not fnmatch.fnmatch(path.name, glob) or not _is_probably_text(path):
            continue
        try:
            for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if query in line:
                    matches.append({"path": str(path), "line": idx, "text": line[:500]})
                    if len(matches) >= max_matches:
                        break
        except OSError:
            continue
    return as_json({"matches": matches, "truncated": len(matches) >= max_matches})


def _find_files(ctx: ToolContext, args: dict[str, Any]) -> str:
    root = ctx.resolve_path(args.get("path"))
    pattern = args.get("pattern") or "*"
    max_results = int(args.get("max_results") or 200)
    results = []
    for path in root.rglob(pattern) if root.is_dir() else []:
        if len(results) >= max_results:
            break
        results.append(str(path))
    return as_json({"files": results, "truncated": len(results) >= max_results})


def _run_command(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = args["command"]
    timeout = int(args.get("timeout") or ctx.config.command_timeout)
    completed = subprocess.run(
        command,
        cwd=str(ctx.cwd),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return trim_output(
        as_json(
            {
                "command": command,
                "cwd": str(ctx.cwd),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    )


def _git_status(ctx: ToolContext) -> str:
    completed = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=str(ctx.cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=min(ctx.config.command_timeout, 30),
    )
    return trim_output(
        as_json(
            {
                "cwd": str(ctx.cwd),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    )


def _change_dir(ctx: ToolContext, args: dict[str, Any]) -> str:
    path = ctx.resolve_path(args["path"])
    if not path.is_dir():
        raise ToolError(f"Not a directory: {path}")
    ctx.cwd = path
    return as_json({"cwd": str(ctx.cwd)})


def command_preview(command: str) -> str:
    if os.name == "nt":
        return command
    try:
        return " ".join(shlex.split(command))
    except Exception:
        return command
