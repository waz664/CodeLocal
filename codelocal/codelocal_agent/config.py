from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8090/v1"
DEFAULT_MODEL = "qwen2.5-coder-7b-q4km"


def config_dir() -> Path:
    if os.name == "nt":
        root = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(root) / "CodeLocal"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "codelocal"


@dataclass
class AgentConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048
    max_turns: int = 12
    command_timeout: int = 120
    approval_mode: str = "ask"
    allow_outside_root: bool = False
    show_tool_io: bool = True

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> AgentConfig:
    path = config_path()
    if not path.exists():
        return AgentConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AgentConfig()
    valid = {field.name for field in AgentConfig.__dataclass_fields__.values()}
    filtered: dict[str, Any] = {k: v for k, v in data.items() if k in valid}
    return AgentConfig(**filtered)


def save_config(config: AgentConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
