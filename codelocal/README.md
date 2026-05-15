# CodeLocal

CodeLocal is a small local coding and task assistant CLI for OpenAI-compatible
endpoints. It is designed for local models that are useful but not frontier
scale: the agent takes small steps, uses explicit tools, and keeps approvals
visible.

Default endpoint:

```text
http://127.0.0.1:8090/v1
```

Default model:

```text
qwen2.5-coder-7b-q4km
```

## Run

From this directory:

```bash
python3 -m codelocal_agent
```

One-shot prompt:

```bash
python3 -m codelocal_agent "Inspect this repo and summarize it"
```

Use a different workspace root:

```bash
python3 -m codelocal_agent --root /path/to/project
```

## Interactive Commands

```text
/help
/exit
/config
/set base_url http://192.168.128.113:8090/v1
/set model qwen2.5-coder-7b-q4km
/mode ask
/mode auto
/models
/tools
/cd PATH
/pwd
/clear
/save
```

## Tools

The model can call tools to:

- inspect the environment
- list directories
- read files
- inspect file metadata
- write files
- replace exact text
- search text
- find files
- check git status
- run shell commands
- change the agent working directory

By default, file writes, text replacements, and shell commands ask for approval.
Use `/mode auto` only in a workspace you are comfortable letting the local model
modify.

## Notes

This is intentionally simpler than Claude Code or OpenCode. It favors predictable
behavior with smaller local models over complex orchestration.

Some local endpoints and models do not always emit perfectly structured OpenAI
tool calls. CodeLocal includes a small compatibility parser for common textual
tool-call forms such as `functions.read_file:` followed by JSON arguments.
