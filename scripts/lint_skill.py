"""Lint skills/reasoning-failure-analysis/SKILL.md: frontmatter, length, and tool names."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "reasoning-failure-analysis" / "SKILL.md"
SCHEMAS = ROOT / "tests" / "mcp_server" / "fixtures" / "tool_schemas.json"
MAX_LINES = 150
BACKTICKED_CALL = re.compile(r"`(\w+)\(")
TOOL_LIKE = re.compile(r"\b(list_collections|find_traces|get_trace|check_answer|accuracy_breakdown|[a-z]+_[a-z_]+)\(")


def tool_names() -> set[str]:
    data = json.loads(SCHEMAS.read_text())
    if isinstance(data, dict):
        return set(data)
    return {item["name"] for item in data}


def main() -> int:
    errors: list[str] = []
    text = SKILL.read_text()
    lines = text.splitlines()
    if len(lines) >= MAX_LINES:
        errors.append(f"{SKILL.name} has {len(lines)} lines; the limit is under {MAX_LINES}")
    if not lines or lines[0].strip() != "---" or "---" not in [x.strip() for x in lines[1:]]:
        errors.append("missing YAML frontmatter delimited by --- lines")
        body = text
    else:
        end = [x.strip() for x in lines[1:]].index("---") + 1
        front: dict[str, str] = {}
        for line in lines[1:end]:
            key, sep, value = line.partition(":")
            if sep and not line.startswith((" ", "\t")):
                front[key.strip()] = value.strip()
        if front.get("name") != "reasoning-failure-analysis":
            errors.append(f"frontmatter name must be 'reasoning-failure-analysis', got {front.get('name')!r}")
        if not front.get("description"):
            errors.append("frontmatter description is missing or empty (it must be on one line)")
        body = "\n".join(lines[end + 1 :])
    known = tool_names()
    used = set(BACKTICKED_CALL.findall(body)) | set(TOOL_LIKE.findall(body))
    for name in sorted(used - known):
        errors.append(f"`{name}(` is not a tool in {SCHEMAS.name}; known tools: {', '.join(sorted(known))}")
    for e in errors:
        print(f"skill-lint: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
