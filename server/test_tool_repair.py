"""The two ways a tool call came back wrong end to end, and their repairs.

Both were seen through a real client on 2026-09-12. Neither is recoverable by
the model: the client rejects the call, the model retries, the errors pile into
the context, and after six attempts the reasoning degenerated into a loop.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DSV41_MOCK", "1")
from server.app import State  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "ask_user",
    "parameters": {"type": "object",
                   "properties": {"questions": {"type": "array"}},
                   "required": ["questions"]}}}]
SCH = State._schemas(TOOLS)
fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"     got  {got!r}\n     want {want!r}")
        fails.append(name)


# 1. the whole object wrapped in a parameter the schema never declared
check("unwraps an invented `arguments` wrapper",
      State._repair_args("ask_user", {"arguments": '{"questions": [{"q": "which?"}]}'}, SCH),
      {"questions": [{"q": "which?"}]})
check("unwraps it when already a dict",
      State._repair_args("ask_user", {"arguments": {"questions": [1]}}, SCH),
      {"questions": [1]})

# 2. parameter names the schema does not have
check("drops parameters the schema does not declare",
      State._repair_args("ask_user", {"questions": [1], "extra": 2}, SCH),
      {"questions": [1]})

# and the things it must NOT touch
check("leaves a correct call alone",
      State._repair_args("ask_user", {"questions": [1]}, SCH), {"questions": [1]})
check("leaves an unknown tool alone",
      State._repair_args("mystery", {"whatever": 1}, SCH), {"whatever": 1})
check("does not unwrap when the inner keys do not fit either",
      State._repair_args("ask_user", {"nope": '{"other": 1}'}, SCH), {})

# 3. the invalid escape the model emits: a raw backslash before a normal letter
check("repairs an invalid \\H escape",
      State._loads_lenient(r'{"a": "C:\Home"}'), {"a": r"C:\Home"})
check("leaves valid escapes intact",
      State._loads_lenient(r'{"a": "line\nbreak", "b": "q\"q"}'), {"a": "line\nbreak", "b": 'q"q'})
try:
    State._loads_lenient('{"a": ')
    check("still raises on genuinely broken json", False, True)
except json.JSONDecodeError:
    check("still raises on genuinely broken json", True, True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
