"""Every kwarg the launcher can hand the engine must be a parameter the engine has.

`start.sh` builds `--engine-kwargs` as JSON from .env, `server/app.py` splats it
into `V41Engine(...)`, and a name that is not in the signature is a TypeError
three minutes into a load -- after the weights are on the GPU. The engine
imports torch, so this reads the signature with `ast` instead and runs anywhere.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def engine_params() -> set:
    tree = ast.parse(open(os.path.join(ROOT, "engine/v41_engine.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "V41Engine":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    a = fn.args
                    return {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs} - {"self"}
    raise AssertionError("V41Engine.__init__ not found")


def launcher_keys(path: str) -> set:
    """The JSON keys the shell script can put in --engine-kwargs."""
    return set(re.findall(r'EK=\"\$EK\\\"([a-z_]+)\\\"', open(os.path.join(ROOT, path)).read()))


params = engine_params()
fails = []
for script in ("start.sh", "scripts/entrypoint.sh"):
    keys = launcher_keys(script)
    if not keys:
        print(f"FAIL {script}: no engine-kwargs keys found -- has the format changed?")
        fails.append(script)
        continue
    missing = sorted(keys - params)
    print(f"{'ok  ' if not missing else 'FAIL'} {script}: {len(keys)} kwargs"
          f"{'' if not missing else '  MISSING FROM V41Engine.__init__: ' + ', '.join(missing)}")
    if missing:
        fails.append(script)

# the engine's own CLI splats its argparse into the same constructor
src = open(os.path.join(ROOT, "engine/v41_engine.py")).read()
call = src[src.index("eng = V41Engine(a.model_dir"):]
cli = set(re.findall(r"^\s*([a-z_]+)=a\.[a-z_]+,", call[:call.index(")\n")], re.M))
missing = sorted(cli - params)
print(f"{'ok  ' if not missing else 'FAIL'} engine CLI: {len(cli)} kwargs"
      f"{'' if not missing else '  MISSING: ' + ', '.join(missing)}")
if missing:
    fails.append("engine CLI")

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
