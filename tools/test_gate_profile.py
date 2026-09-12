"""The gate has to fail the outputs that actually came out of a starved keep-set.

Every failing sample below is a real shape, not an invented one: the `* { }`
style block repeated to the cap, "Let me write." forever with an answer of
length zero, a title written with runs of U+2011, a last paragraph that is one
sentence over and over. A gate that passes these is worse than no gate, because
it is evidence that the keep-set is sound.

The other half of the job is not failing GOOD output. A repetition rule strict
enough to catch four-word think-block cycles will flag three similar event
handlers if it is written carelessly, and a corrupted-run rule will flag `===`
and `---` in every page ever written. So each check gets a passing sample too,
and the passing samples are deliberately the awkward ones.

No network, no torch, no model: the checks are pure functions of the text.
"""
import ast
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_profile as G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def passes(check_name, text, want=None):
    ok, why = G.CHECKS[check_name](text, want)
    return ok, why


# =============================================================================
# samples
# =============================================================================

GOOD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tic-Tac-Toe</title>
<style>
  :root { --bg: #101418; --fg: #e8eef4; --line: #2b3440; --win: #7fd1b9; }
  body { margin: 0; background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; }
  main { display: grid; place-items: center; min-height: 100vh; gap: 16px; }
  .grid { display: grid; grid-template-columns: repeat(3, 72px); grid-auto-rows: 72px; gap: 6px; }
  .cell { background: var(--line); border: 0; border-radius: 8px; font-size: 32px; color: inherit; }
  .cell:focus-visible { outline: 2px solid var(--win); outline-offset: 2px; }
  .cell[disabled] { cursor: default; opacity: 0.7; }
  #status { min-height: 1.4em; font-size: 18px; letter-spacing: 0.02em; }
  .reset { padding: 8px 18px; border-radius: 999px; border: 1px solid var(--line); color: inherit; }
</style>
</head>
<body>
<main>
  <div id="status">X to play</div>
  <div class="grid" id="grid"></div>
  <button class="reset" id="reset">New game</button>
</main>
<script>
  const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  let board, turn;

  function reset() {
    board = Array(9).fill("");
    turn = "X";
    render();
  }

  function winner() {
    for (const [a, b, c] of LINES) {
      if (board[a] && board[a] === board[b] && board[b] === board[c]) return board[a];
    }
    return board.every(Boolean) ? "draw" : null;
  }

  function play(i) {
    if (board[i] || winner()) return;
    board[i] = turn;
    turn = turn === "X" ? "O" : "X";
    render();
  }

  function render() {
    const grid = document.getElementById("grid");
    grid.replaceChildren(...board.map((value, i) => {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.textContent = value;
      cell.addEventListener("click", () => play(i));
      return cell;
    }));
    const won = winner();
    document.getElementById("status").textContent =
      won === "draw" ? "Drawn game" : won ? won + " wins" : turn + " to play";
  }

  document.getElementById("reset").addEventListener("click", reset);
  reset();
</script>
</body>
</html>
"""

# The one that started this. Empty rules, then the same <style> block again --
# which is also why the page ends up with two of them.
STYLE_LOOP = """<!doctype html>
<html>
<head>
<style>
* { }
.board { }
.cell { }
* { }
</style>
<style>
* { }
.board { }
.cell { }
* { }
</style>
</head>
<body><div id="board"></div></body>
</html>
"""

GOOD_CSS = """```css
:root {
  --card-bg: #ffffff;
  --card-fg: #16202b;
  --card-radius: 12px;
  --card-pad: 20px;
  --card-ring: #3a7afe;
}
.card {
  background: var(--card-bg);
  color: var(--card-fg);
  border-radius: var(--card-radius);
  padding: var(--card-pad);
  box-shadow: 0 1px 2px rgba(0,0,0,.08);
  transition: transform .15s ease;
}
.card:hover { transform: translateY(-2px); }
.card:focus-visible { outline: 2px solid var(--card-ring); outline-offset: 3px; }
@media (prefers-color-scheme: dark) {
  :root { --card-bg: #131922; --card-fg: #e6edf5; }
}
```"""

EMPTY_CSS = ".card { }\n.card:hover { }\n.card__title { }\n.card__body { }\n"

GOOD_JS = """```js
function debounce(fn, wait) {
  let timer = null;
  let lastArgs = null;
  let lastThis = null;

  const debounced = function (...args) {
    lastArgs = args;
    lastThis = this;
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn.apply(lastThis, lastArgs);
    }, wait);
  };

  debounced.cancel = function () {
    clearTimeout(timer);
    timer = null;
    lastArgs = null;
  };

  debounced.flush = function () {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
    fn.apply(lastThis, lastArgs);
  };

  return debounced;
}

const onResize = debounce(() => console.log(window.innerWidth), 150);
window.addEventListener("resize", onResize);
```"""

# Control flow only: to a regex `if (ready) {` looks exactly like a definition,
# and this is what the exclusion list is for.
BRANCHES_ONLY_JS = """```js
if (ready) {
  start();
}
for (let i = 0; i < 10; i++) {
  step(i);
}
while (pending) {
  drain();
}
switch (mode) {
  case "a": break;
}
```"""

GOOD_PY = """```python
import argparse
import os


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                yield path, os.stat(path, follow_symlinks=False).st_size
            except OSError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    for path, size in sorted(walk(args.root), key=lambda p: -p[1])[: args.n]:
        print(f"{size:>12,}  {path}")


if __name__ == "__main__":
    main()
```"""

PLACEHOLDER_PY = """```python
import os


def walk(root):
    ...


def main():
    ...
```"""

GOOD_SQL = """```sql
SELECT c.name,
       o.id,
       o.placed_at,
       SUM(o.total) OVER (PARTITION BY c.id ORDER BY o.placed_at) AS running_total
FROM orders o
JOIN customers c ON c.id = o.customer_id
QUALIFY ROW_NUMBER() OVER (PARTITION BY c.id ORDER BY o.placed_at DESC) <= 3;
```"""

NO_WINDOW_SQL = "SELECT c.name, o.id FROM orders o JOIN customers c ON c.id = o.customer_id;"

GOOD_YAML = """```yaml
x-common: &common
  environment:
    LOG_LEVEL: info
    TZ: UTC
  healthcheck:
    test: ["CMD", "curl", "-fsS", "http://localhost:8080/healthz"]
    interval: 10s
services:
  api:
    <<: *common
    image: example/api:1.4.0
  worker:
    <<: *common
    image: example/worker:1.4.0
  gateway:
    <<: *common
    image: example/gateway:1.4.0
```"""

NO_ANCHOR_YAML = """```yaml
services:
  api:
    image: example/api:1.4.0
    environment:
      LOG_LEVEL: info
  worker:
    image: example/worker:1.4.0
    environment:
      LOG_LEVEL: info
  gateway:
    image: example/gateway:1.4.0
    environment:
      LOG_LEVEL: info
```"""

GOOD_TS = """```ts
export function groupBy<T, K extends keyof T>(
  items: readonly T[],
  key: K,
): Record<string, T[]> {
  const out: Record<string, T[]> = {};
  for (const item of items) {
    const bucket = String(item[key]);
    (out[bucket] ??= []).push(item);
  }
  return out;
}
```"""

ANY_TS = """```ts
export function groupBy(items: any[], key: any): any {
  const out = {};
  return out;
}
```"""

GOOD_PROSE = """A cache earns its keep only when the work it avoids is larger than the work it
adds, and a hit rate says nothing about either quantity. Each lookup costs a
hash, a lock and a round trip to wherever the entry lives, and that cost is paid
on every request, including the nineteen in twenty that hit.

The miss is where the arithmetic turns. If the backing store is fast and the
cache sits across a network, five misses in a hundred can cost more than the
ninety-five hits saved, because the miss path now runs twice: once to discover
the absence and once to fetch. Systems in that regime get faster when the cache
is removed.
"""

ONE_PARAGRAPH = "A cache is only worth its cost when the work avoided exceeds the work added."

# The 5,000-character answer whose last paragraph was one sentence, again and
# again. The sentence is long enough that the universal n-gram window tiles it.
REPEATED_TAIL = GOOD_PROSE + "\n" + ("The cache must be warmed before the first request arrives "
                                     "or the latency budget is spent on the miss path. " * 5)

# Shorter than the n-gram window, so only the prose rule can see it.
SHORT_REPEAT = """Caches trade memory for time, and the trade is not always good.

The cache must be warm. Requests arrive faster than entries expire, so the
working set never settles. The cache must be warm.
"""

FRENCH = """Un cache très efficace peut malgré tout ralentir un système, parce que chaque
requête paie le coût de la recherche avant de savoir si elle sera servie. Cette
dépense est constante, tandis que le gain dépend de ce qui est évité.

Nous avons mesuré ce cas avec un magasin local rapide et un cache distant. Les
défauts de cache coûtaient alors deux allers-retours au lieu d'un, et le
système allait plus vite sans cache du tout.
"""

# Asked for French, answered in Italian: the documented failure, and the one a
# structural check cannot see. Written long enough to clear the prose shape, so
# the only thing left to fail it is the language itself.
ITALIAN_INSTEAD = """Una cache molto efficiente può comunque rendere un sistema più lento. Ogni
richiesta paga il costo della ricerca prima di sapere se sarà servita. Questo
costo è costante e non dipende dal risultato.

Il guadagno invece dipende da quello che viene evitato. Quando il negozio
locale è molto veloce il guadagno sparisce del tutto. Abbiamo misurato lo
stesso caso con una cache remota. Il sistema andava più veloce senza.
"""

ARABIC = """الذاكرة المؤقتة ذات نسبة الإصابة العالية قد تجعل النظام أبطأ، لأن كل طلب يدفع
تكلفة البحث قبل أن يعرف إن كان سيُخدم من الذاكرة أم لا. هذه التكلفة ثابتة.

أما الفائدة فتعتمد على العمل الذي تم تفاديه. وعندما يكون المخزن المحلي سريعًا
والذاكرة المؤقتة بعيدة عبر الشبكة، فإن النظام يصبح أسرع بدونها تمامًا.
"""

# The `general` profile's answer to an Arabic prompt: not Arabic at all.
LATIN_COLLAGE = """Pamięć podręczna e um sistema mai rapid, dar nu intotdeauna melhor.

O cache pode ser mais lento, jeżeli o armazenamento local este rapid si reteaua
adauga un koszt na każde zapytanie, care nu poate fi evitat.
"""

JAPANESE = """ヒット率が高いキャッシュでも、システムが遅くなることがあります。理由は単純で、
すべてのリクエストが検索の費用を先に払うからです。この費用は一定です。

一方で得られる利益は、避けられた作業の大きさに依存します。ローカルの保存先が
十分に速く、キャッシュがネットワークの向こうにある場合、利益は消えてしまいます。
"""

# Han without kana is Chinese, and answering a Japanese prompt in Chinese is the
# neighbour-language drift a thin keep-set produces.
CHINESE_INSTEAD = """命中率很高的缓存仍然可能让系统变慢，因为每个请求都要先付出查找的代价。

这个代价是固定的，而收益取决于被避免的工作量。当本地存储足够快时，收益就消失了。
"""

GOOD_SHELL = """```bash
#!/usr/bin/env bash
set -euo pipefail
IFS=$'\\n\\t'

root="${1:?usage: $0 DIR}"
find "$root" -type f -size +100M -printf '%s\\t%p\\n' \\
  | sort -rn \\
  | head -n 10 \\
  | awk -F'\\t' '{ printf "%.1f GB\\t%s\\n", $1/1073741824, $2 }'
```"""

GOOD_LATEX = r"""```latex
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\title{Cache cost}
\begin{document}
\maketitle
\section{Model}
\label{sec:model}
The expected cost per request is
\begin{equation}
\label{eq:cost}
C = p\,c_{\text{hit}} + (1-p)\,c_{\text{miss}}.
\end{equation}
\section{Measurements}
Equation~\ref{eq:cost} is evaluated in the table below.
\begin{tabular}{lr}
\toprule
hit rate & cost \\
\midrule
0.95 & 1.4 \\
\bottomrule
\end{tabular}
\end{document}
```"""

GOOD_GO = """```go
package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

type createReq struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

func Handle(store Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var in createReq
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			writeErr(w, http.StatusBadRequest, "malformed body")
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		id, err := store.Create(ctx, in.Name, in.Email)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, "could not create")
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"id": id})
	}
}
```"""

# =============================================================================
# the domain checks
# =============================================================================

check("html_page passes a real page", *passes("html_page", GOOD_HTML, ["<title", "grid"]))
ok, why = passes("html_page", STYLE_LOOP)
check("html_page fails the `* { }` style loop", not ok, why)
ok, why = passes("html_page", GOOD_HTML.replace("<!doctype html>\n", ""))
check("html_page fails a page with no doctype", not ok, why)

check("css_block passes a real component", *passes("css_block", GOOD_CSS,
                                                   ["--", "prefers-color-scheme"]))
ok, why = passes("css_block", EMPTY_CSS)
check("css_block fails rules with nothing in them", not ok, why)

check("js_function passes a debounce", *passes("js_function", GOOD_JS, ["cancel", "flush"]))
ok, why = passes("js_function", BRANCHES_ONLY_JS)
check("js_function is not fooled by `if (x) {`", not ok, why)

check("python_script passes a walker", *passes("python_script", GOOD_PY, ["argparse", "except"]))
ok, why = passes("python_script", PLACEHOLDER_PY)
check("python_script fails `...` bodies", not ok, why)

check("sql_query passes a windowed join", *passes("sql_query", GOOD_SQL))
ok, why = passes("sql_query", NO_WINDOW_SQL)
check("sql_query fails a join with no window function", not ok, why)

check("yaml_doc passes anchors and aliases", *passes("yaml_doc", GOOD_YAML))
ok, why = passes("yaml_doc", NO_ANCHOR_YAML)
check("yaml_doc fails copy-paste instead of an anchor", not ok, why)
ok, why = passes("yaml_doc", GOOD_YAML.replace("  api:", "\tapi:"))
check("yaml_doc fails a tab in the indentation", not ok, why)

check("ts_generic passes a constrained generic", *passes("ts_generic", GOOD_TS, ["Record<", "keyof"]))
ok, why = passes("ts_generic", ANY_TS)
check("ts_generic fails `any`", not ok, why)

check("code_block passes a Go handler",
      *passes("code_block", GOOD_GO, ["package ", "func ", "err != nil", "http."]))
ok, why = passes("code_block", GOOD_GO, ["package ", "func ", "err != nil", "database/sql"])
check("code_block fails when a required construct is absent", not ok, why)

check("shell_pipeline passes a quoted pipeline",
      *passes("shell_pipeline", GOOD_SHELL, ["set -", "IFS", "find"]))
UNQUOTED_SHELL = GOOD_SHELL.replace('"$root"', "$root").replace('"${1:?usage: $0 DIR}"', "$1")
ok, why = passes("shell_pipeline", UNQUOTED_SHELL)
check("shell_pipeline fails an unquoted expansion", not ok, why)

check("latex_doc passes a compilable article", *passes("latex_doc", GOOD_LATEX,
                                                       ["\\section", "\\label"]))
ok, why = passes("latex_doc", GOOD_LATEX.replace("\\end{document}", ""))
check("latex_doc fails a truncated document", not ok, why)

check("prose passes two real paragraphs", *passes("prose", GOOD_PROSE))
ok, why = passes("prose", ONE_PARAGRAPH)
check("prose fails a one-line answer", not ok, why)
ok, why = passes("prose", SHORT_REPEAT)
check("prose fails a sentence repeated under the n-gram window", not ok, why)

FR_MARKERS = ["qui", "pour", "dans", "avec", "nous", "cette", "parce"]
check("prose_markers passes French", *passes("prose_markers", FRENCH, FR_MARKERS))
ok, why = passes("prose_markers", ITALIAN_INSTEAD, FR_MARKERS)
check("prose_markers fails Italian answered to a French prompt", not ok, why)

check("prose_script passes Arabic", *passes("prose_script", ARABIC, "arabic"))
ok, why = passes("prose_script", LATIN_COLLAGE, "arabic")
check("prose_script fails the Latin-script collage", not ok, why)
check("prose_script passes Japanese", *passes("prose_script", JAPANESE, "kana"))
ok, why = passes("prose_script", CHINESE_INSTEAD, "kana")
check("prose_script fails Chinese answered to a Japanese prompt", not ok, why)

check("numeric_answer passes the trap answer",
      *passes("numeric_answer", "The ball costs $0.05, not the ten cents it looks like.",
              ["0.05", "5 cents"]))
ok, why = passes("numeric_answer", "The ball costs $0.10 and the bat costs $1.00.",
                 ["0.05", "5 cents"])
check("numeric_answer fails the obvious wrong number", not ok, why)

# =============================================================================
# the universal checks
# =============================================================================

check("clean output has nothing universal against it",
      G.universal("Let me lay out the grid first, then the win lines.", GOOD_HTML, "stop", True) == [],
      str(G.universal("Let me lay out the grid first, then the win lines.", GOOD_HTML, "stop", True)))

bad = G.universal("thinking", GOOD_PROSE, "length", True)
check("finish_reason other than stop is a failure", any("length" in b for b in bad), "; ".join(bad))

note = ("[stopped: the model began repeating itself and was cut off before it produced an "
        "answer. Lower the reasoning effort, or turn thinking off.]")
bad = G.universal("I keep. I write.", note, "length", True)
check("the server's degeneration note is a failure",
      any("cut the generation off" in b for b in bad), "; ".join(bad))

bad = G.universal("I keep. I write. " * 20, GOOD_PROSE, "stop", True)
check("a think-block cycle is caught in the reasoning",
      any(b.startswith("reasoning loops") for b in bad), "; ".join(bad))

bad = G.universal("plan", "I'll write the code now. " + "Let me write. " * 40, "length", True)
check("the `Let me write.` loop is caught in the answer",
      any(b.startswith("answer loops") for b in bad), "; ".join(bad))

bad = G.universal("plan", REPEATED_TAIL, "stop", True)
check("the repeated-paragraph tail is caught",
      any(b.startswith("answer loops") for b in bad), "; ".join(bad))

# The model corrupts a word, then loops trying to repair it. The run rule sees
# the first symptom, the n-gram rule sees the second.
corrupt = "<h1>Tic‑‑‑‑Tac‑‑‑‑Toe</h1>"
bad = G.universal("plan", GOOD_PROSE + "\n\n" + corrupt, "stop", True)
check("a run of U+2011 inside a word is caught",
      any("corrupted run" in b for b in bad), "; ".join(bad))
check("the same shape in ASCII is caught",
      G.corrupt_run("<h1>Tic----Tac----Toe</h1>") is not None)
repair = "I meant color-scheme, not color-s-s-mode. " * 6
bad = G.universal("plan", GOOD_PROSE + "\n\n" + repair, "stop", True)
check("the loop the model enters repairing a corrupted token is caught",
      any(b.startswith("answer loops") for b in bad), "; ".join(bad))

check("`===` and `...` in real code are not corrupted runs",
      G.corrupt_run(GOOD_HTML) is None, str(G.corrupt_run(GOOD_HTML)))
check("a markdown rule is not a corrupted run", G.corrupt_run("text\n\n---------\n\nmore") is None)
check("a box-drawing rule is not a corrupted run",
      G.corrupt_run("| a | b |\n──────────") is None)
check("three blank lines are not a corrupted run", G.corrupt_run("a\n\n\n\nb") is None)
# A numeric range and a writer's ellipsis are welded between alphanumerics and are not runs;
# `overflow 6...10` on a ring-buffer answer was the false positive that found this.
check("a numeric range with an ellipsis is not a corrupted run",
      G.corrupt_run("push 1..5, overflow 6...10 returns false") is None)
check("an ellipsis between words is not a corrupted run", G.corrupt_run("wait...no, that fires twice") is None)
check("five or more welded dots still are", G.corrupt_run("wait.....no") is not None)

bad = G.universal("I thought about the grid, the win lines and the reset button at length.",
                  "", "stop", True)
check("reasoning with an empty answer is named think-exit",
      any(b.startswith("think-exit") for b in bad), "; ".join(bad))
bad = G.universal("", "", "stop", False)
check("an empty answer with thinking off is just empty",
      any(b == "empty answer" for b in bad), "; ".join(bad))
bad = G.universal("", GOOD_PROSE, "stop", True)
check("thinking on with no reasoning at all is a failure",
      any("reasoning is empty" in b for b in bad), "; ".join(bad))
check("thinking off does not require reasoning",
      G.universal("", GOOD_PROSE, "stop", False) == [])

check("the n-gram rule tolerates repeated boilerplate below three copies",
      G.repeated_ngram(("document.getElementById('a').addEventListener('click', function () "
                        "{ send('a'); }); ") * 2) is None)

# =============================================================================
# the suite
# =============================================================================

allp = [p for prompts in G.PROMPTS.values() for p in prompts] + G.ALWAYS
check("every prompt names a check that exists",
      all(p["check"] in G.CHECKS for p in allp),
      ", ".join(sorted({p["check"] for p in allp if p["check"] not in G.CHECKS})))
check("prompt names are unique", len({p["name"] for p in allp}) == len(allp))
check("every prompt carries text", all(p.get("prompt", "").strip() for p in allp))

prompts, silent = G.suite(["html", "css", "nosuchtopic"])
check("a suite is the union of its topics' prompts, plus the always-on ones",
      {p["name"] for p in prompts} == {"html-page", "css-card", "reason-bat-ball", "reason-machines"},
      ", ".join(sorted(p["name"] for p in prompts)))
check("a topic with no prompts is reported, not silently dropped", silent == ["nosuchtopic"])


def catalogue() -> set:
    """The topic names corpus/fetch_topics.py can gather, read without importing
    it -- a topic added to the corpus with no prompt here would otherwise be
    gated by nothing at all and still come out green."""
    tree = ast.parse(open(os.path.join(ROOT, "corpus/fetch_topics.py")).read())
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("LANGS", "DOMAINS", "CODE"):
            out |= {k.value for k in node.value.keys}
    return out


cat = catalogue()
check("the topic catalogue was found", len(cat) >= 30, f"{len(cat)} topics")
check("every catalogue topic has at least one prompt",
      not (cat - set(G.PROMPTS)), f"no prompts for: {', '.join(sorted(cat - set(G.PROMPTS)))}")

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    f.write('{"css": [{"name": "x", "prompt": "y", "check": "no_such_check"}]}')
    bad_file = f.name
try:
    G.read_prompts_file(bad_file)
    check("a prompts file naming an unknown check is refused", False)
except ValueError as e:
    check("a prompts file naming an unknown check is refused", "no_such_check" in str(e))
finally:
    os.unlink(bad_file)

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
