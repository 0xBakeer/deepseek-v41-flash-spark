"""test_tune_atlas.py -- the `a` key: the legend, the popup, the loopback server.

Three things can quietly stop being true. The key can fall off the legend at a width nobody
tests, which makes a feature nobody can find. The popup can be drawn without the port, which
makes it useless -- the port is the whole message. And the server can bind something other
than a free loopback port, which is either a crash or, much worse, a directory of this
checkout offered to the network.

So the legend is read at the widths the draw tests use, the popup is rendered at all seven of
them, and the server is really started on a real socket, really fetched from, and shut down.
No terminal, no torch, no browser.
"""
import curses
import json
import os
import shutil
import sys
import tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_export as AX  # noqa: E402
import budget as B  # noqa: E402
import tune as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail and not ok else ''}")
    if not ok:
        fails.append(name)


T.C.update({k: 0 for k in ("accent", "good", "warn", "bad", "muted", "bright", "on")})
curses.doupdate = lambda: None

stats = os.path.join(ROOT, "results/keepsets/topics/coverage.json")
index = B.TopicIndex(stats, "saliency") if os.path.exists(stats) else None
host = B.Host("gb10-test", 130.6e9, 118.6e9, True)
SIZES = [(24, 80), (30, 100), (32, 104), (40, 140), (24, 70), (60, 200), (20, 70)]


def state(view="easy", **kw):
    st = T.State(host, index, stats, 0.36, 32768, "cb3",
                 sorted(index.topics)[:3] if index else [], rank="maxmin", source="saliency")
    st.view = view
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def rows(st, h, w):
    g = T._Grid(h, w)
    T.draw(g, st)
    return [g.row(y) for y in range(h)]


# --- the label, exactly, on both screens ------------------------------------
# It is a credit as much as a key, so the wording is fixed and the test says so.
check("the key legend entry is the credit", T.ATLAS_KEY == "a  Weight Atlas by alesha-pro",
      repr(T.ATLAS_KEY))
for view in ("easy", "advanced"):
    for w in (80, 96):
        line = rows(state(view), 30, w)[29]
        check(f"the {view} key line carries it at {w} columns", T.ATLAS_KEY in line, repr(line))
        check(f"  and still fits {w} columns", len(line) <= w, f"{len(line)} chars")

# --- the handle in the title bar --------------------------------------------
# The owner asked for it by name, and it has to survive every width, including the
# narrowest one this tool will draw at all.
for view in ("easy", "advanced"):
    for h, w in SIZES:
        head = rows(state(view), h, w)[0]
        check(f"the {view} title bar names the repo and the handle at {w}x{h}",
              "deepseek-v41-flash-spark · 0xbakeer" in head, repr(head))

# --- the popup ---------------------------------------------------------------
PORT = 54321
for view in ("easy", "advanced"):
    for h, w in SIZES:
        body = rows(state(view, atlas=PORT), h, w)
        bad = []
        if not any(T.ATLAS_LABEL in r for r in body):
            bad.append("no title")
        if not any(f"served on http://127.0.0.1:{PORT}/" in r for r in body):
            bad.append("no url")
        if not any(f"ssh -L {PORT}:127.0.0.1:{PORT} <host>" in r for r in body):
            bad.append("no tunnel line")
        if not any("any key to close" in r for r in body):
            bad.append("no way out")
        if any(len(r) > w for r in body):
            bad.append("a row is wider than the window")
        check(f"the {view} popup says the port at {w}x{h}", not bad, "; ".join(bad))

# the popup must not name a real host: the line is a template a reader fills in
body = "\n".join(rows(state(atlas=PORT), 40, 140))
check("the tunnel line names no real host", "<host>" in body and ".local" not in body)

# --- the server ---------------------------------------------------------------
# A real socket, on a real free port, serving the two files the page cannot open without.
srv = T.AtlasServer()
try:
    port = srv.start()
    check("it picks a port", isinstance(port, int) and port > 0, str(port))
    check("  on loopback and nowhere else", srv._srv.server_address[0] == "127.0.0.1",
          str(srv._srv.server_address))
    check("  and says so in the url", srv.url == f"http://127.0.0.1:{port}/", srv.url)

    want = ["index.html"]
    if os.path.exists(AX.STATS_DEFAULT) and os.path.exists(AX.GATES_DEFAULT):
        # exactly what `a` does before it serves: the data is generated, gitignored, and
        # re-made whenever the trace behind it moves
        if AX.is_stale():
            AX.export(profiles=T.PROFILES)
        # manifest.json is the model picker and sits ABOVE the slug (src/data.ts
        # loadManifest); the per-model files sit under it
        want += ["models/manifest.json", f"models/{AX.SLUG}/atlas.jsonl"]
    else:
        print(f"note: no {os.path.relpath(AX.STATS_DEFAULT, ROOT)} here, so only the page is fetched")
    for rel in want:
        try:
            with urllib.request.urlopen(srv.url + rel, timeout=10) as r:
                code, payload = r.status, r.read()
        except Exception as e:  # noqa: BLE001
            code, payload = f"{type(e).__name__}: {e}", b""
        check(f"GET /{rel} is 200", code == 200, str(code))
        if rel.endswith("index.html"):
            check("  and it is the vendored page", b"<title>Weight Atlas</title>" in payload)
            check("  which fetches nothing from the network",
                  b"fonts.googleapis.com" not in payload and b"fonts.gstatic.com" not in payload)
        if rel.endswith("manifest.json"):
            man = json.loads(payload)
            check("  and the manifest opens on one model",
                  len(man) == 1 and man[0]["slug"] == AX.SLUG, json.dumps(man)[:80])
        if rel.endswith("atlas.jsonl"):
            check("  and the inventory is one tensor per line",
                  payload.count(b"\n") == len(AX.weight_inventory()),
                  f"{payload.count(chr(10).encode())} lines")

    # a second press is the same server, not a second one
    check("pressing it again keeps the same port", srv.start() == port)
finally:
    srv.stop()
check("and it stops", srv.port is None)

# --- staleness ---------------------------------------------------------------
# `a` re-exports only when the picture is older than what it is a picture of. The inputs are
# the trace, the gate index and tune.py -- the last because PROFILES decides which ten
# keep-sets are drawn.
src = AX.inputs()
check("the export is derived from the trace, the gates and the profiles",
      [os.path.basename(p) for p in src] == ["coverage.json", "gates.json", "tune.py"],
      str(src))

tmp = tempfile.mkdtemp(prefix="atlas-stale-")
try:
    check("an export that does not exist is stale", AX.is_stale(tmp))
    d = os.path.join(tmp, AX.SLUG)
    os.makedirs(d)
    for p in (os.path.join(tmp, "manifest.json"), os.path.join(d, "insights.json"),
              os.path.join(d, "atlas.jsonl")):
        open(p, "w").write("{}")
    check("  one newer than every input is not", not AX.is_stale(tmp))
    os.utime(os.path.join(d, "atlas.jsonl"), (0, 0))
    check("  and one file left behind makes the whole export stale", AX.is_stale(tmp))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
