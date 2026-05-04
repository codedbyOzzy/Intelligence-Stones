"""Mind Stone v1.1 functional tests."""

from mind_stone import MindStone
import os

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAIL += 1

# ── Test 1: session initialises at 1 ───────────────────────────────────────────
stone = MindStone(path=".test_v11.json", session_gap_minutes=30)
for _ in range(5):
    stone.observe("show me a python async example", "Here is an example...")

s1 = stone.session_summary()
check("T1a session_count starts at 1",    s1["total_sessions"] == 1)
check("T1b current_session_turns == 5",   s1["current_session_turns"] == 5)

# ── Test 2: new session detected after gap ─────────────────────────────────────
stone.profile.last_observe_ts -= 31 * 60   # fake a 31-min gap
result = stone.observe("how does this work?", "It works by...", verbose=True)
s2 = stone.session_summary()
check("T2a session_count incremented",    s2["total_sessions"] == 2)
check("T2b current_session_turns reset",  s2["current_session_turns"] == 1)
check("T2c verbose returns session key",  "session" in result)
check("T2d verbose returns signals key",  "signals" in result)
check("T2e verbose returns profile key",  "profile" in result)
check("T2f verbose session.is_new True",  result["session"].get("is_new") is True)

# ── Test 3: directive activates after enough turns ─────────────────────────────
stone2 = MindStone(path=".test_v11b.json")
for _ in range(15):
    stone2.observe("too long just show code example please", "OK here is code...")
directive = stone2.get_style_directive()
check("T3a directive non-empty after 15", bool(directive))
check("T3b directive is a string",        isinstance(directive, str))

# ── Test 4: session_summary keys ──────────────────────────────────────────────
ss = stone2.session_summary()
expected = {"session_number", "current_session_turns", "minutes_since_last_turn",
            "total_sessions", "total_turns_all_time"}
check("T4  session_summary keys correct", expected == set(ss.keys()))

# ── Test 5: summary has version + sessions ────────────────────────────────────
summ = stone2.summary()
check("T5a summary has version field",    "version" in summ)
check("T5b summary has sessions field",   "sessions" in summ)

# ── Test 6: no-gap same session doesn't increment ─────────────────────────────
stone3 = MindStone(path=".test_v11c.json", session_gap_minutes=30)
stone3.observe("python example", "Here...")
stone3.observe("shorter please", "OK")
stone3.observe("got it", "Good")
s3 = stone3.session_summary()
check("T6  single session stays at 1",   s3["total_sessions"] == 1)
check("T6b total_turns == 3",            s3["total_turns_all_time"] == 3)

# ── Test 7: session_gap_minutes=0 disables detection ─────────────────────────
stone4 = MindStone(path=".test_v11d.json", session_gap_minutes=0)
stone4.observe("hello", "hi")
stone4.profile.last_observe_ts -= 9999  # huge gap — should NOT trigger
stone4.observe("world", "ok")
s4 = stone4.session_summary()
check("T7  gap=0 never increments",      s4["total_sessions"] == 1)

# ── Cleanup ────────────────────────────────────────────────────────────────────
for f in [".test_v11.json", ".test_v11b.json", ".test_v11c.json", ".test_v11d.json"]:
    if os.path.exists(f):
        os.remove(f)

print(f"\n{PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    raise SystemExit(1)
