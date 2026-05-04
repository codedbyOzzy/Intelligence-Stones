"""
Mind Stone v1.2 functional test suite.

Covers all v1.2 fixes (thread safety, type hints, satisfied threshold,
tech_amplifier, question detection) plus the edge cases flagged in the
code-review report: corrupt JSON, normalise_fn exception, 500-turn drift,
concurrent writes, I/O error on save.
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

from mind_stone import MindStone, SignalConfig

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
        FAIL += 1


def tmp(suffix: str) -> str:
    return f".test_v12_{suffix}.json"


def cleanup(*paths: str) -> None:
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


# ── Thread safety ──────────────────────────────────────────────────────────────
print("\n--- Thread safety ---")

def test_concurrent_writes():
    path = tmp("concurrent")
    stone = MindStone(path=path, save_every=1)
    errors = []

    def worker():
        try:
            for _ in range(50):
                stone.observe("show me a python async example", "Here you go...")
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    check("T01 no exceptions under concurrent observe()", len(errors) == 0,
          str(errors[:1]))
    check("T02 total_turns == 400 (8×50)",
          stone.profile.total_turns == 400,
          f"got {stone.profile.total_turns}")
    cleanup(path)

test_concurrent_writes()


def test_concurrent_read_write():
    path = tmp("rw")
    stone = MindStone(path=path)
    errors = []

    def writer():
        for _ in range(30):
            stone.observe("python gpu cuda async", "ok")

    def reader():
        for _ in range(30):
            try:
                stone.summary()
                stone.get_style_directive()
            except Exception as exc:
                errors.append(str(exc))

    ts = [threading.Thread(target=writer) for _ in range(3)]
    ts += [threading.Thread(target=reader) for _ in range(3)]
    for t in ts: t.start()
    for t in ts: t.join()

    check("T03 no exceptions during concurrent read/write", len(errors) == 0,
          str(errors[:1]))
    cleanup(path)

test_concurrent_read_write()


# ── Corrupt JSON recovery ──────────────────────────────────────────────────────
print("\n--- Corrupt JSON recovery ---")

def test_corrupt_json():
    path = tmp("corrupt")
    # Write invalid JSON
    Path(path).write_text("{this is not valid json!!}", encoding="utf-8")
    stone = MindStone(path=path)
    check("T04 loads fresh profile on corrupt JSON",
          stone.profile.total_turns == 0)
    check("T05 can observe() after corrupt load",
          stone.observe("hello", "hi") is None)
    cleanup(path)

test_corrupt_json()


def test_truncated_json():
    path = tmp("truncated")
    Path(path).write_text('{"verbosity": 0.3, "tech_depth":', encoding="utf-8")
    stone = MindStone(path=path)
    check("T06 loads fresh profile on truncated JSON",
          stone.profile.total_turns == 0)
    cleanup(path)

test_truncated_json()


def test_empty_file():
    path = tmp("empty")
    Path(path).write_text("", encoding="utf-8")
    stone = MindStone(path=path)
    check("T07 loads fresh profile on empty file",
          stone.profile.total_turns == 0)
    cleanup(path)

test_empty_file()


def test_wrong_types_json():
    path = tmp("wrongtype")
    # verbosity as string instead of float
    Path(path).write_text(
        json.dumps({"verbosity": "very_verbose", "total_turns": "ten"}),
        encoding="utf-8",
    )
    stone = MindStone(path=path)
    check("T08 loads fresh profile on type-error JSON",
          stone.profile.total_turns == 0)
    cleanup(path)

test_wrong_types_json()


# ── normalise_fn exception handling ───────────────────────────────────────────
print("\n--- normalise_fn exception handling ---")

def test_normalise_fn_exception():
    def bad_normalise(text: str) -> str:
        raise RuntimeError("normalise crashed")

    cfg = SignalConfig(
        neg_verbosity    = frozenset({"shorter"}),
        pos_verbosity    = frozenset(),
        example_signals  = frozenset(),
        theory_signals   = frozenset(),
        satisfied_tokens = frozenset({"ok"}),
        tech_words       = frozenset({"python"}),
        normalise_fn     = bad_normalise,
    )
    path = tmp("norm_exc")
    stone = MindStone(path=path, config=cfg)
    try:
        stone.observe("shorter please", "ok")
        check("T09 observe() survives normalise_fn exception", True)
        check("T10 profile still updated after exception",
              stone.profile.total_turns == 1)
    except Exception as exc:
        check("T09 observe() survives normalise_fn exception", False, str(exc))
        check("T10 profile still updated after exception", False)
    cleanup(path)

test_normalise_fn_exception()


# ── 500-turn drift ─────────────────────────────────────────────────────────────
print("\n--- 500-turn drift ---")

def test_500_turn_drift():
    path = tmp("drift")
    stone = MindStone(path=path, save_every=500)

    # First 250 turns: verbose, theory, non-technical user
    for _ in range(250):
        stone.observe("elaborate more please, why does this work", "Detailed theory...")

    mid = stone.profile.verbosity
    mid_tech = stone.profile.tech_depth
    mid_example = stone.profile.example_bias

    # Next 250 turns: terse, example, high-tech user
    for _ in range(250):
        stone.observe("show me python cuda code shorter", "```python\n...```")

    final = stone.profile

    check("T11 total_turns == 500",        final.total_turns == 500)
    check("T12 profile stays in [0,1]",
          all(0.0 <= v <= 1.0 for v in [
              final.verbosity, final.tech_depth,
              final.example_bias, final.follow_up_rate,
          ]))
    check("T13 verbosity shifted toward terse (final < mid)",
          final.verbosity < mid)
    check("T14 tech_depth shifted toward expert",
          final.tech_depth > mid_tech)
    check("T15 example_bias shifted toward examples",
          final.example_bias > mid_example)
    cleanup(path)

test_500_turn_drift()


# ── v1.2 satisfied threshold (wc <= 6) ────────────────────────────────────────
print("\n--- Satisfied threshold (v1.2: wc <= 6) ---")

def test_satisfied_threshold():
    path = tmp("sat")
    stone = MindStone(path=path)

    # 5-word satisfied message (was rejected at wc<=4, now accepted at wc<=6)
    r = stone.observe("got it that was perfect", "Good.", verbose=True)
    check("T16 5-word satisfied message detected",
          r["signals"]["follow_up_rate"]["satisfied"] is True)

    # 6-word satisfied message
    r2 = stone.observe("ok got it thank you", "Good.", verbose=True)
    check("T17 6-word satisfied message detected",
          r2["signals"]["follow_up_rate"]["satisfied"] is True)

    # 7-word message: too long to be satisfied
    r3 = stone.observe("ok got it thanks a lot man", "Good.", verbose=True)
    check("T18 7-word message not counted as satisfied",
          r3["signals"]["follow_up_rate"]["satisfied"] is False)

    cleanup(path)

test_satisfied_threshold()


# ── Question detection prevents false satisfied ────────────────────────────────
print("\n--- Question detection (v1.2) ---")

def test_question_detection():
    path = tmp("qdtect")
    stone = MindStone(path=path)

    # Ends with "?" → not satisfied even if token present
    r = stone.observe("got it but why?", "Because...", verbose=True)
    check("T19 'got it but why?' not counted as satisfied",
          r["signals"]["follow_up_rate"]["satisfied"] is False)
    check("T20 is_question=True for '?'-ending",
          r["signals"]["follow_up_rate"]["is_question"] is True)

    # Contains question word → not satisfied
    r2 = stone.observe("ok but how", "Like this...", verbose=True)
    check("T21 'ok but how' not counted as satisfied",
          r2["signals"]["follow_up_rate"]["satisfied"] is False)

    # Pure satisfied, no question → satisfied
    r3 = stone.observe("perfect thanks", "Good.", verbose=True)
    check("T22 'perfect thanks' counted as satisfied",
          r3["signals"]["follow_up_rate"]["satisfied"] is True)

    cleanup(path)

test_question_detection()


# ── tech_amplifier parameter ──────────────────────────────────────────────────
print("\n--- tech_amplifier parameter (v1.2) ---")

def test_tech_amplifier():
    path_lo = tmp("amp_lo")
    path_hi = tmp("amp_hi")

    stone_lo = MindStone(path=path_lo, tech_amplifier=2.0)
    stone_hi = MindStone(path=path_hi, tech_amplifier=16.0)

    # Use a low-density tech message so the amplifier difference is visible
    # before both values get clamped to 1.0
    # "python" = 1 tech word out of 5 total -> ratio 0.2
    # amp 2.0:  signal = min(1.0, 0.2 * 2.0)  = 0.40  -> EMA stays below amp-16 value
    # amp 16.0: signal = min(1.0, 0.2 * 16.0) = 1.0   -> EMA pulled toward 1.0
    msg = "tell me about python today"
    for _ in range(5):
        stone_lo.observe(msg, "ok")
        stone_hi.observe(msg, "ok")

    check("T23 higher tech_amplifier -> higher tech_depth",
          stone_hi.profile.tech_depth > stone_lo.profile.tech_depth,
          f"lo={stone_lo.profile.tech_depth:.3f} hi={stone_hi.profile.tech_depth:.3f}")
    check("T24 both tech_depth values in [0,1]",
          0 <= stone_lo.profile.tech_depth <= 1 and
          0 <= stone_hi.profile.tech_depth <= 1)
    cleanup(path_lo, path_hi)

test_tech_amplifier()


# ── I/O error on save ─────────────────────────────────────────────────────────
print("\n--- I/O error on save ---")

def test_save_io_error():
    path = tmp("ioerr")
    stone = MindStone(path=path, save_every=1)

    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        try:
            stone.observe("hello", "hi")
            check("T25 observe() survives save I/O error", True)
        except Exception as exc:
            check("T25 observe() survives save I/O error", False, str(exc))

    check("T26 profile still updated despite save error",
          stone.profile.total_turns == 1)
    cleanup(path)

test_save_io_error()


# ── type hint: SignalConfig.normalise_fn ──────────────────────────────────────
print("\n--- Type hint correctness ---")

def test_type_hints():
    import inspect
    from typing import get_type_hints
    from mind_stone import SignalConfig
    hints = get_type_hints(SignalConfig)
    hint_str = str(hints.get("normalise_fn", ""))
    check("T27 normalise_fn hint contains 'Callable'",
          "Callable" in hint_str, f"got: {hint_str}")

test_type_hints()


# ── Backward compatibility: v1.0 profile loads fine ──────────────────────────
print("\n--- Backward compatibility ---")

def test_v10_profile_loads():
    path = tmp("v10compat")
    # Write a minimal v1.0-style profile (no session fields)
    v10 = {
        "verbosity": 0.25,
        "tech_depth": 0.80,
        "example_bias": 0.70,
        "follow_up_rate": 0.40,
        "peak_hours": [14, 15],
        "hour_counts": {"14": 5, "15": 3},
        "total_turns": 30,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    Path(path).write_text(json.dumps(v10), encoding="utf-8")

    stone = MindStone(path=path)
    check("T28 v1.0 profile loads without error",
          stone.profile.total_turns == 30)
    check("T29 v1.0 profile verbosity preserved",
          abs(stone.profile.verbosity - 0.25) < 0.001)
    check("T30 v1.0 profile session_count defaults to 0",
          stone.profile.session_count == 0)
    cleanup(path)

test_v10_profile_loads()


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'-' * 46}")
print(f"  {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    raise SystemExit(1)
