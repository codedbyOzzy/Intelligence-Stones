# 🧠 Mind Stone — Adaptive Intelligence Profile

> *"Most AI assistants know **what** to say. Mind Stone teaches them **how** to say it — for you, specifically."*

A lightweight, zero-dependency module that silently observes every conversation turn and builds a quantified model of how a user communicates. It then generates a short style directive that shapes the AI assistant's tone, depth, and format — without the user ever configuring anything.

---

## The problem it solves

Every person communicates differently:

- **Alex** sends three-word messages and gets frustrated by long answers
- **Sam** asks "why?" after every reply and wants theory before examples  
- **Jordan** pastes GPU code and expects expert-level responses without explanation

Standard AI assistants treat them all the same. Mind Stone doesn't.

---

## How it works

```
Each conversation turn:
  user_message + assistant_message
         │
         ▼
  ┌─────────────────────────────────┐
  │        Mind Stone Engine        │
  │                                 │
  │  Signal detection               │
  │    "too long" → verbosity ↓     │
  │    "show me"  → example_bias ↑  │
  │    "cuda gpu" → tech_depth ↑    │
  │    "got it"   → satisfaction ↑  │
  │                                 │
  │  EMA update (α = 0.12)          │
  │  Persistence every 5 turns      │
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │     Intelligence Profile        │
  │                                 │
  │  verbosity:      0.21  (terse)  │
  │  tech_depth:     0.87  (expert) │
  │  example_bias:   0.74  (←examp) │
  │  follow_up_rate: 0.62           │
  │  confidence:     68%            │
  └──────────────┬──────────────────┘
                 │
                 ▼
  get_style_directive()
  ──────────────────────────────────────
  "This user prefers concise answers.
   Use domain terminology freely.
   Lead with a code example."
  ──────────────────────────────────────
         │
         ▼ (injected into system prompt)
  LLM call  →  calibrated response
```

### Profile dimensions

| Dimension | Low (0) | High (1) | How it's learned |
|-----------|---------|----------|-----------------|
| `verbosity` | Terse, direct | Detailed, thorough | Explicit signals ("shorter", "elaborate") + message length |
| `tech_depth` | Plain language | Expert vocabulary | Technical word density in user messages |
| `example_bias` | Theory first | Examples first | "show me" / "why does it" signals |
| `follow_up_rate` | Satisfied by first reply | Always asks more | Short positive replies vs long follow-ups |

### Learning curve

```
Turns:     0    5   12   25   40   55+
Confidence: 0%  0% 14%  40%  70% 100%
Directive:  ─── ─── ON  ─── ─── ───▶
```

Style directives activate at ~12 turns. Full calibration takes ~55 turns of natural conversation.

---

## Quick start

No installation required — just copy `mind_stone.py` into your project.

```python
from mind_stone import MindStone

stone = MindStone()   # loads .mind_stone.json if it exists

# After every conversation turn:
stone.observe(user_message, assistant_message)

# Before every LLM call:
directive = stone.get_style_directive()   # "" until enough data
if directive:
    system_prompt += "\n\n" + directive
```

---

## OpenAI integration

```python
from openai import OpenAI
from mind_stone import MindStone

client = OpenAI()
stone  = MindStone()

BASE_PROMPT = "You are a helpful assistant."
history     = []

def chat(user_message: str) -> str:
    # 1. Inject adaptive style directive
    system = BASE_PROMPT
    directive = stone.get_style_directive()
    if directive:
        system += "\n\n" + directive

    # 2. Call LLM
    messages = [{"role": "system", "content": system}] + history
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = response.choices[0].message.content

    # 3. Update history and observe
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    stone.observe(user_message, reply)   # ← one line to learn

    return reply
```

---

## Language customisation

Mind Stone ships with English signals. Use another language by passing a `SignalConfig`:

```python
from mind_stone import MindStone, SignalConfig

MY_CONFIG = SignalConfig(
    neg_verbosity    = frozenset({"kurzer", "zu lang", "fass dich"}),       # German: shorter
    pos_verbosity    = frozenset({"mehr details", "erklar mir", "weiter"}),  # German: more
    example_signals  = frozenset({"beispiel", "zeig mir", "code"}),
    theory_signals   = frozenset({"warum", "wie funktioniert", "erklare"}),
    satisfied_tokens = frozenset({"ok", "danke", "verstanden", "gut"}),
    tech_words       = frozenset({"python", "api", "docker", "gpu"}),
    normalise_fn     = None,   # German uses ASCII — no normalisation needed
)

stone = MindStone(config=MY_CONFIG)
```

For languages with non-ASCII characters (Turkish, French, etc.) — provide a `normalise_fn` that strips diacritics. See [`signals_turkish.py`](signals_turkish.py) for a complete example.

---

## API reference

### `MindStone(path, config, ema_alpha, min_confidence, save_every, session_gap_minutes)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.mind_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `ema_alpha` | `0.12` | Learning rate — lower is more stable |
| `min_confidence` | `0.15` | Confidence threshold before directives activate |
| `save_every` | `5` | Persist to disk every N turns |
| `session_gap_minutes` | `30` | Gap (minutes) that marks a new session boundary; `0` disables |
| `tech_amplifier` | `8.0` | Multiplier on tech-word ratio; higher = more sensitive to technical vocabulary |

### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
# → None normally; dict with full signal report when verbose=True

stone.get_style_directive() -> str   # "" or 1-4 sentence directive
stone.summary()             -> dict  # human-readable profile snapshot
stone.session_summary()     -> dict  # v1.1: current session metadata
stone.reset()                        # clear profile + delete file
stone.profile                        # raw IntelligenceProfile dataclass
```

#### `observe(verbose=True)` report structure

```python
{
    "session": {
        "is_new":    bool,   # True when a session boundary was crossed
        "number":    int,    # current session number (starts at 1)
        "turn":      int,    # turn within the current session
        "alpha_used": float, # effective EMA alpha (dampened during ramp-up)
    },
    "signals": {
        "neg_verbosity":  bool,
        "pos_verbosity":  bool,
        "example":        bool,
        "theory":         bool,
        "satisfied":      bool,
        "tech_word_count": int,
    },
    "profile": { ... },   # same as summary()
}
```

#### `session_summary()` structure

```python
{
    "session_number":        int,    # current session index
    "current_session_turns": int,    # turns in the ongoing session
    "minutes_since_last_turn": float,
    "total_sessions":        int,    # all sessions ever recorded
    "total_turns_all_time":  int,
}
```

### `SignalConfig`

```python
@dataclass
class SignalConfig:
    neg_verbosity:    frozenset   # "shorter", "too long", ...
    pos_verbosity:    frozenset   # "elaborate", "more detail", ...
    example_signals:  frozenset   # "show me", "example", ...
    theory_signals:   frozenset   # "why", "how does it work", ...
    satisfied_tokens: frozenset   # "ok", "got it", "thanks", ...
    tech_words:       frozenset   # vocabulary indicating expertise
    normalise_fn:     callable | None
```

---

## Design decisions

**Why EMA instead of a counter?**  
A simple counter weights day-1 behaviour forever. EMA ensures recent turns matter more — if a user's preferences shift, the profile adapts within ~15 turns.

**Why 0.12 as the default alpha?**  
At α=0.12, a single strong signal moves the profile by ~12%, a weak signal by ~5%. This is stable enough to ignore one-off turns ("shorter" said once while in a hurry) without being so slow that it takes 100+ turns to reflect genuine preferences.

**Why a confidence threshold?**  
With fewer than ~12 observations, any signal is statistically noisy. Injecting directives too early risks reinforcing a false impression. The threshold ensures the directive represents a genuine pattern.

**Why session-dampened EMA? (v1.1)**  
The first few turns of a new session often don't represent a user's real preferences — they may be rushed, testing something, or just warming up. Dampening the EMA alpha by 50% for the first 3 turns of each session prevents one atypical session from corrupting a profile built over months of data.

**Why not use embeddings or ML?**  
Zero-dependency is a design goal. The EMA approach works well for the 4–6 dimensions that matter for communication style, runs in microseconds, and produces auditable, interpretable profiles. A black-box model would be harder to debug and overkill for this task.

---

## Changelog

### v1.2.0
- **Thread-safe** — all public methods (`observe`, `get_style_directive`, `summary`, `session_summary`, `reset`) are protected by `threading.Lock`. Safe to share a single instance across threads (e.g. async web servers).
- **Correct type hint** — `normalise_fn` is now typed as `Optional[Callable[[str], str]]` instead of `Optional[object]`
- **`tech_amplifier` parameter** — replaces the hardcoded `*8` multiplier; tune sensitivity to technical vocabulary per use-case (`MindStone(tech_amplifier=5.0)`)
- **Satisfied threshold raised** — `wc <= 4` raised to `wc <= 6`; "got it that was perfect" now correctly registers as satisfied
- **Question detection** — `follow_up_rate` now checks for `?` and question words (`why`, `how`, `what`, …); "got it, but why?" is no longer counted as satisfied
- **`normalise_fn` fault tolerance** — exceptions raised by `normalise_fn` are caught; observation continues with unnormalised text instead of crashing
- Added `test_v12.py`: 30 functional tests covering all v1.2 fixes plus edge cases (corrupt JSON, I/O error on save, 500-turn drift, concurrent writes, `normalise_fn` exception)

### v1.1.0
- **Session awareness** — automatic session boundary detection (`session_gap_minutes` parameter)
- **Session-dampened EMA** — alpha halved for the first 3 turns of a new session to prevent atypical sessions from corrupting long-term profiles
- **`observe(verbose=True)`** — returns a structured per-turn signal report (session info, signals detected, profile snapshot)
- **`session_summary()`** — new method returning current session number, turn count, minutes since last turn, and lifetime totals
- **Temporal directive** — `get_style_directive()` adds a note when the user is active outside their detected peak hours
- **`summary()`** now includes `version` and `sessions` fields
- Backward-compatible JSON format — v1.0 profile files load without changes

### v1.0.0
- Initial release
- EMA-based profile (verbosity, tech_depth, example_bias, follow_up_rate)
- Persistence via JSON, `SignalConfig` for multilingual support
- English defaults (`EN_CONFIG`) + Turkish reference implementation (`TR_CONFIG`)

---

## Files

```
mind_stone.py          core module — copy this into your project
signals_turkish.py     Turkish signal sets (reference implementation)
example.py             basic, v1.1, OpenAI, and Turkish usage examples
test_v11.py            v1.1 functional test suite (16 tests)
test_v12.py            v1.2 functional test suite (30 tests)
```

---

## Requirements

Python 3.9+. No third-party packages.

---

## Contributing

Signal sets for more languages are very welcome. Copy `signals_turkish.py`, adapt the frozensets for your language, and open a pull request.

---

## License

MIT — free to use, modify, and distribute in personal and commercial projects.
