# Intelligence Stones

> *"Most AI assistants know what to say. These teach them how to say it — and whether it actually landed."*

A growing collection of lightweight, zero-dependency Python modules that give AI assistants genuine understanding of the people they talk to.

Each stone is a standalone drop-in. No framework. No configuration. No external dependencies.  
Together, they build something more complete.

---

## The Collection

| Stone | Status | What it learns |
|-------|--------|----------------|
| [**Mind Stone**](#-mind-stone) | `v1.2.0` ✅ | *How* the user communicates — style, depth, pace |
| [**Echo Stone**](#-echo-stone) | `v1.0.0` ✅ | *Whether* the user actually understood |
| **Bond Stone** | 🔒 in development | — |
| **Intuition Stone** | 🔒 in development | — |

Each stone operates independently. Each one makes the assistant measurably better at a specific dimension of human communication. The full picture emerges when they work together.

---

## 🧠 Mind Stone

> *The user's communication fingerprint — learned silently, applied automatically.*

Observes every conversation turn and builds a quantified model of how a user communicates. Generates a short style directive injected into the system prompt — shaping tone, depth, and format without the user ever configuring anything.

### How it works

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
  │  example_bias:   0.74  (examp.) │
  │  follow_up_rate: 0.62           │
  │  confidence:     68%            │
  └──────────────┬──────────────────┘
                 │
                 ▼
  get_style_directive()
  ─────────────────────────────────────────
  "This user prefers concise answers.
   Use domain terminology freely.
   Lead with a code example."
  ─────────────────────────────────────────
         │
         ▼ (injected into system prompt)
  LLM call  →  calibrated response
```

### Profile dimensions

| Dimension | Low (0) | High (1) | Learned from |
|-----------|---------|----------|-------------|
| `verbosity` | Terse, direct | Detailed, thorough | Explicit signals + message length |
| `tech_depth` | Plain language | Expert vocabulary | Technical word density |
| `example_bias` | Theory first | Examples first | "show me" / "why does it" signals |
| `follow_up_rate` | Satisfied by first reply | Always asks more | Short confirmations vs long follow-ups |

### Learning curve

```
Turns:      0    5   12   25   40   55+
Confidence: 0%  0%  14%  40%  70% 100%
Directive:  ───────── ON ──────────────▶
```

### Quick start

```python
from mind_stone import MindStone

stone = MindStone()

# After every conversation turn:
stone.observe(user_message, assistant_message)

# Before every LLM call:
directive = stone.get_style_directive()   # "" until ~12 turns
if directive:
    system_prompt += "\n\n" + directive
```

### API

#### `MindStone(path, config, ema_alpha, min_confidence, save_every, session_gap_minutes, tech_amplifier)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.mind_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `ema_alpha` | `0.12` | Learning rate |
| `min_confidence` | `0.15` | Threshold before directives activate (~12 turns) |
| `save_every` | `5` | Persist to disk every N turns |
| `session_gap_minutes` | `30` | Gap that marks a new session boundary |
| `tech_amplifier` | `8.0` | Sensitivity to technical vocabulary |

#### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
stone.get_style_directive() -> str
stone.summary()             -> dict
stone.session_summary()     -> dict
stone.reset()
```

#### `observe(verbose=True)` report

```python
{
  "session":  {"is_new": bool, "number": int, "turn": int, "alpha_used": float},
  "signals":  {"verbosity": {...}, "tech_depth": {...}, ...},
  "profile":  { ... }
}
```

---

## 📡 Echo Stone

> *The assistant spoke. But did the user understand?*

Analyses the user's reaction to each response and detects comprehension patterns that neither the user nor a standard AI would explicitly flag. Translates these patterns into a directive that shapes *how the assistant explains* — not just what it says.

### The problem it solves

```
Standard flow:
  FRIDAY explains X  →  User says "ok got it"  →  FRIDAY moves on

What actually happened:
  User says "ok got it"  →  30 seconds later: "wait, how does X work again?"
                                                            ↑
                                          Echo Stone caught this.
```

### Detected patterns

| Signal | What happened |
|--------|--------------|
| `explicit_confusion` | User directly says they didn't understand |
| `overload_deflect` | Long response → 1–3 word reply (cognitive shutdown) |
| `deepening` | User asks a deeper question — they understood and want more |
| `rephrase` | User asks the same question in different words |
| `false_confirmation` | User confirms, then returns to the same topic |
| `genuine_confirmation` | User confirms and moves to a genuinely different topic |

### Comprehension profile

| Dimension | Low (0) | High (1) |
|-----------|---------|----------|
| `comprehension_rate` | Rarely understands first try | Always gets it first try |
| `false_confirm_rate` | Confirmations are genuine | Often confirms without understanding |
| `overload_rate` | Handles complexity well | Easily overwhelmed by long responses |
| `depth_rate` | Stays surface level | Digs deeper every time |

### Quick start

```python
from echo_stone import EchoStone

stone = EchoStone()

# Same interface as Mind Stone:
stone.observe(user_message, assistant_message)

# Before every LLM call:
directive = stone.get_comprehension_directive()   # "" until ~8 turns
if directive:
    system_prompt += "\n\n" + directive
```

### Running both stones together

```python
from mind_stone import MindStone
from echo_stone import EchoStone

mind = MindStone()
echo = EchoStone()

def chat(user_message: str, assistant_message: str) -> None:
    # One call each — that's all it takes
    mind.observe(user_message, assistant_message)
    echo.observe(user_message, assistant_message)

def get_directives() -> str:
    parts = []
    d = mind.get_style_directive()
    if d: parts.append(d)
    d = echo.get_comprehension_directive()
    if d: parts.append(d)
    return "\n\n".join(parts)
```

### API

#### `EchoStone(path, config, ema_alpha, min_confidence, save_every, rephrase_threshold, overload_word_count)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `.echo_stone.json` | Profile persistence path |
| `config` | `EN_CONFIG` | Language signal sets |
| `ema_alpha` | `0.15` | Learning rate |
| `min_confidence` | `0.12` | Threshold before directives activate (~8 turns) |
| `save_every` | `5` | Persist to disk every N turns |
| `rephrase_threshold` | `0.38` | Jaccard similarity for rephrase detection |
| `overload_word_count` | `120` | Response word count that triggers overload check |

#### Methods

```python
stone.observe(user_message, assistant_message, verbose=False)
stone.get_comprehension_directive() -> str
stone.summary()                     -> dict
stone.reset()
```

#### `observe(verbose=True)` report

```python
{
  "signal":     str | None,   # detected comprehension signal
  "is_confirm": bool,         # was this message a confirmation?
  "profile":    { ... }       # same as summary()
}
```

---

## Language customisation

Both stones ship with English signal sets. Any language is supported by passing a config object:

```python
from mind_stone import MindStone, SignalConfig
from echo_stone import EchoStone, EchoConfig

# Example: German
mind = MindStone(config=SignalConfig(
    neg_verbosity    = frozenset({"kurzer", "zu lang"}),
    pos_verbosity    = frozenset({"mehr details", "erklar mir"}),
    example_signals  = frozenset({"beispiel", "zeig mir"}),
    theory_signals   = frozenset({"warum", "wie funktioniert"}),
    satisfied_tokens = frozenset({"ok", "danke", "verstanden"}),
    tech_words       = frozenset({"python", "api", "docker"}),
    normalise_fn     = None,
))

echo = EchoStone(config=EchoConfig(
    confusion_signals   = frozenset({"verstehe nicht", "nochmal", "was meinst du"}),
    confirmation_tokens = frozenset({"ok", "verstanden", "danke", "gut"}),
    deepen_signals      = frozenset({"also wenn", "was ware wenn", "bedeutet das"}),
    normalise_fn        = None,
))
```

For non-ASCII languages (Turkish, French, etc.) — provide a `normalise_fn` that strips diacritics.  
See [`signals_turkish.py`](signals_turkish.py) for a complete reference implementation.

---

## What's coming

Two more stones are in development. They're not announced yet — but here's what they're designed to solve.

---

**Bond Stone**

Every conversation has context the user never restates: the project they're building, the person they mentioned two weeks ago, the constraint they explained once and assumed you'd remember.

Bond Stone builds a persistent, structured model of the user's world — not as raw chat history, but as a live knowledge graph that every response can silently query.

*In development.*

---

**Intuition Stone**

Experienced human assistants don't just answer the current question. They know where the conversation is going — and quietly prepare for it.

Intuition Stone learns the shape of conversations: which questions lead to which follow-ups, which topics always resurface, what the user is actually trying to solve when they ask what they ask.

*In development.*

---

When all four stones are in place, the picture looks like this:

```
Mind Stone      → the assistant speaks in a way that fits you
Echo Stone      → the assistant knows whether it worked
Bond Stone      → the assistant knows your world
Intuition Stone → the assistant knows where you're going

Together        → an assistant that genuinely understands you
```

No magic. No large models. No embeddings. Just careful observation, accumulated over time.

---

## Design decisions

**Why EMA instead of a counter?**  
A simple counter weights day-1 behaviour forever. EMA ensures recent turns matter more — if preferences shift, the profile adapts within ~15 turns.

**Why 0.12 as the default alpha for Mind Stone?**  
At α=0.12, a single strong signal moves the profile ~12%. Stable enough to ignore one-off turns, fast enough to reflect genuine preference shifts.

**Why Jaccard similarity for rephrase detection in Echo Stone?**  
Zero-dependency constraint rules out embeddings. Jaccard on content words (stopwords removed) is fast, interpretable, and works well for the 4–8 word messages that typically trigger rephrase detection.

**Why not use embeddings or ML?**  
Zero-dependency is a hard design constraint. Both stones run in microseconds, produce auditable profiles, and require no model downloads or API calls. The EMA approach is sufficient for the dimensions that matter.

---

## Files

```
mind_stone.py          Mind Stone core module
echo_stone.py          Echo Stone core module
signals_turkish.py     Turkish signal sets (reference implementation for both stones)
example.py             Usage examples
test_v11.py            Mind Stone v1.1 test suite (16 tests)
test_v12.py            Mind Stone v1.2 test suite (30 tests)
```

---

## Requirements

Python 3.9+. No third-party packages.

---

## Contributing

Signal sets for more languages are very welcome.  
Copy `signals_turkish.py`, adapt the frozensets for your language, and open a pull request.

---

## License

MIT — free to use, modify, and distribute.
