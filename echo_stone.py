"""
Echo Stone — Comprehension Pattern Detector
============================================
The companion module to Mind Stone.

Mind Stone learns *how* a user communicates.
Echo Stone learns *whether they actually understood*.

Every conversation turn, Echo Stone analyses the user's reaction
to the assistant's previous response and detects:

  - Did they say "got it" but ask the same question again?       → false confirmation
  - Did they rephrase the same question three different ways?    → real confusion
  - Did a long response produce a one-word reply?                → cognitive overload
  - Did they build on the answer with a deeper question?         → genuine understanding

These patterns are tracked across sessions and translated into a
comprehension directive that shapes how the assistant explains —
not just what it says, but whether it's actually landing.

Usage (same interface as Mind Stone)
-------------------------------------
    from echo_stone import EchoStone

    stone = EchoStone()                               # loads saved profile

    # After each conversation turn (same call as Mind Stone):
    stone.observe(user_message, assistant_message)

    # Before each LLM call:
    directive = stone.get_comprehension_directive()   # "" until enough data
    if directive:
        system_prompt += "\\n\\n" + directive

No external dependencies. Pure Python standard library.
Works with any LLM backend. Designed to run alongside Mind Stone.

Language customisation
-----------------------
    from echo_stone import EchoStone, EchoConfig
    from signals_turkish import TR_ECHO_CONFIG

    stone = EchoStone(config=TR_ECHO_CONFIG)

See signals_turkish.py for a complete non-English reference implementation.

v1.0 signals
-------------
  explicit_confusion    User explicitly says they didn't understand
  overload_deflect      Long response → very short reply (cognitive overload)
  deepening             User builds a deeper question on the answer
  rephrase              User asks the same thing in different words
  false_confirmation    User confirms, then returns to the same topic
  genuine_confirmation  User confirms and genuinely moves on

License: MIT
"""

from __future__ import annotations

__version__ = "1.0.0"

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EchoConfig:
    """Language-specific signal sets for Echo Stone.

    Attributes
    ----------
    confusion_signals    Phrases meaning "I didn't understand" or "explain again"
    confirmation_tokens  Short tokens meaning "understood" / "got it"
    deepen_signals       Phrases meaning "building on that..." / "so that means..."
    normalise_fn         Optional callable: (str) -> str.
                         Strip diacritics before matching (e.g. Turkish, French).
                         Receives a lowercase string, must return a string.
    """
    confusion_signals:   frozenset                       = field(default_factory=frozenset)
    confirmation_tokens: frozenset                       = field(default_factory=frozenset)
    deepen_signals:      frozenset                       = field(default_factory=frozenset)
    normalise_fn:        Optional[Callable[[str], str]]  = None


# ── Default English signal sets ───────────────────────────────────────────────

_EN_CONFUSION = frozenset({
    "i dont understand", "i don't understand", "didn't get it", "dont get it",
    "what do you mean", "what does that mean", "can you repeat", "say that again",
    "explain again", "explain differently", "simpler", "in simpler terms",
    "i'm lost", "im lost", "confused", "makes no sense", "what", "huh",
    "can you clarify", "not following", "lost me",
})

_EN_CONFIRM = frozenset({
    "ok", "got it", "makes sense", "understood", "thanks", "thank you",
    "perfect", "great", "clear", "alright", "sure", "yep", "yes", "nice",
    "cool", "that helps", "good", "awesome",
})

_EN_DEEPEN = frozenset({
    "so that means", "so if", "what if", "does that mean", "in that case",
    "building on that", "following that logic", "so then", "and what about",
    "what about", "one more question", "to take it further", "going deeper",
    "what happens when", "what about when", "what would happen",
})

EN_CONFIG = EchoConfig(
    confusion_signals   = _EN_CONFUSION,
    confirmation_tokens = _EN_CONFIRM,
    deepen_signals      = _EN_DEEPEN,
    normalise_fn        = None,
)


# ── Stop words (ignored in rephrase / overlap detection) ─────────────────────

_EN_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "i", "you", "he", "she", "it", "we", "they", "my", "your",
    "this", "that", "what", "how", "why", "when", "where", "which",
    "and", "or", "but", "for", "not", "so", "at", "in", "on", "to",
    "of", "with", "can", "could", "just", "also", "very",
})


# ── Profile data structure ────────────────────────────────────────────────────

@dataclass
class ComprehensionProfile:
    """Quantified model of how a user processes information.

    All float attributes use Exponential Moving Average (EMA) updates.
    Values are bounded to [0.0, 1.0].

    Dimensions
    ----------
    comprehension_rate  0 = rarely understands on first try, 1 = always does
    false_confirm_rate  0 = confirmations are genuine, 1 = often false
    overload_rate       0 = handles complexity well, 1 = easily overwhelmed
    depth_rate          0 = stays surface level, 1 = digs deeper every time
    """

    comprehension_rate: float = 0.50
    false_confirm_rate: float = 0.50
    overload_rate:      float = 0.50
    depth_rate:         float = 0.50

    total_turns: int   = 0
    created_at:  float = field(default_factory=time.time)
    updated_at:  float = field(default_factory=time.time)

    def confidence(self) -> float:
        """Profile reliability, 0→1.

        Active at ~8 turns, fully calibrated at ~40 turns.
        Comprehension directives only activate above the confidence threshold.
        """
        return min(1.0, max(0.0, (self.total_turns - 3) / 37))

    def as_dict(self) -> dict:
        return asdict(self)


# ── Core engine ────────────────────────────────────────────────────────────────

class EchoStone:
    """Comprehension pattern detector.

    Identical interface to MindStone — drop it into the same observe() call.
    Internally tracks the previous turn to detect two-turn comprehension patterns.

    Thread-safe.

    Parameters
    ----------
    path : Path | str
        Where to persist the profile (JSON). Default: ``.echo_stone.json``.
    config : EchoConfig
        Language-specific signal sets. Default: English (EN_CONFIG).
    ema_alpha : float
        Learning rate. Default: 0.15.
    min_confidence : float
        Confidence threshold before directives activate. Default: 0.12 (~8 turns).
    save_every : int
        Persist to disk every N turns. Default: 5.
    rephrase_threshold : float
        Jaccard similarity threshold above which two messages are considered
        rephrases of the same question. Default: 0.38.
    overload_word_count : int
        Assistant response word count above which a very short follow-up
        is treated as a cognitive overload signal. Default: 120.
    """

    def __init__(
        self,
        path:                str | Path  = ".echo_stone.json",
        config:              EchoConfig  = EN_CONFIG,
        ema_alpha:           float       = 0.15,
        min_confidence:      float       = 0.12,
        save_every:          int         = 5,
        rephrase_threshold:  float       = 0.38,
        overload_word_count: int         = 120,
    ) -> None:
        self._path               = Path(path)
        self._config             = config
        self._alpha              = ema_alpha
        self._min_conf           = min_confidence
        self._save_every         = save_every
        self._rephrase_threshold = rephrase_threshold
        self._overload_wc        = overload_word_count
        self._lock               = threading.Lock()
        self.profile             = self._load()

        # Two-turn context (runtime only — never persisted)
        self._prev_user:        Optional[str] = None
        self._prev_assistant:   Optional[str] = None
        self._prev_was_confirm: bool          = False
        self._prev_topic_words: frozenset     = frozenset()
        # Topic words at the moment of the last confirmation.
        # Used to detect false confirmations one turn later.
        self._confirmed_topic:  frozenset     = frozenset()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> ComprehensionProfile:
        if not self._path.exists():
            return ComprehensionProfile()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return ComprehensionProfile(
                comprehension_rate = float(data.get("comprehension_rate", 0.50)),
                false_confirm_rate = float(data.get("false_confirm_rate", 0.50)),
                overload_rate      = float(data.get("overload_rate",      0.50)),
                depth_rate         = float(data.get("depth_rate",         0.50)),
                total_turns        = int(data.get("total_turns",          0)),
                created_at         = float(data.get("created_at",         time.time())),
                updated_at         = float(data.get("updated_at",         time.time())),
            )
        except Exception:
            return ComprehensionProfile()

    def _save(self) -> None:
        """Write profile to disk. Lock must already be held."""
        try:
            self.profile.updated_at = time.time()
            self._path.write_text(
                json.dumps(self.profile.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[EchoStone] Save error: {exc}")

    # ── Core API ──────────────────────────────────────────────────────────────

    def observe(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool = False,
    ) -> Optional[dict]:
        """Update the comprehension profile from one conversation turn.

        Same interface as MindStone.observe() — call after every exchange.
        Thread-safe.

        The first call stores the turn for context; pattern analysis starts
        from the second call onward (two-turn lookahead).

        Parameters
        ----------
        user_message       The user's raw message text.
        assistant_message  The assistant's full response text.
        verbose            If True, return a signal report dict instead of None.

        Returns
        -------
        None by default. A dict when verbose=True:
            {
              "signal":     str | None,   # detected comprehension signal
              "is_confirm": bool,         # was this message a confirmation?
              "profile":    dict,         # same as summary()
            }
        """
        with self._lock:
            return self._observe_locked(user_message, assistant_message, verbose)

    def _observe_locked(
        self,
        user_message:      str,
        assistant_message: str,
        verbose:           bool,
    ) -> Optional[dict]:
        p   = self.profile
        cfg = self._config

        # Normalise and tokenise
        um = (user_message or "").strip().lower()
        if callable(cfg.normalise_fn):
            try:
                um = cfg.normalise_fn(um)
            except Exception:
                pass   # normalise failure: continue with raw text

        um_words = frozenset(re.findall(r"\w+", um))
        stopwords = _EN_STOPWORDS
        um_content = um_words - stopwords

        # ── Signal detection (requires at least one prior turn) ───────────────
        signal: Optional[str] = None

        if self._prev_user is not None:

            # 1. Explicit confusion — highest priority
            if _matches(um, um_words, cfg.confusion_signals):
                p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.25)
                p.overload_rate      = _ema(p.overload_rate,      1.0, 0.18)
                signal = "explicit_confusion"

            # 2. Cognitive overload — checked before confirmation
            # A very short reply after a long response signals shutdown, not understanding.
            elif (self._prev_assistant and
                  len(self._prev_assistant.split()) > self._overload_wc and
                  len(um.split()) <= 3):
                p.overload_rate = _ema(p.overload_rate, 1.0, 0.20)
                signal = "overload_deflect"

            # 3. Deepening — user builds on the answer
            elif _matches(um, um_words, cfg.deepen_signals):
                p.comprehension_rate = _ema(p.comprehension_rate, 1.0, 0.20)
                p.depth_rate         = _ema(p.depth_rate,         1.0, 0.22)
                signal = "deepening"

            # 4. Rephrase — same question, different words
            elif _rephrase_score(um_content, self._prev_topic_words) >= self._rephrase_threshold:
                p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.18)
                signal = "rephrase"
                if self._prev_was_confirm:
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 1.0, 0.28)
                    signal = "false_confirmation"

            # 5. Post-confirmation topic overlap → false or genuine confirmation
            elif self._prev_was_confirm:
                overlap = _rephrase_score(um_content, self._confirmed_topic)
                if overlap >= 0.18:
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 1.0, 0.22)
                    p.comprehension_rate = _ema(p.comprehension_rate, 0.0, 0.14)
                    signal = "false_confirmation_soft"
                else:
                    p.false_confirm_rate = _ema(p.false_confirm_rate, 0.0, 0.14)
                    p.comprehension_rate = _ema(p.comprehension_rate, 1.0, 0.14)
                    signal = "genuine_confirmation"

            # 6. Neutral — mild positive comprehension signal
            else:
                p.comprehension_rate = _ema(p.comprehension_rate, 0.65, 0.07)
                signal = "neutral"

            p.comprehension_rate = _clamp(p.comprehension_rate)
            p.false_confirm_rate = _clamp(p.false_confirm_rate)
            p.overload_rate      = _clamp(p.overload_rate)
            p.depth_rate         = _clamp(p.depth_rate)

        # ── Update state for next turn ────────────────────────────────────────
        wc = len(um.split())
        is_confirm = (
            wc <= 6
            and bool(um_words & cfg.confirmation_tokens)
            and not um.rstrip().endswith("?")
        )
        if is_confirm:
            self._confirmed_topic = self._prev_topic_words
        else:
            self._confirmed_topic = frozenset()

        self._prev_was_confirm  = is_confirm
        self._prev_user         = um
        self._prev_assistant    = assistant_message
        self._prev_topic_words  = um_content

        p.total_turns += 1
        if p.total_turns % self._save_every == 0:
            self._save()

        if verbose:
            return {
                "signal":     signal,
                "is_confirm": is_confirm,
                "profile":    self._summary_locked(),
            }
        return None

    def get_comprehension_directive(self) -> str:
        """Return a system-prompt fragment based on the comprehension profile.

        Returns "" until enough data has been observed (~8 turns).
        Append to your system prompt before each LLM call.
        Thread-safe.
        """
        with self._lock:
            p = self.profile
            if p.confidence() < self._min_conf:
                return ""

            conf  = p.confidence()
            lines: list[str] = []

            if p.false_confirm_rate > 0.58 and conf > 0.20:
                lines.append(
                    "This user sometimes signals understanding without fully grasping "
                    "the concept. After complex explanations, add a brief comprehension "
                    "checkpoint (e.g. a one-sentence summary or 'Does that make sense?')."
                )

            if p.comprehension_rate < 0.35 and conf > 0.25:
                lines.append(
                    "This user often needs more than one pass to fully understand. "
                    "Use concrete analogies, avoid abstract explanations, "
                    "and break multi-part answers into numbered steps."
                )

            if p.overload_rate > 0.62 and conf > 0.20:
                lines.append(
                    "Long responses tend to overwhelm this user. "
                    "Prefer shorter, focused answers and offer to elaborate on request."
                )

            if p.depth_rate > 0.65 and conf > 0.30:
                lines.append(
                    "This user actively builds on explanations. "
                    "You can be denser and anticipate the likely follow-up depth question."
                )

            if not lines:
                return ""

            header = "[Comprehension guidance -- internal directive, do not repeat to the user]\n"
            return header + "\n".join(f"* {l}" for l in lines)

    def summary(self) -> dict:
        """Human-readable profile summary. Thread-safe."""
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        p = self.profile
        return {
            "version":            __version__,
            "observations":       p.total_turns,
            "confidence":         f"{p.confidence() * 100:.0f}%",
            "comprehension_rate": _label(p.comprehension_rate, "low", "moderate", "high"),
            "false_confirm_rate": _label(p.false_confirm_rate, "rare", "moderate", "frequent"),
            "overload_rate":      _label(p.overload_rate,      "resilient", "moderate", "sensitive"),
            "depth_rate":         _label(p.depth_rate,         "surface", "moderate", "deep"),
        }

    def reset(self) -> None:
        """Clear the profile and delete the saved file. Thread-safe."""
        with self._lock:
            self.profile            = ComprehensionProfile()
            self._prev_user         = None
            self._prev_assistant    = None
            self._prev_was_confirm  = False
            self._prev_topic_words  = frozenset()
            self._confirmed_topic   = frozenset()
            if self._path.exists():
                self._path.unlink()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ema(current: float, new_val: float, alpha: float) -> float:
    return current * (1 - alpha) + new_val * alpha


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _rephrase_score(words_a: frozenset, words_b: frozenset) -> float:
    """Jaccard similarity between two content-word sets."""
    if not words_a or not words_b:
        return 0.0
    inter = len(words_a & words_b)
    union = len(words_a | words_b)
    return inter / union if union else 0.0


def _matches(text: str, tokens: frozenset, signals: frozenset) -> bool:
    for sig in signals:
        if " " in sig:
            if sig in text:
                return True
        else:
            if sig in tokens:
                return True
    return False


def _label(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.65:
        return high
    return mid
