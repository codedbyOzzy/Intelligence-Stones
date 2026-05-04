"""
Turkish signal sets for Mind Stone and Echo Stone.

Usage
-----
    from mind_stone import MindStone
    from echo_stone import EchoStone
    from signals_turkish import TR_CONFIG, TR_ECHO_CONFIG

    mind = MindStone(config=TR_CONFIG)
    echo = EchoStone(config=TR_ECHO_CONFIG)

Note: All signal strings are stored in ASCII (diacritics stripped) because
both engines normalise user input via the ``normalise_fn`` before matching.
This avoids encoding mismatches on Windows terminals.
"""

from mind_stone import SignalConfig
from echo_stone import EchoConfig

# ── Normalisation ─────────────────────────────────────────────────────────────

# The source string below contains Turkish characters intentionally —
# they are the *keys* of the mapping table that strips diacritics.
# Without them, _norm_tr() cannot convert Turkish input to ASCII for matching.
_TR_MAP = str.maketrans(
    "çğışöüÇĞİŞÖÜ",   # Turkish diacritics (source)
    "cgisouCGISOu",    # ASCII equivalents  (target)
)


def _norm_tr(text: str) -> str:
    """Strip Turkish diacritics for ASCII comparison."""
    return text.translate(_TR_MAP)


# ── Mind Stone signal sets (all ASCII after normalisation) ────────────────────

_TR_NEG_VERBOSITY = frozenset({
    "kisalt", "kisa tut", "kisa kes", "cok uzun", "uzun oldu",
    "daha kisa", "ozet gec", "ozetle", "ozet ver", "sadece soyle",
    "gerek yok", "yeter", "dur", "tamam tamam",
})

_TR_POS_VERBOSITY = frozenset({
    "devam et", "daha fazla", "detaylandir", "acar misin", "biraz daha",
    "anlat", "acikla", "devam", "genislet", "anlat bakalim", "detayli anlat",
    "tam anlat", "tam olarak", "nasil yani",
})

_TR_EXAMPLE_SIGNALS = frozenset({
    "ornek ver", "ornek goster", "nasil gorunur", "nasil yapilir",
    "goster bana", "mesela", "ornek", "kod goster", "nasil kullanilir",
    "pratikte", "uygulamada", "demo", "somut",
})

_TR_THEORY_SIGNALS = frozenset({
    "neden", "nasil calisir", "mantigi ne", "arkasinda ne var",
    "neden boyle", "ne ise yarar", "ne icin", "temeli ne",
    "ilkesi", "prensibi", "amaci ne", "sebebi ne",
})

_TR_SATISFIED_TOKENS = frozenset({
    "tamam", "anladim", "tesekkurler", "super", "harika", "mukemmel",
    "ok", "oldu", "guzel", "evet", "tamamdir", "tmm", "iyi", "sag ol",
})

_TR_TECH_WORDS = frozenset({
    # Programming languages (same in Turkish context)
    "python", "javascript", "typescript", "rust", "golang", "java",
    "sql", "bash", "powershell",
    # English tech terms used in Turkish
    "api", "async", "await", "thread", "queue", "class", "function",
    "json", "yaml", "regex", "token", "stream", "buffer",
    "gpu", "cuda", "cpu", "ram", "embedding", "vector", "model",
    "inference", "fine-tune", "rag", "prompt", "llm",
    "docker", "kubernetes", "git", "linux",
    "neural", "transformer", "gradient",
    # Turkish technical vocabulary
    "fonksiyon", "degisken", "dongu", "sinif", "nesne", "dizi",
    "veritabani", "sorgu", "sunucu", "istemci", "protokol",
    "algoritma", "bellek", "islemci", "hata", "debug", "test",
})

TR_CONFIG = SignalConfig(
    neg_verbosity    = _TR_NEG_VERBOSITY,
    pos_verbosity    = _TR_POS_VERBOSITY,
    example_signals  = _TR_EXAMPLE_SIGNALS,
    theory_signals   = _TR_THEORY_SIGNALS,
    satisfied_tokens = _TR_SATISFIED_TOKENS,
    tech_words       = _TR_TECH_WORDS,
    normalise_fn     = _norm_tr,
)


# ── Echo Stone signal sets ────────────────────────────────────────────────────

_TR_ECHO_CONFUSION = frozenset({
    # Explicit confusion
    "anlamadim", "anlayamadim", "anlasilmadi", "anlamiyorum",
    "tekrar", "tekrar anlatir misin", "tekrar aciklar misin",
    "baska turlu", "farkli anlatir misin", "daha basit",
    "ne demek", "ne anlama geliyor", "kafam karisti",
    "nasil yani", "yani ne", "ne demeye calisiyor",
    # Implicit confusion
    "peki ama", "ama nasil", "ama neden",
})

_TR_ECHO_CONFIRM = frozenset({
    "tamam", "anladim", "ok", "oldu", "tamamdir", "tmm",
    "peki", "guzel", "super", "harika", "anlasild",
    "mantikli", "evet", "yes", "iyi", "sag ol",
})

_TR_ECHO_DEEPEN = frozenset({
    # Building on the answer
    "peki ya", "ya da", "bir de", "ya su durumda",
    "yani demek ki", "demek ki", "o zaman",
    "su anlama mi geliyor", "soyle mi anlayacagiz",
    "bu da mi", "bu durum icin de", "benzer sekilde",
    "daha ileri gidersek", "daha da",
    "onu da sorayim", "bir sorum daha",
    "peki o zaman", "o halde",
})

TR_ECHO_CONFIG = EchoConfig(
    confusion_signals   = _TR_ECHO_CONFUSION,
    confirmation_tokens = _TR_ECHO_CONFIRM,
    deepen_signals      = _TR_ECHO_DEEPEN,
    normalise_fn        = _norm_tr,
)
