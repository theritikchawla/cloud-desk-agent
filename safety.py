"""Safety checks that run in normal Python code, NOT inside the AI prompt.

A prompt alone can be tricked. These checks are a second layer:
  * detect_injection        - blocks "ignore your instructions" style messages
  * neutralise_context      - removes such lines from documentation before it reaches Gemini
  * find_sensitive / redact - spots passwords and card numbers so we never store or send them
  * is_human_request        - spots "I want to talk to a person"
  * check_answer            - inspects Gemini's answer before the customer sees it

None of this is perfect. It is a prototype, not a production security system.
"""
import hashlib
import re

# A secret word hidden in the system prompt. If it ever appears in an answer, the prompt leaked.
CANARY = "CANARY-" + hashlib.sha256(b"clouddesk-prompt-canary").hexdigest()[:10]

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+|my\s+)*(previous|prior|above|earlier)?\s*(instructions|rules|prompts?|guidelines)",
    r"disregard\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above)?\s*(instructions|rules|prompts?|guidelines)",
    r"forget\s+(all\s+|everything\s+|your\s+|the\s+)*(previous\s+|prior\s+)?(instructions|rules|prompts?)",
    r"forget\s+everything\b",
    r"override\s+(your|the|all)\s+(instructions|rules|safety|guidelines|settings)",
    r"(reveal|show|print|repeat|tell|display|leak|give)\s+(me\s+)?(your|the)\s+(hidden\s+|secret\s+|system\s+|internal\s+)*(prompt|instructions|rules|api\s*key|secrets?|password|configuration)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions|rules)",
    r"\bsystem\s+prompt\b",
    r"you\s+are\s+now\s+(?!a\s+customer)",
    r"\b(developer|debug|jailbreak|god)\s+mode\b",
    r"\bjailbreak\b",
    r"\bpretend\s+(that\s+)?you\s+(have\s+no|are\s+not|aren't|can)\b",
    r"new\s+instructions?\s*:",
]
_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PASSWORD = re.compile(r"(\b(?:my\s+|the\s+|new\s+|old\s+)?pass(?:word|code)\s*(?:is|=|:)\s*)(\S+)", re.IGNORECASE)
_CVV = re.compile(r"(\b(?:cvv|cvc|security\s+code)\s*(?:is|=|:)?\s*)(\d{3,4})\b", re.IGNORECASE)
_HUMAN = re.compile(
    r"\b(human|real\s+person|live\s+agent|support\s+agent|customer\s+service|representative|"
    r"talk\s+to\s+(?:someone|somebody|a\s+person|an\s+agent)|speak\s+(?:to|with)\s+(?:someone|somebody|a\s+person|an\s+agent)|"
    r"create\s+(?:a\s+)?(?:support\s+)?ticket|open\s+(?:a\s+)?(?:support\s+)?ticket|escalate)\b",
    re.IGNORECASE,
)


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def neutralise_context(text: str) -> str:
    """Remove lines of documentation that try to give the AI orders."""
    kept = []
    for line in text.splitlines():
        kept.append("[removed suspicious line]" if detect_injection(line) else line)
    return "\n".join(kept)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _card_matches(text: str):
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            yield m


def find_sensitive(text: str) -> list[str]:
    """Return which kinds of sensitive data are in the text: 'card', 'cvv', 'password'."""
    found = []
    if any(True for _ in _card_matches(text)):
        found.append("card")
    if _CVV.search(text):
        found.append("cvv")
    if _PASSWORD.search(text):
        found.append("password")
    return found


def redact(text: str) -> str:
    """Hide sensitive values so they are not stored in the database."""
    for m in sorted(_card_matches(text), key=lambda m: m.start(), reverse=True):
        text = text[: m.start()] + "[REDACTED CARD]" + text[m.end():]
    text = _CVV.sub(r"\1[REDACTED]", text)
    return _PASSWORD.sub(r"\1[REDACTED]", text)


def is_human_request(text: str) -> bool:
    return bool(_HUMAN.search(text))


# ---------- checking the AI's answer ----------
_MONEY = re.compile(r"([$€£])\s?(\d[\d,]*(?:\.\d+)?)")
_QUANTITY = re.compile(r"(\d+(?:\.\d+)?)[\s-]*(days?|hours?|minutes?|business\s+days?|%|percent)", re.IGNORECASE)
_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def _money_set(text: str) -> set[str]:
    out = set()
    for symbol, number in _MONEY.findall(text):
        number = number.replace(",", "")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        out.add(symbol + number)
    return out


def _quantity_set(text: str) -> set[tuple[str, str]]:
    out = set()
    for number, unit in _QUANTITY.findall(text):
        unit = unit.lower().replace("business ", "")
        unit = "day" if unit.startswith("day") else "hour" if unit.startswith("hour") else \
            "minute" if unit.startswith("minute") else "%"
        out.add((str(float(number)).rstrip("0").rstrip("."), unit))
    return out


def check_answer(answer: str, context: str, api_key: str = "") -> tuple[bool, str]:
    """Return (is_safe, reason). Runs before an answer is shown or saved."""
    if api_key and api_key in answer:
        return False, "secret"
    if _KEY_SHAPE.search(answer) or CANARY in answer:
        return False, "secret"
    # Every price and every number of days/hours/percent must appear in the documentation we sent.
    if not _money_set(answer) <= _money_set(context):
        return False, "unsupported_number"
    if not _quantity_set(answer) <= _quantity_set(context):
        return False, "unsupported_number"
    return True, ""
