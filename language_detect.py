"""Which language is the candidate actually speaking? (from the transcript of what they just said)

Used to let the interview follow the candidate: if they start answering in Hindi, the interviewer's voice, writing and
listening move to Hindi without being asked. Pure text rules (no model, no network), so it is fast and testable.

  en        English
  hi        Hindi, mostly Devanagari
  hinglish  a mix: Devanagari with Latin words, or Hindi written in Roman letters ("mujhe yeh project pasand hai")

It only returns a code (or None when the utterance is too short to judge). Nothing the candidate says is ever copied
into a prompt from here.
"""
import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z]+|[ऀ-ॿ]+")

# Common Hindi words as they are typed in Roman letters. Words that are also ordinary English words (main, to, me, par,
# the, hi...) are deliberately left out so English sentences are not mistaken for Hindi.
_ROMAN_HINDI = frozenset("""
hai hain hoon hu ho tha thi the hoga hogi hote hota hoti raha rahi rahe kiya kiye kar karna karte karta karti karenge
nahi nahin mat na kya kyun kyu kaise kab kahan kaun kitna kitne yeh ye woh wo hum tum aap apna apni apne mera meri
mere mujhe mujhko tumhe aapko humein hume unhe usne maine humne aur lekin magar ya bhi sirf bahut thoda zyada kam
accha acha achha theek thik sahi galat abhi phir fir kal aaj ab jab tab agar toh isliye kyunki kyonki mein ka ki ke
ko se ne tak liye bina saath baad pehle pahle andar bahar upar neeche bol bolo bolna baat batao bataiye samajh samjha
pata kaam naukri sawal jawab chahiye chahta chahti sakta sakti sakte lagta lagti pasand
""".split())

# Very common English function words, to recognise plain English.
_ENGLISH = frozenset("""
the a an is are was were be been am i you we they he she it my your our their this that these those and or but if so
because with without for from to of in on at by as do does did have has had not no yes can could would should will
what when where why how which who about into over after before then than also just very more most some any
""".split())

MIN_WORDS = 3


def classify_language(text: str) -> str | None:
    """'en' | 'hi' | 'hinglish', or None when there is too little to judge."""
    if not text:
        return None
    words = _WORD.findall(text)
    if len(words) < MIN_WORDS:
        return None
    dev = len(_DEVANAGARI.findall(text))
    lat = len(_LATIN.findall(text))
    letters = dev + lat
    if letters == 0:
        return None
    dev_ratio = dev / letters
    if dev_ratio >= 0.85:
        return "hi"
    if dev_ratio >= 0.15:
        return "hinglish"                                      # Devanagari and Latin in one utterance
    latin_words = [w.lower() for w in words if _LATIN.match(w)]
    hindi_hits = sum(1 for w in latin_words if w in _ROMAN_HINDI)
    english_hits = sum(1 for w in latin_words if w in _ENGLISH)
    if hindi_hits >= 2 and hindi_hits / max(len(latin_words), 1) >= 0.25 and hindi_hits >= english_hits:
        return "hinglish"                                      # Hindi typed in Roman letters
    return "en"


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


class LanguageFollower:
    """Decides, turn by turn, whether the interview language should follow what the candidate is speaking.

    Cautious on purpose, because a wrong switch is more annoying than a missed one:
      * short replies ("yes", "okay", a name) are never judged;
      * English -> Hindi/Hinglish needs a clear utterance (4+ words) or two in a row;
      * Hindi/Hinglish -> English needs a longer English utterance (6+ words) twice in a row, because technical answers
        inside a Hinglish interview are often mostly English;
      * after any switch (automatic or requested) it waits two turns before it will switch again.
    """
    COOLDOWN_TURNS = 2

    def __init__(self):
        self._pending = 0
        self._turns_since_switch = self.COOLDOWN_TURNS

    def note_switch(self) -> None:
        """Call whenever the language changed (including an explicit request)."""
        self._pending = 0
        self._turns_since_switch = 0

    def observe(self, text: str, current: str) -> str | None:
        """The language to switch to for this utterance, or None to stay."""
        self._turns_since_switch += 1
        detected = classify_language(text)
        if detected is None:
            return None
        in_english = current == "en"
        speaking_english = detected == "en"
        if in_english == speaking_english:
            self._pending = 0                                    # same language family: nothing to do
            return None
        if self._turns_since_switch <= self.COOLDOWN_TURNS:
            return None
        self._pending += 1
        words = word_count(text)
        needed_words, needed_repeats = (4, 1) if in_english else (6, 2)
        if (words >= needed_words and self._pending >= needed_repeats) or self._pending >= 2 + (0 if in_english else 1):
            self._pending = 0
            return detected
        return None
