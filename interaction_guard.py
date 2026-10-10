"""Deterministic classification of what the candidate just said.

This runs BEFORE any LLM call. Its job is to keep the interview an interview: requests for the
answer, hints, feedback, previous/upcoming questions, topic changes, unrelated chat and prompt
injection are recognised here and answered with fixed, pre-approved text (see refusals.py) - so
they can never be argued, coaxed or injected into an LLM response.

Coverage: English, Hindi (Devanagari) and romanised Hindi/Hinglish, since the interview can run in
any of them and STT output for Hindi is frequently mangled.

Design rules that keep this from misfiring on genuine answers:
  * Request-style intents only fire on SHORT utterances. A long utterance is an answer, even if
    it happens to contain the words "hint", "answer" or "skip list".
  * Prompt-injection phrases fire at any length, but on a long utterance they only flag the turn
    as suspicious (the answer is still evaluated, and its text is never shown to the composer).
"""
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from language_policy import noise_reason


class Intent(str, Enum):
    ANSWER = "answer"
    EMPTY = "empty"
    REPEAT = "repeat"
    SKIP = "skip"
    END_INTERVIEW = "end_interview"
    REQUEST_ANSWER = "request_answer"
    REQUEST_HINT = "request_hint"
    REQUEST_EVALUATION = "request_evaluation"
    REQUEST_PREVIOUS = "request_previous"
    REQUEST_UPCOMING = "request_upcoming"
    REQUEST_EXPLANATION = "request_explanation"
    CHANGE_TOPIC = "change_topic"
    UNRELATED = "unrelated"
    PROMPT_INJECTION = "prompt_injection"
    # "I don't know": a genuine (non-)answer. Scored, but never pressed with a follow-up.
    DONT_KNOW = "dont_know"
    # "Can I speak in Hindi?" - a request about the interview itself, never an answer to the question.
    LANGUAGE_REQUEST = "language_request"
    # A request for a language the interview does not support, or a transcript in a script no supported language uses.
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    # "Are you there?", "Can you hear me?" - the candidate checking the connection, not answering anything.
    PRESENCE_CHECK = "presence_check"
    # "Give me a second", "let me think", "my internet is slow" - a request for time, not an answer.
    NEED_TIME = "need_time"


# Intents answered with a fixed refusal/redirect and never scored as an answer.
RESTRICTED_INTENTS = frozenset({
    Intent.REQUEST_ANSWER, Intent.REQUEST_HINT, Intent.REQUEST_EVALUATION, Intent.REQUEST_PREVIOUS,
    Intent.REQUEST_UPCOMING, Intent.REQUEST_EXPLANATION, Intent.CHANGE_TOPIC, Intent.UNRELATED,
    Intent.PROMPT_INJECTION,
})
# Turns that are not an attempt to answer the current question (excluded from scoring).
NON_ANSWER_INTENTS = RESTRICTED_INTENTS | {Intent.EMPTY, Intent.REPEAT, Intent.SKIP, Intent.END_INTERVIEW,
                                           Intent.LANGUAGE_REQUEST, Intent.UNSUPPORTED_LANGUAGE,
                                           Intent.PRESENCE_CHECK, Intent.NEED_TIME}

MAX_REQUEST_WORDS = 35
MAX_CONTROL_WORDS = 20  # skip / end-interview phrases
MAX_SKIP_WORDS = 14


@dataclass(frozen=True)
class Classification:
    intent: Intent
    rule: str = ""
    suspicious: bool = False  # injection-like text inside an otherwise normal answer


def _p(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


_INJECTION = _p(
    r"\bignore\s+(?:all\s+|any\s+|the\s+|your\s+|my\s+|these\s+|those\s+)*(?:previous|prior|above|earlier|preceding|initial|original|interview)\s+(?:instructions?|rules?|prompts?|directions?|guidelines?|constraints?)",
    r"\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+)*(?:previous\s+|prior\s+|above\s+|earlier\s+)?(?:instructions?|rules?|prompts?|guidelines?)",
    r"\bforget\s+(?:all\s+|everything\s+|that\s+|about\s+|your\s+|the\s+)*(?:previous\s+|prior\s+|these\s+)?(?:instructions?|rules?|prompt|this\s+is\s+an?\s+interview|you\s+are\s+an?\s+(?:ai\s+)?interviewer)",
    r"\b(?:reveal|show|print|repeat|tell|display|leak|share|give|read|output)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+|hidden\s+|initial\s+|original\s+|secret\s+|internal\s+)(?:prompt|instructions?|rules|guidelines)",
    # Only a request AT the interviewer: "your system prompt" / "what is the system prompt". A candidate
    # describing their own work ("I keep the system prompt in a config service") is an answer.
    r"\byour\s+(?:system\s+|hidden\s+|secret\s+)?(?:prompt|instructions?)\b",
    r"\bwhat\s+(?:is|are|'s)\s+(?:the\s+)?system\s+prompt\b|^\W*system\s+prompt\W*$",
    r"\bdeveloper\s+mode\b|\bjailbreak\b|\bdo\s+anything\s+now\b",
    r"\byou\s+are\s+(?:now|no\s+longer)\s+(?:a|an|my|the|in|free|not|dan|unrestricted|allowed)\b",
    r"\bfrom\s+now\s+on,?\s+you\b",
    # "imagine we have a million users" / "pretend the dependency is down" are normal engineering speech.
    r"\b(?:pretend|imagine)\s+(?:that\s+)?you(?:'re|\s+are|\s+were|\s+have\s+no)\b",
    r"(?:^|[.!?]\s+|\bnow\s+|\byou\s+(?:must|should|will)\s+)(?:please\s+)?(?:act|behave|roleplay|role-play)\s+(?:as|like)\b",
    r"\bact\s+as\s+(?:a|an|my)\s+(?:tutor|teacher|assistant|chat\s?bot|friend|coach|mentor|hr|recruiter|expert|different)",
    r"\b(?:override|bypass|disable|turn\s+off)\s+(?:the\s+|your\s+)?(?:rules|restrictions|safety|guidelines|interview\s+mode|filters?)",
    r"\bthis\s+is\s+not\s+an?\s+interview\b|\bstop\s+being\s+an?\s+interviewer\b|\byou\s+are\s+not\s+an?\s+interviewer\b|\bforget\s+that\s+this\s+is\s+an?\s+interview\b",
    r"\b(?:give|award|assign|rate|mark|score|grade)\s+(?:me\s+|this\s+|my\s+(?:answer|response|interview)\s+)?(?:full|100|hundred|maximum|max\b|highest|top\s+score|perfect|10\s*/\s*10|ten\s+out\s+of\s+ten|a\s+(?:good|high|great|perfect)\s+(?:score|rating|grade))",
    r"\b(?:pass|hire|select|shortlist|recommend)\s+me\b",
    r"\bmark\s+(?:this|it|my\s+answer)\s+as\s+(?:correct|right|perfect|excellent)",
    r"\bi\s+(?:deserve|should\s+get)\s+(?:a\s+|the\s+)?(?:pass|full\s+marks|100|high\s+score|the\s+job)",
    # Hindi / Hinglish
    r"(?:पिछले|पिछली|पहले\s+के|सभी)\s*(?:सभी\s+)?(?:निर्देश|निर्देशों|नियम|इंस्ट्रक्शन)(?:\s+को)?\s*(?:भूल|अनदेखा|इग्नोर|ignore)",
    r"सिस्टम\s*प्रॉम्प्ट",
    r"नियम\s*(?:भूल|तोड़|इग्नोर)",
    r"(?:मुझे|मुझको)\s*(?:पूरे|फुल|सौ|100)\s*(?:नंबर|अंक|मार्क्स)",
    r"\b(?:pichhle|pichle|purane|sabhi)\s+(?:sabhi\s+)?(?:nirdesh|instructions?|rules?)\b.{0,15}\b(?:ignore|bhool|bhul)",
    r"\b(?:instructions?|rules?|nirdesh|niyam)\s+(?:ko\s+)?(?:ignore|bhool|bhul)",
    r"\bmujhe\s+(?:full|poore|pure|100|sau)\s+(?:marks|number|score)",
)

_END_INTERVIEW = _p(
    r"\b(?:end|stop|finish|terminate|quit|exit)\s+(?:the\s+|this\s+|my\s+)?(?:interview|session|meeting)\b",
    r"\bi\s+(?:want|wish|would\s+like|'d\s+like)\s+to\s+(?:stop|end|withdraw|quit|leave|exit)\b",
    r"\bi\s+(?:don't|do\s+not)\s+want\s+to\s+continue\b",
    r"\bcan\s+we\s+stop\s+here\b|\blet'?s\s+stop\s+here\b",
    r"\bi\s+need\s+to\s+leave\b|\bi\s+have\s+to\s+go\s+now\b",
    # Devanagari, loose around the "इंट..." stem of "interview" (STT mangles the spelling).
    r"\bइंट\w*.{0,12}(?:खत्म|खतम|ख़त्म|बंद|समाप्त)",
    r"(?:खत्म|खतम|ख़त्म|बंद|समाप्त).{0,12}\bइंट\w*",
    r"\bइंट\w*.{0,20}नहीं\s*(?:देना|करनी|करना|चाहिए)",
    r"नहीं\s*(?:देना|करनी|करना).{0,20}\bइंट\w*",
    r"आगे\s*(?:नहीं|मत)\s*बढ़|मुझे\s*जाना\s*है|मुझे\s*रुकना\s*है|बस\s*करो|बस\s*कीजिए",
    r"\binterview\w*.{0,12}(?:khatam|khatm|khtam|band|samapt)\b",
    r"(?:khatam|khatm|khtam|band|samapt).{0,12}\binterview\w*\b",
    r"\binterview\w*.{0,20}nahi[nṃ]?\s*(?:dena|karni|karna)\b",
    r"nahi[nṃ]?\s*(?:dena|karni|karna).{0,20}\binterview\w*\b",
    r"\baage\s*(?:nahi|mat)\s*badh|\bmujhe\s+jana\s+hai\b|\bbas\s+karo\b",
)

_SKIP = _p(
    r"^(?:please\s+|ok(?:ay)?,?\s+|hmm,?\s+)?(?:can\s+we\s+|could\s+we\s+|let'?s\s+|i\s+(?:want|would\s+like|'d\s+like)\s+to\s+|just\s+)?(?:skip|pass)(?:\s+on)?(?:\s+(?:this|that|it))?(?:\s+(?:one|question))?(?:\s+please)?$",
    r"\b(?:can|could)\s+we\s+(?:skip|move\s+on)\b",
    r"\bnext\s+question\s+please\b|\bpass\s+on\s+this\s+one\b|\b(?:go|move)\s+to\s+the\s+next\s+(?:question|one)\b",
    r"\bi\s+(?:want|would\s+like)\s+to\s+skip\b",
    r"स्किप|छोड़\s*(?:दो|दीजिए)\s*(?:यह\s*सवाल)?|अगला\s*सवाल\s*(?:पूछ|दीजिए|दो)|यह\s*सवाल\s*छोड़",
    r"\bskip\s*(?:kar\w*|kijiye)\b|\bagla\s*(?:sawaal|savaal|question)\s*(?:puchho|poochho|do|dijiye)\b",
)

_PREVIOUS = _p(
    r"\b(?:what|which)\s+(?:was|were|is)\s+(?:the\s+|my\s+)?(?:previous|last|earlier|first|second|third|fourth|fifth|prior|preceding|\d+(?:st|nd|rd|th))\s+(?:question|answer)s?\b",
    r"\b(?:answer|solution)\s+(?:to|for|of)\s+(?:the\s+)?(?:previous|last|earlier|first|second|third|prior)\s+question",
    r"\b(?:go|come|move|get)\s+back\s+to\s+(?:the\s+)?(?:previous|last|first|earlier|question\s+\d)",
    r"\b(?:repeat|tell\s+me|remind\s+me\s+of|read)\s+(?:me\s+)?(?:the\s+)?(?:previous|last|earlier|first|second|third)\s+question",
    r"^(?:the\s+)?(?:previous|last|earlier)\s+question(?:\s+please)?\s*\??$",
    r"\b(?:what|which)\s+(?:was|is|were)\s+question\s+(?:number\s+)?\d+\b|^question\s+(?:number\s+)?\d+\s*\??$",
    r"(?:पिछले|पिछला|पहले\s+(?:वाले|वाला))\s*(?:सवाल|प्रश्न)",
    r"\b(?:pichhla|pichhle|pichla|pehle\s+wala|pehle\s+wale)\s+(?:sawal|savaal|question)\b",
)

_UPCOMING = _p(
    r"\b(?:what|which)(?:'s|\s+is|\s+will|\s+would|\s+are)?\s+(?:the\s+|your\s+)?(?:next|upcoming|following|coming|remaining|other|later|final)\s+(?:questions?|topics?)\b",
    r"\bwhat\s+(?:will|would|are|do)\s+you\s+(?:going\s+to\s+|gonna\s+|plan\s+to\s+)?ask(?:\s+me)?(?:\s+next|\s+later)?\b",
    r"^\W*(?:so\s+|ok(?:ay)?,?\s+)?what(?:'s|\s+is)\s+next\s*\??\W*$",
    r"\b(?:tell|show|give)\s+me\s+(?:the\s+|your\s+)?(?:next|upcoming|remaining|all|other)\s+questions?\b",
    r"\bhow\s+many\s+(?:more\s+)?questions?\b",
    r"\b(?:what|which)\s+topics?\s+(?:will|are|would)\s+(?:you|we|be|there)\b",
    r"(?:अगला|अगले)\s*(?:सवाल|प्रश्न)\s*(?:क्या|कौन\s*सा|बताइए|बताओ)",
    r"\bagla\s+(?:sawal|savaal|question)\s+(?:kya|kaun|batao|bataiye)\b",
    r"आगे\s*(?:क्या|कौन\s*सा)\s*(?:पूछ|सवाल)",
)

_EVALUATION = _p(
    r"\b(?:why|how)\s+(?:is|was|are|were)\s+(?:my|the|that|this)\s+(?:answer|response|reply)\b",
    r"\b(?:was|is)\s+(?:my|that|this|the)\s+(?:answer|response|reply)\s+(?:right|correct|wrong|incorrect|good|bad|ok|okay|fine|enough)\b",
    r"\b(?:is|was)\s+(?:that|this|it)\s+(?:right|correct|wrong|incorrect|good|ok|okay)\s*\??$",
    r"\bhow\s+(?:did|am|do)\s+i\s+(?:do|doing|perform|performing|score|fare)\b",
    r"^\W*(?:what(?:'s|\s+is|\s+was)\s+)?my\s+(?:score|marks|rating|result|grade|performance|evaluation|feedback|status)\b",
    r"\b(?:tell|give|show)\s+me\s+(?:my\s+)?(?:score|marks|rating|result|feedback|performance|grade)\b",
    r"\b(?:did|will|can)\s+i\s+(?:pass|fail|clear|qualify|get\s+(?:selected|the\s+job|hired))\b",
    r"\bam\s+i\s+(?:selected|hired|shortlisted|through|passing|failing|doing\s+(?:well|ok|okay|fine|good))\b",
    r"^(?:any\s+|some\s+)?feedback(?:\s+please|\s+on\s+.{1,30})?\??$",
    r"\bwhat\s+(?:did|do)\s+i\s+(?:miss|get\s+wrong|need\s+to\s+improve)\b",
    r"\bwhere\s+did\s+i\s+(?:go\s+wrong|make\s+(?:a\s+)?mistake)\b",
    r"\bwhat(?:'s|\s+is|\s+was)\s+wrong\s+(?:with|in)\s+my\b",
    r"\bwhat\s+(?:are|is)\s+(?:the\s+)?evaluation\s+criteri",
    r"\bhow\s+(?:are|is|will)\s+(?:you|this|i|my\s+\w+)\s+(?:be\s+)?(?:evaluat|scor|grad|judg|mark)",
    r"\bwhat\s+are\s+you\s+(?:evaluat|scor|judg|looking\s+for)",
    r"मेरा\s*(?:जवाब|उत्तर)\s*(?:गलत|सही|ठीक)",
    r"मेरा\s*(?:स्कोर|मार्क्स|नंबर|रिजल्ट)",
    r"\bmera\s+(?:jawab|jawaab|answer|uttar)\s+(?:galat|sahi|theek|kaisa)",
    r"\bmera\s+(?:score|marks|number|result)\b",
    r"\bmain\s+(?:pass|fail|select)\b|\bkaisa\s+(?:raha|gaya)\b",
)

_REQUEST_ANSWER = _p(
    r"\b(?:what(?:'s|\s+is|\s+was|\s+would\s+be)|tell\s+me|give\s+me|show\s+me|share|say)\s+(?:me\s+)?(?:the\s+)?(?:correct\s+|right\s+|actual\s+|exact\s+|model\s+|expected\s+|proper\s+|real\s+)?(?:answer|solution)\b",
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell|give|share|say|provide|show)(?:\s+me)?\s+(?:the\s+)?(?:correct\s+|right\s+)?(?:answer|solution)\b",
    r"\bi\s+(?:want|need|would\s+like|'d\s+like)\s+(?:to\s+know\s+)?the\s+(?:answer|solution)\b",
    r"\b(?:what|which)\s+is\s+the\s+(?:correct|right)\s+(?:answer|way|approach|solution)\b",
    r"सही\s*(?:उत्तर|जवाब)\s*(?:क्या|बताइए|बताओ|बता\s*दो|बताएं)",
    r"(?:उत्तर|जवाब)\s*(?:क्या\s*है|बताइए|बताओ|बता\s*दीजिए)",
    r"\banswer\s+(?:kya\s+hai|batao|bataiye|bata\s+do|bata\s+dijiye)\b",
    r"\bsahi\s+(?:jawab|jawaab|answer|uttar)\s+(?:kya|batao|bataiye)\b",
)

_REQUEST_HINT = _p(
    r"\b(?:give|provide|share|need|want|got\s+any|any)\s+(?:me\s+)?(?:a\s+|an\s+|some\s+|any\s+)?(?:small\s+|little\s+|quick\s+|tiny\s+)?(?:hint|clue|tip|pointer|nudge)s?\b",
    r"\b(?:can|could)\s+(?:i|you)\s+(?:get|have|give\s+me)\s+(?:a\s+|an\s+|some\s+)?(?:small\s+|little\s+|quick\s+)?(?:hint|clue|tip|pointer|help)\b",
    r"^(?:a\s+|any\s+)?(?:hint|clue)s?(?:\s+please)?\??$",
    r"\b(?:help|guide)\s+me\s+(?:with|on|to)\b",
    r"\bcan\s+you\s+help\s+me\s+(?:with|answer|to\s+answer|understand)\b",
    r"\bgive\s+me\s+(?:a\s+)?(?:direction|idea|starting\s+point)\b",
    r"(?:हिंट|इशारा|संकेत|सुराग)\s*(?:दो|दीजिए|दीजिये|चाहिए|दे\s*दो|दे\s*दीजिए)",
    r"कोई\s*(?:हिंट|संकेत|इशारा)",
    r"(?:मदद|हेल्प)\s*(?:कीजिए|कीजिये|करो|कर\s*दो|चाहिए)",
    r"\b(?:thoda\s+)?(?:hint|clue)\s+(?:do|dijiye|chahiye|dena|de\s+do)\b",
    r"\bmadad\s+(?:kijiye|karo|chahiye|kar\s+do)\b",
)

_CHANGE_TOPIC = _p(
    r"\b(?:can|could|shall|may)\s+(?:we|i|you)\s+(?:please\s+)?(?:talk|speak|discuss|chat|move|switch|change|go|start)(?:\s+about)?\s+(?:something|anything|another|a\s+different|some\s+other|other|a\s+new)\b",
    r"\b(?:let'?s|lets)\s+(?:talk|speak|discuss|chat|switch|change|move)\s+(?:about|to|on)\s+(?:something|another|a\s+different|other|some\s+other)",
    r"\bchange\s+(?:the\s+|this\s+)?(?:topic|subject)\b",
    r"\bi\s+(?:don't|do\s+not)\s+want\s+to\s+(?:talk|discuss|answer)\s+(?:about\s+)?(?:this|that)\b",
    r"\b(?:talk|speak)\s+about\s+something\s+else\b|\bdifferent\s+topic\b",
    r"कुछ\s*और\s*(?:बात|बातें)|विषय\s*बदल",
    r"\btopic\s+badal|\bkuch\s+aur\s+(?:baat|baatein)\b|\bdoosri\s+baat\b",
)

_UNRELATED = _p(
    r"\bhow(?:'s|\s+is)\s+the\s+(?:weather|temperature)\b|\bwhat(?:'s|\s+is)\s+the\s+(?:weather|temperature)\b|\b(?:is\s+it|will\s+it)\s+rain",
    r"\b(?:tell|sing|play)\s+(?:me\s+)?(?:a\s+)?(?:joke|song|story|poem)\b|\bsing\s+(?:a\s+)?song\b",
    r"^\W*(?:who|what|which|tell\s+me|did\s+you|have\s+you|do\s+you)\b.{0,40}\b(?:cricket|ipl|football|match\s+score|movie|film|bollywood)\b",
    r"^\W*(?:what|whats|what's|tell\s+me|give\s+me|how)\b.{0,30}\b(?:stock\s+price|bitcoin|crypto\s+price|recipe|horoscope)\b",
    r"\bwhat\s+(?:time|day|date)\s+is\s+it\b|\bwho\s+(?:won|is\s+the\s+(?:prime\s+minister|president))\b",
    r"मौसम|चुटकुला|गाना\s*सुना|क्रिकेट|फिल्म",
    r"\bmausam\b|\bjoke\s+suna|\bgaana\s+suna",
)

_REPEAT = _p(
    # A request ("repeat that", "please rephrase the question"), not the verb inside an answer ("we repeat the call").
    r"^\W*(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:repeat|rephrase|reword)\b(?!.{0,20}\b(?:previous|last|earlier|first|second|third)\b)",
    r"\b(?:repeat|rephrase|reword)\s+(?:that|it|this|the\s+question|your\s+question|please|again)\b(?!.{0,20}\b(?:previous|last|earlier|first|second|third)\b)",
    # "once more" inside an answer ("retry the failed ones once more") is not a request: only as one, or as the closer.
    r"\bsay\s+(?:that|it)\s+again\b|\bcome\s+again\b"
    r"|^\W*(?:(?:can|could)\s+you\s+)?(?:please\s+)?(?:say\s+(?:that|it)\s+)?(?:once\s+more|one\s+more\s+time)\b"
    r"|\b(?:once\s+more|one\s+more\s+time|again)\s+please\W*$",
    r"^(?:pardon|sorry|what|huh|excuse\s+me)\s*[?.!]*$",
    r"\bi\s+(?:didn't|did\s+not|couldn't|could\s+not|missed|lost)\s+(?:catch|hear|get|understand|follow)\b",
    r"\b(?:didn't|did\s+not)\s+(?:catch|hear|get|understand)\s+(?:that|the\s+question|you|it)\b",
    r"\bwhat\s+(?:was|is)\s+the\s+question\b|\bwhat\s+did\s+you\s+(?:say|ask)\b",
    r"\bi\s+(?:don'?t|do\s+not)\s+(?:understand|get)\s+(?:the|this|that|your|what\s+you)\b|\bdon'?t\s+know\s+what\s+you\s+mean\b",
    r"\b(?:can|could)\s+you\s+(?:please\s+)?(?:ask|say)\s+(?:that|it|the\s+question)\s+(?:again|differently|another\s+way)\b",
    r"\bask\s+(?:it|that|the\s+question)\s+(?:again|differently)\b",
    # audio trouble on the candidate's side: the sensible reply is to say it again
    r"\b(?:can't|cannot|couldn't|could\s+not|can\s+not|not\s+able\s+to)\s+hear\s+(?:you|that|anything|properly|clearly)\b",
    r"\b(?:your\s+)?(?:voice|audio|sound)\s+(?:is\s+|was\s+)?(?:breaking|cutting|not\s+clear|unclear|low|choppy|lagging)\b",
    r"\bspeak\s+(?:a\s+(?:little|bit)\s+)?(?:slower|slowly|louder|clearly|up)\b|\bslow\s+down\b",
    r"आवाज़\s*(?:नहीं|साफ़\s*नहीं)\s*आ\s*रही|धीरे\s*(?:बोलिए|बोलो)|ज़ोर\s*से\s*(?:बोलिए|बोलो)",
    r"\bawaaz\s+(?:nahi|saaf\s+nahi)\s+aa\s+rahi\b|\bdheere\s+bol", 
    r"(?:दोबारा|फिर\s*से|एक\s*बार\s*फिर|रिपीट)\s*(?:बोलिए|बोलो|बताइए|बताओ|कहिए|कीजिए|करो|कर\s*दीजिए|पूछिए)",
    r"(?:सुनाई|समझ)\s*नहीं\s*(?:आया|दिया|आई)",
    r"\bdobara\s+(?:bolo|boliye|batao|bataiye|poochho|puchiye)\b|\bphir\s+se\s+(?:bolo|boliye|batao|poochho|puchiye)\b",
    r"\bsamajh\s+nahi\s+aaya\b|\bsunai\s+nahi\s+diya\b|\brepeat\s+kar(?:o|iye)\b",
)

_EXPLANATION = _p(
    r"^(?:please\s+)?(?:can|could|would)\s+you\s+(?:please\s+)?(?:explain|teach|define|describe|elaborate\s+on|walk\s+me\s+through|help\s+me\s+understand)\s+(?!the\s+question\b)",
    r"^(?:please\s+)?(?:explain|teach\s+me|define|describe)\s+(?!the\s+question\b)\S+(?!.*\b(?:i|we)\s+(?:would|will|can|'d|use|used|have)\b)",
    r"\bwhat\s+does\s+.{1,40}\s+mean\b",
    r"^(?:what|whats|what's)\s+(?:is|are|does|do)\s+(?:a\s+|an\s+|the\s+)?[\w\-.+#/]+(?:\s+[\w\-.+#/]+){0,4}\s*\??$",
    r"\b(?:meaning|definition)\s+of\b",
    r"(?:.{1,40})\s*(?:क्या|kya)\s*(?:होता|hota)\s*(?:है|hai)\s*\??$",
    r"\b(?:iska|iske)\s+(?:matlab|arth)\b|(?:मतलब|अर्थ)\s*(?:क्या|बताइए|बताओ)",
)

# Whole-utterance "I don't know" answers. Anchored at the start, short, and rejected if the candidate
# goes on to attempt an answer ("not sure, but maybe it uses hashing") - that is a partial answer.
# Short utterances only (see classify): a long answer that happens to contain "let me think" is an answer.
_PRESENCE = _p(
    r"\bare\s+you\s+(?:still\s+)?(?:there|listening|here|with\s+me)\b",
    r"\bcan\s+(?:you|u)\s+(?:hear|see)\s+me\b", r"\bcould\s+you\s+hear\s+me\b", r"\bcan\s+you\s+hear\b",
    r"\bis\s+(?:anyone|anybody|someone)\s+(?:there|here)\b", r"\bam\s+i\s+(?:audible|visible|clear)\b",
    r"^\W*(?:hello|hi|hey)\b[\W\s]*(?:(?:hello|hi|hey)\b[\W\s]*)+$",
    r"(?:आप|आपको)\s*(?:मुझे\s*)?(?:सुन\s*(?:पा\s*रहे|रहे|सकते)|सुनाई)", r"सुनाई\s*दे\s*रहा", r"कोई\s*है\s*\??",
    r"\bsunai\s+de\s+raha\b", r"\bkya\s+aap\s+(?:sun|hain)\b",
)
_NEED_TIME = _p(
    r"\b(?:give|gimme)\s+me\s+(?:a\s+|one\s+)?(?:second|sec|moment|minute|min|bit|while|some\s+time)\b",
    r"\b(?:just\s+)?(?:one|a)\s+(?:second|moment|minute|sec)\b", r"\bhold\s+on\b",
    r"\bwait\s+(?:a\s+)?(?:second|moment|minute|please)\b", r"\b(?:let\s+me|lemme)\s+(?:think|see|recall|remember)\b",
    r"\bcan\s+i\s+(?:take|have)\s+(?:a\s+)?(?:second|moment|minute|min|some\s+time)\b",
    r"\bi\s+need\s+(?:a\s+)?(?:second|moment|minute|some\s+time|to\s+think)\b",
    r"\bmy\s+(?:mic|microphone|internet|connection|wifi|network|audio|camera)\s+(?:is|isn't|is\s+not|has|keeps|was)\b",
    r"(?:एक|थोड़ा)\s*(?:मिनट|सेकंड|रुकिए|समय)", r"रुकिए|रुको|सोचने\s*दीजिए",
    r"\bek\s+(?:minute|second|min)\b", r"\bthoda\s+(?:time|ruk|sochne)\b", r"\bsochne\s+do\b", r"\bruk(?:o|iye|iyega)\b",
)
MAX_PRESENCE_WORDS = 9

_DONT_KNOW = _p(
    r"^(?:(?:um+|uh+|well|honestly|sorry|actually|so)[,.\s]+)*(?:i\s+)?(?:really\s+|honestly\s+)?"
    r"(?:(?:don'?t|do\s+not)\s+know|have\s+no\s+idea|no\s+idea|(?:am|i'?m)\s+not\s+sure|not\s+sure|"
    r"(?:can'?t|cannot)\s+(?:say|recall|remember)|(?:haven'?t|have\s+not|never)\s+(?:worked|used|done|tried|heard|come\s+across))\b",
    r"^(?:मुझे\s+)?(?:नहीं\s+पता|पता\s+नहीं|मालूम\s+नहीं|नहीं\s+मालूम|कोई\s+आइडिया\s+नहीं|याद\s+नहीं)",
    r"^(?:mujhe\s+)?(?:pata\s+nahi|nahi\s+pata|malum\s+nahi|nahi\s+malum|idea\s+nahi|yaad\s+nahi)\b",
)
_PARTIAL_ANSWER_HINTS = re.compile(
    r"\b(?:but|however|maybe|perhaps|probably|i\s+think|i\s+guess|it\s+(?:is|might|could)|would|might)\b|"
    r"don'?t\s+know\s+what\s+(?:you|that|this|it)\b|लेकिन|मगर|शायद|\bshayad\b|\blekin\b|\bpar\b", re.I)
MAX_DONT_KNOW_WORDS = 10

# Order matters: more specific intents first.
_RULES: list[tuple[Intent, list[re.Pattern]]] = [
    (Intent.REQUEST_PREVIOUS, _PREVIOUS),
    (Intent.REQUEST_UPCOMING, _UPCOMING),
    (Intent.REQUEST_EVALUATION, _EVALUATION),
    (Intent.REQUEST_ANSWER, _REQUEST_ANSWER),
    (Intent.REQUEST_HINT, _REQUEST_HINT),
    (Intent.CHANGE_TOPIC, _CHANGE_TOPIC),
    (Intent.UNRELATED, _UNRELATED),
    (Intent.REPEAT, _REPEAT),
    (Intent.REQUEST_EXPLANATION, _EXPLANATION),
]

# Whisper-style hallucinations on near-silence, and pure fillers: not an answer.
_FILLER_ONLY = re.compile(
    r"^(?:u+h+m*|u+m+|h+m+|m+|a+h+|e+r+|uh[- ]huh|mm[- ]hmm)$", re.I,
)
_HALLUCINATIONS = frozenset({
    "thank you", "thank you.", "thanks for watching", "thank you for watching", "you", "bye", "bye bye",
})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def contains_injection(text: str) -> bool:
    text = _normalize(text)
    return any(p.search(text) for p in _INJECTION)


def _matches(patterns: list[re.Pattern], text: str) -> str | None:
    for i, p in enumerate(patterns):
        if p.search(text):
            return f"#{i}"
    return None


# Wording that marks someone ANSWERING (describing what they would/did do), as opposed to asking the interviewer for
# something. Used with overlap against the current question, never on its own.
_ANSWER_CUES = re.compile(
    r"\b(?:i|we)\s+(?:would|will|can|could|should|'d|'ll|use|used|have|had|am|was|were|do|did|need|first|then|also|usually|"
    r"generally|typically|think|believe|built|build|designed|design|implemented|implement|handle|handled|check|checked|add|added)\b"
    r"|\b(?:because|so\s+that|for\s+example|for\s+instance|in\s+order\s+to|which\s+means|that\s+way|as\s+a\s+result|"
    r"first(?:ly)?|secondly|finally|then\s+(?:i|we)|after\s+that)\b",
    re.I)
MIN_SUBSTANTIVE_WORDS = 9


def is_substantive_answer(text: str, context: "Iterable[str] | None") -> bool:
    """True when the utterance reads as a genuine attempt at THIS question: long enough, phrased as an answer, and
    sharing real vocabulary with the question, its topic and its expected topics. Pure lexical code - no LLM.

    A request for a hint or the answer to the interviewer is short and has no overlap worth the name; a technical
    answer that happens to contain "repeat", "my score" or "film" shares several concepts with the question.
    """
    if not context or word_count(text) < MIN_SUBSTANTIVE_WORDS:
        return False
    from topic_guard import stems          # local import: topic_guard is optional, lexical, and cheap
    anchors = set()
    for item in context:
        anchors |= stems(str(item))
    overlap = len(stems(text) & anchors)
    cues = bool(_ANSWER_CUES.search(text))
    return overlap >= 3 or (cues and overlap >= 1)


def classify(text: str, *, context: "Iterable[str] | None" = None) -> Classification:
    """What kind of turn is this? `context` (the current question's text/topic/expected topics) lets a substantive,
    on-topic answer skip the request-style rules, which only make sense for short utterances aimed at the interviewer."""
    text = _normalize(text)
    words = word_count(text)
    stripped = text.strip(" .,!?;:-—")

    if not any(ch.isalnum() for ch in text):
        return Classification(Intent.EMPTY, "no-alnum")
    if words <= 3 and (_FILLER_ONLY.match(stripped) or stripped.lower() in _HALLUCINATIONS):
        return Classification(Intent.EMPTY, "filler")

    noise = noise_reason(text)
    if noise:
        return Classification(Intent.EMPTY, noise)

    substantive = is_substantive_answer(text, context)
    if words <= MAX_PRESENCE_WORDS and not substantive:
        for presence_intent, presence_patterns in ((Intent.PRESENCE_CHECK, _PRESENCE), (Intent.NEED_TIME, _NEED_TIME)):
            rule = _matches(presence_patterns, text)
            if rule:
                return Classification(presence_intent, rule)
    if any(p.search(text) for p in _INJECTION):
        if words <= MAX_REQUEST_WORDS and not substantive:
            return Classification(Intent.PROMPT_INJECTION, "injection")
        # A long (or substantive, on-topic) answer that happens to contain injection-like text - e.g. a candidate
        # explaining prompt-injection defences: still evaluate it as an answer, but flag it so its raw text is never
        # passed on to the composer.
        return Classification(Intent.ANSWER, "injection-in-long-answer", suspicious=True)

    if words <= MAX_CONTROL_WORDS:
        rule = _matches(_END_INTERVIEW, text)
        if rule:
            return Classification(Intent.END_INTERVIEW, rule)
    if words <= MAX_SKIP_WORDS:
        rule = _matches(_SKIP, text)
        if rule:
            return Classification(Intent.SKIP, rule)

    if words <= MAX_REQUEST_WORDS and not substantive:
        for intent, patterns in _RULES:
            rule = _matches(patterns, text)
            if rule:
                return Classification(intent, rule)

    if words <= MAX_DONT_KNOW_WORDS and not _PARTIAL_ANSWER_HINTS.search(text):
        rule = _matches(_DONT_KNOW, text)
        if rule:
            return Classification(Intent.DONT_KNOW, rule)

    return Classification(Intent.ANSWER)
