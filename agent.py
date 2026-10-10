"""The realtime AI interviewer: LiveKit glue around the interview engine.

Architecture (see conductor.py for the turn pipeline):
  * Deterministic code owns interview STATE - which question, follow-up counts, timers, when it
    ends. It is persisted through the API, so a restarted worker resumes the interview.
  * LLMs are used for exactly two narrow, validated jobs: evaluating an answer (policy.py) and
    phrasing the next spoken turn (composer.py). They never choose an action or change state.
  * Everything the interviewer says goes through one path - Conductor -> _speak -> session.say -
    and the session has NO conversational LLM configured, so free-form, unvalidated speech is
    structurally impossible rather than merely discouraged.
  * Audio: Sarvam Bulbul v3 over WebSocket streaming (sarvam_tts_plugin.py), with Edge TTS as a
    last-resort provider fallback. Barge-in uses LiveKit's interruption handling.
"""
import asyncio
import json
import logging
import os
import time
from collections import deque

import httpx
from livekit import rtc
from livekit.agents import (
    Agent, AgentSession, ChatContext, ChatMessage, JobContext, StopResponse, WorkerOptions, cli, inference, tts,
)
from livekit.plugins import elevenlabs as lk_elevenlabs
from livekit.plugins import groq as lk_groq
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from conductor import InterviewConductor, Speech
from config import (AGENT_NAME, AGENT_SERVICE_KEY, API_BASE_URL, ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_PRONUNCIATION_DICTIONARY_ID,
                    ELEVENLABS_PRONUNCIATION_VERSION_ID, ELEVENLABS_VOICE_ID, ELEVENLABS_VOICE_ID_FEMALE,
                    ELEVENLABS_VOICE_ID_MALE, GROQ_API_KEY, AUTO_LANGUAGE,
                    OPENAI_API_KEY, SARVAM_API_KEY, TTS_PROVIDER, TTS_VOICE, VOICE_PROVIDER_FALLBACK)
from edge_tts_plugin import EdgeTTS
from languages import LANGUAGE_CONFIGS, normalize_language
import refusals
import turn_trace
from observability import attach_metrics_logging
from roles import is_plain_language_role
from runner import InterviewRunner
from sarvam_tts_plugin import SarvamTTS, speech_units
from speech_prep import prepare_for_speech
from refusals import INTERVIEWER_NAMES
from voice_config import VoiceConfig, resolve_voice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workmate-agent")


CANDIDATE_WAIT_S = float(os.getenv("CANDIDATE_WAIT_S") or 45)      # max wait for the candidate before greeting anyway
CANDIDATE_SETTLE_S = float(os.getenv("CANDIDATE_SETTLE_S") or 1.2)  # let their browser subscribe to the audio first

# Never sent to an LLM (the session has none). LiveKit's Agent requires the field.
_AGENT_DESCRIPTION = "Controlled interview conductor: every spoken turn is produced and validated by the interview engine."

# Turn handling. LiveKit's defaults are used except where an interview clearly needs different behaviour: a candidate
# thinking mid-answer pauses far longer than a casual speaker (see TURN_MIN_DELAY_S / TURN_MAX_DELAY_S below).
# Interruptions (barge-in) stay enabled with LiveKit's adaptive detection. Preemptive generation is off: nothing here
# generates speech speculatively from partial transcripts.
def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# How long the candidate may stay silent before their answer is treated as finished.
#   min_delay: silence required even when the end-of-turn model believes the sentence is complete. 0.4 s (the old
#              value) committed a turn after almost any sentence end, so a candidate who finished a sentence and then
#              thought for 1-2 s was cut off and the next fragment was answered separately.
#   max_delay: the longest the detector will wait when the model thinks the thought is unfinished.
# Both are adjustable per deployment: TURN_MIN_DELAY_S / TURN_MAX_DELAY_S. Lower them for snappier replies, raise them
# for slower, more reflective speakers. The candidate can always end their answer at once with "Done speaking".
TURN_MIN_DELAY_S = _float_env("TURN_MIN_DELAY_S", 2.0)
TURN_MAX_DELAY_S = _float_env("TURN_MAX_DELAY_S", 6.0)

TURN_HANDLING = {
    "endpointing": {"min_delay": TURN_MIN_DELAY_S, "max_delay": max(TURN_MAX_DELAY_S, TURN_MIN_DELAY_S)},
    "interruption": {"enabled": True},
    "preemptive_generation": {"enabled": False},
}


def make_turn_handling() -> dict:
    """TURN_HANDLING plus an EXPLICIT end-of-turn detector.

    Left unset, LiveKit picks one by run mode: `agent.py dev` uses LiveKit's CLOUD detector, and every
    turn waits up to 1.0 s for that network prediction (we logged "eot prediction timed out" with
    timeout=1.0, then a fall back to the local model mid-interview); `agent.py start` uses the LOCAL
    model. Dev and production therefore behaved differently and neither was a deliberate choice.

    TURN_DETECTOR=local (default) - on-device model: no network round trip in the speech path, no 1 s
                                     stall, identical in dev and prod (costs ~108 MB resident per interview)
    TURN_DETECTOR=vad             - no model at all: the turn ends after the endpointing delay of silence
                                     (fastest, least memory, but cuts off candidates who pause mid-thought)
    TURN_DETECTOR=cloud           - LiveKit's cloud model (with local fallback)
    """
    mode = (os.getenv("TURN_DETECTOR") or "local").strip().lower()
    handling = dict(TURN_HANDLING)
    if mode == "vad":
        handling["turn_detection"] = "vad"
    elif mode == "cloud":
        handling["turn_detection"] = inference.TurnDetector(version="v1")
    else:
        handling["turn_detection"] = inference.TurnDetector(version="v1-mini")
    return handling


class InterviewerAgent(Agent):
    def __init__(self, runner: InterviewRunner, *, language: str, role: str, experience_level: str,
                 plain_language: bool, interviewer_name: str = "Aarav", publish=None, candidate_ready=None,
                 on_language_change=None, auto_language: bool = False, supported_languages=None):
        super().__init__(instructions=_AGENT_DESCRIPTION)
        self.runner = runner
        self._publish = publish   # async (sender, text) -> None: live transcript to the candidate's screen
        self._caption_tasks: set = set()
        # Awaitable that returns once a candidate is actually in the room (see entrypoint). Without it the
        # greeting is spoken the moment the interviewer joins - into an empty room if the candidate is slower.
        self._candidate_ready = candidate_ready
        # Every caption line so far (bounded). Data messages are not replayed to late joiners, so this is
        # re-sent when a candidate connects; otherwise a reload or reconnect leaves the Transcript tab empty.
        self._caption_log: deque = deque(maxlen=200)
        self.conductor = InterviewConductor(
            runner, self._speak, language=language, role=role, experience_level=experience_level,
            plain_language=plain_language, interviewer_name=interviewer_name, on_language_change=on_language_change,
            auto_language=auto_language, supported_languages=supported_languages)

    async def on_enter(self):
        if self._candidate_ready is not None:
            await self._candidate_ready()
        await self.conductor.start()

    def _publish_line(self, sender: str, text: str) -> None:
        """Send a live-caption line to the candidate's screen WITHOUT delaying speech or the turn: it
        runs as its own task, and a failed caption must never affect the interview."""
        if not (self._publish and text and text.strip()):
            return
        line = (sender, text.strip(), time.time())
        self._caption_log.append(line)
        self._spawn_caption(*line)

    def _spawn_caption(self, sender: str, text: str, ts: float) -> None:
        async def _send() -> None:
            try:
                await self._publish(sender, text, ts)
            except Exception as exc:  # noqa: BLE001
                logger.debug("transcript publish failed: %s", exc)

        task = asyncio.get_running_loop().create_task(_send())
        self._caption_tasks.add(task)
        task.add_done_callback(self._caption_tasks.discard)

    async def replay_captions(self) -> int:
        """Re-send the caption history, in order, to a candidate who just connected. Returns the count."""
        if not self._publish:
            return 0
        sent = 0
        for sender, text, ts in list(self._caption_log):
            try:
                await self._publish(sender, text, ts)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("caption replay stopped: %s", exc)
                break
        return sent

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        heard = new_message.text_content or ""
        turn_trace.mark("turn_committed")
        self._publish_line("candidate", heard)
        await self.conductor.handle_candidate_turn(heard)
        raise StopResponse()  # every reply is produced by the conductor; never by a free-form LLM

    async def _speak(self, speech: Speech) -> None:
        text = prepare_for_speech(speech.text)
        if not text:
            logger.warning("nothing speakable for action=%s", speech.action)
            return
        self._publish_line("agent", text)
        turn_trace.mark("say_called")
        handle = self.session.say(text, allow_interruptions=speech.interruptible, add_to_chat_ctx=False)
        await handle  # returns when playout finishes or the candidate barges in
        if handle.interrupted:
            logger.info("event=interruption interview=%s action=%s question=%s", self.runner.interview_id,
                        speech.action, speech.question_id)


STT_VOCABULARY_HINT = ("Job interview with Arav from WorkmateIQ. Hindi, English, Hinglish, project, React, Python, Java, API, "
                       "database, team, manager, experience.")


class VoiceProviderError(RuntimeError):
    """The interview's chosen voice provider cannot be used (for example its credentials are missing)."""


# ElevenLabs models that only speak English: using one for a Hindi/Hinglish interview produces English-accented noise.
ELEVENLABS_ENGLISH_ONLY_MODELS = frozenset({"eleven_monolingual_v1", "eleven_turbo_v2", "eleven_flash_v2"})
# Models that accept an explicit language code. For these the language is PINNED, so the engine does not guess it from
# the text of each sentence (the cause of an English sentence being read with Urdu-style pronunciation, and of
# romanised Hindi being read as English).
ELEVENLABS_LANGUAGE_CODE_MODELS = frozenset({"eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"})


def is_candidate_identity(identity: str) -> bool:
    """The API names the candidate's LiveKit identity ``candidate-<id>``; nothing else may end the candidate's turn."""
    return bool(identity) and identity.startswith("candidate-")


def elevenlabs_language(language: str) -> str:
    """ISO 639-1 code ElevenLabs expects for an interview language (Hinglish is voiced as Hindi)."""
    return "en" if language == "en" else "hi"


def supported_languages_for(provider: str, model: str = "") -> frozenset[str]:
    """The interview languages this provider/model can actually SPEAK. A switch to anything else is refused with a
    clear message instead of being attempted with a voice that cannot pronounce it."""
    if provider == "elevenlabs" and (model or "") in ELEVENLABS_ENGLISH_ONLY_MODELS:
        return frozenset({"en"})
    return frozenset(LANGUAGE_CONFIGS)


def build_tts(voice: VoiceConfig) -> tuple[tts.TTS, SarvamTTS | None]:
    """Returns (tts for the session, the Sarvam instance if used so its stats can be reported).

    Only the provider recorded on the interview is used. If it cannot be (no credentials, or it fails) that is
    reported as an error - it is NEVER silently replaced by a different provider. The one exception is an explicit,
    opt-in setting: VOICE_PROVIDER_FALLBACK=edge keeps a Sarvam interview audible with Edge TTS if Sarvam is down."""
    provider = voice.provider
    if provider == "sarvam":
        if not SARVAM_API_KEY:
            raise VoiceProviderError("Voice provider 'sarvam' is selected but SARVAM_API_KEY is not configured.")
        sarvam = SarvamTTS(SARVAM_API_KEY, voice)
        sarvam.prewarm()   # connection is ready before the first sentence is spoken
        if VOICE_PROVIDER_FALLBACK == "edge":
            return tts.FallbackAdapter([sarvam, EdgeTTS(voice=voice.edge_voice)], max_retry_per_tts=0), sarvam
        return sarvam, sarvam
    if provider == "elevenlabs":
        if not ELEVENLABS_API_KEY:
            raise VoiceProviderError("Voice provider 'elevenlabs' is selected but ELEVENLABS_API_KEY is not configured.")
        if voice.language not in supported_languages_for("elevenlabs", ELEVENLABS_MODEL):
            raise VoiceProviderError(
                f"ElevenLabs model '{ELEVENLABS_MODEL}' only speaks English but this interview is in "
                f"{voice.language}. Set ELEVENLABS_MODEL to a multilingual model (e.g. eleven_flash_v2_5).")
        options = {"api_key": ELEVENLABS_API_KEY, "model": ELEVENLABS_MODEL}
        if ELEVENLABS_MODEL in ELEVENLABS_LANGUAGE_CODE_MODELS:
            options["language"] = elevenlabs_language(voice.language)
        if ELEVENLABS_PRONUNCIATION_DICTIONARY_ID and ELEVENLABS_PRONUNCIATION_VERSION_ID:
            options["pronunciation_dictionary_locators"] = [lk_elevenlabs.PronunciationDictionaryLocator(
                pronunciation_dictionary_id=ELEVENLABS_PRONUNCIATION_DICTIONARY_ID,
                version_id=ELEVENLABS_PRONUNCIATION_VERSION_ID)]
        per_gender = ELEVENLABS_VOICE_ID_FEMALE if voice.gender == "female" else ELEVENLABS_VOICE_ID_MALE
        voice_id = per_gender or ELEVENLABS_VOICE_ID
        if voice_id:
            options["voice_id"] = voice_id
        return lk_elevenlabs.TTS(**options), None
    if provider == "openai" and OPENAI_API_KEY:        # legacy value from the old TTS_PROVIDER switch
        return lk_openai.TTS(api_key=OPENAI_API_KEY, voice=TTS_VOICE), None
    return EdgeTTS(voice=voice.edge_voice), None        # legacy/dev: TTS_PROVIDER=edge and no admin selection


def _another_agent_present(room: rtc.Room) -> bool:
    return any(p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT for p in room.remote_participants.values())


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Two interviewers in one room means two voices over each other. The API avoids dispatching a
    # second one; this is the backstop if it happens anyway.
    if _another_agent_present(ctx.room):
        logger.warning("another agent is already in room %s - exiting instead of doubling up", ctx.room.name)
        ctx.shutdown(reason="duplicate agent")
        return

    interview_id = ctx.room.name.replace("interview-", "")
    async with httpx.AsyncClient(timeout=15.0) as http:
        # The full plan (question rubrics, expected topics...) is served only to this service, over the
        # authenticated agent-context endpoint. The public interview endpoint no longer carries it.
        resp = await http.get(f"{API_BASE_URL}/v1/interviews/{interview_id}/agent-context",
                              headers={"X-Agent-Key": AGENT_SERVICE_KEY} if AGENT_SERVICE_KEY else {})
        resp.raise_for_status()
        data = resp.json()

        language = normalize_language(data.get("language"))
        role_name = data.get("role", "")
        plain_language = is_plain_language_role(role_name) if role_name else False
        voice_prefs = data.get("voice") or {}
        # The provider recorded when this interview was created (an admin changing the setting mid-interview does not
        # affect it); older interviews without one use the environment default.
        voice_provider = (data.get("voice_provider") or TTS_PROVIDER).lower()
        logger.info("interview %s uses voice provider %s", interview_id, voice_provider)
        voice = resolve_voice(language, voice_prefs.get("gender"), voice_prefs.get("speaker"),
                              voice_prefs.get("pace"), provider=voice_provider)

        runner = InterviewRunner(
            http, interview_id, data["plan"], candidate_name=data.get("candidate", ""),
            coverage_threshold=data.get("coverage_threshold", 0.7),
            max_followups_per_question=data.get("max_followups_per_question", 2),
            depth_probe_enabled=data.get("depth_probe_enabled", True),
            duration_minutes=data.get("duration_minutes") or 30)
        resumed = runner.restore(data.get("progress"))
        logger.info("interview start id=%s language=%s role=%r plain_language=%s resumed=%s voice=%s",
                    interview_id, LANGUAGE_CONFIGS[language]["label"], role_name, plain_language, resumed,
                    voice.as_dict())
        await runner.start()

        tts_engine, sarvam = build_tts(voice)
        if sarvam is not None:
            # Fixed interviewer lines (refusals, closing...) are synthesised once and then replayed from
            # memory. The most common ones are pre-synthesised in the background after the greeting.
            all_units = [u for text in refusals.fixed_phrases(language) for u in speech_units(prepare_for_speech(text))]
            essential = [u for text in refusals.fixed_phrases(language, essential_only=True)
                         for u in speech_units(prepare_for_speech(text))]
            sarvam.set_cacheable(all_units)
        # A short vocabulary hint: Whisper spells the words it is told to expect. Without it "Hinglish" came out as
        # "हुजुबुरी" and a request to change language was missed.
        # With AUTO_LANGUAGE the recogniser is NOT told which language to expect: it detects it for every utterance, so a
        # candidate who starts speaking Hindi is heard as Hindi (a recogniser locked to English turns Hindi into English-
        # sounding nonsense). The conductor then follows the language they speak. AUTO_LANGUAGE=false restores one fixed
        # language per interview.
        stt_engine = lk_groq.STT(model="whisper-large-v3-turbo", api_key=GROQ_API_KEY,
                                 language=LANGUAGE_CONFIGS[language]["stt_language"], prompt=STT_VOCABULARY_HINT,
                                 detect_language=AUTO_LANGUAGE)
        current_voice = voice

        async def apply_language(new_language: str) -> None:
            """The candidate asked to change language: retune how the interviewer LISTENS (speech recognition) and how it
            SPEAKS (voice), for the rest of this session. Writing and what is saved are handled by the conductor."""
            nonlocal current_voice
            cfg = LANGUAGE_CONFIGS[normalize_language(new_language)]
            if not AUTO_LANGUAGE:                         # with auto-detection the recogniser already follows the speaker
                stt_engine.update_options(language=cfg["stt_language"])
            current_voice = resolve_voice(new_language, voice_prefs.get("gender"), voice_prefs.get("speaker"),
                                          voice_prefs.get("pace"), provider=voice_provider)
            if voice_provider == "elevenlabs" and ELEVENLABS_MODEL in ELEVENLABS_LANGUAGE_CODE_MODELS \
                    and hasattr(tts_engine, "update_options"):
                tts_engine.update_options(language=elevenlabs_language(normalize_language(new_language)))
            if sarvam is not None:
                await sarvam.switch_voice(current_voice)
                units = [u for t in refusals.fixed_phrases(new_language) for u in speech_units(prepare_for_speech(t))]
                sarvam.set_cacheable(units)
            logger.info("interview %s: speech now %s (listening %s, voice %s)", interview_id, cfg["label"],
                        cfg["stt_language"], current_voice.language_code)

        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            stt=stt_engine,
            tts=tts_engine,
            turn_handling=make_turn_handling(),
        )
        attach_metrics_logging(session, interview_id, lambda: {
            "question": runner.state.question_id if runner.state else None, "phase": runner.phase})
        vad_opts = getattr(ctx.proc.userdata["vad"], "_opts", None)
        tracer = turn_trace.TurnTracer(interview_id, getattr(vad_opts, "min_silence_duration", None))
        turn_trace.install(tracer)
        tracer.attach(session)
        if sarvam is not None:
            warmed = False

            def prewarm_after_greeting(event) -> None:
                nonlocal warmed
                if not warmed and event.old_state == "speaking" and event.new_state == "listening":
                    warmed = True       # the greeting is done, the candidate is answering: idle time
                    asyncio.get_running_loop().create_task(sarvam.prewarm_phrases(essential))

            session.on("agent_state_changed", prewarm_after_greeting)
        # The candidate left (closed tab, refreshed page): end this job WITHOUT completing the
        # interview - the next connection resumes it from the server-held progress.
        session.on("close", lambda event: runner.done.set())

        async def publish_transcript(sender: str, text: str, ts: float | None = None) -> None:
            # The candidate room's transcript tab listens for this topic and payload shape.
            await ctx.room.local_participant.publish_data(
                json.dumps({"sender": sender, "text": text, "timestamp": ts or time.time()}),
                reliable=True, topic="transcript")

        last_progress: dict = {}

        def publish_progress(payload: dict) -> None:
            """Tell the candidate's screen which question is active (topic 'progress'). Fire and forget."""
            last_progress.clear()
            last_progress.update(payload, timestamp=time.time())

            async def _send() -> None:
                try:
                    await ctx.room.local_participant.publish_data(json.dumps(last_progress), reliable=True, topic="progress")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("progress publish failed: %s", exc)

            asyncio.get_running_loop().create_task(_send())

        runner.on_progress = publish_progress

        def candidate_in_room() -> bool:
            return any(p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
                       for p in ctx.room.remote_participants.values())

        async def wait_for_candidate() -> None:
            """Greet only once a candidate is in the room, plus a moment for their browser to subscribe to
            the interviewer's audio. Bounded, so a candidate who never arrives can't hang the job."""
            loop = asyncio.get_running_loop()
            deadline = loop.time() + CANDIDATE_WAIT_S
            while loop.time() < deadline:
                if candidate_in_room():
                    await asyncio.sleep(CANDIDATE_SETTLE_S)
                    return
                await asyncio.sleep(0.25)
            logger.warning("no candidate joined %s within %.0fs - greeting anyway", ctx.room.name, CANDIDATE_WAIT_S)

        interviewer = InterviewerAgent(runner, language=language, role=role_name,
                                       experience_level=data.get("experience_level") or "",
                                       plain_language=plain_language,
                                       interviewer_name=INTERVIEWER_NAMES.get(voice.gender, "Aarav"),
                                       publish=publish_transcript, candidate_ready=wait_for_candidate,
                                       on_language_change=apply_language, auto_language=AUTO_LANGUAGE,
                                       supported_languages=supported_languages_for(voice_provider, ELEVENLABS_MODEL))

        def on_participant_connected(participant) -> None:
            if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
                return

            async def _replay() -> None:
                await asyncio.sleep(1.0)               # let the new participant's data channel open
                count = await interviewer.replay_captions()
                if last_progress:                       # a reload mid-interview: put the timeline back where it was
                    await ctx.room.local_participant.publish_data(json.dumps(last_progress), reliable=True, topic="progress")
                if count:
                    logger.info("replayed %d caption line(s) to %s", count, participant.identity)

            asyncio.get_running_loop().create_task(_replay())

        ctx.room.on("participant_connected", on_participant_connected)
        await session.start(agent=interviewer, room=ctx.room)
        runner.record_tts_metadata_nowait({"voice": voice.as_dict(), "resumed": resumed})

        def on_data_received(packet):
            # "Done speaking" button: background noise or a long pause can fool VAD into thinking the
            # candidate is still talking. This lets the candidate force the turn to end right now.
            if packet.topic == "commit_turn":
                # Only the candidate may end their own turn: another participant (or a stray data message) must not
                # be able to cut an answer short. The identity is the one the API put in the candidate's token.
                sender = getattr(getattr(packet, "participant", None), "identity", "") or ""
                if not is_candidate_identity(sender):
                    logger.warning("event=commit_turn_ignored reason=not_candidate interview=%s", interview_id)
                    return
                logger.info("event=commit_turn_received interview=%s", interview_id)
                try:
                    session.commit_user_turn(transcript_timeout=3.0, stt_flush_duration=1.0)
                except Exception:
                    logger.exception("commit_user_turn failed")

        ctx.room.on("data_received", on_data_received)

        # Keep the job (and the shared httpx client) alive until /complete has been posted or the
        # session has ended.
        await runner.done.wait()

        if sarvam is not None:
            runner.record_tts_metadata_nowait({"tts_stats": sarvam.stats.snapshot()})
        await runner.flush_pending()


def prewarm(proc):
    # Runs once when the worker process starts (not per job): loading the Silero VAD model here
    # means every interview after the first skips the load.
    # min_silence_duration: how long the candidate must be silent before VAD reports "stopped speaking".
    # Silero's default is 0.55 s - MORE than the 0.4 s endpointing delay in TURN_HANDLING, so the 0.4 s
    # setting never took effect and every reply waited at least 0.55 s. 0.4 s lines the two up; the
    # end-of-turn model still protects a candidate who pauses mid-thought (it holds the turn open up to
    # max_delay). Set VAD_MIN_SILENCE_S=0.55 to restore the previous behaviour.
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.getenv("VAD_MIN_SILENCE_S") or 0.4))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=AGENT_NAME,
        # Default load_threshold (0.7) marks the worker "unavailable" above 70% reported CPU load.
        # On constrained hosts just loading the VAD model pushes past that, so the worker refuses
        # every job. Raise it close to the maximum so it keeps accepting jobs.
        load_threshold=0.99,
        # `dev` mode defaults num_idle_processes to 0 (4 in `start`/prod), so every job cold-spawns
        # a subprocess and races the 10s default initialize_process_timeout. On a busy machine
        # (screen share, video call) that spawn + VAD load loses the race: "no process became
        # available after 3 attempts" and the candidate never hears the agent join. One warm
        # process and a 60s init budget fix it without touching job handling.
        num_idle_processes=1,
        initialize_process_timeout=60.0,
    ))
