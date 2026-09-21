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
from config import API_BASE_URL, ELEVENLABS_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, SARVAM_API_KEY, TTS_PROVIDER, TTS_VOICE
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

AGENT_NAME = "workmate-interviewer"

# Never sent to an LLM (the session has none). LiveKit's Agent requires the field.
_AGENT_DESCRIPTION = "Controlled interview conductor: every spoken turn is produced and validated by the interview engine."

# Turn handling. LiveKit's defaults are used except where an interview clearly needs different
# behaviour: a candidate thinking mid-answer pauses far longer than a casual speaker, so the
# maximum wait for a turn the end-of-turn model judges unfinished is raised from 3s to 5s (the
# minimum delay stays 0.5s, so normal turns are not slowed). Interruptions (barge-in) stay enabled
# with LiveKit's adaptive detection. Preemptive generation is off: nothing here generates
# speech speculatively from partial transcripts.
TURN_HANDLING = {
    "endpointing": {"min_delay": 0.4, "max_delay": 3.0},
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
                 plain_language: bool, interviewer_name: str = "Aarav", publish=None):
        super().__init__(instructions=_AGENT_DESCRIPTION)
        self.runner = runner
        self._publish = publish   # async (sender, text) -> None: live transcript to the candidate's screen
        self._caption_tasks: set = set()
        self.conductor = InterviewConductor(
            runner, self._speak, language=language, role=role, experience_level=experience_level,
            plain_language=plain_language, interviewer_name=interviewer_name)

    async def on_enter(self):
        await self.conductor.start()

    def _publish_line(self, sender: str, text: str) -> None:
        """Send a live-caption line to the candidate's screen WITHOUT delaying speech or the turn: it
        runs as its own task, and a failed caption must never affect the interview."""
        if not (self._publish and text and text.strip()):
            return

        async def _send() -> None:
            try:
                await self._publish(sender, text.strip())
            except Exception as exc:  # noqa: BLE001
                logger.debug("transcript publish failed: %s", exc)

        task = asyncio.get_running_loop().create_task(_send())
        self._caption_tasks.add(task)
        task.add_done_callback(self._caption_tasks.discard)

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
            logger.info("barge-in during action=%s question=%s", speech.action, speech.question_id)


def build_tts(voice: VoiceConfig) -> tuple[tts.TTS, SarvamTTS | None]:
    """Returns (tts for the session, the Sarvam instance if used so its stats can be reported)."""
    if voice.provider == "sarvam" and SARVAM_API_KEY:
        sarvam = SarvamTTS(SARVAM_API_KEY, voice)
        sarvam.prewarm()   # connection is ready before the first sentence is spoken
        # If Sarvam is completely unreachable, keep the interview audible with a different provider
        # rather than going silent. Retries already happen inside the plugin, so none here.
        return tts.FallbackAdapter([sarvam, EdgeTTS(voice=voice.edge_voice)], max_retry_per_tts=0), sarvam
    if voice.provider == "elevenlabs" and ELEVENLABS_API_KEY:
        return lk_elevenlabs.TTS(api_key=ELEVENLABS_API_KEY, voice=TTS_VOICE), None
    if voice.provider == "openai" and OPENAI_API_KEY:
        return lk_openai.TTS(api_key=OPENAI_API_KEY, voice=TTS_VOICE), None
    return EdgeTTS(voice=voice.edge_voice), None


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
        resp = await http.get(f"{API_BASE_URL}/v1/interviews/{interview_id}")
        resp.raise_for_status()
        data = resp.json()

        language = normalize_language(data.get("language"))
        role_name = data.get("role", "")
        plain_language = is_plain_language_role(role_name) if role_name else False
        voice_prefs = data.get("voice") or {}
        voice = resolve_voice(language, voice_prefs.get("gender"), voice_prefs.get("speaker"),
                              voice_prefs.get("pace"), provider=TTS_PROVIDER)

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
        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            stt=lk_groq.STT(model="whisper-large-v3-turbo", api_key=GROQ_API_KEY,
                            language=LANGUAGE_CONFIGS[language]["stt_language"]),
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

        async def publish_transcript(sender: str, text: str) -> None:
            # The candidate room's transcript tab listens for this topic and payload shape.
            await ctx.room.local_participant.publish_data(
                json.dumps({"sender": sender, "text": text, "timestamp": time.time()}),
                reliable=True, topic="transcript")

        await session.start(
            agent=InterviewerAgent(runner, language=language, role=role_name,
                                   experience_level=data.get("experience_level") or "",
                                   plain_language=plain_language,
                                   interviewer_name=INTERVIEWER_NAMES.get(voice.gender, "Aarav"),
                                   publish=publish_transcript),
            room=ctx.room,
        )
        runner.record_tts_metadata_nowait({"voice": voice.as_dict(), "resumed": resumed})

        def on_data_received(packet):
            # "Done speaking" button: background noise or a long pause can fool VAD into thinking the
            # candidate is still talking. This lets the candidate force the turn to end right now.
            if packet.topic == "commit_turn":
                logger.info("received manual commit_turn signal from candidate")
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
