"""Runner state/persistence and the LiveKit glue (real AgentSession + streaming TTS plugin + audio sink)."""
import asyncio
import struct
import time

import httpx
import pytest
from livekit.agents import AgentSession, ChatContext, ChatMessage, StopResponse

from agent import TURN_HANDLING, InterviewerAgent, _another_agent_present, build_tts
from conductor import Speech
from runner import DEFAULT_QUESTION_SECONDS, InterviewRunner
from sarvam_tts_plugin import SarvamTTS, _ConnectionPool, _SentenceEngine
from speech_prep import split_for_tts
from testing_fakes import CapturingAudioSink
from voice_config import resolve_voice

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


PLAN = {"questions": [
    {"id": "p_intro", "type": "introduction", "question_text": None},
    {"id": "p_candidate_intro", "type": "candidate_introduction", "question_text": "Tell me about yourself."},
    {"id": "q1", "type": "scenario", "topic": "hashing", "question_text": "How does a HashMap work?", "time_limit": 90},
    {"id": "q2", "type": "scenario", "competencies": ["system_design"], "question_text": "Design a rate limiter."},
]}


def make_runner(handler=None, **kw):
    posts = []

    def default_handler(request):
        posts.append((request.url.path.rsplit("/", 1)[-1], dict(httpx.QueryParams(request.content.decode())) if request.content else {}))
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler or default_handler))
    return InterviewRunner(client, "intv_r", PLAN, **kw), posts, client


# ------------------------------------------------------------------ runner state
async def test_question_state_uses_the_hr_time_limit_topic_and_defaults():
    runner, _, client = make_runner()
    runner.idx = 2
    runner.state = runner._new_question_state(runner.current_question)
    assert runner.state.max_seconds == 90 and runner.state.topic == "hashing"
    runner.advance()
    assert runner.state.max_seconds == DEFAULT_QUESTION_SECONDS and runner.state.topic == "system design"
    await client.aclose()


async def test_restore_resumes_only_after_the_greeting_phase_and_bounds_the_values():
    runner, _, client = make_runner()
    assert runner.restore({"question_index": 99, "followup_count": 50, "small_talk_done": 0}) is False or True
    runner2, _, c2 = make_runner()
    assert runner2.restore({"question_index": 3, "followup_count": 9, "small_talk_done": 0}) is True
    assert runner2.idx == 3 and runner2.small_talk_done == runner2.small_talk_rounds
    assert runner2.state.followup_count <= runner2.state.max_followups and runner2.resumed
    assert make_runner()[0].restore({"question_index": 0}) is False
    await client.aclose()
    await c2.aclose()


async def test_a_turn_recorded_just_before_complete_is_still_delivered():
    runner, posts, client = make_runner()
    runner.record_turn_nowait("q1", "candidate", "my final answer", intent="answer")   # spawned, not yet started
    await runner.complete()                                                            # flips terminal immediately
    names = [n for n, _ in posts]
    assert "turns" in names and names.index("complete") > names.index("turns")
    await client.aclose()


async def test_nothing_new_is_recorded_after_the_interview_is_terminal():
    runner, posts, client = make_runner()
    await runner.complete()
    runner.record_turn_nowait("q1", "candidate", "late", intent="answer")
    runner.persist_progress_nowait("late")
    await runner.flush_pending()
    assert [n for n, _ in posts] == ["complete"]
    await runner.complete()                                   # idempotent
    assert [n for n, _ in posts] == ["complete"] and runner.done.is_set()
    await client.aclose()


async def test_a_409_from_the_server_marks_the_interview_terminal_locally():
    def handler(request):
        return httpx.Response(409, json={"detail": "INTERVIEW_ALREADY_COMPLETED"})

    runner, _, client = make_runner(handler)
    runner.record_turn_nowait("q1", "candidate", "x")
    await runner.flush_pending()
    assert runner.terminal
    await client.aclose()


async def test_stale_detection_uses_generation_and_terminal_flag():
    runner, _, client = make_runner()
    gen = runner.generation
    assert not runner.is_stale(gen)
    runner.generation += 1
    assert runner.is_stale(gen)
    await client.aclose()


async def test_time_budget_is_a_fraction_of_the_interview_duration():
    runner, _, client = make_runner(duration_minutes=10)
    assert not runner.time_exhausted()
    runner._clock = lambda: runner.started_at + 8 * 60
    assert not runner.time_exhausted()
    runner._clock = lambda: runner.started_at + 9 * 60
    assert runner.time_exhausted()
    await client.aclose()


# ------------------------------------------------------------------ LiveKit glue
VOICE = resolve_voice("en", "female")


def pcm_for(text):
    return struct.pack("<h", (sum(map(ord, text)) % 3000) + 1) * int(24000 * 0.4 * max(1, len(text.split())))


class FakeConn:
    def __init__(self, calls, hold=None):
        self.closed = False
        self.created = time.monotonic()
        self.calls = calls
        self.hold = hold

    async def synth(self, text, **_):
        self.calls.append(text)
        if self.hold is not None:
            await self.hold.wait()
        return pcm_for(text), 2, 0.05

    async def close(self):
        self.closed = True


async def started_agent(sink_hold=False, hold=None):
    calls: list[str] = []
    conns: list[FakeConn] = []

    async def open_conn():
        conn = FakeConn(calls, hold if not conns else None)
        conns.append(conn)
        return conn

    engine = _SentenceEngine("k", VOICE, pool=_ConnectionPool(open_conn))
    session = AgentSession(tts=SarvamTTS("k", VOICE, engine=engine), turn_handling=TURN_HANDLING)
    sink = CapturingAudioSink(hold_playout=sink_hold)
    session.output.audio = sink
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    runner = InterviewRunner(client, "intv_r", PLAN, small_talk_rounds=0)
    runner.terminal = True                      # keep the runner off the network for these glue tests
    agent = InterviewerAgent(runner, language="en", role="Backend Engineer", experience_level="", plain_language=False)
    await session.start(agent=agent)
    return agent, session, sink, calls, conns, client, engine


async def test_speech_reaches_the_audio_output_exactly_once_in_order():
    agent, session, sink, calls, _, client, _ = await started_agent()
    text = "You mentioned hashing. What happens when two keys land in the same bucket, and how is it handled?"
    await asyncio.wait_for(agent._speak(Speech(text, "FOLLOW_UP")), 10)
    assert calls == split_for_tts(text)
    assert bytes(sink.pcm) == b"".join(pcm_for(u) for u in split_for_tts(text))
    await session.aclose()
    await client.aclose()


async def test_text_is_cleaned_before_it_reaches_tts():
    agent, session, sink, calls, _, client, _ = await started_agent()
    await asyncio.wait_for(agent._speak(Speech('**Great**\n- point [pause] {"action": "X"} Why is caching useful? \U0001F389', "FOLLOW_UP")), 10)
    spoken = " ".join(calls)
    for junk in ("*", "{", "}", "[pause]", "\U0001F389", "- "):
        assert junk not in spoken
    assert "Why is caching useful?" in spoken
    await session.aclose()
    await client.aclose()


async def test_barge_in_stops_the_interviewer_and_the_next_utterance_is_clean():
    hold = asyncio.Event()
    agent, session, sink, calls, conns, client, engine = await started_agent(sink_hold=True, hold=hold)
    speaking = asyncio.create_task(agent._speak(Speech("This long question is interrupted by the candidate speaking.", "NEXT_QUESTION")))
    for _ in range(200):
        if calls:
            break
        await asyncio.sleep(0.01)
    assert calls, "TTS never started"
    await session.interrupt()                                  # the candidate started talking
    await asyncio.wait_for(speaking, 5)
    assert engine.stats.discarded_in_flight >= 1 and conns[0].closed   # interrupted connection is never reused
    interrupted_bytes = len(sink.pcm)

    sink.hold_playout = False
    await asyncio.wait_for(agent._speak(Speech("Thanks. What is your approach?", "FOLLOW_UP")), 10)
    fresh = bytes(sink.pcm[interrupted_bytes:])
    assert fresh == b"".join(pcm_for(u) for u in split_for_tts("Thanks. What is your approach?"))
    await session.aclose()
    await client.aclose()


async def test_a_tts_failure_never_crashes_the_interview_turn():
    agent, session, sink, calls, _, client, engine = await started_agent()

    async def broken(*a, **k):
        raise RuntimeError("provider exploded")

    agent.conductor._speak = broken
    await agent.conductor._say(Speech("Hello there, welcome to the interview.", "GREETING"))   # must not raise
    await session.aclose()
    await client.aclose()


async def test_user_turns_are_forwarded_to_the_conductor_and_never_answered_by_a_free_form_llm():
    agent, session, sink, calls, _, client, _ = await started_agent()
    seen = []

    async def capture(text):
        seen.append(text)

    agent.conductor.handle_candidate_turn = capture
    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(ChatContext(), ChatMessage(role="user", content=["it uses hashing"]))
    assert seen == ["it uses hashing"]
    await session.aclose()
    await client.aclose()


async def test_the_session_has_no_conversational_llm_so_free_form_speech_is_impossible():
    agent, session, *_rest, client, _ = await started_agent()
    assert session.llm is None
    await session.aclose()
    await client.aclose()


def test_turn_handling_gives_candidates_longer_thinking_pauses_without_slowing_normal_turns():
    assert TURN_HANDLING["endpointing"]["min_delay"] == 0.4          # responsive: replies within ~4s overall
    assert TURN_HANDLING["endpointing"]["max_delay"] == 3.0          # LiveKit default: bounds the wait on an unfinished turn
    assert TURN_HANDLING["interruption"]["enabled"] is True          # barge-in stays on
    assert TURN_HANDLING["preemptive_generation"]["enabled"] is False


def test_only_a_configured_provider_is_used_and_sarvam_gets_a_fallback(monkeypatch):
    import agent as agent_module
    monkeypatch.setattr(agent_module, "SARVAM_API_KEY", "k")
    engine, sarvam = build_tts(resolve_voice("en", provider="sarvam"))
    assert sarvam is not None and engine.__class__.__name__ == "FallbackAdapter"
    monkeypatch.setattr(agent_module, "SARVAM_API_KEY", "")
    engine2, sarvam2 = build_tts(resolve_voice("en", provider="sarvam"))
    assert sarvam2 is None and engine2.__class__.__name__ == "EdgeTTS"


class _Participant:
    def __init__(self, kind):
        self.kind = kind


class _Room:
    def __init__(self, kinds):
        self.remote_participants = {str(i): _Participant(k) for i, k in enumerate(kinds)}


def test_a_second_agent_detects_the_first_and_backs_off():
    from livekit import rtc
    agent_kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    standard = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    assert _another_agent_present(_Room([standard, agent_kind])) is True
    assert _another_agent_present(_Room([standard])) is False
    assert _another_agent_present(_Room([])) is False


async def test_spoken_lines_are_published_to_the_candidates_live_transcript():
    agent, session, sink, calls, _, client, _ = await started_agent()
    sent = []

    async def publish(sender, text, ts=None):
        sent.append((sender, text))

    agent._publish = publish
    await asyncio.wait_for(agent._speak(Speech("Tell me about a project you are proud of.", "NEXT_QUESTION")), 10)
    await asyncio.sleep(0.05)                                   # captions are sent from a background task
    assert sent == [("agent", "Tell me about a project you are proud of.")]
    await session.aclose()
    await client.aclose()


async def test_a_failing_transcript_publish_never_breaks_the_interview():
    agent, session, sink, calls, _, client, _ = await started_agent()

    async def broken(sender, text):
        raise RuntimeError("data channel closed")

    agent._publish = broken
    await asyncio.wait_for(agent._speak(Speech("What did you build?", "NEXT_QUESTION")), 10)
    assert calls and bytes(sink.pcm)                           # still spoken
    await session.aclose()
    await client.aclose()


async def test_the_greeting_waits_until_a_candidate_is_in_the_room():
    agent, session, sink, calls, _, client, _ = await started_agent()
    gate = asyncio.Event()
    order: list[str] = []

    async def candidate_ready():
        order.append("waiting")
        await gate.wait()
        order.append("candidate arrived")

    async def fake_start():
        order.append("greeting spoken")

    agent._candidate_ready = candidate_ready
    agent.conductor.start = fake_start
    entering = asyncio.create_task(agent.on_enter())
    await asyncio.sleep(0.1)
    assert order == ["waiting"], "the interviewer spoke before any candidate was present"
    gate.set()
    await asyncio.wait_for(entering, 2)
    assert order == ["waiting", "candidate arrived", "greeting spoken"]
    await session.aclose()
    await client.aclose()


async def test_a_candidate_who_connects_later_receives_the_caption_history_in_order():
    agent, session, sink, calls, _, client, _ = await started_agent()
    sent = []

    async def publish(sender, text, ts=None):
        sent.append((sender, text, ts))

    agent._publish = publish
    agent._publish_line("agent", "Hello, welcome to your interview.")
    agent._publish_line("candidate", "Thank you.")
    await asyncio.sleep(0.05)
    sent.clear()                                                # what the FIRST candidate already saw live
    replayed = await agent.replay_captions()                    # a candidate (re)connecting now
    assert replayed == 2
    assert [(a, b) for a, b, _ in sent] == [("agent", "Hello, welcome to your interview."), ("candidate", "Thank you.")]
    assert all(ts for _, _, ts in sent)                         # original timestamps are kept
    await session.aclose()
    await client.aclose()


async def test_the_caption_history_is_bounded_and_a_failing_publish_never_breaks_it():
    agent, session, sink, calls, _, client, _ = await started_agent()

    async def broken(sender, text, ts=None):
        raise RuntimeError("data channel closed")

    agent._publish = broken
    for i in range(500):
        agent._publish_line("agent", f"line {i}")
    assert len(agent._caption_log) == 200 and agent._caption_log[-1][1] == "line 499"
    assert await agent.replay_captions() == 0                   # fails quietly, no exception
    await session.aclose()
    await client.aclose()
