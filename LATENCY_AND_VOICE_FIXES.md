# Voice Latency & Reliability Fixes

Scope: the realtime interview voice path (LiveKit + Sarvam TTS + Groq STT/LLM). Everything here lives in
`workmate-iq-agent`. Status is **Done** (implemented and tested), **Pending** (decision or work left) or
**Not doing** (considered and rejected, with the reason).

Numbers marked *measured* come from real calls to Sarvam / Groq on the development Mac (17 GB RAM, calm
load unless stated). Anything not measured is marked **NOT MEASURED** rather than estimated.

---

## 1. The pipeline and where time goes

```
Candidate stops speaking
  → VAD confirms silence              400 ms now (was 550, Silero default)      config
  → Groq Whisper (batch STT)          ~300 ms (measured); 1144 ms seen under load
  → end-of-turn detector              local model now; cloud one stalled 1.0 s when it timed out
  → judge LLM                         median 1015 ms (measured)
  → composer LLM                      474-858 ms (measured)
  → Sarvam first audio                ~340-560 ms now  (was 540-2363 ms: whole sentence)   measured
  → LiveKit publish / playback        NOT MEASURED  (turn_trace: publish_ms)
  → network to the candidate          NOT MEASURED
```

End-to-end time-to-first-audio (TTFA) in a live interview is **NOT MEASURED yet** - the `turn_trace`
line below will report it on the next interview.

---

## 2. Fixes

| # | Priority | Status | Change | Where |
|---|---|---|---|---|
| F1 | P0 | **Done** | Stream Sarvam audio **chunk by chunk** instead of waiting for the whole sentence | `sarvam_tts_plugin.py` |
| F2 | P0 | **Done** | Choose the turn detector explicitly (local by default) | `agent.py:make_turn_handling` |
| F3 | P0 | **Done** | Per-turn latency trace (`turn_trace` log line) | `turn_trace.py`, `agent.py`, `conductor.py` |
| F4 | P2 | **Done** | VAD silence 0.55 s -> 0.4 s (configurable) | `agent.py:prewarm` |
| F5 | P2 | **Done** | Live captions no longer block speech | `agent.py:_publish_line` |
| F6 | P2 | **Done** | Fixed interviewer lines are cached (refusals, closing...) | `sarvam_tts_plugin.py`, `refusals.py` |
| F7 | P1 | **Evaluated - not adopted** | Faster judge model tested; kept the current one (see below) | `llm_provider.py` |
| F8 | P2 | **Not doing** | Batch STT measured ~0.3 s; no evidence it justifies a provider change | `agent.py` |
| F9 | P3 | **Documented** | Local turn model costs ~108 MB resident per interview; `TURN_DETECTOR=vad` is the lever | scale planning |

### F1 - Chunk-level TTS streaming (largest avoidable delay)

- **Problem.** `_WSConn.synth` collected every chunk and returned after Sarvam's `final` event, and the
  stream only pushed audio after that. Sarvam produces the first chunk in ~340 ms but a 130-character
  sentence takes ~1.7 s to finish (it generates ~5x faster than it plays). Every reply waited for the
  whole first sentence.
- **Change.**
  - `_WSConn.synth_stream` yields PCM as it arrives; `synth` is now a wrapper.
  - `_SentenceEngine.synthesize_stream`: retries (ws -> ws -> REST) are allowed **only while nothing has been
    yielded**. After audio started, a failure ends that sentence (logged as `tts_sentence_cut`, counted in
    `mid_sentence_failures`) instead of replaying it, so the candidate never hears a duplicate.
  - `_SarvamSynthesizeStream._run`: one queue per sentence. Later sentences are prefetched in parallel (max 3)
    **after sentence 1 has produced its first audio**, and released strictly in speaking order. A 250 ms
    pre-roll protects against an unusually slow first chunk.
  - Barge-in cancels every producer and closes every in-flight connection (no stale audio).
- **Measured (real Sarvam, median of 5, ms, first audio):**

  | Utterance | Before | After |
  |---|---|---|
  | Follow-up, 1 sentence | 1304 / 2363 | 479 / 557 |
  | Next question, 2 sentences | 627 / 1051 | 432 / 632 |
  | Greeting, 4 sentences | 540 / 1312 | 669 / 598 |

  (Two runs; the network was noisier in the second. The first version of the prefetch made the greeting
  ~130 ms *slower* because all sentences competed for the network at once - fixed by starting later sentences
  only after sentence 1 has audio.)
- **Trade-off.** A drop mid-sentence now cuts that sentence rather than replaying it. It is rare (the
  connection was healthy enough to start) and visible in `mid_sentence_failures`.
- **Tests.** `test_sarvam_tts.py` - first audio before the sentence finishes, no replay after a mid-sentence
  failure, clean retry before the first chunk, in-order delivery with prefetch, cancellation closes all
  connections.

### F2 - Explicit turn detector

- **Problem.** `TURN_HANDLING` never set `turn_detection`, so LiveKit chose by run mode: `agent.py dev` uses
  the **cloud** detector, and each turn waits up to **1.0 s** for that network prediction (logged:
  `eot prediction timed out`, then a mid-interview fall back to the local model). `agent.py start` uses the
  local model. Dev and production therefore behaved differently.
- **Change.** `make_turn_handling()` picks it on purpose via `TURN_DETECTOR`:
  - `local` (default) - on-device model, no network in the speech path, same in dev and prod.
  - `vad` - no model; fastest and lightest, but cuts off candidates who pause mid-thought.
  - `cloud` - LiveKit's cloud model.
- **Not measured.** The local model's per-turn inference time. `turn_commit_ms` in the trace will show it.

### F3 - Per-turn latency trace

One structured line per turn, no transcript text or personal data. The values below are **illustrative
only** (they show the format); real values appear after the next live interview:

```
turn_trace {"turn_id":"intv_x-3","kind":"candidate_turn","stt_final_ms":310,"turn_commit_ms":520,
 "judge_ms":1010,"compose_ms":640,"logic_ms":1700,"pre_tts_ms":2260,"tts_ttfa_ms":430,"tts_buffer_ms":0,
 "publish_ms":90,"ttfa_ms":2790,"vad_min_silence_ms":400}
```

A stage whose marks are missing is `null` (not measured), never a guess. `ttfa_ms` is counted from when VAD
*confirmed* the end of speech; the candidate actually stopped about `vad_min_silence_ms` earlier.

Read it with:

```bash
grep -a turn_trace /private/tmp/worker.log | tail -20
```

### F4 - VAD silence

Silero's `min_silence_duration` defaults to 0.55 s, **longer** than the 0.4 s endpointing delay in
`TURN_HANDLING`, so the 0.4 s setting never took effect and every reply waited at least 0.55 s. Now
`VAD_MIN_SILENCE_S` (default 0.4). This is the one behavioural change to watch: if candidates get cut off
mid-thought, set `VAD_MIN_SILENCE_S=0.55`.

### F5 - Captions off the speech path

`session.say` used to wait for `publish_data` (a network send) before starting. Captions are now sent from a
background task; a failed caption never affects the interview.

### F6 - Fixed-phrase cache

Refusals, "let's try another question", "shall we begin?" and the closing lines never change, yet each play
went to Sarvam. They are now synthesised once and replayed from memory.

- Only lines registered as fixed are cached; personalised text (the greeting with the candidate's name) and
  anything from the LLM is never cached. A line cut short by a failure is never cached.
- Bounded: 48 entries / 8 MB per interview process.
- After the greeting finishes, the ~21 most common lines are pre-synthesised in the background on their own
  connection (so it can never take the warm connection a live reply needs).
- **Measured (real Sarvam):** first play 452 ms, replay **5 ms**; 21 lines pre-synthesised in ~10 s, 3.0 MB.

### F7 - Judge -> composer (evaluated, not adopted)

The composer needs the judge's output (which topics were covered, what the candidate said), so the two calls
cannot simply run in parallel. Speculative composition would spend extra tokens on every turn against the Groq
rate limit and could not use the judge's grounded concepts; streaming the composer into TTS is invasive for a
small gain. The remaining option was a faster judge model, which I tested with real calls on the 10 golden
scenarios in `eval_live.py` (5 s pacing, no failed calls on either model):

| | gpt-oss-120b (current) | gpt-oss-20b |
|---|---|---|
| median latency | 978 ms | 686 ms |
| max latency | 2641 ms | 2272 ms |
| same intent and coverage as 120b | - | **9 / 10** |

The one disagreement is the important one: for a **confidently wrong answer** the 20b judge returned intent
`off_topic` (120b: `answer`). An `off_topic` verdict makes the interviewer redirect the candidate instead of
scoring the answer and following up, which would mishandle exactly the candidates who answer wrongly. The ~290 ms
saving is not worth that, so the default stays. `JUDGE_MODEL=openai/gpt-oss-20b` remains available for load or
rate-limit reasons; re-run the comparison before using it.

### F8 - Streaming STT (Pending)

Groq Whisper is batch (~300 ms measured). A streaming STT would deliver partial finals earlier, but that is a
provider change; the measured 300 ms does not justify it yet. Revisit if `stt_final_ms` in `turn_trace` is high.

### F9 - Scale note (Pending)

`TURN_DETECTOR=local` keeps a ~108 MB model resident per interview process. At high concurrency consider
`TURN_DETECTOR=vad`, or size instances for it.

---

## 3. Findings that are not code

- **This Mac is memory-starved.** 17 GB RAM, ~60 MB free, 5.4 GB compressed, 2.7 GB of swap in use, Chrome
  holding ~7.4 GB; load average 44-84. The agent and API together use ~50 MB. This causes the stutter and the
  slow starts seen in development (earlier logs: Silero 48 s behind real time, TTS 13.8 s for 11 s of audio).
  It will not exist on a dedicated server.
- **Docker Mongo** sampled at ~100% CPU and competes for the same CPU.
- **Not the cause:** Sarvam (first chunk ~340 ms), Groq Whisper (~300 ms), the event loop (no blocking calls;
  LLM calls run in `to_thread`; DB writes are fire-and-forget), conversation history (the judge and composer
  prompts are closed-context, with no history).

## 4. Not doing

- Switching TTS/STT providers - the measurements do not show a provider problem.
- Raising RAM - the agent uses ~50 MB; the pressure is other applications on the dev machine.
- Rewriting the pipeline - the shape is fine; the gains were in buffering and configuration.
- Filler phrases ("umm...") to hide latency - banned by the interviewer rules.

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TURN_DETECTOR` | `local` | `local` / `vad` / `cloud` |
| `VAD_MIN_SILENCE_S` | `0.4` | silence before "candidate stopped speaking" |
| `TTS_SPEAKER_OVERRIDE`, `TTS_PACE_OVERRIDE`, `TTS_GENDER` | unset | voice overrides |
| `JUDGE_MODEL`, `COMPOSER_MODEL` | see `llm_provider.py` | per-role LLM |

## 6. Validation plan

1. Restart the worker and run one interview. Then `grep -a turn_trace /private/tmp/worker.log`.
2. Targets: `tts_ttfa_ms` under ~600, `judge_ms + compose_ms` near 1.5 s, `tts_buffer_ms` near 0.
3. Compare `TURN_DETECTOR=vad` and `VAD_MIN_SILENCE_S=0.55` against the defaults.
4. Watch `mid_sentence_failures` and `cache_hits` in the `tts_session_summary` line at the end of a session.
5. Before a real launch, load-test a few simultaneous interviews on the target instance.

## 7. Files

New: `turn_trace.py`, `test_turn_trace.py`. Changed: `sarvam_tts_plugin.py`, `agent.py`, `conductor.py`,
`refusals.py`, `test_sarvam_tts.py`, `.env.example`. Full agent suite: 447 tests passing.
