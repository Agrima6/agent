# Deploying the AI Interview Agent to AWS EC2

## Why deploy

The voice breaks on the development Mac because it runs out of **RAM** (not disk): 17 GB total, ~60 MB free,
5.4 GB compressed, 2.7 GB of swap, with Chrome alone using ~7 GB. The agent itself needs ~50 MB idle. A
dedicated server removes that contention. It will not, by itself, remove network or model latency - see
`LATENCY_AND_VOICE_FIXES.md`.

## What runs where

```
Candidate browser ──(audio/video)──▶  LiveKit Cloud  ◀──(outbound only)──  worker container  ┐
                                                                                            ├─ one EC2 server
Interview backend (client-service) ──HTTP──▶  api container (:8000)  ◀───  worker container ┘
                                                     │
                                                postgres container
```

- The candidate's browser talks to **LiveKit Cloud**, never to your server.
- The **worker** only makes *outbound* connections (LiveKit, Sarvam, Groq). No inbound port is needed for voice.
- The only inbound traffic is the interview backend calling the **API** on port 8000.

## 1. Choose the server

| Setting | Recommendation | Why |
|---|---|---|
| Region | **ap-south-1 (Mumbai)** | your LiveKit project is in "India South" and Sarvam is in India; a nearby region cuts round trips in every turn |
| Instance | **c6i.xlarge** (4 vCPU, 8 GB) to start | a live interview is one process; roughly 10 simultaneous interviews is an *estimate*, load-test before relying on it. Graviton (`c7g.xlarge`) also works: all packages have arm64 wheels |
| OS | Ubuntu 22.04 or 24.04 | |
| Disk | 30 GB gp3 | logs are capped (20 MB x 5 per container) |
| Security group | inbound **22** from your IP only; **do not open 8000** to the internet | reach the API via SSH tunnel or a rule limited to the backend's IP |

The interview limit is usually the AI providers, not the server: check the Groq, Sarvam and LiveKit plan
limits for the number of concurrent interviews you need.

## 2. Prepare the server

```bash
ssh ubuntu@<server-ip>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit        # log in again so the group applies
```

## 3. Get the code

```bash
ssh ubuntu@<server-ip>
git clone https://github.com/Agrima6/agent.git workmate-iq-agent
cd workmate-iq-agent && git checkout feature/ai-interview-hardening
```

## 4. Configure secrets

```bash
cd deploy
cp .env.production.example .env.production
nano .env.production            # fill in every value
chmod 600 .env.production
```

Use long random values for `POSTGRES_PASSWORD` and `AGENT_SERVICE_KEY` (`openssl rand -hex 32`). The
`AGENT_SERVICE_KEY` must match what the interview backend sends. **Rotate any API key that has been pasted into
a chat or shared** (the Sarvam key used during development was) and put the new one here.

## 5. Start it

```bash
cd ~/workmate-iq-agent/deploy
docker compose --env-file .env.production up -d --build
docker compose ps                                   # all three: running / healthy
docker compose logs worker | grep "registered worker"
```

`registered worker` means the worker reached LiveKit. The first build takes a few minutes.

## 6. Point the interview backend at it

**Quick and secure, no config change** (good for a demo): from the machine running client-service,

```bash
ssh -N -L 8000:127.0.0.1:8010 ubuntu@<server-ip>      # 8010 = API_HOST_PORT on the server (use 8000 if it is free there)
```

client-service keeps `AGENT_SERVICE_URL=http://localhost:8000`, and stop the local agent so the port is free.

**Permanent:** put nginx (or an AWS load balancer) with HTTPS in front of port 8000, restrict the security group
to the backend, and set `AGENT_SERVICE_URL=https://agent.<your-domain>` and the same `AGENT_SERVICE_KEY` in
client-service.

## 7. Verify with a real interview

1. Start the demo interview and let the interviewer greet you.
2. Read the per-turn timings: `docker compose logs -f worker | grep turn_trace`
3. Look at `ttfa_ms`, `tts_ttfa_ms`, `judge_ms`, `compose_ms` (see `LATENCY_AND_VOICE_FIXES.md`).

## 8. Day-to-day

| Task | Command (in `deploy/`) |
|---|---|
| Logs | `docker compose logs -f worker` / `api` |
| Update | `git pull && docker compose --env-file .env.production up -d --build` |
| Restart | `docker compose restart worker` |
| Add capacity | `docker compose --env-file .env.production up -d --scale worker=2` (also grow the instance) |
| Roll back | `git checkout <previous-commit>` then the update command |
| Back up the database | `docker compose exec postgres pg_dump -U agent agent > backup.sql` |

On update, the worker's 5-minute stop grace period lets interviews already in progress finish first.

## 9. Things to know at cutover

- The server starts with an **empty database**. Interviews in progress on the Mac are not carried over.
- Backend records that cache old agent IDs (`agentRoleId`, `agentCandidateId`, `agentInterviewId`) are
  recreated automatically on the next start when the agent answers "not found".
- Only `livekit-plugins-openai` was missing from `requirements.txt`; it is fixed, and every package has a
  prebuilt Linux wheel for Python 3.12 (x86_64 and arm64), checked by resolving them for both.
- Not verified here: a full image build and a live interview on the server. Do both once (steps 5 and 7).

## 10. Keep local and server agents apart

Your Mac and the server share one LiveKit project. If both use the same agent name, an interview created by the
**local** API can be picked up by the **server** worker (or the reverse); the worker then looks the interview up in
the wrong database, finds nothing, and no interviewer ever speaks. Always give the server its own name
(`AGENT_NAME=workmate-prod` in `.env.production`) and keep the default name for local development. Never run a
local agent API on port 8000 while the SSH tunnel is supposed to own that port.

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `failed to connect to livekit, retrying` | outbound 443 blocked, or a brief DNS problem; it retries by itself. Check `curl -I $LIVEKIT_URL` from the server |
| worker exits at once | missing/invalid `LIVEKIT_*` values; read `docker compose logs worker` |
| interviewer never joins | worker not `registered`; the API refuses the backend (`AGENT_SERVICE_KEY` mismatch); a local agent API is answering on port 8000 instead of the tunnel; or `docker compose logs api` shows `agent dispatch FAILED` |
| voice still choppy | check `docker stats` for CPU/RAM, then `turn_trace` for which stage is slow |
| out of disk | logs are capped; check `docker system df` and remove old images |
