# Deploying TriageForge on Render (free tier)

Nothing in this repo deploys automatically. These are the manual steps to stand up
the stack defined in `render.yaml`. No paid plan is needed.

## Why Render

`render.yaml` is a Render Blueprint that describes all three pieces in one file,
each on a free plan:

| Service | Render type | Plan |
|---|---|---|
| `triageforge-api` | Web service, Docker runtime (the root `Dockerfile`) | free |
| `triageforge-ui` | Static site (Vite build of `frontend/`) | free |
| `triageforge-sessions` | Key Value (Redis-compatible), internal only | free |

The API and Key Value run in `singapore` (the closest Render region to India).
They must share a region to talk over Render's private network; to move them,
change both `region` fields in `render.yaml`. The static site is served from
Render's CDN and has no region. Every service deploys from the `main` branch.

The backend container was measured at 200 to 249 MiB resident after a request
(`eval/docker_verification.txt`, `eval/docker_e2e_verification.txt`), which fits
the free instance's 512 MB. Dependencies in `requirements.txt` are pinned to the
versions every test and eval ran on, so a Render build installs exactly those.

Free-tier behaviour to expect:
- The free web service spins down after a period without traffic; the first request
  after that waits for a cold start (the container also spawns the MCP subprocess
  on startup).
- Free Key Value has no persistence. Sessions already expire after `SESSION_TTL_S`
  (24 h), so a restart only drops in-flight conversations.
- Groq's free tier still applies: 200K tokens per model per day per Groq account.
  The blueprint sets `GROQ_MAX_RATE_LIMIT_WAIT_S=60`, so when every key is
  rate-limited a visitor gets a 503 within about a minute instead of a 5 minute wait.
- The first API build downloads ChromaDB's embedding model (about 80 MB). It uses
  `scripts/fetch_embedding_model.py`, which resumes a broken download instead of
  restarting, so a slow link makes the build slower, not failed.

## Steps

1. **Push the repo to GitHub** (Render deploys from a Git repo).

2. **Create the Blueprint.** In the Render dashboard: New > Blueprint, connect the
   GitHub repo, select the branch. Render reads `render.yaml` and lists the three
   services.

3. **Fill in the secrets** Render asks for (every `sync: false` entry):

   | Service | Variable | Value |
   |---|---|---|
   | api | `GROQ_API_KEY` | your Groq key |
   | api | `TRIAGEFORGE_API_KEYS` | one or more random strings, comma-separated (for example from `python -c "import secrets; print(secrets.token_urlsafe(24))"`) |
   | api | `ALLOWED_ORIGINS` | the UI's URL, for example `https://triageforge-ui.onrender.com` (use a placeholder now if the URL is not known yet, see step 5) |
   | api | `LANGSMITH_API_KEY` | optional; leave empty to keep tracing off |
   | ui | `VITE_API_URL` | the API's URL, for example `https://triageforge-api.onrender.com` |
   | ui | `VITE_API_KEY` | one of the `TRIAGEFORGE_API_KEYS` values |

   `REDIS_URL` is wired automatically from the Key Value instance. Extra Groq keys
   (`GROQ_API_KEY_2`, ...) and any other setting from `.env.example` can be added
   under the API service's Environment tab.

   `VITE_API_KEY` is compiled into the public JavaScript bundle, so anyone who opens
   the site can read it. Treat it as a demo key: keep `RATE_LIMIT_PER_MIN` low and
   rotate it if it is abused.

4. **Apply the Blueprint** and wait for the API's Docker build (it installs the
   requirements and embeds the knowledge base during the build) and the static
   site build.

5. **Fix up the URLs.** Render shows each service's `onrender.com` URL. If they
   differ from what you entered:
   - set `ALLOWED_ORIGINS` on the API to the UI's exact origin and save (the API
     restarts);
   - set `VITE_API_URL` on the UI and trigger a manual deploy. Vite inlines it at
     build time, so the UI must be rebuilt for a change to take effect.

6. **Verify** (replace the host and key):

   ```
   curl https://triageforge-api.onrender.com/health
   curl https://triageforge-api.onrender.com/metrics | head
   curl -X POST https://triageforge-api.onrender.com/ticket \
     -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
     -d '{"ticket_id":"t1","user_message":"How do I reset my password?","session_id":"s1"}'
   ```

   Expected: `{"status":"ok"}`, Prometheus text, then a `TriageResponse`. A request
   without the header should return 401. Then open the UI URL and submit a ticket;
   the trace should stream in step by step.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Browser console shows a CORS error | `ALLOWED_ORIGINS` does not exactly match the UI origin (scheme, host, no trailing slash) |
| UI calls `localhost` | `VITE_API_URL` was not set before the static build; set it and redeploy the UI |
| 401 from the UI | `VITE_API_KEY` is not one of `TRIAGEFORGE_API_KEYS` |
| 503 with `Retry-After` | every Groq key hit its rate limit; wait or add more keys |
| First request is very slow | free instance cold start after idling |
