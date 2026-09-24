# What's needed on the H100 (illyad)

Hand this to whoever administers the GPU box — or paste it to an agent running on it.

---

## Context

The H100 runs **Ollama only**. It performs inference for a YouTube learning assistant whose backend runs on a **separate server**, reachable over **Tailscale** (both machines are already on the same tailnet).

The GPU box does not run the application, store data, or accept public traffic. It only answers inference requests from the backend.

**Nothing here requires root**, except where explicitly noted. Ollama is already installed and running.

---

## What the backend sends

Two kinds of request, both to the standard Ollama API:

| Endpoint | Purpose | Shape |
|---|---|---|
| `POST /api/embed` | Index a video transcript | Batches of up to 32 text chunks, ~1,800 characters each. One video = 15–60 chunks total, once per video (cached afterwards). |
| `POST /api/chat` | Answer a question | Streaming. Prompt is ~5,000 tokens (system prompt + retrieved transcript excerpts + conversation history). Response streams token by token. |

Load is light and bursty: a few seconds of embedding when a video is opened, then a streaming generation per question. This is a single-user workload.

---

## Requirements

### 1. Two models must be pulled

```bash
ollama pull nomic-embed-text     # ~275MB — REQUIRED
ollama pull llama3.1:70b         # ~42GB — or another chat model
```

**The embedding model is not optional and cannot be a chat model.** They do different jobs. `nomic-embed-text` is specifically required because of its context length — see §2.

Verify:

```bash
ollama list
```

### 2. The embedding model must handle 1,800-character inputs

This is the requirement most likely to be missed, because violating it **fails silently**.

Embedding models have a maximum sequence length and quietly discard anything beyond it — no error, no warning. The backend sends transcript chunks of up to 1,800 characters (~450 tokens). If the model truncates below that, roughly the back third of every chunk is dropped before embedding. Search then matches against partial text while the language model receives the full chunk: subtly wrong answers, nothing in any log.

| Model | Max tokens | Verdict |
|---|---|---|
| `nomic-embed-text` | 8,192 | ✅ Recommended |
| `mxbai-embed-large` | 512 | ⚠️ Workable |
| `all-minilm:l6-v2` | **256** | ❌ Truncates — do not use without lowering chunk size |

If only `all-minilm` is available, say so — the backend can lower its chunk size to compensate, at some cost to retrieval quality.

The backend detects this automatically at startup and reports it, so a mistake here is caught rather than silently degrading.

### 3. Reachability from the backend over the tailnet

Ollama listens on `127.0.0.1:11434` by default. **In kernel-mode Tailscale, loopback services are not reachable over the tailnet** — a connection to the node's `100.x.y.z` address arrives on `tailscale0`, where nothing is listening.

One of these is needed. In order of preference:

1. **Bind Ollama to the tailnet address** (preferred — needs control of the daemon):
   ```
   OLLAMA_HOST=100.x.y.z:11434
   ```
   Do **not** use `0.0.0.0` unless the machine has no public interface. See §5.

2. **`tailscale serve`** (needs root or operator rights):
   ```bash
   tailscale serve --bg --tcp 11434 tcp://127.0.0.1:11434
   ```

3. **Userspace forwarder** (needs neither; provided as `scripts/h100-tailnet-forward.py`):
   ```bash
   nohup python3 h100-tailnet-forward.py > forward.log 2>&1 &
   ```
   Binds the tailnet IP and forwards to loopback. Python standard library only.

Success looks like this, **run from the backend server**:

```bash
curl -s http://100.x.y.z:11434/api/tags
```

> **Addendum — confirmed against a live Ollama instance (v0.13.5), applies here too:**
> Ollama rejects any request whose `Host` header isn't `localhost`/`127.0.0.1` (or the exact value of `OLLAMA_HOST`, if set) with an **empty `403`** — no body, no log line, `curl -s` just returns nothing. This is DNS-rebinding protection, and it bites options **2 and 3** above: neither changes what Ollama itself is bound to (still `127.0.0.1:11434`), so a client dialing `100.x.y.z:11434` sends `Host: 100.x.y.z:11434`, which Ollama silently rejects.
> - **Option 1 is unaffected** — Ollama trusts the address it was told to bind to (`OLLAMA_HOST`), so a request to that same address matches.
> - **Option 2** (`tailscale serve --tcp`) needs a proxy in front that rewrites the `Host` header to `localhost:11434` before it reaches Ollama, or it will hit this silently.
> - **Option 3**: `scripts/h100-tailnet-forward.py` (in this handoff bundle) already does this rewrite — it's not a plain forwarder, use it as provided rather than a generic TCP proxy.
>
> If `curl -s http://100.x.y.z:11434/api/tags` from the backend returns nothing with exit code 0, this is almost certainly why. Confirm with `curl -i` (look for `403`) or reproduce locally on the H100 with `curl -i -H "Host: 100.x.y.z:11434" http://127.0.0.1:11434/api/tags`.

### 4. Recommended Ollama settings

These improve responsiveness but are not required. They need the ability to restart the daemon:

```bash
OLLAMA_KEEP_ALIVE=-1        # keep the model resident; avoids a ~60s reload
OLLAMA_NUM_PARALLEL=4       # embedding batches may overlap a streaming answer
OLLAMA_MAX_LOADED_MODELS=2  # chat + embedding models resident together
```

`OLLAMA_KEEP_ALIVE` is **already sent per request** by the backend, so it is not needed server-side. The other two cannot be set per request.

### 5. Security

**Ollama has no authentication of any kind.** Anyone who can reach port 11434 can run inference on the GPU, and pull or delete models.

- Do **not** expose 11434 to the public internet.
- Do **not** bind `0.0.0.0` if the machine has a public interface.
- Tailnet-only access (via any of the §3 options) is correct and sufficient.

### 6. VRAM sharing

Open Notebook reportedly runs on the same host and uses the same Ollama instance. Two concerns:

- `llama3.1:70b` (~42GB) is safe on an 80GB card alongside the KV cache at 8k context.
- `mistral-large` (~73GB) is **not** — it leaves no room for a KV cache and risks eviction or OOM if anything else loads.

Check what is currently resident:

```bash
curl -s http://127.0.0.1:11434/api/ps
nvidia-smi
```

If the GPU is shared, prefer the 70B (or an 8B) over `mistral-large`.

---

## Definition of done

All of these must succeed:

```bash
# On the H100
ollama list                      # shows a chat model AND nomic-embed-text
curl -s http://127.0.0.1:11434/api/tags

# From the BACKEND server — this is the one that matters
curl -s http://100.x.y.z:11434/api/tags
```

Then, on the backend, `npm run preflight` exercises the entire path — embedding capacity, transcript extraction, indexing, retrieval, and a streaming answer — and reports precisely which stage fails.

---

## Not needed

To be explicit, the GPU box does **not** need:

- Node.js, the application code, or any database
- Inbound public ports, a domain, or TLS
- Docker
- Root, unless you choose option 1 or 2 in §3
- Persistent storage beyond the model blobs
