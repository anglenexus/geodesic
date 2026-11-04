# Kimi K3 on one 8x B300 node (Modal): serving and profiling plan

Goal: serve Kimi K3 with SGLang + FlashInfer on the cheapest single node that holds it, and run the same two
profiling phases as Qwen3.5-27B (PR #22), each capped at **15 minutes** so the whole GPU session fits in about
**one hour** (~$64). Results are small and committed to `bench_results/`; a blog source follows the run, as
`blog_source.md` did for Qwen.

## 1. Why this needs its own plan

The method carries over from Qwen3.5-27B; the model, the number formats, the kernels and the infrastructure
do not.

| | Qwen3.5-27B (PR #22) | Kimi K3 (this plan) |
|---|---|---|
| Size | 27B dense | 2.8T total, ~104B active per token (MoE: 896 routed experts, 16 per token, 2 shared) |
| Layers | 64: 48 Gated DeltaNet + 16 GQA attention | 93: 69 KDA (Kimi Delta Attention) + 24 MLA (gated multi-head latent attention) |
| Weights | three checkpoints to compare (AWQ int4, FP8, BF16) | one native checkpoint: routed experts in **MXFP4**, everything else BF16 (1.56 TB). FP8 (~2.8 TB) and BF16 (~5.6 TB) don't fit one node |
| 4-bit math | Marlin: 4-bit weights are unpacked to BF16 in software (Hopper has no 4-bit math) | Blackwell **FP4 tensor cores**: FlashInfer's trtllm-gen MXFP4 MoE kernel multiplies 4-bit values directly |
| Attention cache | paged KV, 64 KiB/token | MLA latent cache: 512 + 64 values per token per layer, **~27 KiB/token** for all 24 layers in BF16 |
| Recurrent state | DeltaNet, ~150 MB/request, capped the A100 at 9 requests | KDA, **~434 MB/request** in FP32 (69 layers x 96 heads x 128 x 128), sharded 8 ways by TP8 |
| GPUs | 1 (A100-40GB, H200) | 8 (one B300 node, NVLink), tensor parallel 8, optionally decode context parallel 8 |
| Host | a rented VM: tmux, SSH, local disk | a Modal container: weights in a Volume (~2.5 GB/s), SSH built from sshd + `modal.forward` |

### Number formats in one place

- **MXFP4** (OCP microscaling): each weight is an E2M1 4-bit float (values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), packed **two per byte**. Every block of **32** weights shares one **E8M0** scale (a power of two, 1 byte). Cost: 4 + 8/32 = **4.25 bits per weight**. Kimi K3 was trained with it (quantization-aware from SFT on), with MXFP8 activations, so serving in MXFP4 is not a lossy afterthought.
- **NVFP4** (NVIDIA): the same E2M1 values, blocks of **16** with an **FP8 E4M3** scale (fractional, not just powers of two) plus one FP32 scale per tensor: **4.5 bits per weight**, finer scaling. `nvidia/Kimi-K3-NVFP4` is a post-training conversion and needs SGLang's NVFP4 dev image.
- **MXFP8 / FP8 E4M3**: 8-bit activations (and an optional FP8 KV cache), which B300 also multiplies natively.
- **AWQ int4 (Qwen, for contrast)**: 4-bit integers with a BF16 scale per 128 weights, unpacked by Marlin to BF16 before the multiply.

What we expect, to check against the measurements:
- 1.445 TB of the 1.56 TB are the routed experts (92 MoE layers x 896 experts x 3 x 3584 x 3072 weights at 4.25 bits); the other ~115 GB (attention, shared experts, the dense first layer, embeddings) are BF16 and **all of it is read for every token**.
- So at one request, a decode step reads ~110 GB of BF16 against ~26 GB of FP4 experts (16 per layer): the BF16 parts dominate single-stream speed. As the batch grows, more distinct experts are touched (all 896 at a few dozen requests), and the FP4 expert reads take over. Per GPU (TP8) the floor is ~17 GB per step at ~8 TB/s, ~2 ms: an upper bound of a few hundred tokens/s for one stream.
- MLA's cache is so small that memory goes to the KDA state, as with Qwen on the A100: the cap on running requests is set by state slots (SGLang's B300 note: 101 admitted requests for MXFP4 without speculative decoding).

## 2. Infrastructure: Modal

- **GPU:** `gpu="B300:8"`, one machine, 8 x B300 SXM6 (268 GiB each, 2.15 TiB total, driver 610.57, CUDA 13.3), checked available on 2026-11-03 (placed in 5 s). $7.10 per GPU-hour, per-second billing.
- **Image:** `lmsysorg/sglang:v0.5.23-cu130` (current release, CUDA 13) plus openssh-server, tmux, pandas, plotly; our scripts are mounted at `/root/serving`.
- **Weights:** Volume `kimi-k3-weights`, filled once by 12 CPU-only containers (`modal run modal_kimi_k3.py::download`, ~$1). Not gated on Hugging Face, no token needed. Storage beyond the free 1 TiB costs ~$1.30/day: delete the Volume after the runs (`modal volume delete kimi-k3-weights`).
- **Results:** Volume `kimi-k3-results` (committed every minute), pulled into `bench_results/` with `modal run modal_kimi_k3.py::fetch`.
- **Tunnels:** the container runs sshd with the laptop's public key and exposes port 22 through `modal.forward(22, unencrypted=True)`; the run prints `ssh -N -p <port> -L 30000:... -L 8000:... root@<host>`. Then `smoke_test.py` and the dashboards work exactly as with the Qwen VMs. The API is not exposed publicly.

## 3. Cost, limits and kill switches

| Item | Cost |
|---|---|
| GPU session, planned (~62 min) | 8 x $7.10 x 1.03 h + ~$5 CPU/memory = **~$64** |
| Kill switch: Modal stops the container at 120 min | **~$128** worst case |
| Weight download (CPU) | ~$1 |
| Dev check (1 x B300, ~3 min) | ~$0.40 |
| Weight storage | ~$1.30/day until deleted |

- **Workspace usage limit: $42 today (blocking).** Modal raises it after successful payments (the first at $10 of charges) but doesn't say by how much. The run needs ~$70 (planned) to ~$130 (kill switch) of headroom. **Action:** ask Modal support (dashboard chat or support@modal.com) to raise the zouyouzhi usage limit to $200, or confirm it has auto-raised past $150 before the run. Optionally set a spend limit of $150 under Settings → Usage & Billing as a second ceiling.
- **Kill switches, inner to outer:**
  1. Each phase stops itself at 15 min (`PHASE_MIN`); the driver exits after a 15-min hold window (`HOLD_MIN`) or when `/tmp/STOP` exists.
  2. The Modal function's `timeout` (`KILL_MIN=120`) kills the container whatever the scripts do.
  3. The workspace usage/spend limit.
- Starter plan: 10 GPUs at once (we use 8), $30/month of credits.

## 4. The one-hour run

```
0:00  container up, sshd + tunnel printed                     (modal run --detach modal_kimi_k3.py::run)
0:01  Phase 4 kernels, GPUs idle, cap 15 min                  fi_bench_kimi.py mla, fp4
0:16  server start, Balanced profile, ~13 min                 serve_kimi_k3.sh start (1.56 TB at ~2.5 GB/s)
0:30  Phase 3, cap 15 min, server already up                  run_bench.py --model kimi-k3 --reuse-server
0:45  export + dashboards                                     export_results.py, make_dashboard.py, fi_bench_kimi.py report
0:47  hold window, 15 min: smoke tests and dashboards         python smoke_test.py --model kimi-k3 --no-think
1:02  exit
```

A server restart costs ~13 minutes (weight reading), so the one-hour run tests **one server setting**.
Comparisons that need a restart are optional extras that fit under the 2-hour kill switch, one per ~15 min
(`modal run --detach modal_kimi_k3.py::run --steps all+low_latency`, see 5.3).

## 5. Tests

### 5.1 Server (`serve_kimi_k3.sh`)

SGLang's verified B300 1x8 "Balanced" recipe, plus our metrics and limits:

```
sglang serve /models/Kimi-K3 --trust-remote-code --served-model-name kimi-k3 \
  --tp-size 8 --dcp-size 8 --mem-fraction-static 0.85 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --context-length 65536 --max-running-requests 128 --cuda-graph-max-bs 128 \
  --enable-metrics --enable-cache-report --host 127.0.0.1 --port 30000
```

No `--moe-runner-backend` and no attention-backend flags: on Blackwell SGLang picks FlashInfer's MXFP4
trtllm-gen MoE and the trtllm_mla attention kernels itself, and the startup log records which. Thinking is
switched off per request with `chat_template_kwargs: {"thinking": false}`.

### 5.2 Phase 4: kernels (`fi_bench_kimi.py`, 15 min, before the server starts)

| Test | What | Why |
|---|---|---|
| **4a MLA decode** | FlashInfer's MLA decode kernel at Kimi's shapes (latent 512 + rope 64), BF16 vs FP8 KV, for batch x context (1 x 8K, 16 x 8K, 64 x 4K, 16 x 32K), in two layouts: **TP8** (12 heads per GPU, full context) and **DCP8** (all 96 heads, 1/8 of the context) | DCP is what the Balanced profile adds: the same arithmetic, but each GPU reads 1/8 of the cache instead of a full copy. Shows where it pays (long contexts) |
| **4b FP4 GEMM** | One expert's matrices (gate+up 6144 x 3584, down 3584 x 3072) for 1-4096 tokens in BF16, FP8, MXFP4 and NVFP4 (FlashInfer `mm_fp4`), plus the BF16 attention projections | The heart of the post: what Blackwell's FP4 tensor cores buy over BF16/FP8 at decode (memory-bound) and prefill (compute-bound) batch sizes. Contrast with Qwen on Hopper, where Marlin 4-bit fell to 0.5x BF16 at prefill |
| **4c first-use cost** | First-call time of every kernel above in a fresh FlashInfer cache, vs steady state | Qwen's JIT finding, re-checked on Blackwell (trtllm-gen ships many kernels prebuilt) |

The KDA state budget is read from the server's startup log in Phase 3 (state slots, admitted requests,
KV tokens), the same way the DeltaNet budget was for Qwen.

### 5.3 Phase 3: serving (`run_bench.py --model kimi-k3`, 15 min, one setting)

- Workloads as for Qwen: agent (16 shared 4K system prompts + 128-token question), chat (1K in), long (8K in); 256 output tokens, thinking off.
- Sweep: agent at 1, 16, 64 concurrent; chat at 64; long at 16. Then the 37-prompt tool-call check, then whatever stress tests still fit (cold start, Poisson arrivals, p99.9, burst).
- `--reuse-server`: the server started at 0:16 is reused, so the 15 minutes are spent on tests, not on loading weights.
- Optional restarts (2-hour cap only): `low_latency` (plain TP8, no DCP), `kda_bf16` (`--mamba-ssm-dtype bfloat16`, about half the state per request), `kv_fp8` (`--kv-cache-dtype fp8_e4m3`).
- Out of scope for this session: NVFP4 (another image and 1.5 TB of weights), DSPARK/DFLASH speculative decoding (draft models), multi-node.

### 5.4 Smoke tests and dashboards (hold window)

```bash
ssh -N -p <port> -L 30000:127.0.0.1:30000 -L 8000:127.0.0.1:8000 -o StrictHostKeyChecking=no root@<host>
python smoke_test.py --model kimi-k3 --no-think       # chat, tool call, prefix-cache hit
open http://127.0.0.1:8000/                           # Phase 3 dashboard.html and Phase 4 fi_report.html
```

## 6. Status (2026-11-03)

- Image `lmsysorg/sglang:v0.5.23-cu130` built on Modal: torch 2.14.1 (CUDA 13.0), FlashInfer 0.7.2, SGLang 0.5.23;
  every flag the plan uses exists (`--dcp-size`, `--mamba-ssm-dtype`, `--kv-cache-dtype fp8_e4m3`, `--cuda-graph-max-bs-decode`).
- Weights downloaded: 96/96 shards, 1,561 GB, in 12 min (12 CPU containers, ~2.2 GB/s together).
- Preflight passed: `bench_serving` has every flag `run_bench.py` uses; Kimi's TikToken tokenizer loads (163,840 tokens).
- Dev check on 1 x B300 passed (Phase 4 in quick form, report built). Kernels are timed inside CUDA graphs, as the
  server runs them: timed eagerly, FlashInfer's FP4 GEMM showed ~230 us of per-call host overhead that hid the kernel.
  First numbers (16 requests x 8K context; one 6144 x 3584 expert matrix):
  - MLA decode per layer: TP8 39 us (BF16 cache), 18.5 us (FP8 cache); DCP8 13 us.
  - 256 tokens: MXFP4 2.3 PFLOPS, NVFP4 1.9, FP8 1.8, BF16 1.2. At 1 token every format takes 4-8 us: one expert
    matrix is too small for 4-bit weights to show a bandwidth advantage on its own.
- Spent so far: ~$1 (covered by credits). Blocking: the $42 usage limit (section 3).

### Run attempts (2026-11-03)

- **Modal, attempt 1:** Phase 4 finished in ~1 min (results kept, backed up to S3). The server start read weights at
  ~0.4 GB/s: SGLang's default multi-threaded loader reads every shard on every one of the 8 ranks (12.5 TB from a
  Volume that gives ~2.2 GB/s at best). Fix: `--weight-loader-prefetch-checkpoints --weight-loader-prefetch-num-threads 1`
  (each rank prefetches 1/8 into the shared page cache; 1.56 TB read once, ~13 min).
- **The container was preempted** ~6 min into loading. Modal GPU functions are always preemptible (`nonpreemptible`
  is not available for GPUs), restart with the same input, and restart the timeout too. Stopped; ~$10.56 spent in total.
- **Decision: regular (non-preemptible) GPUs only.** Nebius has B300 quota in us-north1, uk-south1, eu-west2 but no
  free 8 x B300, regular or preemptible, and no free 8 x H200 (2026-11-03 afternoon).
- **Now:** `capacity_watch.py` tries a regular Nebius 8 x B300 in the three regions every 15 min and checks Vast.ai
  (plus Prime Intellect / Shadeform with API keys); on success the run starts by itself (`nebius_kimi_k3.sh`,
  kill switch 150 min of VM time, results to S3 every 2 min, resumable after a stop). Dashboard: `.nebius/capacity.html`.
- **Fallback (night):** Modal with a total budget and attempt limit kept on the results Volume (survives restarts),
  Phase 3 resume, Phase 4 skipped when done, and `modal run modal_kimi_k3.py::sync` copying results to S3 every 2 min.

## 7. Steps before the run

1. Usage limit raised past ~$150 (section 3). **Blocking.**
2. `modal run modal_kimi_k3.py::download` (once, ~12 min, CPU only). Done.
3. `modal run modal_kimi_k3.py::preflight` (CPU) and `::dev` (1 x B300, ~3 min). Done.
4. `modal run --detach modal_kimi_k3.py::run`, then the tunnel command from its log (`modal app logs kimi-k3-serving`).
5. `modal run modal_kimi_k3.py::fetch`, commit `bench_results/`, then `blog_source_kimi_k3.md` from the committed results.

## 8. Blog angle

"Serving a 2.8-trillion-parameter open model on one rented node": what one 8 x B300 machine buys, why it is
one node instead of two H200 nodes (memory, native FP4), what MXFP4 is and why a 4-bit model can be the
original rather than a compromise, how the bottleneck moves from BF16 weights to FP4 experts as the batch
grows, and the recurring lesson from Qwen: for hybrid linear-attention models, the recurrent state, not the
KV cache, sets how many users fit. With the cost of the whole profiling session.
