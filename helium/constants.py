"""Helium-v1. Swap REPORT_MODEL or HELIUM_RUNTIME without changing diagnose()."""

REPORT_MODEL = "Qwen/Qwen3.6-27B"
MAX_MODEL_LEN = 8192
GPU = "B300"
# Helium KV pool. Weights stay ~52 GiB; this is not model quality.
KV_CACHE_GIB = 8
KV_CACHE_MEMORY_BYTES = KV_CACHE_GIB * 1024**3
# Qwen3.6 GDN/Mamba: one cache block per in-flight sequence. 8 GiB KV
# only has ~167 blocks; vLLM default max_num_seqs=1024 then fails.
# Helium is 1–2 jobs. Do not raise KV to satisfy the 1024 default.
MAX_NUM_SEQS = 8
# vLLM still checks free >= util * *total* card, even with kv_cache_memory_bytes.
# Helium loads first on an empty GPU. 0.26 ≈ weights + 8 GiB KV.
GPU_MEMORY_UTILIZATION = 0.26
# Nitrogen loads second (Helium already resident). 0.92 of 268 GiB is 246;
# only ~206 GiB is free then. 0.65 * 268 ≈ 174 < 206.
NITROGEN_GPU_MEMORY_UTILIZATION = 0.65
# Third engine: Qwen3.5-9B text for Dev 2. ~19 GiB weights + 4 GiB KV.
# Ceiling 0.35 of the *card*; _share_gpu_utilization also caps to free memory.
OXYGEN_MODEL = "Qwen/Qwen3.5-9B"
OXYGEN_KV_CACHE_GIB = 4
OXYGEN_KV_CACHE_MEMORY_BYTES = OXYGEN_KV_CACHE_GIB * 1024**3
OXYGEN_GPU_MEMORY_UTILIZATION = 0.35
# Fourth engine: Gemma 4 26B-A4B-IT vision (~52 GiB weights + 8 GiB KV).
# Loads last; ceiling 0.32 of the card, also capped to remaining free.
FLUORINE_MODEL = "google/gemma-4-26B-A4B-it"
FLUORINE_KV_CACHE_GIB = 8
FLUORINE_KV_CACHE_MEMORY_BYTES = FLUORINE_KV_CACHE_GIB * 1024**3
FLUORINE_GPU_MEMORY_UTILIZATION = 0.32

MODAL_APP_NAME = "coherence-helium"
HELIUM_CLS_NAME = "HeliumGPU"

# deployed: Cls.from_name after `modal deploy` (min_containers=1, always on).
# ephemeral: app.run() per call (cold start). Env HELIUM_RUNTIME overrides.
HELIUM_RUNTIME = "deployed"
# Cold replica boot of four vLLM engines can take many minutes. Demo capture
# must not sit on diagnose. Env HELIUM_INVOKE_TIMEOUT overrides.
HELIUM_INVOKE_TIMEOUT_S = 40
