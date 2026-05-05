# Vast.ai base image — layers are pre-cached on host machines for fast pulls.
# Built FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 + dev tools + Python 3.12.
# See: https://github.com/vast-ai/base-image
FROM vastai/base-image:cuda-12.8.1-auto

# Override the Vast.ai entrypoint for headless training (no SSH/Jupyter/Portal)
ENTRYPOINT []

# ── Stable layers (cached across rebuilds) ───────────────────────────

# Install ART with ALL backend dependencies (peft, unsloth, vllm, wandb, transformers, trl, etc.)
RUN --mount=type=cache,target=/root/.cache/pip \
    . /venv/main/bin/activate && \
    pip install \
    "openpipe-art[backend]==0.5.17" \
    sentencepiece \
    defusedxml

# Override stock vLLM with ART's custom fork (has Qwen 3.5 LoRA patches not in stock vLLM)
RUN --mount=type=cache,target=/root/.cache/pip \
    . /venv/main/bin/activate && \
    pip install --force-reinstall --no-deps \
    https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0+art1-cp38-abi3-manylinux_2_31_x86_64.whl

# Build llama.cpp for GGUF conversion (base image already has cmake + build-essential).
# Use --no-deps to prevent llama.cpp from downgrading transformers, numpy, protobuf, etc.
RUN cd /opt && git clone https://github.com/ggml-org/llama.cpp && \
    cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc) && \
    . /venv/main/bin/activate && pip install --no-deps -e . && pip install gguf

# ── trl patches (after llama.cpp so patches don't invalidate its cache) ──

# Patch trl: force-disable vllm_ascend detection (NVIDIA systems don't use Ascend backend)
RUN . /venv/main/bin/activate && python3 -c "import trl.import_utils as m; p=m.__file__; c=open(p).read().replace('_vllm_ascend_available = _is_package_available(\"vllm_ascend\")','_vllm_ascend_available = False'); open(p,'w').write(c); print('Patched',p)"

# Patch trl: make optional deps (mergekit, llm_blender) non-fatal in callbacks imports.
# These packages are incompatible with the rest of the stack (transformers 5.x, pydantic, etc.)
# so we wrap their imports in try/except instead of installing them.
RUN . /venv/main/bin/activate && python3 -c "\
import os, trl;\
NL = chr(10);\
p = os.path.join(os.path.dirname(trl.__file__), 'trainer', 'callbacks.py');\
c = open(p).read();\
c = c.replace('from .judges import BasePairwiseJudge',\
    'try:' + NL + '    from .judges import BasePairwiseJudge' + NL +\
    'except (ImportError, RuntimeError):' + NL + '    BasePairwiseJudge = None');\
c = c.replace('from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf',\
    'try:' + NL + '    from ..mergekit_utils import MergeConfig, merge_models, upload_model_to_hf' + NL +\
    'except (ImportError, RuntimeError):' + NL + '    MergeConfig = merge_models = upload_model_to_hf = None');\
open(p, 'w').write(c);\
print('Patched', p)"

# Patch trl: handle missing GuidedDecodingParams from ART's custom vLLM fork.
# The fork (v0.17.0+art1) doesn't have this class but trl expects it.
# Uses regex to preserve original indentation (import may be inside a function/block).
RUN . /venv/main/bin/activate && python3 -c "\
import os, re, trl;\
NL = chr(10);\
p = os.path.join(os.path.dirname(trl.__file__), 'trainer', 'grpo_trainer.py');\
c = open(p).read();\
c = re.sub('([ \\t]*)from vllm\\.sampling_params import GuidedDecodingParams',\
    lambda m: m.group(1)+'try:'+NL+m.group(1)+'    from vllm.sampling_params import GuidedDecodingParams'+NL+m.group(1)+'except ImportError:'+NL+m.group(1)+'    GuidedDecodingParams=None',\
    c);\
open(p, 'w').write(c);\
print('Patched', p)"

# ── Hotfix layer (only this rebuilds when deps change) ──────────────

COPY requirements-hotfix.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    . /venv/main/bin/activate && \
    pip install -r /tmp/requirements-hotfix.txt

# Validate critical imports at build time — catches missing deps NOW
# instead of failing on the instance 15 minutes later.
RUN . /venv/main/bin/activate && \
    python3 -c "import weave; from trl import GRPOConfig, GRPOTrainer; print('All critical imports OK')"

WORKDIR /app
COPY train_2048_cloud.py .
COPY export_model.py .

# Training output directory
VOLUME /app/output

# Default: run training (export_model.main() is called from within train_2048_cloud.py)
CMD ["bash", "-c", ". /venv/main/bin/activate && python3 train_2048_cloud.py"]
