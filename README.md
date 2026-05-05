# agent-rl

Training and evaluating LLM agents to play 2048 using reinforcement learning with [ART](https://github.com/openpipe/art) (OpenPipe).

## Architecture

```
[MacBook Air]                         [Vast.ai RTX 4090 (24GB)]
     |                                        |
     | 1. docker build + push                 |
     |--------------------------------------->|
     |                                        | 2. Container runs:
     |                                        |    - ART LocalBackend
     |                                        |    - vLLM inference + QLoRA training
     |                                        |    - 20 steps x 18 games
     |                                        |    - Merge LoRA + export GGUF
     |                                        |    - Upload GGUF to HuggingFace Hub
     |                                        |
     | 3. Download GGUF from HuggingFace     |
     |    + create LM Studio model wrapper   |
     |<---------------------------------------|
     |                                        | 4. Instance destroyed
     | 5. Load in LM Studio                   |
     |    (appears with thinking toggle)      |
     |    Run 2048-local.ipynb to evaluate    |
```

## How Training Works

### Single-GPU Time-Sharing on RTX 4090 (24GB)

Qwen3-8B in full BF16 is ~16 GiB, which won't fit alongside a training optimizer on 24 GB. The solution is **QLoRA + time-sharing**:

1. **Unsloth loads the model in 4-bit** (~5 GiB GPU) for LoRA training (0.27% of params trainable)
2. **vLLM loads the full BF16 model** (~15.5 GiB GPU) for inference
3. ART's `LocalBackend` time-shares between the two — they never run simultaneously

Memory breakdown at `gpu_memory_utilization=0.75`:
- vLLM model weights (BF16): ~15.5 GiB
- KV cache (14,448 tokens): ~1.98 GiB
- Unsloth 4-bit model: ~0.6 GiB
- Total: ~18 GiB / 23.6 GiB available

### Rollout: How the Model Plays 2048

Each "rollout" is one full game. The model sees the board as text and responds with an XML move:

```
System: You are an excellent 2048 player...

User: _|_|_|_
      _|2|_|_
      _|_|_|_
      _|_|4|2

Assistant: <move>left</move>

User: _|_|_|_
      |2|_|_|_
      _|_|2|_
      _|_|_|2

Assistant: <move>down</move>
...
```

The conversation history grows with each turn. To stay within the 2048-token context window, older turns are truncated — only the last 15 board+move pairs are kept (~830 tokens + system prompt). The full game history is preserved in the trajectory for reward computation.

Qwen3's thinking mode is disabled during training (`enable_thinking: False`) so the model outputs moves directly instead of burning output tokens on internal reasoning.

### Reward Signal

| Outcome | Reward |
|---------|--------|
| Win (reach tile 64) | +2.0 |
| Lose (board fills) | Scaled 0–1 based on max tile + board value |
| Invalid move | -1.0 |

The partial reward for losses uses log-scaled max tile value + 20% board value bonus, giving the model gradient even when it doesn't win.

### GRPO Training Loop

```
for each step (1..20):
    1. Gather: play 18 parallel games via vLLM inference
    2. Train: QLoRA update via unsloth/TRL (learning_rate=5e-6)
    3. Log: metrics to W&B + Weave
```

At step 1, the base Qwen3-8B model already wins some games (reward=2, max_value=64). Training improves the win rate and board strategy over 20 steps.

### Export Pipeline

After training completes, the container automatically:

1. Frees GPU memory (deletes model/backend, runs `gc.collect()` + `torch.cuda.empty_cache()`)
2. Merges LoRA adapter into the base model on CPU
3. Converts to GGUF f16 using llama.cpp
4. Quantizes to Q4_K_M (~4.7 GB)
5. Uploads to [HuggingFace Hub](https://huggingface.co/nathanpua/qwen3-8b-2048-gguf)

## Notebooks

### `2048-cloud.ipynb` - Cloud Training on RTX 4090

Trains **Qwen3-8B** to play 2048 on a rented [Vast.ai](https://vast.ai) RTX 4090 using ART's `LocalBackend`. The container automatically runs training then exports a quantized GGUF model for local inference.

**Setup:**
1. Copy `.env.example` to `.env` and fill in your credentials
2. Install [Docker Desktop](https://docker.com) with buildx support
3. Run `pip install vastai` and authenticate with `vastai set api-key YOUR_KEY`
4. Open the notebook and run all cells

**What it does:**
- Builds a Docker image with CUDA 12.8, ART, custom vLLM fork, and llama.cpp
- Pushes image to Docker Hub
- Launches an interruptable RTX 4090 instance on Vast.ai (80GB disk)
- Trains 20 steps x 18 games/step using GRPO rollouts
- Merges LoRA adapter into base model and converts to GGUF Q4_K_M
- Uploads GGUF to HuggingFace Hub, then downloads locally
- Creates an LM Studio `model.yaml` wrapper so the thinking toggle appears in the UI
- Destroys the instance to stop billing

**Estimated cost:** ~$0.15 for ~1 hour of training + export at ~$0.15/hr

### `2048-local.ipynb` - Local Inference with LM Studio

Evaluates a trained **Qwen3-8B** model running locally via [LM Studio](https://lmstudio.ai). No GPU required — runs on a laptop.

**Setup:**
1. Install [LM Studio](https://lmstudio.ai)
2. Run `2048-cloud.ipynb` to train and download the GGUF model (it will appear as `nathanpua/qwen3-8b-2048` with a thinking toggle)
3. Start the local server: load the model, then click **Start Server**
4. The server runs at `http://localhost:1234/v1` by default
5. Open the notebook and run all cells

**What it does:**
- Plays games of 2048 by sending the board state as text to the model
- The model responds with moves as XML: `<move>left</move>`
- Uses `extract_move()` to find the move in both `content` and `reasoning_content` (handles thinking mode)
- Runs a single live game with move-by-move output, then 10 games in parallel batches of 4
- Tracks win rate, average moves, max tile, and board value

**Thinking mode:** The trained model's chat template supports Qwen3's thinking mode. LM Studio exposes an "Enable Thinking" toggle via the `model.yaml` wrapper created during download. The `extract_move()` function handles both modes transparently.

## Files

| File | Purpose |
|------|---------|
| `2048-cloud.ipynb` | Orchestration: build, launch, monitor, download, cleanup |
| `2048-local.ipynb` | Local inference and evaluation via LM Studio |
| `train_2048_cloud.py` | ART LocalBackend training script (runs inside Docker) |
| `export_model.py` | LoRA merge + GGUF conversion + HuggingFace upload (runs inside Docker) |
| `Dockerfile` | CUDA 12.8 + ART + llama.cpp training environment |
| `requirements-hotfix.txt` | Dependency conflict fixes (layer that rebuilds quickly) |
| `.env.example` | Required environment variables template |

## Game Environment

- **Board**: 4x4 grid where cells contain powers of 2 (or are empty)
- **Moves**: `left`, `right`, `up`, `down` - slides tiles and merges matching adjacent ones
- **Spawning**: After each move, a new tile appears (2 with 90% probability, 4 with 10%)
- **Winning condition**: Reach a tile value of **64** (reduced from 2048 to keep games short)
- **Losing condition**: Board fills up without reaching 64
