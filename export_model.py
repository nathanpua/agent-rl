"""Merge LoRA adapter into base model and convert to GGUF for LM Studio."""
import os
import subprocess
import sys

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
BASE_MODEL = "Qwen/Qwen3-8B"


def find_latest_checkpoint():
    """Find the latest checkpoint directory."""
    checkpoints_dir = f"{OUTPUT_DIR}/.art/2048/models/qwen3-2048/checkpoints"
    if not os.path.exists(checkpoints_dir):
        print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
        print("Did training complete successfully?")
        sys.exit(1)

    # Checkpoints are named with zero-padded numbers: 0001, 0002, ..., 0019
    checkpoints = [
        d for d in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, d)) and d.isdigit()
    ]
    if not checkpoints:
        print("ERROR: No checkpoints found")
        sys.exit(1)

    # Sort numerically (0019 > 0005)
    checkpoints.sort(key=lambda d: int(d))
    latest = os.path.join(checkpoints_dir, checkpoints[-1])
    print(f"Latest checkpoint: {latest}")
    return latest


def main():
    adapter_path = find_latest_checkpoint()

    print("Loading base model and LoRA adapter...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged_path = f"{OUTPUT_DIR}/merged"
    merged.save_pretrained(merged_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")

    print("Converting to GGUF (f16)...")
    f16_gguf = f"{OUTPUT_DIR}/qwen3-8b-2048-f16.gguf"
    subprocess.run(
        [
            "python3",
            "/opt/llama.cpp/convert_hf_to_gguf.py",
            merged_path,
            "--outtype",
            "f16",
            "--outfile",
            f16_gguf,
        ],
        check=True,
    )
    print(f"F16 GGUF saved: {f16_gguf}")

    # Free disk space before quantization: remove merged HF model (~18GB)
    import shutil
    shutil.rmtree(merged_path, ignore_errors=True)
    print("Cleaned up merged model to free disk space")

    print("Quantizing to Q4_K_M...")
    q4_gguf = f"{OUTPUT_DIR}/qwen3-8b-2048-Q4_K_M.gguf"
    subprocess.run(
        [
            "/opt/llama.cpp/build/bin/llama-quantize",
            f16_gguf,
            q4_gguf,
            "Q4_K_M",
        ],
        check=True,
    )
    print(f"Q4_K_M GGUF saved: {q4_gguf}")

    # Clean up intermediate f16 GGUF (~18GB)
    os.remove(f16_gguf)

    # Upload to HuggingFace Hub for download (SSH not available with runtype="args")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("\nUploading Q4_K_M GGUF to HuggingFace Hub...")
        from huggingface_hub import HfApi

        HF_REPO = os.environ.get("HF_REPO", "nathanpua/qwen3-8b-2048-gguf")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
        api.upload_file(
            path_or_fileobj=q4_gguf,
            path_in_repo="qwen3-8b-2048-Q4_K_M.gguf",
            repo_id=HF_REPO,
            repo_type="model",
        )
        print(f"Uploaded to: https://huggingface.co/{HF_REPO}")
    else:
        print("WARNING: HF_TOKEN not set, skipping HuggingFace upload")

    print(f"\nExport complete! Output: {q4_gguf}")


if __name__ == "__main__":
    main()
