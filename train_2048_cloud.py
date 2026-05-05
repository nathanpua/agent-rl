"""Train Qwen 3 8B to play 2048 using ART LocalBackend on Vast.ai RTX 4090."""
import asyncio
import math
import os
import random
import string
from defusedxml.ElementTree import fromstring as xml_fromstring
from typing import Literal, TypedDict

import art
from art.dev import InternalModelConfig, EngineArgs
from art.local import LocalBackend
from art.utils.strip_logprobs import strip_logprobs
from openai import AsyncOpenAI
from pydantic import BaseModel
import requests
import weave

# ─── Config ───────────────────────────────────────────────────────────
WINNING_VALUE = 64
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
TRAINING_STEPS = int(os.environ.get("TRAINING_STEPS", "20"))
GAMES_PER_STEP = int(os.environ.get("GAMES_PER_STEP", "18"))
MAX_HISTORY_TURNS = 15  # Keep last 15 board+move pairs (~830 tokens, well under 1920 input limit)

random.seed(42)


# Warn if WANDB_API_KEY is missing
if not os.environ.get("WANDB_API_KEY"):
    print("WARNING: WANDB_API_KEY not set. Training will work but metrics won't be logged to W&B.")

# Initialize Weave for LLM call tracing (entity auto-detected from WANDB_API_KEY)
weave.init("2048", settings={"print_call_link": False}, postprocess_output=strip_logprobs)


# ─── 2048 Game Environment ────────────────────────────────────────────
class TwentyFortyEightGame(TypedDict):
    id: str
    board: list[list[int | None]]


def populate_random_cell(game: TwentyFortyEightGame) -> None:
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    random_clear_coordinates = random.choice(all_clear_coordinates)
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }
    populate_random_cell(game)
    populate_random_cell(game)
    return game


def render_board(game: TwentyFortyEightGame) -> str:
    board = game["board"]
    max_cell_width = max(
        [len(str(cell)) for row in board for cell in row if cell is not None]
    )
    board_str = ""
    for row in board:
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


def condense_sequence(sequence: list[int | None]) -> list[int | None]:
    condensed_sequence = []
    gapless_sequence = [cell for cell in sequence if cell is not None]
    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1
    return condensed_sequence + [None] * (4 - len(condensed_sequence))


def condense_board(
    game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]
) -> None:
    if direction == "left":
        for row in game["board"]:
            condensed_row = condense_sequence(row)
            for i in range(len(row)):
                row[i] = condensed_row[i]
    if direction == "right":
        for row in game["board"]:
            reversed_row = row[::-1]
            condensed_row = condense_sequence(reversed_row)[::-1]
            for i in range(len(row)):
                row[i] = condensed_row[i]
    if direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]
    if direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            reversed_column = column[::-1]
            condensed_column = condense_sequence(reversed_column)[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


def has_empty_cell(game: TwentyFortyEightGame) -> bool:
    return any(cell is None for row in game["board"] for cell in row)


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    direction = None
    try:
        root = xml_fromstring(move_xml)
        direction = root.text
    except Exception:
        raise ValueError("Invalid xml")
    if direction not in ["left", "right", "up", "down"]:
        raise ValueError("Invalid direction")
    condense_board(game, direction)
    if has_empty_cell(game):
        populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    return max([cell for row in game["board"] for cell in row if cell is not None])


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    if max_cell_value(game) >= WINNING_VALUE:
        return True
    if any(cell is None for row in game["board"] for cell in row):
        return False
    return True


def total_board_value(game: TwentyFortyEightGame) -> int:
    return sum([cell for row in game["board"] for cell in row if cell is not None])


def truncate_messages(messages: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Keep system prompt + most recent turns.

    Drops oldest user/assistant message pairs to prevent exceeding the
    vLLM context limit (2048 tokens ≈ 1920 input after 128 output reserved).
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]
    remaining = messages[1:]

    # Each turn is a user message + assistant response pair (2 messages)
    max_messages = max_turns * 2
    if len(remaining) > max_messages:
        remaining = remaining[-max_messages:]

    return [system_msg] + remaining


# ─── Rollout ──────────────────────────────────────────────────────────
class Scenario2048(BaseModel):
    step: int


@weave.op
@art.retry(exceptions=(requests.ReadTimeout,))
async def rollout(model: art.Model, scenario: Scenario2048) -> art.Trajectory:
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    game = generate_game()
    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are an excellent 2048 player. Always choose the move most likely to "
                    "lead to combine cells to eventually reach the number 2048. Optional moves "
                    "are 'left', 'right', 'up', 'down'. Return your move as an XML object with "
                    "a single property 'move', like so: <move>left</move>"
                ),
            }
        ],
        metadata={
            "game_id": game["id"],
            "step": scenario.step,
        },
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        try:
            messages = truncate_messages(trajectory.messages())
            chat_completion = await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.get_inference_name(),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            print(f"caught exception generating chat completion: {e}")
            raise

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value
            trajectory.metrics["move_number"] = move_number

            if max_value < WINNING_VALUE:
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
            else:
                trajectory.reward = 2
            break

    return trajectory


# ─── Main ─────────────────────────────────────────────────────────────
async def main():
    model = art.TrainableModel(
        name="qwen3-2048",
        project="2048",
        base_model="Qwen/Qwen3-8B",
        _internal_config=InternalModelConfig(
            # No trainer_gpu_ids / inference_gpu_ids — single GPU time-shares automatically
            engine_args=EngineArgs(
                gpu_memory_utilization=0.75,
                max_model_len=2048,
                max_num_seqs=4,
                enforce_eager=True,
            ),
            init_args={
                "load_in_4bit": True,
                "max_seq_length": 2048,
            },
        ),
    )

    backend = LocalBackend(path=f"{OUTPUT_DIR}/.art")
    await model.register(backend)

    print(f"Training for {TRAINING_STEPS} steps, {GAMES_PER_STEP} games per step")
    print(f"Output directory: {OUTPUT_DIR}")

    for i in range(await model.get_step(), TRAINING_STEPS):
        print(f"\n{'='*60}")
        print(f"Step {i + 1}/{TRAINING_STEPS}")
        print(f"{'='*60}")

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, Scenario2048(step=i))
                    for _ in range(GAMES_PER_STEP)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=GAMES_PER_STEP,
        )
        await model.delete_checkpoints("train/reward")

        result = await backend.train(model, train_groups, learning_rate=5e-6)
        await model.log(
            train_groups, metrics=result.metrics, step=result.step, split="train"
        )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    # Free GPU memory so export can load the model on CPU without OOM.
    # The vLLM inference server holds ~15GB GPU; we must release it first.
    print("Cleaning up training resources...")
    del model, backend
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory freed")

    # Run export in the same process — avoids orphaned vLLM child processes
    # and container restarts that occur when using pkill or os._exit(0).
    import export_model
    export_model.main()

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
