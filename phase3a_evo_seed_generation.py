"""Phase 3a -- Evo seed generation.

Generates the initial candidate population with evo-1.5-8k-base. Each sample is
prompted with a prefix taken from a training terminator chosen as target anchor, so
that sampling begins inside terminator-like sequence space rather than in
unconstrained genomic space. Generation is checkpointed and restart-safe.

Run before phase3b_constrained_design.py.
"""

import random

import numpy as np
import pandas as pd
import torch


from evo import Evo, generate

EVO_MODEL_NAME = "evo-1.5-8k-base"

EVO_DEVICE = "cuda:0"

EVO_TEMPERATURE = 0.7

EVO_TOP_K = 4

EVO_TOP_P = 1.0

EVO_BATCH_SIZE = 2

EVO_CACHED_GENERATION = True

EVO_PREPEND_BOS = False

EVO_VERBOSE = 0

EVO_RANDOM_SEED = 2026


def find_project_root():

    if env_root:

        expected_plan = (
            root
            / "phase3_results"
            / "target_sweep_standard_TE_0p00_0p99"
            / "evo"
            / "evo_generation_plan_all_targets.csv"
        )

        if expected_plan.exists():

            return root

    for root in [cwd, *cwd.parents]:

        expected_plan = (
            root
            / "phase3_results"
            / "target_sweep_standard_TE_0p00_0p99"
            / "evo"
            / "evo_generation_plan_all_targets.csv"
        )

        if expected_plan.exists():

            return root

    raise FileNotFoundError(
        "Could not locate Phase-3 Evo plan.\n" "Set PHASE3_PROJECT_ROOT if necessary."
    )


if torch.cuda.is_available():

    pass


try:

    pass

except Exception:

    pass


if not torch.cuda.is_available():

    raise RuntimeError("CUDA is not available in the Evo kernel.")

df_plan = pd.read_csv(PLAN_PATH)

required_columns = [
    "job_id",
    "target_id",
    "target_te",
    "target_model_y",
    "anchor_rank",
    "anchor_sample_idx",
    "prompt",
    "prompt_length",
    "target_length",
    "n_tokens",
    "n_samples",
]

missing = [column for column in required_columns if column not in df_plan.columns]

if missing:

    raise RuntimeError(f"Missing Evo-plan columns: " f"{missing}")

df_plan["prompt"] = df_plan["prompt"].astype(str).str.upper()

DNA = {
    "A",
    "C",
    "G",
    "T",
}

if not (
    df_plan["prompt"]
    .map(lambda sequence: (len(sequence) > 0 and set(sequence).issubset(DNA)))
    .all()
):

    raise RuntimeError("Invalid DNA prompt.")

if not np.array_equal(
    df_plan["prompt"].str.len().to_numpy(), df_plan["prompt_length"].to_numpy()
):

    raise RuntimeError("Prompt-length QA failed.")

if not np.array_equal(
    (df_plan["prompt_length"] + df_plan["n_tokens"]).to_numpy(),
    df_plan["target_length"].to_numpy(),
):

    raise RuntimeError("Target-length arithmetic failed.")

if (df_plan["n_tokens"] <= 0).any():

    raise RuntimeError("Plan contains non-positive n_tokens.")

planned_total = int(df_plan["n_samples"].sum())


def set_seed(seed):

    seed = int(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(EVO_RANDOM_SEED)


try:

    evo_wrapper = Evo(EVO_MODEL_NAME, device=EVO_DEVICE)

except TypeError:

    evo_wrapper = Evo(EVO_MODEL_NAME)

    evo_wrapper.model = evo_wrapper.model.to(EVO_DEVICE)

evo_model = evo_wrapper.model.eval()

evo_tokenizer = evo_wrapper.tokenizer


if RAW_PATH.exists():

    df_existing = pd.read_csv(RAW_PATH)

    records = df_existing.to_dict("records")


else:

    df_existing = pd.DataFrame()

    records = []


def completed_for_job(job_id):

    if len(df_existing) == 0 or "job_id" not in df_existing.columns:

        return 0

    return int(
        (pd.to_numeric(df_existing["job_id"], errors="coerce") == int(job_id)).sum()
    )


def atomic_save(current_records):

    temporary_path = RAW_PATH.with_suffix(".tmp.csv")

    temporary_path.replace(RAW_PATH)


for plan_row_idx, row in df_plan.iterrows():

    job_id = int(row["job_id"])

    requested = int(row["n_samples"])

    done = completed_for_job(job_id)

    if done >= requested:

        continue

    prompt = str(row["prompt"])

    n_tokens = int(row["n_tokens"])

    while done < requested:

        batch_size = min(EVO_BATCH_SIZE, requested - done)

        generation_seed = EVO_RANDOM_SEED + job_id * 100000 + done

        set_seed(generation_seed)

        try:

            continuations, scores = generate(
                [prompt] * batch_size,
                evo_model,
                evo_tokenizer,
                n_tokens=n_tokens,
                temperature=EVO_TEMPERATURE,
                top_k=EVO_TOP_K,
                top_p=EVO_TOP_P,
                batched=True,
                prepend_bos=EVO_PREPEND_BOS,
                cached_generation=EVO_CACHED_GENERATION,
                device=EVO_DEVICE,
                verbose=EVO_VERBOSE,
            )

        except torch.cuda.OutOfMemoryError:

            torch.cuda.empty_cache()

            batch_size = 1

            continuations, scores = generate(
                [prompt],
                evo_model,
                evo_tokenizer,
                n_tokens=n_tokens,
                temperature=EVO_TEMPERATURE,
                top_k=EVO_TOP_K,
                top_p=EVO_TOP_P,
                batched=True,
                prepend_bos=EVO_PREPEND_BOS,
                cached_generation=EVO_CACHED_GENERATION,
                device=EVO_DEVICE,
                verbose=EVO_VERBOSE,
            )

        if not (len(continuations) == len(scores) == batch_size):

            raise RuntimeError("Unexpected Evo output length.")

        for local_index, (continuation, evo_score) in enumerate(
            zip(continuations, scores)
        ):

            continuation_raw = str(continuation)

            continuation_upper = continuation_raw.upper()

            sequence = prompt + continuation_upper

            records.append(
                {
                    "job_id": job_id,
                    "sample_in_job": int(done + local_index),
                    "generation_seed": int(generation_seed),
                    "target_id": str(row["target_id"]),
                    "target_te": float(row["target_te"]),
                    "target_average_strength": float(row["target_average_strength"]),
                    "target_model_y": float(row["target_model_y"]),
                    "anchor_rank": int(row["anchor_rank"]),
                    "anchor_sample_idx": int(row["anchor_sample_idx"]),
                    "prompt": prompt,
                    "prompt_length": len(prompt),
                    "continuation_repr": repr(continuation_raw),
                    "continuation_length": len(continuation_upper),
                    "target_length": int(row["target_length"]),
                    "n_tokens": n_tokens,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "evo_generation_score": float(evo_score),
                    "evo_model_name": EVO_MODEL_NAME,
                    "evo_temperature": EVO_TEMPERATURE,
                    "evo_top_k": EVO_TOP_K,
                    "evo_top_p": EVO_TOP_P,
                    "evo_cached_generation": EVO_CACHED_GENERATION,
                    "evo_prepend_bos": EVO_PREPEND_BOS,
                }
            )

        done += batch_size

        atomic_save(records)

    if (plan_row_idx + 1) % 20 == 0:

        pass

df_raw = pd.DataFrame(records)

atomic_save(records)
