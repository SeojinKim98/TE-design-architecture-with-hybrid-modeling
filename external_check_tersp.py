"""Independent cross-model check with TerSP.

Scores the frozen design set with the published TerSP predictor, applied without
retraining and strictly post hoc. Candidates are annotated in place into the TerSP
region schema using the same hairpin search as Phase 1; candidates with no
schema-compatible hairpin are reported unscorable rather than padded.

Reports agreement at the level of the requested targets and, separately, at the
level of individual candidates.
"""

import sys
import urllib.request
import importlib.util

missing = []

for import_name, pip_name in packages.items():

    if importlib.util.find_spec(import_name) is None:
        missing.append(pip_name)

if missing:

    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *missing,
        ]
    )

else:

    pass

MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "gkundlatsch/TerSP-Web/main/"
    "terminator_strength_predictor_v2.joblib"
)


if not MODEL_PATH.exists():

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


import joblib

try:

    model_data = joblib.load(MODEL_PATH)

except Exception as exc:

    raise RuntimeError(
        "\nTerSP model download succeeded, but the serialized model "
        "could not be loaded in the current Colab environment.\n"
        "Do NOT install old sklearn from source.\n"
        f"Original error: {type(exc).__name__}: {exc}"
    )

if isinstance(model_data, dict):

    if "model" not in model_data:

        raise RuntimeError(
            "The TerSP joblib file is a dictionary but has no 'model' key."
        )

    TERSP_MODEL = model_data["model"]

else:

    TERSP_MODEL = model_data


import numpy as np
import pandas as pd

if len(csv_names) != 1:

    raise RuntimeError(
        "Please upload exactly one CSV:\n"
        "phase3_all_final_candidates_for_external_validation.csv"
    )


df_phase3_all = pd.read_csv(INPUT_CSV)


required_columns = [
    "target_id",
    "target_te",
    "sequence",
]

missing_columns = [
    column for column in required_columns if column not in df_phase3_all.columns
]

if missing_columns:

    raise RuntimeError(f"Missing required columns: {missing_columns}")


def normalize_dna(sequence):

    sequence = str(sequence).upper().strip().replace("U", "T")

    if len(sequence) == 0:

        return None

    if not set(sequence).issubset(set("ACGT")):

        return None

    return sequence


df_phase3_all["sequence"] = df_phase3_all["sequence"].map(normalize_dna)

invalid_mask = df_phase3_all["sequence"].isna()

n_invalid = int(invalid_mask.sum())


if n_invalid > 0:

    raise RuntimeError(
        "Merged Phase-3 file contains invalid DNA sequences. "
        "Stop and inspect the merged file."
    )

df_phase3_all["target_te"] = pd.to_numeric(df_phase3_all["target_te"], errors="raise")

expected_target_te = np.asarray(
    [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.99,
    ],
    dtype=float,
)

observed_target_te = np.sort(df_phase3_all["target_te"].unique())


if len(observed_target_te) != len(expected_target_te) or not np.allclose(
    observed_target_te, expected_target_te, atol=1e-12
):

    raise RuntimeError(
        "The merged file does not contain the expected 11-target TE grid."
    )

if "target_model_y" not in df_phase3_all.columns:

    df_phase3_all["target_model_y"] = -np.log10(1.0 - df_phase3_all["target_te"])

before_dedup = len(df_phase3_all)

duplicate_mask = df_phase3_all.duplicated(
    subset=[
        "target_id",
        "sequence",
    ],
    keep="first",
)

n_duplicates = int(duplicate_mask.sum())

if n_duplicates > 0:

    df_phase3_all = df_phase3_all.loc[~duplicate_mask].copy()

after_dedup = len(df_phase3_all)


df_phase3_all = df_phase3_all.sort_values(
    [
        "target_te",
        "target_id",
    ]
).reset_index(drop=True)

if "external_validation_candidate_id" not in df_phase3_all.columns:

    df_phase3_all["external_validation_candidate_id"] = [
        f"candidate_{index:04d}" for index in range(1, len(df_phase3_all) + 1)
    ]

candidate_summary = (
    df_phase3_all.groupby(
        [
            "target_id",
            "target_te",
        ],
        sort=True,
    )
    .agg(
        N_candidates=("sequence", "size"),
        N_unique_sequences=("sequence", "nunique"),
    )
    .reset_index()
    .sort_values("target_te")
    .reset_index(drop=True)
)


display(candidate_summary)


sequence_lengths = df_phase3_all["sequence"].str.len()


if df_phase3_all["target_id"].nunique() != 11:

    raise RuntimeError("Expected 11 targets after loading.")

if len(df_phase3_all) <= 11:

    pass

else:

    pass


import numpy as np
import pandas as pd


def pair_score(left, right):

    pair = (left, right)

    if pair in [
        ("G", "C"),
        ("C", "G"),
    ]:

        return 3.0

    if pair in [
        ("A", "T"),
        ("T", "A"),
    ]:

        return 2.0

    if pair in [
        ("G", "T"),
        ("T", "G"),
    ]:

        return 1.0

    return 0.0


def find_tersp_compatible_hairpin(
    sequence, upstream_required=8, downstream_required=12
):

    sequence = normalize_dna(sequence)

    if sequence is None:

        return None

    best = None

    best_key = None

    for stem_length in range(4, 14):

        for loop_length in range(3, 9):

            window_length = 2 * stem_length + loop_length

            if window_length > len(sequence):

                continue

            for start in range(len(sequence) - window_length + 1):

                left_start = start

                left_end = left_start + stem_length

                loop_start = left_end

                loop_end = loop_start + loop_length

                right_start = loop_end

                right_end = right_start + stem_length

                if left_start < upstream_required:

                    continue

                if len(sequence) - right_end < downstream_required:

                    continue

                left = sequence[left_start:left_end]

                right = sequence[right_start:right_end]

                scores = np.asarray(
                    [
                        pair_score(left[i], right[stem_length - 1 - i])
                        for i in range(stem_length)
                    ],
                    dtype=float,
                )

                paired_fraction = float(np.mean(scores > 0))

                if paired_fraction < 0.75:

                    continue

                raw_score = float(scores.sum())

                adjusted_score = float(raw_score - 0.5 * abs(loop_length - 4))

                # Exact Phase-1 ordering.
                key = (
                    adjusted_score,
                    paired_fraction,
                    stem_length,
                    -abs(loop_length - 4),
                )

                # strict >
                if best_key is None or key > best_key:

                    best_key = key

                    best = {
                        "left_start": int(left_start),
                        "left_end": int(left_end),
                        "loop_start": int(loop_start),
                        "loop_end": int(loop_end),
                        "right_start": int(right_start),
                        "right_end": int(right_end),
                        "stem_length": int(stem_length),
                        "loop_length": int(loop_length),
                        "pair_fraction": float(paired_fraction),
                        "pairing_score": float(adjusted_score),
                    }

    return best


def sequence_to_tersp_regions(sequence):

    sequence = normalize_dna(sequence)

    if sequence is None:

        return {
            "tersp_eligible": False,
            "tersp_annotation_status": "invalid_sequence",
        }

    geometry = find_tersp_compatible_hairpin(sequence)

    if geometry is None:

        return {
            "tersp_eligible": False,
            "tersp_annotation_status": "no_context_compatible_hairpin",
        }

    a_tract_dna = sequence[geometry["left_start"] - 8 : geometry["left_start"]]

    first_half_dna = sequence[geometry["left_start"] : geometry["left_end"]]

    loop_dna = sequence[geometry["loop_start"] : geometry["loop_end"]]

    second_half_dna = sequence[geometry["right_start"] : geometry["right_end"]]

    u_tract_dna = sequence[geometry["right_end"] : geometry["right_end"] + 12]

    def to_rna(value):

        return value.replace("T", "U")

    return {
        "tersp_eligible": True,
        "tersp_annotation_status": "OK",
        "tersp_a_tract": to_rna(a_tract_dna),
        "tersp_first_half": to_rna(first_half_dna),
        "tersp_loop": to_rna(loop_dna),
        "tersp_second_half": to_rna(second_half_dna),
        "tersp_u_tract": to_rna(u_tract_dna),
        "tersp_stem_length": geometry["stem_length"],
        "tersp_loop_length": geometry["loop_length"],
        "tersp_pair_fraction": geometry["pair_fraction"],
        "tersp_pairing_score": geometry["pairing_score"],
        "tersp_left_start": geometry["left_start"],
        "tersp_right_end": geometry["right_end"],
    }


annotation_rows = []

for i, row in df_phase3_all.iterrows():

    annotation_rows.append(sequence_to_tersp_regions(row["sequence"]))

df_annotation = pd.DataFrame(annotation_rows)

df_phase3_tersp = pd.concat(
    [
        df_phase3_all.reset_index(drop=True),
        df_annotation,
    ],
    axis=1,
)

coverage_by_target = (
    df_phase3_tersp.groupby(
        [
            "target_id",
            "target_te",
        ]
    )
    .agg(
        N_total=("sequence", "size"),
        N_eligible=("tersp_eligible", "sum"),
    )
    .reset_index()
)

coverage_by_target["coverage"] = (
    coverage_by_target["N_eligible"] / coverage_by_target["N_total"]
)


display(coverage_by_target)


import numpy as np
import pandas as pd

from scipy.stats import entropy


def tersp_calculate_features(a_tract, u_tract, stem, loop):

    features = {}

    def nt_percent(seq, nt, length):

        return seq.count(nt) / length if length > 0 else 0

    def count_changes(seq):

        return sum(1 for x, y in zip(seq, seq[1:]) if x != y)

    a_sub = a_tract[2:]

    features.update(
        {
            "A%_total_A_tract": nt_percent(a_tract, "A", 8),
            "C%_A_tract": nt_percent(a_tract, "C", 8),
            "G%_A_tract": nt_percent(a_tract, "G", 8),
            "U%_A_tract": nt_percent(a_tract, "U", 8),
            "A%_6_A_tract": nt_percent(a_sub, "A", 6),
            "C%_6_A_tract": nt_percent(a_sub, "C", 6),
            "A_Tract_state-change": count_changes(a_tract),
        }
    )

    features.update(
        {
            "U%_Total_U_tract": nt_percent(u_tract, "U", 12),
            "G%_U_tract": nt_percent(u_tract, "G", 12),
            "A%_U_tract": nt_percent(u_tract, "A", 12),
            "C%_U_tract": nt_percent(u_tract, "C", 12),
            "A%_6_U_tract": nt_percent(u_tract[:6], "A", 6),
            "U%_6_U_tract": nt_percent(u_tract[:6], "U", 6),
            "U%_10_U_tract": nt_percent(u_tract[:10], "U", 10),
            "U_Tract_state-change": count_changes(u_tract),
        }
    )

    loop_len = len(loop)

    features.update(
        {
            "Tamanho Loop": loop_len,
            "%GC_Loop": (loop.count("G") + loop.count("C")) / loop_len,
        }
    )

    hp_len = len(stem)

    features.update(
        {
            "Tamanho Hairpin sem Loop": hp_len,
            "%G_HP": stem.count("G") / hp_len,
            "%C_HP": stem.count("C") / hp_len,
            "%A_HP": stem.count("A") / hp_len,
            "%U_HP": stem.count("U") / hp_len,
            "HP_S_Loop_state_change": sum(1 for x, y in zip(stem, stem[1:]) if x != y),
            "GC_Inicial_Hairpin": next(
                (i for i, c in enumerate(stem) if c not in ("G", "C")), hp_len
            ),
        }
    )

    features["Entropia_A_tract"] = entropy(
        [
            features["A%_total_A_tract"],
            features["C%_A_tract"],
            features["G%_A_tract"],
            features["U%_A_tract"],
        ],
        base=2,
    )

    features["Entropia_U_tract"] = entropy(
        [
            features["U%_Total_U_tract"],
            features["C%_U_tract"],
            features["G%_U_tract"],
            features["A%_U_tract"],
        ],
        base=2,
    )

    features["Entropia_HP_S_Loop"] = entropy(
        [
            features["%G_HP"],
            features["%C_HP"],
            features["%A_HP"],
            features["%U_HP"],
        ],
        base=2,
    )

    normalized = {
        "Tamanho Loop": (features["Tamanho Loop"] - 3) / (16 - 3),
        "A%_6_A_tract": features["A%_6_A_tract"],
        "C%_6_A_tract": features["C%_6_A_tract"],
        "U%_10_U_tract": features["U%_10_U_tract"],
        "U%_6_U_tract": features["U%_6_U_tract"],
        "A%_6_U_tract": features["A%_6_U_tract"],
        "C%_U_tract": features["C%_U_tract"],
        "Tamanho Hairpin sem Loop": (features["Tamanho Hairpin sem Loop"] - 6)
        / (49 - 6),
        "%GC_Loop": features["%GC_Loop"],
        "Entropia_A_tract": features["Entropia_A_tract"] / 2,
        "Entropia_U_tract": features["Entropia_U_tract"] / 2,
        "Entropia_HP_S_Loop": (features["Entropia_HP_S_Loop"] - 0.998000884)
        / (2.0 - 0.998000884),
        "A_Tract_state-change": features["A_Tract_state-change"] / 7,
        "U_Tract_state-change": features["U_Tract_state-change"] / 11,
        "HP_S_Loop_state_change": (features["HP_S_Loop_state_change"] - 2) / (34 - 2),
        "GC_Inicial_Hairpin": features["GC_Inicial_Hairpin"] / 12,
    }

    # Official clipping
    for key, value in normalized.items():

        normalized[key] = min(max(value, 0.0), 1.0)

    return normalized


prediction_rows = []

for i, row in df_phase3_tersp.iterrows():

    if not row["tersp_eligible"]:

        prediction_rows.append(
            {
                "tersp_status": row["tersp_annotation_status"],
                "tersp_strength": np.nan,
                "tersp_te": np.nan,
                "tersp_model_y": np.nan,
            }
        )

        continue

    stem = row["tersp_first_half"] + row["tersp_second_half"]

    features = tersp_calculate_features(
        row["tersp_a_tract"],
        row["tersp_u_tract"],
        stem,
        row["tersp_loop"],
    )

    X = pd.DataFrame([features])

    if hasattr(TERSP_MODEL, "feature_names_in_"):

        expected = list(TERSP_MODEL.feature_names_in_)

        missing = [feature for feature in expected if feature not in X.columns]

        if missing:

            raise RuntimeError(f"Missing TerSP features: {missing}")

        X = X[expected]

    strength = float(TERSP_MODEL.predict(X)[0])

    if strength != 0:

        te = float(1.0 - 1.0 / strength)

    else:

        te = np.nan

    if strength > 0:

        tersp_model_y = float(np.log10(strength))

    else:

        tersp_model_y = np.nan

    prediction_rows.append(
        {
            "tersp_status": "OK",
            "tersp_strength": strength,
            "tersp_te": te,
            "tersp_model_y": tersp_model_y,
        }
    )

df_tersp_prediction = pd.DataFrame(prediction_rows)

df_validation_all = pd.concat(
    [
        df_phase3_tersp.reset_index(drop=True),
        df_tersp_prediction,
    ],
    axis=1,
)

df_validation_all["target_model_y"] = -np.log10(1.0 - df_validation_all["target_te"])

df_validation_all["tersp_abs_error_te"] = np.abs(
    df_validation_all["tersp_te"] - df_validation_all["target_te"]
)

df_validation_all["tersp_abs_error_model_y"] = np.abs(
    df_validation_all["tersp_model_y"] - df_validation_all["target_model_y"]
)

df_validation_all["tersp_te_in_0_1"] = (df_validation_all["tersp_te"] >= 0) & (
    df_validation_all["tersp_te"] <= 1
)


import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
)

df_valid_all = df_validation_all.loc[
    (df_validation_all["tersp_status"] == "OK")
    & np.isfinite(df_validation_all["tersp_te"])
].copy()

if len(df_valid_all) == 0:

    raise RuntimeError("No valid TerSP predictions.")

summary_rows = []

all_target_groups = df_validation_all.groupby(
    [
        "target_id",
        "target_te",
    ],
    sort=True,
)

for (target_id, target_te), group_all in all_target_groups:

    group_valid = group_all.loc[
        (group_all["tersp_status"] == "OK") & np.isfinite(group_all["tersp_te"])
    ].copy()

    n_total = len(group_all)

    n_valid = len(group_valid)

    if n_valid > 0:

        te_values = group_valid["tersp_te"].to_numpy(dtype=float)

        strength_values = group_valid["tersp_strength"].to_numpy(dtype=float)

        model_y_values = group_valid["tersp_model_y"].to_numpy(dtype=float)

        te_errors = np.abs(te_values - float(target_te))

        target_model_y = float(-np.log10(1.0 - float(target_te)))

        finite_model_y = model_y_values[np.isfinite(model_y_values)]

        summary_rows.append(
            {
                "target_id": target_id,
                "target_te": float(target_te),
                "target_model_y": target_model_y,
                "N_total": int(n_total),
                "N_valid": int(n_valid),
                "coverage": float(n_valid / n_total),
                "tersp_te_median": float(np.median(te_values)),
                "tersp_te_mean": float(np.mean(te_values)),
                "tersp_te_q25": float(np.quantile(te_values, 0.25)),
                "tersp_te_q75": float(np.quantile(te_values, 0.75)),
                "tersp_te_min": float(np.min(te_values)),
                "tersp_te_max": float(np.max(te_values)),
                "median_abs_error_te": float(np.median(te_errors)),
                "mean_abs_error_te": float(np.mean(te_errors)),
                "within_0p05": float(np.mean(te_errors <= 0.05)),
                "within_0p10": float(np.mean(te_errors <= 0.10)),
                "fraction_te_in_0_1": float(
                    np.mean((te_values >= 0) & (te_values <= 1))
                ),
                "tersp_strength_median": float(np.median(strength_values)),
                "tersp_model_y_median": (
                    float(np.median(finite_model_y))
                    if len(finite_model_y) > 0
                    else np.nan
                ),
            }
        )

    else:

        summary_rows.append(
            {
                "target_id": target_id,
                "target_te": float(target_te),
                "target_model_y": float(-np.log10(1.0 - float(target_te))),
                "N_total": int(n_total),
                "N_valid": 0,
                "coverage": 0.0,
                "tersp_te_median": np.nan,
                "tersp_te_mean": np.nan,
                "tersp_te_q25": np.nan,
                "tersp_te_q75": np.nan,
                "tersp_te_min": np.nan,
                "tersp_te_max": np.nan,
                "median_abs_error_te": np.nan,
                "mean_abs_error_te": np.nan,
                "within_0p05": np.nan,
                "within_0p10": np.nan,
                "fraction_te_in_0_1": np.nan,
                "tersp_strength_median": np.nan,
                "tersp_model_y_median": np.nan,
            }
        )

df_target_summary = (
    pd.DataFrame(summary_rows).sort_values("target_te").reset_index(drop=True)
)


display(df_target_summary)

summary_valid = df_target_summary.loc[
    np.isfinite(df_target_summary["tersp_te_median"])
].copy()

if len(summary_valid) >= 3:

    target_values = summary_valid["target_te"].to_numpy(dtype=float)

    median_external = summary_valid["tersp_te_median"].to_numpy(dtype=float)

    pearson_te = pearsonr(target_values, median_external)

    spearman_te = spearmanr(target_values, median_external)

    correct = 0

    total = 0

    for i in range(len(target_values)):

        for j in range(i + 1, len(target_values)):

            target_delta = target_values[i] - target_values[j]

            prediction_delta = median_external[i] - median_external[j]

            if target_delta == 0:

                continue

            total += 1

            if np.sign(target_delta) == np.sign(prediction_delta):

                correct += 1

    rank_accuracy = correct / total if total > 0 else np.nan

    model_y_mask = np.isfinite(summary_valid["tersp_model_y_median"])

    model_y_summary = summary_valid.loc[model_y_mask]

    if len(model_y_summary) >= 3:

        spearman_model_y = float(
            spearmanr(
                model_y_summary["target_model_y"],
                model_y_summary["tersp_model_y_median"],
            ).statistic
        )

        pearson_model_y = float(
            pearsonr(
                model_y_summary["target_model_y"],
                model_y_summary["tersp_model_y_median"],
            ).statistic
        )

    else:

        spearman_model_y = np.nan

        pearson_model_y = np.nan

    target_level_metrics = {
        "N_targets_evaluable": int(len(summary_valid)),
        "Target_median_TE_MAE": float(np.mean(np.abs(target_values - median_external))),
        "Target_median_TE_MedianAE": float(
            np.median(np.abs(target_values - median_external))
        ),
        "Target_median_TE_Pearson": float(pearson_te.statistic),
        "Target_median_TE_Pearson_p": float(pearson_te.pvalue),
        "Target_median_TE_Spearman": float(spearman_te.statistic),
        "Target_median_TE_Spearman_p": float(spearman_te.pvalue),
        "Target_median_pairwise_ranking_accuracy": float(rank_accuracy),
        "Target_median_model_y_Pearson": pearson_model_y,
        "Target_median_model_y_Spearman": spearman_model_y,
    }

    for key, value in target_level_metrics.items():

        if isinstance(value, float):

            pass

        else:

            pass


else:

    target_level_metrics = {}


import numpy as np
import pandas as pd

plot_summary = (
    df_target_summary.loc[np.isfinite(df_target_summary["tersp_te_median"])]
    .copy()
    .sort_values("target_te")
    .reset_index(drop=True)
)

target_order = plot_summary["target_te"].tolist()

target_labels = [f"{value:.2f}" for value in target_order]

boxplot_values = []

for target_te in target_order:

    values = df_valid_all.loc[
        df_valid_all["target_te"] == target_te, "tersp_te"
    ].to_numpy(dtype=float)

    boxplot_values.append(values)

positions = np.arange(len(target_order))


out_of_range_summary = (
    df_valid_all.assign(
        outside_TE_domain=lambda x: ((x["tersp_te"] < 0) | (x["tersp_te"] > 1))
    )
    .groupby(
        [
            "target_id",
            "target_te",
        ]
    )
    .agg(
        N_valid=("tersp_te", "size"),
        N_outside_0_1=("outside_TE_domain", "sum"),
    )
    .reset_index()
    .sort_values("target_te")
)


display(out_of_range_summary)

target_te = plot_summary["target_te"].to_numpy(dtype=float)

median_te = plot_summary["tersp_te_median"].to_numpy(dtype=float)

q25_te = plot_summary["tersp_te_q25"].to_numpy(dtype=float)

q75_te = plot_summary["tersp_te_q75"].to_numpy(dtype=float)

median_bias = median_te - target_te

q25_bias = q25_te - target_te

q75_bias = q75_te - target_te

bias_yerr = np.vstack(
    [
        median_bias - q25_bias,
        q75_bias - median_bias,
    ]
)


model_y_summary = plot_summary.loc[
    np.isfinite(plot_summary["tersp_model_y_median"])
].copy()

x_model_y = model_y_summary["target_model_y"].to_numpy(dtype=float)

y_model_y = model_y_summary["tersp_model_y_median"].to_numpy(dtype=float)

fit_slope, fit_intercept = np.polyfit(x_model_y, y_model_y, 1)

fit_x = np.linspace(x_model_y.min(), x_model_y.max(), 200)

fit_y = fit_slope * fit_x + fit_intercept


pearson_value = target_level_metrics.get("Target_median_model_y_Pearson", np.nan)

spearman_value = target_level_metrics.get("Target_median_model_y_Spearman", np.nan)

metric_text = f"Pearson r = {pearson_value:.3f}\n" f"Spearman ρ = {spearman_value:.3f}"


files.download(CANDIDATE_OUT)

files.download(TARGET_OUT)

files.download(METRICS_OUT)
