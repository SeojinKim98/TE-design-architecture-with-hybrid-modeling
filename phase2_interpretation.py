"""Phase 2 -- interpretation of the frozen predictor and design-rule extraction.

Reloads the 50 Phase-1 fold models, verifies they reproduce the stored artifacts,
then computes held-out permutation importance (per descriptor and per group),
Spearman associations with Benjamini-Hochberg correction, and exhaustive single
nucleotide in silico mutagenesis aligned to the detected hairpin.

Outputs: importance and association tables, hairpin-aligned mutagenesis summaries,
and phase2_design_knowledge.json -- the design rules consumed by Phase 3.
"""

import json
import joblib
import cloudpickle

import numpy as np
import pandas as pd

required_files = {
    "manifest": "phase1_manifest.json",
    "bio": "phase1_biophysical_features.csv",
    "nt": "phase1_nt_embeddings.npy",
    "dataset": "phase1_cleaned_dataset.csv",
    "oof": "phase1_repeated_oof_predictions.csv",
    "repeat": "phase1_repeat_metrics.csv",
    "fold": "phase1_fold_metrics.csv",
    "extractor": "phase1_feature_extractor.pkl",
}


resolved_paths = {}

for key, filename in required_files.items():

    resolved_paths[key] = path


missing = [path for path in resolved_paths.values() if not os.path.exists(path)]

if missing:

    raise FileNotFoundError("Missing Phase-1 assets:\n" + "\n".join(missing))


if os.path.exists(cv_summary_path):

    df_cv_summary = pd.read_csv(cv_summary_path)


else:

    df_cv_summary = None


with open(resolved_paths["manifest"], "r") as f:

    phase1_manifest = json.load(f)

NT_MODEL_NAME = phase1_manifest["nt_model_name"]

MID_LAYER_IDS = tuple(phase1_manifest["mid_layer_ids"])

LLM_DIM = int(phase1_manifest["nt_embedding_dim"])

BIO_DIM = int(phase1_manifest["biophysical_dim"])

bio_feature_names = list(phase1_manifest["biophysical_feature_names"])

df_bio = pd.read_csv(resolved_paths["bio"])

X_nt = np.load(resolved_paths["nt"])

df_data = pd.read_csv(resolved_paths["dataset"])

df_oof = pd.read_csv(resolved_paths["oof"])

df_repeat = pd.read_csv(resolved_paths["repeat"])

df_fold = pd.read_csv(resolved_paths["fold"])

with open(resolved_paths["extractor"], "rb") as f:

    extract_rich_ecoli_features = cloudpickle.load(f)

required_oof_columns = [
    "actual",
    "mean_oof_prediction",
    "oof_prediction_sd",
    "sequence",
]

missing_oof_columns = [col for col in required_oof_columns if col not in df_oof.columns]

if missing_oof_columns:

    raise ValueError("Missing columns from OOF file: " f"{missing_oof_columns}")

y = df_oof["actual"].to_numpy(dtype=np.float64)

oof_pred = df_oof["mean_oof_prediction"].to_numpy(dtype=np.float64)

oof_sd = df_oof["oof_prediction_sd"].to_numpy(dtype=np.float64)

sequences = df_oof["sequence"].astype(str).str.strip().str.upper().tolist()

missing_bio_features = [
    feature for feature in bio_feature_names if feature not in df_bio.columns
]

if missing_bio_features:

    raise ValueError("Biophysical feature mismatch:\n" f"{missing_bio_features}")

X_bio_raw = df_bio[bio_feature_names].to_numpy(dtype=np.float64)

X_raw = np.hstack([X_nt, X_bio_raw])

model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "model_repeat_*_fold_*.joblib")))

if len(model_paths) == 0:

    raise FileNotFoundError(f"No saved models found in {MODEL_DIR}")

assert X_nt.shape[1] == LLM_DIM, f"Expected NT dim {LLM_DIM}, " f"found {X_nt.shape[1]}"

assert X_bio_raw.shape[1] == BIO_DIM, (
    f"Expected BIO dim {BIO_DIM}, " f"found {X_bio_raw.shape[1]}"
)

assert X_raw.shape[1] == (LLM_DIM + BIO_DIM)

assert len(y) == len(X_raw)
assert len(y) == len(sequences)

assert len(bio_feature_names) == 29

test_features = extract_rich_ecoli_features(sequences[0])

if list(test_features.keys()) != bio_feature_names:

    raise ValueError(
        "Feature extractor output order " "does not match Phase-1 saved feature order."
    )


if df_cv_summary is not None:

    pass

if len(model_paths) != 50:

    pass


for i, feature in enumerate(bio_feature_names, start=1):

    pass


class LeakageSafeStackingRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        preprocessor=None,
        base_estimators=None,
        final_estimator=None,
        inner_splits=5,
        random_state=42,
    ):

        self.preprocessor = preprocessor
        self.base_estimators = base_estimators
        self.final_estimator = final_estimator
        self.inner_splits = inner_splits
        self.random_state = random_state

    def predict(self, X):

        X = np.asarray(X)

        X_transformed = self.preprocessor_.transform(X)

        base_predictions = np.column_stack(
            [estimator.predict(X_transformed) for _, estimator in self.base_estimators_]
        )

        return self.final_estimator_.predict(base_predictions)


def make_regression_bins(y, q=5, n_splits=5):

    bins = pd.qcut(pd.Series(y), q=q, labels=False, duplicates="drop")

    if bins.nunique() < 2:
        return None

    if bins.value_counts().min() < n_splits:
        return None

    return bins.to_numpy()


SEEDS = df_repeat.sort_values("Repeat")["Seed"].astype(int).tolist()

N_SPLITS = 5
N_BINS = 5

target_bins = make_regression_bins(y, q=N_BINS, n_splits=N_SPLITS)

fold_records = []

oof_models_by_sample = [[] for _ in range(len(y))]

for repeat_idx, seed in enumerate(SEEDS, start=1):

    if target_bins is not None:

        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        split_iterator = splitter.split(X_raw, target_bins)

    else:

        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        split_iterator = splitter.split(X_raw)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iterator, start=1):

        model_path = os.path.join(
            MODEL_DIR,
            (f"model_repeat_" f"{repeat_idx:02d}_" f"fold_{fold_idx:02d}.joblib"),
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        saved = joblib.load(model_path)

        model = saved["model"]

        record = {
            "repeat": repeat_idx,
            "seed": seed,
            "fold": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "model": model,
            "path": model_path,
        }

        fold_records.append(record)

        for sample_idx in val_idx:

            oof_models_by_sample[sample_idx].append(model)


model_counts = np.array([len(x) for x in oof_models_by_sample])


if not np.all(model_counts == len(SEEDS)):
    raise RuntimeError("Each sample should have exactly " "10 OOF models.")


reconstructed_sum = np.zeros(len(y), dtype=float)

reconstructed_count = np.zeros(len(y), dtype=int)

for record in fold_records:

    val_idx = record["val_idx"]

    model = record["model"]

    pred = model.predict(X_raw[val_idx])

    reconstructed_sum[val_idx] += pred

    reconstructed_count[val_idx] += 1

reconstructed_oof = reconstructed_sum / reconstructed_count

max_diff = np.max(np.abs(reconstructed_oof - oof_pred))


if max_diff > 1e-6:

    pass

else:

    pass


import time
import numpy as np

N_CHECK_MODELS = 5

check_records = fold_records[:N_CHECK_MODELS]

all_differences = []

for i, record in enumerate(check_records, start=1):

    start_time = time.time()

    model = record["model"]
    val_idx = record["val_idx"]

    pred = model.predict(X_raw[val_idx])

    repeat_idx = record["repeat"]
    seed = record["seed"]

    possible_columns = [
        f"oof_seed_{seed}",
        f"oof_repeat_{repeat_idx:02d}",
        f"oof_repeat_{repeat_idx}",
    ]

    saved_column = None

    for col in possible_columns:

        if col in df_oof.columns:
            saved_column = col
            break

    if saved_column is not None:

        saved_pred = df_oof.loc[val_idx, saved_column].to_numpy()

        max_diff = np.max(np.abs(pred - saved_pred))

        all_differences.append(max_diff)

        status = "✅" if max_diff < 1e-6 else "⚠️"

    else:

        pass


if len(all_differences) > 0:

    if max(all_differences) < 1e-6:

        pass

else:

    pass


bio_corr = df_bio[bio_feature_names].corr(method="spearman")


corr_pairs = []

for i in range(len(bio_feature_names)):

    for j in range(i + 1, len(bio_feature_names)):

        rho = bio_corr.iloc[i, j]

        if abs(rho) >= 0.80:

            corr_pairs.append(
                {
                    "feature_1": bio_feature_names[i],
                    "feature_2": bio_feature_names[j],
                    "spearman_rho": rho,
                }
            )

df_corr_pairs = pd.DataFrame(corr_pairs)


display(df_corr_pairs)


import time
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

PERM_CACHE = "phase2_biophysical_permutation_importance.csv"

PARTIAL_CACHE = PERM_CACHE + ".partial"

N_PERM = 10
RANDOM_STATE = 42

CHUNK_CONDITIONS = 50

FORCE_RECOMPUTE = False


def predict_from_transformed(model, X_transformed):

    base_predictions = np.column_stack(
        [estimator.predict(X_transformed) for _, estimator in model.base_estimators_]
    )

    final_prediction = model.final_estimator_.predict(base_predictions)

    return final_prediction


def format_time(seconds):

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f} min"

    hours = minutes / 60

    return f"{hours:.2f} h"


def deterministic_permutation(
    n_samples, random_state, repeat, fold, feature_idx, perm_idx
):

    seed_sequence = np.random.SeedSequence(
        [int(random_state), int(repeat), int(fold), int(feature_idx), int(perm_idx)]
    )

    rng = np.random.default_rng(seed_sequence)

    return rng.permutation(n_samples)


if FORCE_RECOMPUTE:

    for path in [PERM_CACHE, PARTIAL_CACHE]:

        if os.path.exists(path):

            os.remove(path)


if os.path.exists(PERM_CACHE) and not FORCE_RECOMPUTE:

    df_perm = pd.read_csv(PERM_CACHE)

else:

    total_start = time.perf_counter()

    if os.path.exists(PARTIAL_CACHE):

        df_partial = pd.read_csv(PARTIAL_CACHE)

        perm_records = df_partial.to_dict("records")

    else:

        df_partial = pd.DataFrame()

        perm_records = []

    completed_fold_keys = set()

    if len(df_partial) > 0:

        fold_counts = df_partial.groupby(["Repeat", "Fold"]).size()

        for key, count in fold_counts.items():

            if count >= BIO_DIM:

                completed_fold_keys.add((int(key[0]), int(key[1])))

    total_folds = len(fold_records)

    conditions_per_fold = BIO_DIM * N_PERM

    for rec_idx, record in enumerate(fold_records, start=1):

        repeat = int(record["repeat"])

        fold = int(record["fold"])

        fold_key = (repeat, fold)

        if fold_key in completed_fold_keys:

            continue

        fold_start = time.perf_counter()

        model = record["model"]

        val_idx = record["val_idx"]

        X_val_raw = X_raw[val_idx]

        y_val = y[val_idx]

        n_val = len(y_val)

        step_start = time.perf_counter()

        X_t = model.preprocessor_.transform(X_val_raw)

        X_t = np.asarray(X_t)

        selector = model.preprocessor_.named_transformers_["llm"].named_steps["select"]

        n_llm_selected = int(selector.get_support().sum())

        expected_dim = n_llm_selected + BIO_DIM

        if X_t.shape[1] != expected_dim:

            raise RuntimeError(
                "Transformed dimension mismatch: "
                f"{X_t.shape[1]} "
                f"!= {expected_dim}"
            )

        baseline_start = time.perf_counter()

        baseline_pred = predict_from_transformed(model, X_t)

        baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))

        conditions = []

        for bio_idx, feature_name in enumerate(bio_feature_names):

            for perm_idx in range(N_PERM):

                conditions.append((bio_idx, feature_name, perm_idx))

        n_conditions = len(conditions)

        n_chunks = int(np.ceil(n_conditions / CHUNK_CONDITIONS))

        per_feature_deltas = {feature: [] for feature in bio_feature_names}

        for chunk_idx, chunk_start in enumerate(
            range(0, n_conditions, CHUNK_CONDITIONS), start=1
        ):

            chunk_t0 = time.perf_counter()

            chunk_end = min(chunk_start + CHUNK_CONDITIONS, n_conditions)

            chunk_conditions = conditions[chunk_start:chunk_end]

            n_chunk_conditions = len(chunk_conditions)

            X_chunk = np.tile(X_t, (n_chunk_conditions, 1))

            for local_idx, (bio_idx, feature_name, perm_idx) in enumerate(
                chunk_conditions
            ):

                transformed_col_idx = n_llm_selected + bio_idx

                row_start = local_idx * n_val

                row_end = row_start + n_val

                order = deterministic_permutation(
                    n_samples=n_val,
                    random_state=RANDOM_STATE,
                    repeat=repeat,
                    fold=fold,
                    feature_idx=bio_idx,
                    perm_idx=perm_idx,
                )

                X_chunk[row_start:row_end, transformed_col_idx] = X_t[
                    order, transformed_col_idx
                ]

            chunk_pred = predict_from_transformed(model, X_chunk)

            chunk_pred = chunk_pred.reshape(n_chunk_conditions, n_val)

            for local_idx, (bio_idx, feature_name, perm_idx) in enumerate(
                chunk_conditions
            ):

                perm_rmse = np.sqrt(mean_squared_error(y_val, chunk_pred[local_idx]))

                per_feature_deltas[feature_name].append(perm_rmse - baseline_rmse)

            chunk_elapsed = time.perf_counter() - chunk_t0

            fold_fraction = chunk_end / n_conditions

            global_fraction = ((rec_idx - 1) + fold_fraction) / total_folds

        for feature_name in bio_feature_names:

            deltas = np.asarray(per_feature_deltas[feature_name], dtype=float)

            if len(deltas) != N_PERM:

                raise RuntimeError(
                    f"{feature_name}: "
                    f"expected {N_PERM} permutations, "
                    f"found {len(deltas)}"
                )

            perm_records.append(
                {
                    "Repeat": repeat,
                    "Seed": int(record["seed"]),
                    "Fold": fold,
                    "Feature": feature_name,
                    "Baseline_RMSE": float(baseline_rmse),
                    "Delta_RMSE": float(deltas.mean()),
                    "Permutation_SD": float(deltas.std(ddof=1)),
                    "N_LLM_Selected": n_llm_selected,
                }
            )

        df_partial = pd.DataFrame(perm_records)

        df_partial = df_partial.drop_duplicates(
            subset=["Repeat", "Fold", "Feature"], keep="last"
        )

        fold_elapsed = time.perf_counter() - fold_start

        elapsed_total = time.perf_counter() - total_start

        completed_now = rec_idx

        avg_fold_time = elapsed_total / max(completed_now, 1)

        remaining = total_folds - rec_idx

        estimated_remaining = avg_fold_time * remaining

    df_perm = pd.DataFrame(perm_records)

    df_perm = (
        df_perm.drop_duplicates(subset=["Repeat", "Fold", "Feature"], keep="last")
        .sort_values(["Repeat", "Fold", "Feature"])
        .reset_index(drop=True)
    )

    expected_rows = len(fold_records) * BIO_DIM

    if len(df_perm) != expected_rows:

        pass

    else:

        if os.path.exists(PARTIAL_CACHE):

            os.remove(PARTIAL_CACHE)


df_perm_summary = (
    df_perm.groupby("Feature")["Delta_RMSE"]
    .agg(Mean="mean", SD="std", Median="median")
    .reset_index()
    .sort_values("Mean", ascending=False)
)


TOP_N = 20

plot_df = df_perm_summary.head(TOP_N).sort_values("Mean", ascending=True)


display(df_perm_summary.head(15))


import time
import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "Sequence composition": [
        "seq_len",
        "gc_content",
        "freq_A",
        "freq_C",
        "freq_G",
        "freq_T",
        "count_poly_t_4",
        "max_poly_t_run",
    ],
    "3-mer motifs": [
        "kmer_TTT",
        "kmer_AAA",
        "kmer_GCG",
        "kmer_CGC",
        "kmer_TTG",
        "kmer_TTA",
        "kmer_GCT",
        "kmer_AGC",
    ],
    "Stem-loop descriptors": [
        "stem_length",
        "loop_length",
        "stem_pairing_score",
        "gu_wobble_count",
        "loop_proximal_gc_pair_fraction",
    ],
    "Context / Poly-T position": [
        "upstream_a_richness",
        "positional_poly_t_score",
        "spacer_length",
        "stem_polyT_coupling",
    ],
    "Interaction descriptors": [
        "stem_polyT_interaction",
        "compact_gc_hairpin_score",
        "polyT_spacer_proximity",
        "polyT_position_interaction",
    ],
}

for group_name, features in FEATURE_GROUPS.items():

    missing = [feature for feature in features if feature not in bio_feature_names]

    if missing:

        raise ValueError(f"{group_name} contains " f"missing features: {missing}")

GROUP_CACHE = "phase2_group_permutation_importance.csv"

GROUP_PARTIAL_CACHE = GROUP_CACHE + ".partial"

N_GROUP_PERM = 10

GROUP_RANDOM_STATE = 2026

GROUP_CHUNK_CONDITIONS = 10

FORCE_RECOMPUTE_GROUP = False

if FORCE_RECOMPUTE_GROUP:

    for path in [GROUP_CACHE, GROUP_PARTIAL_CACHE]:

        if os.path.exists(path):

            os.remove(path)


if os.path.exists(GROUP_CACHE) and not FORCE_RECOMPUTE_GROUP:

    df_group_perm = pd.read_csv(GROUP_CACHE)

else:

    total_start = time.perf_counter()

    if os.path.exists(GROUP_PARTIAL_CACHE):

        df_group_partial = pd.read_csv(GROUP_PARTIAL_CACHE)

        group_records = df_group_partial.to_dict("records")

    else:

        df_group_partial = pd.DataFrame()

        group_records = []

    completed_group_folds = set()

    if len(df_group_partial) > 0:

        counts = df_group_partial.groupby(["Repeat", "Fold"]).size()

        for key, count in counts.items():

            if count >= len(FEATURE_GROUPS):

                completed_group_folds.add((int(key[0]), int(key[1])))

    total_folds = len(fold_records)

    for rec_idx, record in enumerate(fold_records, start=1):

        repeat = int(record["repeat"])

        fold = int(record["fold"])

        fold_key = (repeat, fold)

        if fold_key in completed_group_folds:

            continue

        fold_start = time.perf_counter()

        model = record["model"]

        val_idx = record["val_idx"]

        X_val_raw = X_raw[val_idx]

        y_val = y[val_idx]

        n_val = len(y_val)

        X_t = model.preprocessor_.transform(X_val_raw)

        X_t = np.asarray(X_t)

        selector = model.preprocessor_.named_transformers_["llm"].named_steps["select"]

        n_llm_selected = int(selector.get_support().sum())

        baseline_pred = predict_from_transformed(model, X_t)

        baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))

        group_column_indices = {}

        for group_name, features in FEATURE_GROUPS.items():

            bio_indices = [bio_feature_names.index(feature) for feature in features]

            transformed_indices = [n_llm_selected + bio_idx for bio_idx in bio_indices]

            group_column_indices[group_name] = transformed_indices

        conditions = []

        group_names = list(FEATURE_GROUPS.keys())

        for group_idx, group_name in enumerate(group_names):

            for perm_idx in range(N_GROUP_PERM):

                conditions.append((group_idx, group_name, perm_idx))

        n_conditions = len(conditions)

        n_chunks = int(np.ceil(n_conditions / GROUP_CHUNK_CONDITIONS))

        per_group_deltas = {group_name: [] for group_name in group_names}

        for chunk_idx, chunk_start in enumerate(
            range(0, n_conditions, GROUP_CHUNK_CONDITIONS), start=1
        ):

            chunk_t0 = time.perf_counter()

            chunk_end = min(chunk_start + GROUP_CHUNK_CONDITIONS, n_conditions)

            chunk_conditions = conditions[chunk_start:chunk_end]

            n_chunk = len(chunk_conditions)

            X_chunk = np.tile(X_t, (n_chunk, 1))

            for local_idx, (group_idx, group_name, perm_idx) in enumerate(
                chunk_conditions
            ):

                row_start = local_idx * n_val

                row_end = row_start + n_val

                cols = group_column_indices[group_name]

                # Deterministic JOINT permutation
                order = deterministic_permutation(
                    n_samples=n_val,
                    random_state=GROUP_RANDOM_STATE,
                    repeat=repeat,
                    fold=fold,
                    feature_idx=group_idx,
                    perm_idx=perm_idx,
                )

                target_rows = np.arange(row_start, row_end)

                X_chunk[np.ix_(target_rows, cols)] = X_t[np.ix_(order, cols)]

            chunk_pred = predict_from_transformed(model, X_chunk)

            chunk_pred = chunk_pred.reshape(n_chunk, n_val)

            for local_idx, (group_idx, group_name, perm_idx) in enumerate(
                chunk_conditions
            ):

                perm_rmse = np.sqrt(mean_squared_error(y_val, chunk_pred[local_idx]))

                per_group_deltas[group_name].append(perm_rmse - baseline_rmse)

            chunk_elapsed = time.perf_counter() - chunk_t0

            fold_progress = chunk_end / n_conditions

            overall_progress = (rec_idx - 1 + fold_progress) / total_folds

        for group_name in group_names:

            deltas = np.asarray(per_group_deltas[group_name], dtype=float)

            group_records.append(
                {
                    "Repeat": repeat,
                    "Seed": int(record["seed"]),
                    "Fold": fold,
                    "Group": group_name,
                    "Baseline_RMSE": float(baseline_rmse),
                    "Delta_RMSE": float(deltas.mean()),
                    "Permutation_SD": float(deltas.std(ddof=1)),
                    "N_LLM_Selected": n_llm_selected,
                }
            )

        df_group_partial = pd.DataFrame(group_records)

        df_group_partial = df_group_partial.drop_duplicates(
            subset=["Repeat", "Fold", "Group"], keep="last"
        )

        fold_elapsed = time.perf_counter() - fold_start

    df_group_perm = pd.DataFrame(group_records)

    df_group_perm = (
        df_group_perm.drop_duplicates(subset=["Repeat", "Fold", "Group"], keep="last")
        .sort_values(["Repeat", "Fold", "Group"])
        .reset_index(drop=True)
    )

    expected_rows = len(fold_records) * len(FEATURE_GROUPS)

    if len(df_group_perm) != expected_rows:

        pass

    else:

        if os.path.exists(GROUP_PARTIAL_CACHE):

            os.remove(GROUP_PARTIAL_CACHE)


df_group_summary = (
    df_group_perm.groupby("Group")["Delta_RMSE"]
    .agg(Mean="mean", SD="std", Median="median")
    .reset_index()
    .sort_values("Mean", ascending=False)
)


display(df_group_summary)

plot_group = df_group_summary.sort_values("Mean", ascending=True)


def benjamini_hochberg(pvalues):

    pvalues = np.asarray(pvalues, dtype=float)

    n = len(pvalues)

    order = np.argsort(pvalues)

    ranked = pvalues[order]

    q = ranked * n / np.arange(1, n + 1)

    q = np.minimum.accumulate(q[::-1])[::-1]

    q = np.clip(q, 0, 1)

    output = np.empty_like(q)

    output[order] = q

    return output


assoc_records = []

for feature in bio_feature_names:

    x = df_bio[feature].to_numpy()

    rho_actual, p_actual = spearmanr(x, y)

    rho_oof, p_oof = spearmanr(x, oof_pred)

    assoc_records.append(
        {
            "Feature": feature,
            "rho_actual": rho_actual,
            "p_actual": p_actual,
            "rho_oof": rho_oof,
            "p_oof": p_oof,
        }
    )

df_assoc = pd.DataFrame(assoc_records)

df_assoc["q_actual"] = benjamini_hochberg(df_assoc["p_actual"])

df_assoc["q_oof"] = benjamini_hochberg(df_assoc["p_oof"])

df_assoc = df_assoc.merge(
    df_perm_summary[["Feature", "Mean", "SD"]], on="Feature", how="left"
)

df_assoc = df_assoc.rename(
    columns={"Mean": "Permutation_Delta_RMSE", "SD": "Permutation_SD"}
)


display(df_assoc.sort_values("Permutation_Delta_RMSE", ascending=False).head(15))


TOP_FEATURES = df_perm_summary.head(6)["Feature"].tolist()


for ax, feature in zip(axes, TOP_FEATURES):

    tmp = pd.DataFrame({"feature": df_bio[feature], "actual": y, "oof": oof_pred})

    try:

        tmp["bin"] = pd.qcut(tmp["feature"], q=6, duplicates="drop")

    except ValueError:

        continue

    grouped = (
        tmp.groupby("bin", observed=True)
        .agg(
            x=("feature", "median"),
            actual=("actual", "mean"),
            predicted=("oof", "mean"),
            n=("actual", "size"),
        )
        .reset_index()
    )


import numpy as np
import pandas as pd

N_QUANTILES = 5

SPACER_SENTINEL = 15

SPACER_ORDER = ["0 nt", "1 nt", ">=2 nt"]

if "PHASE2_TABLE_DIR" in globals() and PHASE2_TABLE_DIR is not None:
    pass
else:
    pass


interaction_df = df_bio[bio_feature_names].copy()

interaction_df["actual_strength"] = np.asarray(y, dtype=float)

interaction_df["oof_prediction"] = np.asarray(oof_pred, dtype=float)


display(
    interaction_df["spacer_length"].value_counts().sort_index().rename("N").to_frame()
)

n_fallback = int((interaction_df["spacer_length"] == SPACER_SENTINEL).sum())


def make_quantile_group(series, q=5):
    """
    Quantile grouping without artificially forcing duplicated boundaries.

    Returns integer groups:
        0 = lowest
        ...
        q-1 = highest

    If the variable cannot support q distinct bins,
    pd.qcut(..., duplicates='drop') automatically reduces the number.
    """

    valid = series.dropna()

    if valid.nunique() < 2:
        raise ValueError(f"{series.name} has fewer than " "2 unique values.")

    groups = pd.qcut(series, q=q, labels=False, duplicates="drop")

    return groups


def make_spacer_group(series):
    """
    Biologically interpretable spacer grouping.

    spacer_length = 15 is NOT treated as a true 15-nt spacer.
    It is the Phase-1 no-hairpin/fallback sentinel and is excluded.

    Valid groups:
        0 nt
        1 nt
        >=2 nt
    """

    result = pd.Series(np.nan, index=series.index, dtype="object")

    result.loc[series == 0] = "0 nt"

    result.loc[series == 1] = "1 nt"

    result.loc[(series >= 2) & (series < SPACER_SENTINEL)] = ">=2 nt"

    return pd.Categorical(result, categories=SPACER_ORDER, ordered=True)


def prepare_interaction_table(
    df, x_feature, y_feature, target_column, x_mode="quantile", y_mode="quantile", q=5
):

    temp = df[[x_feature, y_feature, target_column]].copy()

    if x_mode == "spacer":

        temp["x_group"] = make_spacer_group(temp[x_feature])

        x_label = f"{x_feature} group"

    else:

        temp["x_group"] = make_quantile_group(temp[x_feature], q=q)

        x_label = f"{x_feature} quantile"

    if y_mode == "spacer":

        temp["y_group"] = make_spacer_group(temp[y_feature])

        y_label = f"{y_feature} group"

    else:

        temp["y_group"] = make_quantile_group(temp[y_feature], q=q)

        y_label = f"{y_feature} quantile"

    temp = temp.dropna(subset=["x_group", "y_group", target_column])

    pivot = temp.pivot_table(
        index="y_group",
        columns="x_group",
        values=target_column,
        aggfunc="mean",
        observed=False,
    )

    counts = temp.pivot_table(
        index="y_group",
        columns="x_group",
        values=target_column,
        aggfunc="count",
        observed=False,
    )

    return {
        "pivot": pivot,
        "counts": counts,
        "x_label": x_label,
        "y_label": y_label,
        "n_samples": len(temp),
    }


INTERACTIONS = [
    {
        "name": "Stem pairing × Poly-T run",
        "x_feature": "stem_pairing_score",
        "y_feature": "max_poly_t_run",
        "x_mode": "quantile",
        "y_mode": "quantile",
    },
    {
        "name": "Stem pairing × Spacer",
        "x_feature": "stem_pairing_score",
        "y_feature": "spacer_length",
        "x_mode": "quantile",
        "y_mode": "spacer",
    },
    {
        "name": "Positional Poly-T × Poly-T run",
        "x_feature": "positional_poly_t_score",
        "y_feature": "max_poly_t_run",
        "x_mode": "quantile",
        "y_mode": "quantile",
    },
]

interaction_results = {}

for interaction in INTERACTIONS:

    name = interaction["name"]

    actual_result = prepare_interaction_table(
        interaction_df,
        x_feature=interaction["x_feature"],
        y_feature=interaction["y_feature"],
        target_column="actual_strength",
        x_mode=interaction["x_mode"],
        y_mode=interaction["y_mode"],
        q=N_QUANTILES,
    )

    oof_result = prepare_interaction_table(
        interaction_df,
        x_feature=interaction["x_feature"],
        y_feature=interaction["y_feature"],
        target_column="oof_prediction",
        x_mode=interaction["x_mode"],
        y_mode=interaction["y_mode"],
        q=N_QUANTILES,
    )

    interaction_results[name] = {"actual": actual_result, "oof": oof_result}


for interaction in INTERACTIONS:

    name = interaction["name"]

    result = interaction_results[name]["oof"]


spacer_result = interaction_results["Stem pairing × Spacer"]["oof"]


display(spacer_result["counts"])


def draw_interaction_heatmap(ax, result, title, vmin, vmax, show_colorbar=True):

    pivot = result["pivot"]

    counts = result["counts"]

    values = pivot.to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    if len(finite_values) > 0:

        midpoint = (vmin + vmax) / 2

    else:

        midpoint = 0

    for i in range(len(pivot.index)):

        for j in range(len(pivot.columns)):

            value = pivot.iloc[i, j]

            if pd.isna(value):
                continue

            n = counts.iloc[i, j]

            text_color = "white" if value < midpoint else "black"

    if show_colorbar:

        pass

    return im


for col_idx, interaction in enumerate(INTERACTIONS):

    name = interaction["name"]

    actual_result = interaction_results[name]["actual"]

    oof_result = interaction_results[name]["oof"]

    actual_values = actual_result["pivot"].to_numpy(dtype=float)

    oof_values = oof_result["pivot"].to_numpy(dtype=float)

    combined_values = np.concatenate(
        [actual_values[np.isfinite(actual_values)], oof_values[np.isfinite(oof_values)]]
    )

    if len(combined_values) == 0:

        vmin = 0
        vmax = 1

    else:

        vmin = float(np.min(combined_values))

        vmax = float(np.max(combined_values))

    # Avoid invalid color range
    if np.isclose(vmin, vmax):

        vmax = vmin + 1e-6


for interaction in INTERACTIONS:

    name = interaction["name"]

    safe_name = name.lower().replace(" ", "_").replace("×", "x").replace("/", "_")

    for target_name in ["actual", "oof"]:

        result = interaction_results[name][target_name]


display(p2)

display(n2)


import numpy as np

CHECK_INDICES = np.unique(
    np.linspace(0, len(sequences) - 1, min(5, len(sequences)), dtype=int)
)

max_abs_diff = 0.0

for sample_idx in CHECK_INDICES:

    seq = sequences[sample_idx]

    recomputed = extract_rich_ecoli_features(seq)

    if list(recomputed.keys()) != bio_feature_names:

        raise RuntimeError("❌ Feature-name/order mismatch " f"at sample {sample_idx}")

    recomputed_array = np.array(
        [recomputed[name] for name in bio_feature_names], dtype=float
    )

    saved_array = X_bio_raw[sample_idx]

    diff = np.max(np.abs(recomputed_array - saved_array))

    max_abs_diff = max(max_abs_diff, diff)


if max_abs_diff < 1e-10:

    pass

else:

    raise RuntimeError("❌ Saved 29D features and " "feature extractor do not match.")


import time
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_start = time.perf_counter()

if "tokenizer" not in globals() or tokenizer is None:

    tokenizer = AutoTokenizer.from_pretrained(NT_MODEL_NAME, trust_remote_code=True)

if "nt_model" not in globals() or nt_model is None:

    nt_model = AutoModelForMaskedLM.from_pretrained(
        NT_MODEL_NAME, trust_remote_code=True
    ).to(device)

nt_model.eval()


if hasattr(nt_model, "base_model"):

    nt_backbone = nt_model.base_model

    nt_backbone.eval()


else:

    nt_backbone = None


USE_AMP = False

# Initially try the backbone
USE_BACKBONE_ONLY = nt_backbone is not None

ISM_NT_BATCH_SIZE = 32


def masked_mean_pool(hidden_state, valid_token_mask):

    mask = valid_token_mask.unsqueeze(-1).to(hidden_state.dtype)

    denominator = mask.sum(dim=1).clamp(min=1.0)

    return (hidden_state * mask).sum(dim=1) / denominator


def extract_nt_embeddings_ism(
    sequence_list, batch_size=32, show_progress=True, progress_prefix="NT"
):

    sequence_list = list(sequence_list)

    n_seq = len(sequence_list)

    if n_seq == 0:

        return np.empty((0, LLM_DIM), dtype=np.float32)

    n_batches = int(np.ceil(n_seq / batch_size))

    embeddings = []

    special_ids = list(tokenizer.all_special_ids)

    start_total = time.perf_counter()

    if show_progress:

        pass

    with torch.inference_mode():

        for batch_idx, start in enumerate(range(0, n_seq, batch_size), start=1):

            batch_start = time.perf_counter()

            batch = sequence_list[start : start + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            inference_model = nt_backbone if USE_BACKBONE_ONLY else nt_model

            outputs = inference_model(
                **inputs, output_hidden_states=True, return_dict=True
            )

            hidden_states = outputs.hidden_states

            input_ids = inputs["input_ids"]

            valid_mask = inputs["attention_mask"].bool()

            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for token_id in special_ids:

                special_mask |= input_ids == token_id

            valid_mask &= ~special_mask

            intermediate = torch.stack(
                [hidden_states[layer_id] for layer_id in MID_LAYER_IDS], dim=0
            ).mean(dim=0)

            mid_pool = masked_mean_pool(intermediate, valid_mask)

            final_pool = masked_mean_pool(hidden_states[-1], valid_mask)

            fused = torch.cat([mid_pool, final_pool], dim=-1)

            embeddings.append(fused.detach().cpu().float().numpy())

            if show_progress:

                batch_elapsed = time.perf_counter() - batch_start

                done = min(start + batch_size, n_seq)

    result = np.vstack(embeddings)

    if show_progress:

        pass

    return result


NT_CHECK_INDICES = np.array([0, len(sequences) // 2, len(sequences) - 1], dtype=int)

NT_CHECK_INDICES = np.unique(NT_CHECK_INDICES)

check_sequences = [sequences[i] for i in NT_CHECK_INDICES]

X_check = extract_nt_embeddings_ism(
    check_sequences,
    batch_size=len(check_sequences),
    show_progress=True,
    progress_prefix="NT sanity",
)

X_reference = X_nt[NT_CHECK_INDICES]

max_abs_diff = np.max(np.abs(X_check - X_reference))

mean_abs_diff = np.mean(np.abs(X_check - X_reference))


if max_abs_diff <= 1e-5:

    pass

else:

    USE_BACKBONE_ONLY = False

    X_check_full = extract_nt_embeddings_ism(
        check_sequences,
        batch_size=len(check_sequences),
        show_progress=True,
        progress_prefix="NT full-model sanity",
    )

    max_abs_diff_full = np.max(np.abs(X_check_full - X_reference))

    if max_abs_diff_full > 1e-5:

        raise RuntimeError(
            "❌ NT embeddings do not match " "saved Phase-1 representation."
        )


import numpy as np
import pandas as pd

ISM_SCOPE = "all"

ISM_N_SAMPLES = 30
ISM_N_BINS = 5
ISM_RANDOM_STATE = 42


def select_ism_samples(y, n_samples=30, n_bins=5, random_state=42):

    rng = np.random.default_rng(random_state)

    bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop")

    selected = []

    unique_bins = sorted(bins.dropna().unique())

    per_bin = max(1, int(np.ceil(n_samples / len(unique_bins))))

    for b in unique_bins:

        idx = np.where(bins.to_numpy() == b)[0]

        n_pick = min(per_bin, len(idx))

        picked = rng.choice(idx, size=n_pick, replace=False)

        selected.extend(picked.tolist())

    return np.array(selected[:n_samples], dtype=int)


if ISM_SCOPE == "all":

    ism_sample_idx = np.arange(len(sequences), dtype=int)


elif ISM_SCOPE == "subset":

    if os.path.exists(ISM_SAMPLE_FILE):

        df_ism_samples = pd.read_csv(ISM_SAMPLE_FILE)

        ism_sample_idx = df_ism_samples["sample_idx"].to_numpy(dtype=int)

    else:

        ism_sample_idx = select_ism_samples(
            y, n_samples=ISM_N_SAMPLES, n_bins=ISM_N_BINS, random_state=ISM_RANDOM_STATE
        )

        df_ism_samples = pd.DataFrame(
            {
                "sample_idx": ism_sample_idx,
                "actual_strength": y[ism_sample_idx],
                "sequence_length": [len(sequences[i]) for i in ism_sample_idx],
            }
        )


else:

    raise ValueError("ISM_SCOPE must be " "'all' or 'subset'.")

total_mutants = int(sum(3 * len(sequences[i]) for i in ism_sample_idx))


import time

import numpy as np
import pandas as pd

ISM_CACHE = "phase2_ism_oof_real_all.csv"

ISM_PARTIAL_CACHE = "phase2_ism_oof_real_all.partial.csv"


FORCE_RECOMPUTE_ISM = False

CACHE_MUTANT_NT = True


def format_duration(seconds):

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f}s"

    if seconds < 3600:
        return f"{seconds / 60:.1f} min"

    return f"{seconds / 3600:.2f} h"


if FORCE_RECOMPUTE_ISM:

    for path in [ISM_CACHE, ISM_PARTIAL_CACHE]:

        if os.path.exists(path):

            os.remove(path)

    if os.path.isdir(ISM_TEMP_DIR):

        shutil.rmtree(ISM_TEMP_DIR)


def ism_cache_is_complete(df_cache, sample_indices):

    expected_samples = set(int(i) for i in sample_indices)

    observed_samples = set(df_cache["sample_idx"].astype(int).unique())

    if expected_samples != observed_samples:

        return False

    for sample_idx in sample_indices:

        expected_rows = 3 * len(sequences[int(sample_idx)])

        observed_rows = len(df_cache[df_cache["sample_idx"] == int(sample_idx)])

        if observed_rows != expected_rows:

            return False

    return True


if os.path.exists(ISM_CACHE) and not FORCE_RECOMPUTE_ISM:

    df_ism_candidate = pd.read_csv(ISM_CACHE)

    if ism_cache_is_complete(df_ism_candidate, ism_sample_idx):

        df_ism = df_ism_candidate

        ISM_NEEDS_COMPUTATION = False

    else:

        ISM_NEEDS_COMPUTATION = True

else:

    ISM_NEEDS_COMPUTATION = True

if ISM_NEEDS_COMPUTATION:

    total_start = time.perf_counter()

    bases = ["A", "C", "G", "T"]

    if os.path.exists(ISM_PARTIAL_CACHE):

        df_partial = pd.read_csv(ISM_PARTIAL_CACHE)

        ism_records = df_partial.to_dict("records")

    else:

        df_partial = pd.DataFrame()

        ism_records = []

    completed_samples = set()

    if len(df_partial) > 0:

        for sample_idx in ism_sample_idx:

            sample_idx = int(sample_idx)

            expected_rows = 3 * len(sequences[sample_idx])

            observed_rows = len(df_partial[df_partial["sample_idx"] == sample_idx])

            if observed_rows == expected_rows:

                completed_samples.add(sample_idx)

    sample_times = []

    for sample_counter, sample_idx in enumerate(ism_sample_idx, start=1):

        sample_idx = int(sample_idx)

        if sample_idx in (completed_samples):

            continue

        sample_start = time.perf_counter()

        wt_seq = sequences[sample_idx]

        seq_len = len(wt_seq)

        sample_models = oof_models_by_sample[sample_idx]

        if len(sample_models) != len(SEEDS):

            raise RuntimeError(
                f"Sample {sample_idx}: "
                f"expected {len(SEEDS)} OOF models, "
                f"found {len(sample_models)}"
            )

        mutant_sequences = []
        mutant_meta = []

        for pos, ref_base in enumerate(wt_seq):

            for alt_base in bases:

                if alt_base == ref_base:
                    continue

                mutant_seq = wt_seq[:pos] + alt_base + wt_seq[pos + 1 :]

                mutant_sequences.append(mutant_seq)

                mutant_meta.append(
                    {
                        "position_0based": pos,
                        "position_1based": pos + 1,
                        "ref": ref_base,
                        "alt": alt_base,
                    }
                )

        n_mutants = len(mutant_sequences)

        if CACHE_MUTANT_NT and os.path.exists(temp_nt_path):

            X_mut_nt = np.load(temp_nt_path)

            if X_mut_nt.shape == (n_mutants, LLM_DIM):

                pass

            else:

                os.remove(temp_nt_path)

                X_mut_nt = extract_nt_embeddings_ism(
                    mutant_sequences,
                    batch_size=ISM_NT_BATCH_SIZE,
                    show_progress=True,
                    progress_prefix=(f"sample {sample_idx}"),
                )

        else:

            X_mut_nt = extract_nt_embeddings_ism(
                mutant_sequences,
                batch_size=ISM_NT_BATCH_SIZE,
                show_progress=True,
                progress_prefix=(f"sample {sample_idx}"),
            )

        if CACHE_MUTANT_NT and not os.path.exists(temp_nt_path):

            pass

        bio_start = time.perf_counter()

        mutant_bio_records = []

        for j, mutant_seq in enumerate(mutant_sequences, start=1):

            mutant_bio_records.append(extract_rich_ecoli_features(mutant_seq))

            if j % 100 == 0 or j == n_mutants:

                pass

        mutant_bio_df = pd.DataFrame(mutant_bio_records)

        X_mut_bio = mutant_bio_df[bio_feature_names].to_numpy(dtype=float)

        X_mut_raw = np.hstack([X_mut_nt, X_mut_bio])

        if X_mut_raw.shape[1] != LLM_DIM + BIO_DIM:

            raise RuntimeError("Mutant hybrid dimension mismatch.")

        X_eval_raw = np.vstack([X_raw[sample_idx : sample_idx + 1], X_mut_raw])

        model_predictions = []

        model_stage_start = time.perf_counter()

        for model_idx, model in enumerate(sample_models, start=1):

            one_model_start = time.perf_counter()

            pred = model.predict(X_eval_raw)

            model_predictions.append(pred)

        model_predictions = np.vstack(model_predictions)

        wt_predictions = model_predictions[:, 0]

        mutant_predictions = model_predictions[:, 1:]

        wt_pred_mean = wt_predictions.mean()

        wt_pred_sd = wt_predictions.std(ddof=1)

        wt_oof_difference = abs(wt_pred_mean - oof_pred[sample_idx])

        if wt_oof_difference > 1e-6:

            pass

        mutant_pred_mean = mutant_predictions.mean(axis=0)

        mutant_pred_sd = mutant_predictions.std(axis=0, ddof=1)

        delta_by_model = mutant_predictions - wt_predictions[:, None]

        delta_mean = delta_by_model.mean(axis=0)

        delta_sd = delta_by_model.std(axis=0, ddof=1)

        sample_records = []

        for mut_idx, meta in enumerate(mutant_meta):

            normalized_position = meta["position_0based"] / max(seq_len - 1, 1)

            sample_records.append(
                {
                    "sample_idx": sample_idx,
                    "sequence_length": seq_len,
                    "actual_strength": float(y[sample_idx]),
                    "wt_prediction": float(wt_pred_mean),
                    "wt_prediction_sd": float(wt_pred_sd),
                    "position_0based": int(meta["position_0based"]),
                    "position_1based": int(meta["position_1based"]),
                    "normalized_position": float(normalized_position),
                    "ref": meta["ref"],
                    "alt": meta["alt"],
                    "mutant_prediction": float(mutant_pred_mean[mut_idx]),
                    "mutant_prediction_sd": float(mutant_pred_sd[mut_idx]),
                    # Primary ISM effect
                    "delta_prediction": float(delta_mean[mut_idx]),
                    "delta_prediction_sd": float(delta_sd[mut_idx]),
                    "abs_delta": float(abs(delta_mean[mut_idx])),
                }
            )

        ism_records.extend(sample_records)

        df_current = pd.DataFrame(ism_records)

        df_current = df_current.drop_duplicates(
            subset=["sample_idx", "position_0based", "alt"], keep="last"
        )

        if CACHE_MUTANT_NT and os.path.exists(temp_nt_path):

            os.remove(temp_nt_path)

        sample_elapsed = time.perf_counter() - sample_start

        sample_times.append(sample_elapsed)

        mean_sample_time = np.mean(sample_times)

        remaining_samples = len(ism_sample_idx) - sample_counter

        rough_remaining = remaining_samples * mean_sample_time

    df_ism = pd.DataFrame(ism_records)

    df_ism = (
        df_ism.drop_duplicates(
            subset=["sample_idx", "position_0based", "alt"], keep="last"
        )
        .sort_values(["sample_idx", "position_0based", "alt"])
        .reset_index(drop=True)
    )

    if not ism_cache_is_complete(df_ism, ism_sample_idx):

        raise RuntimeError(
            "❌ ISM calculation ended but " "some mutation records are missing."
        )

    if os.path.exists(ISM_PARTIAL_CACHE):

        os.remove(ISM_PARTIAL_CACHE)


N_POSITION_BINS = 50

df_ism["position_bin"] = pd.cut(
    df_ism["normalized_position"],
    bins=np.linspace(0, 1, N_POSITION_BINS + 1),
    labels=False,
    include_lowest=True,
)

signed_heatmap = df_ism.pivot_table(
    index="alt", columns="position_bin", values="delta_prediction", aggfunc="mean"
).reindex(["A", "C", "G", "T"])


from scipy.stats import spearmanr

rho_ism_unc, p_ism_unc = spearmanr(df_ism["abs_delta"], df_ism["delta_prediction_sd"])


N_POSITION_BINS = 50

df_ism["position_bin"] = pd.cut(
    df_ism["normalized_position"],
    bins=np.linspace(0, 1, N_POSITION_BINS + 1),
    labels=False,
    include_lowest=True,
)

per_sequence_position = df_ism.groupby(
    ["sample_idx", "position_bin"], as_index=False
).agg(
    Mean_abs_delta=("abs_delta", "mean"),
    Mean_delta=("delta_prediction", "mean"),
    Mean_delta_uncertainty=("delta_prediction_sd", "mean"),
)

positional_ism = (
    per_sequence_position.groupby("position_bin")["Mean_abs_delta"]
    .agg(Mean="mean", Median="median", SD="std", N_sequences="count")
    .reset_index()
)

positional_ism["SEM"] = positional_ism["SD"] / np.sqrt(positional_ism["N_sequences"])

positional_ism["normalized_position"] = (
    positional_ism["position_bin"] + 0.5
) / N_POSITION_BINS


x = positional_ism["normalized_position"].to_numpy()

mean = positional_ism["Mean"].to_numpy()

sem = positional_ism["SEM"].fillna(0).to_numpy()


CANONICAL_GC = {("G", "C"), ("C", "G")}

CANONICAL_AT = {("A", "T"), ("T", "A")}

WOBBLE_GT = {("G", "T"), ("T", "G")}


def infer_hairpin_geometry(sequence):
    """
    Reproduce the heuristic hairpin search used in Phase 1.

    Returns
    -------
    dict with:
        hairpin_detected
        stem_start
        stem_length
        loop_length
        loop_start
        loop_end
        right_stem_start
        hairpin_end_exclusive
        hairpin_anchor_0based

    Alignment convention
    --------------------
    hairpin_anchor_0based =
        final nucleotide of the detected 3' stem arm

    Therefore:
        relative_position = mutation_position - hairpin_anchor

        0  = last nucleotide of hairpin
        <0 = within/upstream of hairpin
        >0 = downstream toward Poly-T
    """

    sequence = sequence.upper().strip()

    seq_len = len(sequence)

    best = None

    for stem_len in range(4, 14):

        for loop_len in range(3, 9):

            window_len = 2 * stem_len + loop_len

            if window_len > seq_len:
                continue

            for start in range(seq_len - window_len + 1):

                left_stem = sequence[start : start + stem_len]

                right_start = start + stem_len + loop_len

                right_stem = sequence[right_start : right_start + stem_len]

                right_rev = right_stem[::-1]

                pairing_score = 0.0
                n_pairs = 0

                for left_base, right_base in zip(left_stem, right_rev):

                    pair = (left_base, right_base)

                    if pair in CANONICAL_GC:

                        pairing_score += 3.0
                        n_pairs += 1

                    elif pair in CANONICAL_AT:

                        pairing_score += 2.0
                        n_pairs += 1

                    elif pair in WOBBLE_GT:

                        pairing_score += 1.0
                        n_pairs += 1

                pair_fraction = n_pairs / stem_len

                if pair_fraction < 0.75:
                    continue

                score = pairing_score - 0.5 * abs(loop_len - 4)

                ranking_key = (score, pair_fraction, stem_len, -abs(loop_len - 4))

                if best is None or ranking_key > best["ranking_key"]:

                    best = {
                        "ranking_key": ranking_key,
                        "start": start,
                        "stem_len": stem_len,
                        "loop_len": loop_len,
                        "score": score,
                    }

    if best is None:

        return {
            "hairpin_detected": False,
            "stem_start": np.nan,
            "stem_length": 0,
            "loop_length": 0,
            "loop_start": np.nan,
            "loop_end": np.nan,
            "right_stem_start": np.nan,
            "hairpin_end_exclusive": np.nan,
            "hairpin_anchor_0based": np.nan,
        }

    stem_start = best["start"]

    stem_len = best["stem_len"]

    loop_len = best["loop_len"]

    loop_start = stem_start + stem_len

    loop_end = loop_start + loop_len - 1

    right_stem_start = loop_start + loop_len

    hairpin_end_exclusive = stem_start + 2 * stem_len + loop_len

    # Last nucleotide of right stem
    hairpin_anchor_0based = hairpin_end_exclusive - 1

    return {
        "hairpin_detected": True,
        "stem_start": int(stem_start),
        "stem_length": int(stem_len),
        "loop_length": int(loop_len),
        "loop_start": int(loop_start),
        "loop_end": int(loop_end),
        "right_stem_start": int(right_stem_start),
        "hairpin_end_exclusive": int(hairpin_end_exclusive),
        "hairpin_anchor_0based": int(hairpin_anchor_0based),
    }


geometry_records = []

ism_unique_samples = sorted(df_ism["sample_idx"].unique())

for sample_idx in ism_unique_samples:

    seq = sequences[int(sample_idx)]

    geom = infer_hairpin_geometry(seq)

    geom["sample_idx"] = int(sample_idx)

    geom["sequence_length"] = len(seq)

    spacer = float(df_bio.iloc[int(sample_idx)]["spacer_length"])

    geom["spacer_length"] = spacer

    if geom["hairpin_detected"]:

        geom["first_downstream_T_relative"] = spacer + 1

    else:

        geom["first_downstream_T_relative"] = np.nan

    geometry_records.append(geom)

df_hairpin_geometry = pd.DataFrame(geometry_records)


display(df_hairpin_geometry.head())


geometry_qc = []

for _, row in df_hairpin_geometry.iterrows():

    sample_idx = int(row["sample_idx"])

    expected_stem = float(df_bio.iloc[sample_idx]["stem_length"])

    expected_loop = float(df_bio.iloc[sample_idx]["loop_length"])

    inferred_stem = float(row["stem_length"])

    inferred_loop = float(row["loop_length"])

    match = np.isclose(expected_stem, inferred_stem) and np.isclose(
        expected_loop, inferred_loop
    )

    geometry_qc.append(
        {
            "sample_idx": sample_idx,
            "expected_stem_length": expected_stem,
            "inferred_stem_length": inferred_stem,
            "expected_loop_length": expected_loop,
            "inferred_loop_length": inferred_loop,
            "match": match,
        }
    )

df_geometry_qc = pd.DataFrame(geometry_qc)

match_rate = df_geometry_qc["match"].mean()


display(df_geometry_qc[~df_geometry_qc["match"]])

if match_rate < 1.0:

    pass


else:

    pass


df_ism_aligned = df_ism.merge(
    df_hairpin_geometry[
        [
            "sample_idx",
            "hairpin_detected",
            "stem_start",
            "stem_length",
            "loop_length",
            "loop_start",
            "loop_end",
            "right_stem_start",
            "hairpin_anchor_0based",
            "spacer_length",
            "first_downstream_T_relative",
        ]
    ],
    on="sample_idx",
    how="left",
)

# Only sequences with detected hairpins
df_ism_aligned = df_ism_aligned[df_ism_aligned["hairpin_detected"]].copy()

df_ism_aligned["hairpin_relative_position"] = (
    df_ism_aligned["position_0based"] - df_ism_aligned["hairpin_anchor_0based"]
)

df_ism_aligned["stem_start_relative"] = (
    df_ism_aligned["stem_start"] - df_ism_aligned["hairpin_anchor_0based"]
)

df_ism_aligned["loop_start_relative"] = (
    df_ism_aligned["loop_start"] - df_ism_aligned["hairpin_anchor_0based"]
)

df_ism_aligned["loop_end_relative"] = (
    df_ism_aligned["loop_end"] - df_ism_aligned["hairpin_anchor_0based"]
)

df_ism_aligned["right_stem_start_relative"] = (
    df_ism_aligned["right_stem_start"] - df_ism_aligned["hairpin_anchor_0based"]
)


ALIGN_MIN = -30
ALIGN_MAX = 20

df_ism_window = df_ism_aligned[
    (df_ism_aligned["hairpin_relative_position"] >= ALIGN_MIN)
    & (df_ism_aligned["hairpin_relative_position"] <= ALIGN_MAX)
].copy()

hairpin_per_sequence = df_ism_window.groupby(
    ["sample_idx", "hairpin_relative_position"], as_index=False
).agg(
    Mean_abs_delta=("abs_delta", "mean"),
    Mean_delta=("delta_prediction", "mean"),
    Mean_delta_uncertainty=("delta_prediction_sd", "mean"),
)

hairpin_aligned_summary = (
    hairpin_per_sequence.groupby("hairpin_relative_position")["Mean_abs_delta"]
    .agg(Mean="mean", Median="median", SD="std", N_sequences="count")
    .reset_index()
)

hairpin_aligned_summary["SEM"] = hairpin_aligned_summary["SD"] / np.sqrt(
    hairpin_aligned_summary["N_sequences"]
)


signed_hairpin_matrix = df_ism_window.pivot_table(
    index="alt",
    columns="hairpin_relative_position",
    values="delta_prediction",
    aggfunc="mean",
).reindex(["A", "C", "G", "T"])


display(hairpin_aligned_summary.head())


# Make heatmap columns continuous
all_positions = np.arange(ALIGN_MIN, ALIGN_MAX + 1)

signed_plot = signed_hairpin_matrix.reindex(columns=all_positions)


zero_col = np.where(all_positions == 0)[0][0] + 0.5


x_pos = hairpin_aligned_summary["hairpin_relative_position"].to_numpy()

mean_sens = hairpin_aligned_summary["Mean"].to_numpy()

sem_sens = hairpin_aligned_summary["SEM"].fillna(0).to_numpy()


# Hairpin 3' end

geom_detected = df_hairpin_geometry[df_hairpin_geometry["hairpin_detected"]].copy()

geom_detected["stem_start_relative"] = (
    geom_detected["stem_start"] - geom_detected["hairpin_anchor_0based"]
)

geom_detected["loop_start_relative"] = (
    geom_detected["stem_start"]
    + geom_detected["stem_length"]
    - geom_detected["hairpin_anchor_0based"]
)

geom_detected["loop_end_relative"] = (
    geom_detected["stem_start"]
    + geom_detected["stem_length"]
    + geom_detected["loop_length"]
    - 1
    - geom_detected["hairpin_anchor_0based"]
)

geom_detected["right_stem_start_relative"] = (
    geom_detected["stem_start"]
    + geom_detected["stem_length"]
    + geom_detected["loop_length"]
    - geom_detected["hairpin_anchor_0based"]
)

median_stem_start = int(np.round(geom_detected["stem_start_relative"].median()))

median_loop_start = int(np.round(geom_detected["loop_start_relative"].median()))

median_loop_end = int(np.round(geom_detected["loop_end_relative"].median()))

median_right_stem_start = int(
    np.round(geom_detected["right_stem_start_relative"].median())
)


valid_t_geom = geom_detected[geom_detected["spacer_length"] < 15]

if len(valid_t_geom) > 0:

    median_first_t = float(valid_t_geom["first_downstream_T_relative"].median())


HOTSPOT_QUANTILE = 0.80

required_columns = ["hairpin_relative_position", "Mean", "Median", "SEM", "N_sequences"]

missing_columns = [
    col for col in required_columns if col not in hairpin_aligned_summary.columns
]

if missing_columns:
    raise ValueError(
        "hairpin_aligned_summary is missing columns: "
        f"{missing_columns}\n"
        f"Available columns: "
        f"{list(hairpin_aligned_summary.columns)}"
    )

hotspot_cutoff = np.quantile(hairpin_aligned_summary["Mean"].dropna(), HOTSPOT_QUANTILE)

hairpin_hotspots = (
    hairpin_aligned_summary[hairpin_aligned_summary["Mean"] >= hotspot_cutoff]
    .sort_values("Mean", ascending=False)
    .copy()
)


display(
    hairpin_hotspots[
        ["hairpin_relative_position", "Mean", "Median", "SEM", "N_sequences"]
    ]
)


top10 = df_perm_summary.head(10).sort_values("Mean", ascending=True)


group_plot = df_group_summary.sort_values("Mean", ascending=True)


top_feature = df_perm_summary.iloc[0]["Feature"]

tmp = pd.DataFrame({"feature": df_bio[top_feature], "actual": y, "pred": oof_pred})

tmp["bin"] = pd.qcut(tmp["feature"], q=7, duplicates="drop")

g = (
    tmp.groupby("bin", observed=True)
    .agg(x=("feature", "median"), actual=("actual", "mean"), pred=("pred", "mean"))
    .reset_index()
)


HIGH_STRENGTH_QUANTILE = 0.80

strength_cutoff = np.quantile(y, HIGH_STRENGTH_QUANTILE)

high_mask = y >= strength_cutoff

candidate_rules = df_assoc[
    (df_assoc["Permutation_Delta_RMSE"] > 0) & (df_assoc["q_actual"] < 0.10)
].sort_values("Permutation_Delta_RMSE", ascending=False)

soft_preferences = {}

for _, row in candidate_rules.head(10).iterrows():

    feature = row["Feature"]

    values_high = df_bio.loc[high_mask, feature].to_numpy()

    direction = "higher_associated" if row["rho_actual"] > 0 else "lower_associated"

    soft_preferences[feature] = {
        "direction": direction,
        "rho_actual": float(row["rho_actual"]),
        "q_actual": float(row["q_actual"]),
        "permutation_delta_rmse": float(row["Permutation_Delta_RMSE"]),
        "high_strength_q25_q75": [
            float(np.quantile(values_high, 0.25)),
            float(np.quantile(values_high, 0.75)),
        ],
        "high_strength_q10_q90": [
            float(np.quantile(values_high, 0.10)),
            float(np.quantile(values_high, 0.90)),
        ],
    }

hotspot_threshold = np.quantile(positional_ism["Mean"], 0.80)

hotspot_df = positional_ism[positional_ism["Mean"] >= hotspot_threshold]

ism_hotspots = [
    {
        "normalized_position": float(row["normalized_position"]),
        "mean_abs_delta": float(row["Mean"]),
    }
    for _, row in hotspot_df.iterrows()
]

group_importance_export = {
    row["Group"]: {"mean_delta_rmse": float(row["Mean"]), "sd": float(row["SD"])}
    for _, row in df_group_summary.iterrows()
}

phase2_design_knowledge = {
    "interpretation_type": "model-derived_and_observational",
    "n_phase1_models": len(fold_records),
    "n_ism_sequences": int(df_ism["sample_idx"].nunique()),
    "soft_feature_preferences": soft_preferences,
    "feature_group_importance": group_importance_export,
    "ism_hotspots": ism_hotspots,
    "important_note": (
        "These rules are model-derived "
        "associations and sensitivities, "
        "not experimentally established "
        "causal constraints."
    ),
}
