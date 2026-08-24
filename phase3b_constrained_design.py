"""Phase 3b -- knowledge-constrained inverse design.

Refines the Evo seed population with a multi-objective genetic algorithm using
non-dominated sorting and crowding-distance ordering (NSGA-II style) over five
minimized objectives. Mutation operator probabilities and the crossover rate are
derived from the Phase-2 evidence scores rather than tuned, so the interpretation
stage determines how the search moves.

Surviving candidates are filtered for support in embedding space, descriptor space
and structure, and for novelty against the training library.

Outputs: the top-ranked candidates per target termination efficiency, exported for
the independent cross-model check.
"""

import json
import random

import numpy as np
import pandas as pd


def find_project_root():

    if env_root:

        if root.exists():
            return root

    for root in [cwd, *cwd.parents]:

        if (root / "phase1_assets").exists() or (root / "phase1_cv_models").exists():
            return root

    return cwd


for path in [
    PHASE3_ROOT,
    METHOD_DIR,
    TARGET_ROOT,
    EVO_DIR,
    GA_ROOT,
    FINAL_ROOT,
    FIGURE_DIR,
]:

    pass


def te_to_average_strength(te):

    te = np.asarray(te, dtype=float)

    if np.any(~np.isfinite(te)):

        raise ValueError("TE contains NaN/Inf.")

    if np.any(te >= 1.0):

        raise ValueError("TE >= 1 requires infinite or invalid Average Strength.")

    average_strength = 1.0 / (1.0 - te)

    if np.any(average_strength <= 0):

        raise ValueError("Non-positive Average Strength.")

    return average_strength


def te_to_model_y(te):

    return np.log10(te_to_average_strength(te))


def model_y_to_average_strength(model_y):

    model_y = np.asarray(model_y, dtype=float)

    return 10.0**model_y


def model_y_to_te(model_y):

    model_y = np.asarray(model_y, dtype=float)

    return 1.0 - 10.0 ** (-model_y)


TARGET_TE_VALUES = np.array(
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

assert len(TARGET_TE_VALUES) == 11

assert np.all(TARGET_TE_VALUES < 1.0)

target_rows = []

for target_rank, target_te in enumerate(TARGET_TE_VALUES):

    target_id = f"te_{target_te:.2f}".replace(".", "p")

    target_average_strength = float(te_to_average_strength(target_te))

    target_model_y = float(te_to_model_y(target_te))

    target_rows.append(
        {
            "target_rank": int(target_rank),
            "target_id": target_id,
            "target_te": float(target_te),
            "target_average_strength": target_average_strength,
            "target_model_y": target_model_y,
            "finite_target": True,
        }
    )

df_targets = pd.DataFrame(target_rows)

df_target_boundary = pd.DataFrame(
    [
        {
            "target_id": "te_1p00",
            "target_te": 1.0,
            "target_average_strength": np.inf,
            "target_model_y": np.inf,
            "finite_target": False,
            "reason": (
                "TE=1 implies Average Strength→∞ "
                "and model_y=log10(Average Strength)→∞."
            ),
        }
    ]
)

roundtrip_te = model_y_to_te(df_targets["target_model_y"].to_numpy(dtype=float))

TARGET_TRANSFORM_MAX_ERROR = float(np.max(np.abs(roundtrip_te - TARGET_TE_VALUES)))

if TARGET_TRANSFORM_MAX_ERROR > 1e-12:

    raise RuntimeError("TE <-> model_y transformation round-trip failed.")


display(df_targets)


EVO_SAMPLES_PER_TARGET = 256

MAX_GA_POPULATION = 96

MAX_GENERATIONS = 40

# Counts newly evaluated GA offspring.
GA_EVAL_BUDGET_PER_SEED = 512

GA_SEEDS = [
    2026,
    777,
    999,
]

MAX_FINAL_REPORT = 30

GLOBAL_RANDOM_SEED = 2026

REQUIRE_FINAL_CANDIDATE_EACH_TARGET = True

random.seed(GLOBAL_RANDOM_SEED)

np.random.seed(GLOBAL_RANDOM_SEED)


import re
import json
import joblib
import cloudpickle

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin


class LeakageSafeStackingRegressor(BaseEstimator, RegressorMixin):

    def fit(self, X, y):

        raise RuntimeError("Compatibility-only class.")

    def predict(self, X):

        for attr in [
            "pipeline_",
            "final_pipeline_",
            "fitted_pipeline_",
            "final_model_",
        ]:

            if hasattr(self, attr):

                obj = getattr(self, attr)

                if obj is not self and hasattr(obj, "predict"):

                    try:

                        return np.asarray(obj.predict(X), dtype=float)

                    except Exception:
                        pass

        base_models = None

        for attr in [
            "base_models_",
            "base_estimators_",
            "estimators_",
            "fitted_estimators_",
            "models_",
        ]:

            if hasattr(self, attr):

                base_models = getattr(self, attr)

                break

        if base_models is None:

            raise AttributeError("Could not locate fitted base models.")

        if isinstance(base_models, dict):

            base_models = list(base_models.values())

        cleaned_models = []

        for item in base_models:

            if isinstance(item, tuple):

                item = item[-1]

            if isinstance(item, dict):

                for key in [
                    "model",
                    "estimator",
                    "pipeline",
                ]:

                    if key in item:

                        item = item[key]

                        break

            cleaned_models.append(item)

        preprocessor = None

        for attr in [
            "preprocessor_",
            "preprocessing_",
            "transformer_",
        ]:

            if hasattr(self, attr):

                preprocessor = getattr(self, attr)

                break

        X_transformed = None

        if preprocessor is not None:

            try:

                X_transformed = preprocessor.transform(X)

            except Exception:

                X_transformed = None

        base_predictions = []

        for model in cleaned_models:

            try:

                prediction = model.predict(X)

            except Exception:

                if X_transformed is None:
                    raise

                prediction = model.predict(X_transformed)

            base_predictions.append(np.asarray(prediction, dtype=float).reshape(-1))

        meta_X = np.column_stack(base_predictions)

        meta_model = None

        for attr in [
            "meta_model_",
            "meta_estimator_",
            "final_estimator_",
            "meta_learner_",
        ]:

            if hasattr(self, attr):

                meta_model = getattr(self, attr)

                break

        if meta_model is None:

            raise AttributeError("Could not locate fitted meta learner.")

        return np.asarray(meta_model.predict(meta_X), dtype=float)


BIO_FEATURE_NAMES = [
    "seq_len",
    "gc_content",
    "freq_A",
    "freq_C",
    "freq_G",
    "freq_T",
    "count_poly_t_4",
    "max_poly_t_run",
    "kmer_TTT",
    "kmer_AAA",
    "kmer_GCG",
    "kmer_CGC",
    "kmer_TTG",
    "kmer_TTA",
    "kmer_GCT",
    "kmer_AGC",
    "stem_length",
    "loop_length",
    "stem_pairing_score",
    "gu_wobble_count",
    "loop_proximal_gc_pair_fraction",
    "upstream_a_richness",
    "positional_poly_t_score",
    "spacer_length",
    "stem_polyT_coupling",
    "stem_polyT_interaction",
    "compact_gc_hairpin_score",
    "polyT_spacer_proximity",
    "polyT_position_interaction",
]

assert len(BIO_FEATURE_NAMES) == 29

bio_feature_names = BIO_FEATURE_NAMES.copy()


for path in [
    CLEANED_PATH,
    BIO_PATH,
    NT_PATH,
    OOF_PATH,
    EXTRACTOR_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(path)

df_train = pd.read_csv(CLEANED_PATH)

df_bio = pd.read_csv(BIO_PATH)

df_oof = pd.read_csv(OOF_PATH)

X_nt_train = np.load(NT_PATH)

with open(EXTRACTOR_PATH, "rb") as f:

    feature_extractor = cloudpickle.load(f)

phase1_manifest = {}

if PHASE1_MANIFEST_PATH.exists():

    with open(PHASE1_MANIFEST_PATH, "r") as f:

        phase1_manifest = json.load(f)


def normalize_column_name(name):

    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_column(df, aliases):

    normalized = {normalize_column_name(column): column for column in df.columns}

    for alias in aliases:

        key = normalize_column_name(alias)

        if key in normalized:

            return normalized[key]

    for alias in aliases:

        key = normalize_column_name(alias)

        for normalized_name, original_name in normalized.items():

            if key in normalized_name:

                return original_name

    return None


TRAIN_ID_COLUMN = resolve_column(
    df_train,
    [
        "id",
    ],
)

TRAIN_SEQUENCE_COLUMN = resolve_column(
    df_train,
    [
        "sequence",
    ],
)

TRAIN_MODEL_Y_COLUMN = resolve_column(
    df_train,
    [
        "actual",
    ],
)


OOF_ID_COLUMN = resolve_column(
    df_oof,
    [
        "id",
    ],
)

OOF_SEQUENCE_COLUMN = resolve_column(
    df_oof,
    [
        "sequence",
    ],
)

OOF_TRUE_MODEL_Y_COLUMN = resolve_column(
    df_oof,
    [
        "actual",
    ],
)

OOF_PRED_MODEL_Y_COLUMN = resolve_column(
    df_oof,
    [
        "mean_oof_prediction",
    ],
)


required_columns = {
    "TRAIN_SEQUENCE_COLUMN": TRAIN_SEQUENCE_COLUMN,
    "TRAIN_MODEL_Y_COLUMN": TRAIN_MODEL_Y_COLUMN,
    "OOF_SEQUENCE_COLUMN": OOF_SEQUENCE_COLUMN,
    "OOF_TRUE_MODEL_Y_COLUMN": OOF_TRUE_MODEL_Y_COLUMN,
    "OOF_PRED_MODEL_Y_COLUMN": OOF_PRED_MODEL_Y_COLUMN,
}

missing_columns = [name for name, value in required_columns.items() if value is None]

if missing_columns:

    raise RuntimeError("Missing required Phase-1 columns:\n" f"{missing_columns}")

sequences_train = (
    df_train[TRAIN_SEQUENCE_COLUMN].astype(str).str.upper().str.strip().to_numpy()
)

sequences_oof = (
    df_oof[OOF_SEQUENCE_COLUMN].astype(str).str.upper().str.strip().to_numpy()
)

if len(sequences_train) != len(sequences_oof):

    raise RuntimeError("Train/OOF row-count mismatch.")

if not np.array_equal(sequences_train, sequences_oof):

    mismatch_indices = np.flatnonzero(sequences_train != sequences_oof)

    raise RuntimeError(
        "Train/OOF sequence order mismatch.\n"
        f"First mismatches: "
        f"{mismatch_indices[:10].tolist()}"
    )

TRAIN_OOF_ID_ALIGNMENT = None

if TRAIN_ID_COLUMN is not None and OOF_ID_COLUMN is not None:

    TRAIN_OOF_ID_ALIGNMENT = bool(
        np.array_equal(
            df_train[TRAIN_ID_COLUMN].astype(str).to_numpy(),
            df_oof[OOF_ID_COLUMN].astype(str).to_numpy(),
        )
    )

model_y_train = pd.to_numeric(df_train[TRAIN_MODEL_Y_COLUMN], errors="raise").to_numpy(
    dtype=float
)

model_y_oof_true = pd.to_numeric(
    df_oof[OOF_TRUE_MODEL_Y_COLUMN], errors="raise"
).to_numpy(dtype=float)

model_y_oof_pred = pd.to_numeric(
    df_oof[OOF_PRED_MODEL_Y_COLUMN], errors="raise"
).to_numpy(dtype=float)

for name, values in [
    ("model_y_train", model_y_train),
    ("model_y_oof_true", model_y_oof_true),
    ("model_y_oof_pred", model_y_oof_pred),
]:

    if not np.all(np.isfinite(values)):

        raise RuntimeError(f"{name} contains NaN/Inf.")

TRAIN_VS_OOF_MODEL_Y_MAX_ERROR = float(np.max(np.abs(model_y_train - model_y_oof_true)))


if TRAIN_VS_OOF_MODEL_Y_MAX_ERROR > 1e-10:

    raise RuntimeError("Train/OOF model_y mismatch.")


average_strength_train = model_y_to_average_strength(model_y_train)

te_train = model_y_to_te(model_y_train)


if SAVED_ORIGINAL_EFFICIENCY_AVAILABLE:

    saved_original_efficiency = pd.to_numeric(
        df_train[MISLEADING_SAVED_EFFICIENCY_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(saved_original_efficiency)):

        pass

    else:

        if SAVED_ORIGINAL_EFFICIENCY_VS_MODEL_Y_MAX_ERROR <= 1e-10:

            pass

        elif SAVED_ORIGINAL_EFFICIENCY_VS_TE_MAX_ERROR <= 1e-8:

            pass

        else:

            pass

        if SAVED_ORIGINAL_EFFICIENCY_STATUS == "numerically_duplicates_model_y":

            pass

        df_saved_column_diagnostic = pd.DataFrame(
            {
                "row_idx": np.arange(len(model_y_train), dtype=int),
                "model_y_actual": model_y_train,
                "reconstructed_average_strength": average_strength_train,
                "reconstructed_te": te_train,
                "saved_original_efficiency": saved_original_efficiency,
                "saved_minus_model_y": (saved_original_efficiency - model_y_train),
                "saved_minus_te": (saved_original_efficiency - te_train),
                "sequence": sequences_train,
            }
        )

        if TRAIN_ID_COLUMN is not None:

            df_saved_column_diagnostic.insert(1, "id", df_train[TRAIN_ID_COLUMN].values)


missing_bio_features = [
    feature for feature in bio_feature_names if feature not in df_bio.columns
]

if missing_bio_features:

    raise RuntimeError("Missing Phase-1 29D descriptors:\n" f"{missing_bio_features}")

X_bio_train = df_bio[bio_feature_names].to_numpy(dtype=float)

n_train = len(sequences_train)

if X_nt_train.shape != (n_train, 2048):

    raise RuntimeError(f"Unexpected NT shape: " f"{X_nt_train.shape}")

if X_bio_train.shape != (n_train, 29):

    raise RuntimeError(f"Unexpected bio shape: " f"{X_bio_train.shape}")

model_paths = sorted(PHASE1_MODEL_DIR.glob("model_repeat_*_fold_*.joblib"))

if len(model_paths) != 50:

    raise RuntimeError(
        "Expected exactly 50 Phase-1 models.\n" f"Found: {len(model_paths)}"
    )

MODEL_ARTIFACTS = []

for path in model_paths:

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):

        raise RuntimeError(f"Unexpected artifact format: " f"{path}")

    if "model" not in artifact:

        raise RuntimeError(f"`model` missing: " f"{path}")

    artifact["_path"] = str(path)

    MODEL_ARTIFACTS.append(artifact)

MODELS_50 = [artifact["model"] for artifact in MODEL_ARTIFACTS]

nt_names = sorted(
    set(
        str(artifact.get("nt_model_name"))
        for artifact in MODEL_ARTIFACTS
        if artifact.get("nt_model_name") is not None
    )
)

if len(nt_names) != 1:

    raise RuntimeError(f"Inconsistent NT metadata: " f"{nt_names}")

NT_MODEL_NAME = nt_names[0]

mid_layer_sets = sorted(
    set(tuple(artifact.get("mid_layer_ids", [])) for artifact in MODEL_ARTIFACTS)
)

if len(mid_layer_sets) != 1:

    raise RuntimeError("Inconsistent intermediate-layer metadata.")

MID_LAYER_IDS = list(mid_layer_sets[0])

target_modes = sorted(
    set(
        str(artifact.get("target_mode"))
        for artifact in MODEL_ARTIFACTS
        if artifact.get("target_mode") is not None
    )
)


if {str(mode).strip().lower() for mode in target_modes} == {"as_is"}:

    pass


X_raw_train = np.hstack(
    [
        X_nt_train,
        X_bio_train,
    ]
)

if X_raw_train.shape != (n_train, 2077):

    raise RuntimeError(f"Unexpected raw dimension: " f"{X_raw_train.shape}")

smoke_prediction = np.asarray(
    MODELS_50[0].predict(X_raw_train[:2]), dtype=float
).reshape(-1)

if smoke_prediction.shape != (2,) or not np.all(np.isfinite(smoke_prediction)):

    raise RuntimeError("Phase-1 prediction smoke test failed.")

TRAIN_MODEL_Y_MIN = float(model_y_train.min())

TRAIN_MODEL_Y_MAX = float(model_y_train.max())

TRAIN_AVERAGE_STRENGTH_MIN = float(average_strength_train.min())

TRAIN_AVERAGE_STRENGTH_MAX = float(average_strength_train.max())

TRAIN_TE_MIN = float(te_train.min())

TRAIN_TE_MAX = float(te_train.max())

df_targets["inside_training_model_y_range"] = (
    df_targets["target_model_y"] >= TRAIN_MODEL_Y_MIN
) & (df_targets["target_model_y"] <= TRAIN_MODEL_Y_MAX)

df_targets["inside_training_te_range"] = (df_targets["target_te"] >= TRAIN_TE_MIN) & (
    df_targets["target_te"] <= TRAIN_TE_MAX
)

df_targets["design_regime"] = np.where(
    df_targets["inside_training_model_y_range"], "interpolative", "extrapolative"
)


df_phase1_target_reconstruction = pd.DataFrame(
    {
        "sample_idx": np.arange(n_train, dtype=int),
        "sequence": sequences_train,
        "model_y": model_y_train,
        "average_strength": average_strength_train,
        "conventional_te": te_train,
        "oof_pred_model_y": model_y_oof_pred,
        "oof_pred_te": model_y_to_te(model_y_oof_pred),
    }
)

if TRAIN_ID_COLUMN is not None:

    df_phase1_target_reconstruction.insert(1, "id", df_train[TRAIN_ID_COLUMN].values)


PHASE1_TARGET_QA = {
    "model_target_definition": "model_y = actual = log10(Average Strength)",
    "conventional_te_definition": "TE = 1 - 10**(-model_y)",
    "train_vs_oof_model_y_max_abs_error": TRAIN_VS_OOF_MODEL_Y_MAX_ERROR,
    "exported_original_efficiency_status": SAVED_ORIGINAL_EFFICIENCY_STATUS,
    "exported_original_efficiency_vs_model_y_max_abs_error": SAVED_ORIGINAL_EFFICIENCY_VS_MODEL_Y_MAX_ERROR,
    "exported_original_efficiency_vs_te_max_abs_error": SAVED_ORIGINAL_EFFICIENCY_VS_TE_MAX_ERROR,
    "exported_original_efficiency_used_downstream": False,
}


display(
    df_targets[
        [
            "target_id",
            "target_te",
            "target_average_strength",
            "target_model_y",
            "design_regime",
        ]
    ]
)


import json
import warnings

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

FEATURE_TO_GROUP = {}

for group_name, features in FEATURE_GROUPS.items():

    for feature in features:

        FEATURE_TO_GROUP[feature] = group_name


def normalize_name(value):

    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def compact_name(value):

    return "".join(
        character for character in normalize_name(value) if character.isalnum()
    )


def find_col(df, exact=None, token_sets=None):

    exact = exact or []

    token_sets = token_sets or []

    normalized = {normalize_name(column): column for column in df.columns}

    compact = {compact_name(column): column for column in df.columns}

    for candidate in exact:

        normalized_candidate = normalize_name(candidate)

        if normalized_candidate in normalized:

            return normalized[normalized_candidate]

        compact_candidate = compact_name(candidate)

        if compact_candidate in compact:

            return compact[compact_candidate]

    for tokens in token_sets:

        normalized_tokens = [normalize_name(token) for token in tokens]

        for normalized_column, original_column in normalized.items():

            if all(token in normalized_column for token in normalized_tokens):

                return original_column

    return None


def collect_phase2_csvs():

    paths = []

    for root in PHASE2_SEARCH_ROOTS:

        if not root.exists():

            continue

        for path in root.rglob("*.csv"):

            path_text = str(path.resolve()).lower()

            if "phase3_results" in path_text:

                continue

            if path not in paths:

                paths.append(path)

    return paths


def find_exact_phase2_file(filename, required=True):

    matches = [
        path for path in ALL_PHASE2_CSVS if path.name.lower() == str(filename).lower()
    ]

    if len(matches) == 0:

        if required:

            for path in sorted(ALL_PHASE2_CSVS):

                if "phase2" in path.name.lower():

                    pass

            raise FileNotFoundError(
                f"Could not locate required Phase-2 file:\n" f"{filename}"
            )

        return None

    matches = sorted(
        matches,
        key=lambda path: (
            len(path.parts),
            str(path),
        ),
    )

    if len(matches) > 1:

        warnings.warn(
            (f"Multiple copies of {filename} found. " f"Using:\n{matches[0]}")
        )

    return matches[0]


df_pi_raw = pd.read_csv(PI_PATH)


PI_FEATURE_COL = find_col(
    df_pi_raw,
    exact=[
        "feature",
        "feature_name",
    ],
)

PI_DELTA_COL = find_col(
    df_pi_raw,
    exact=[
        "delta_rmse",
        "importance",
        "permutation_importance",
    ],
    token_sets=[
        [
            "delta",
            "rmse",
        ],
        [
            "permutation",
            "importance",
        ],
    ],
)

if PI_FEATURE_COL is None:

    raise RuntimeError(
        "Could not resolve PI feature column.\n"
        f"Columns: {df_pi_raw.columns.tolist()}"
    )

if PI_DELTA_COL is None:

    raise RuntimeError(
        "Could not resolve PI effect column.\n" f"Columns: {df_pi_raw.columns.tolist()}"
    )

df_pi_raw[PI_DELTA_COL] = pd.to_numeric(df_pi_raw[PI_DELTA_COL], errors="coerce")

df_pi_summary = (
    df_pi_raw.dropna(
        subset=[
            PI_FEATURE_COL,
            PI_DELTA_COL,
        ]
    )
    .groupby(PI_FEATURE_COL, as_index=False)
    .agg(
        pi_mean=(PI_DELTA_COL, "mean"),
        pi_sd=(PI_DELTA_COL, "std"),
        pi_n=(PI_DELTA_COL, "size"),
        pi_stability=(
            PI_DELTA_COL,
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0)),
        ),
    )
)

df_pi_summary["pi_sd"] = df_pi_summary["pi_sd"].fillna(0.0)

df_pi_summary["pi_positive"] = df_pi_summary["pi_mean"].clip(lower=0)

df_pi_summary["pi_percentile"] = df_pi_summary["pi_positive"].rank(
    pct=True, method="average"
)


def looks_like_association_table(df):

    feature_col = find_col(
        df,
        exact=[
            "feature",
            "feature_name",
        ],
    )

    actual_rho_col = find_col(
        df,
        exact=[
            "actual_rho",
            "rho_actual",
            "actual_spearman",
        ],
        token_sets=[
            [
                "actual",
                "rho",
            ],
            [
                "actual",
                "spearman",
            ],
        ],
    )

    oof_rho_col = find_col(
        df,
        exact=[
            "oof_rho",
            "rho_oof",
            "oof_spearman",
        ],
        token_sets=[
            [
                "oof",
                "rho",
            ],
            [
                "oof",
                "spearman",
            ],
        ],
    )

    return (
        feature_col is not None
        and actual_rho_col is not None
        and oof_rho_col is not None
    )


association_candidates = []

for path in ALL_PHASE2_CSVS:

    if path in [
        PI_PATH,
        GROUP_PI_PATH,
        ISM_RECORDS_PATH,
        ISM_SUMMARY_PATH,
        ISM_HOTSPOTS_PATH,
        RAW_ISM_PATH,
    ]:

        continue

    try:

        header = pd.read_csv(path, nrows=10)

    except Exception:

        continue

    if not looks_like_association_table(header):

        continue

    filename = path.name.lower()

    filename_score = sum(
        keyword in filename
        for keyword in [
            "association",
            "correlation",
            "spearman",
            "feature",
        ]
    )

    try:

        row_count = len(pd.read_csv(path))

    except Exception:

        row_count = 0

    association_candidates.append(
        (
            filename_score,
            row_count,
            path,
        )
    )

if not association_candidates:

    raise FileNotFoundError(
        "Could not locate the Phase-2 actual/OOF " "association table."
    )

association_candidates = sorted(
    association_candidates,
    key=lambda item: (
        item[0],
        item[1],
    ),
    reverse=True,
)


df_assoc = pd.read_csv(ASSOC_PATH)


ASSOC_FEATURE_COL = find_col(
    df_assoc,
    exact=[
        "feature",
        "feature_name",
    ],
)

ACTUAL_RHO_COL = find_col(
    df_assoc,
    exact=[
        "actual_rho",
        "rho_actual",
        "actual_spearman",
    ],
    token_sets=[
        [
            "actual",
            "rho",
        ],
        [
            "actual",
            "spearman",
        ],
    ],
)

OOF_RHO_COL = find_col(
    df_assoc,
    exact=[
        "oof_rho",
        "rho_oof",
        "oof_spearman",
    ],
    token_sets=[
        [
            "oof",
            "rho",
        ],
        [
            "oof",
            "spearman",
        ],
    ],
)

ACTUAL_Q_COL = find_col(
    df_assoc,
    exact=[
        "actual_q",
        "actual_fdr",
        "actual_qvalue",
        "actual_q_value",
        "actual_bh_fdr",
    ],
    token_sets=[
        [
            "actual",
            "fdr",
        ],
        [
            "actual",
            "bh",
        ],
    ],
)

ACTUAL_P_COL = find_col(
    df_assoc,
    exact=[
        "actual_p",
        "actual_pvalue",
        "actual_p_value",
    ],
    token_sets=[
        [
            "actual",
            "p",
        ]
    ],
)

if ASSOC_FEATURE_COL is None or ACTUAL_RHO_COL is None or OOF_RHO_COL is None:

    raise RuntimeError(
        "Could not resolve association columns.\n"
        f"Columns: {df_assoc.columns.tolist()}"
    )


def bh_fdr(p_values):

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)

    order = np.argsort(p_values)

    adjusted_sorted = p_values[order] * n / np.arange(1, n + 1)

    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]

    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    result = np.empty_like(adjusted_sorted)

    result[order] = adjusted_sorted

    return result


actual_rho = pd.to_numeric(df_assoc[ACTUAL_RHO_COL], errors="coerce")

oof_rho = pd.to_numeric(df_assoc[OOF_RHO_COL], errors="coerce")

if ACTUAL_Q_COL is not None:

    actual_q = pd.to_numeric(df_assoc[ACTUAL_Q_COL], errors="coerce").to_numpy(
        dtype=float
    )

elif ACTUAL_P_COL is not None:

    actual_p = pd.to_numeric(df_assoc[ACTUAL_P_COL], errors="coerce")

    actual_q = np.full(len(actual_p), np.nan, dtype=float)

    finite_mask = actual_p.notna().to_numpy()

    actual_q[finite_mask] = bh_fdr(actual_p[finite_mask].to_numpy(dtype=float))

else:

    raise RuntimeError("Association table lacks actual q/FDR " "and actual p-value.")

df_assoc_small = pd.DataFrame(
    {
        "feature": df_assoc[ASSOC_FEATURE_COL],
        "actual_rho": actual_rho,
        "oof_rho": oof_rho,
        "actual_q": actual_q,
    }
)

df_feature_evidence = (
    pd.DataFrame({"feature": bio_feature_names})
    .merge(
        df_pi_summary.rename(columns={PI_FEATURE_COL: "feature"}),
        on="feature",
        how="left",
    )
    .merge(df_assoc_small, on="feature", how="left")
)

for column in [
    "pi_mean",
    "pi_sd",
    "pi_n",
    "pi_stability",
    "pi_positive",
    "pi_percentile",
    "actual_rho",
    "oof_rho",
    "actual_q",
]:

    if column not in (df_feature_evidence.columns):

        df_feature_evidence[column] = np.nan

actual_sign = np.sign(df_feature_evidence["actual_rho"].fillna(0))

oof_sign = np.sign(df_feature_evidence["oof_rho"].fillna(0))

df_feature_evidence["same_direction"] = (actual_sign == oof_sign) & (actual_sign != 0)

df_feature_evidence["association_convergence"] = np.where(
    df_feature_evidence["same_direction"],
    np.sqrt(np.abs(df_feature_evidence["actual_rho"] * df_feature_evidence["oof_rho"])),
    0.0,
)

df_feature_evidence["fdr_support"] = 1.0 - df_feature_evidence["actual_q"].fillna(
    1.0
).clip(0, 1)

for column in [
    "pi_percentile",
    "pi_stability",
    "association_convergence",
    "fdr_support",
]:

    df_feature_evidence[column] = df_feature_evidence[column].fillna(0).clip(0, 1)

evidence_components = df_feature_evidence[
    [
        "pi_percentile",
        "pi_stability",
        "association_convergence",
        "fdr_support",
    ]
].to_numpy(dtype=float)

df_feature_evidence["feature_evidence"] = np.prod(evidence_components, axis=1) ** (
    1.0 / evidence_components.shape[1]
)

if df_feature_evidence["feature_evidence"].sum() <= 0:

    raise RuntimeError("Integrated feature evidence sums to zero.")

df_feature_evidence["feature_group"] = df_feature_evidence["feature"].map(
    FEATURE_TO_GROUP
)

df_group_raw = pd.read_csv(GROUP_PI_PATH)


GROUP_COL = find_col(
    df_group_raw,
    exact=[
        "group",
        "feature_group",
    ],
)

GROUP_DELTA_COL = find_col(
    df_group_raw,
    exact=[
        "delta_rmse",
        "importance",
        "permutation_importance",
    ],
    token_sets=[
        [
            "delta",
            "rmse",
        ],
        [
            "permutation",
            "importance",
        ],
    ],
)

if GROUP_COL is None or GROUP_DELTA_COL is None:

    raise RuntimeError(
        "Could not resolve group PI columns.\n"
        f"Columns: {df_group_raw.columns.tolist()}"
    )

df_group_raw[GROUP_DELTA_COL] = pd.to_numeric(
    df_group_raw[GROUP_DELTA_COL], errors="coerce"
)

df_group_evidence = (
    df_group_raw.dropna(
        subset=[
            GROUP_COL,
            GROUP_DELTA_COL,
        ]
    )
    .groupby(GROUP_COL, as_index=False)
    .agg(
        group_pi_mean=(GROUP_DELTA_COL, "mean"),
        group_pi_stability=(
            GROUP_DELTA_COL,
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0)),
        ),
    )
)

df_group_evidence["positive_pi"] = df_group_evidence["group_pi_mean"].clip(lower=0)

df_group_evidence["pi_percentile"] = df_group_evidence["positive_pi"].rank(
    pct=True, method="average"
)

df_group_evidence["group_evidence"] = np.sqrt(
    df_group_evidence["pi_percentile"] * df_group_evidence["group_pi_stability"]
)

df_group_evidence = df_group_evidence.rename(columns={GROUP_COL: "group"})

interaction_mask = (
    df_group_evidence["group"].astype(str).str.lower().str.contains("interaction")
)

total_group_evidence = float(df_group_evidence["group_evidence"].clip(lower=0).sum())

if total_group_evidence <= 0:

    raise RuntimeError("Group evidence sums to zero.")

if not interaction_mask.any():

    raise RuntimeError("Interaction descriptor group not found.")

P_CROSSOVER = float(
    df_group_evidence.loc[interaction_mask, "group_evidence"].sum()
    / total_group_evidence
)

df_ism_records = pd.read_csv(ISM_RECORDS_PATH)

df_ism_summary = pd.read_csv(ISM_SUMMARY_PATH)

if ISM_HOTSPOTS_PATH is not None:

    df_ism_hotspots = pd.read_csv(ISM_HOTSPOTS_PATH)

else:

    df_ism_hotspots = pd.DataFrame()


if len(df_ism_hotspots) > 0:

    pass


ISM_REL_COL = find_col(
    df_ism_records,
    exact=[
        "relative_position",
        "relative_pos",
        "rel_position",
        "rel_pos",
        "hairpin_relative_position",
        "hairpin_rel_position",
        "aligned_position",
    ],
    token_sets=[
        [
            "relative",
            "position",
        ],
        [
            "hairpin",
            "relative",
        ],
    ],
)

ISM_ALT_COL = find_col(
    df_ism_records,
    exact=[
        "alt",
        "alt_base",
        "mutant_base",
        "mutation_base",
    ],
    token_sets=[
        [
            "alt",
            "base",
        ],
        [
            "mutant",
            "base",
        ],
    ],
)

ISM_DELTA_COL = find_col(
    df_ism_records,
    exact=[
        "delta_prediction",
        "delta_pred",
        "delta_oof",
        "delta_model_y",
        "delta_y",
        "delta",
    ],
    token_sets=[
        [
            "delta",
            "prediction",
        ],
        [
            "delta",
            "pred",
        ],
        [
            "delta",
            "oof",
        ],
        [
            "delta",
            "model",
        ],
    ],
)

ISM_SAMPLE_COL = find_col(
    df_ism_records,
    exact=[
        "sample_idx",
        "sample_index",
        "row_idx",
    ],
    token_sets=[
        [
            "sample",
            "idx",
        ],
        [
            "sample",
            "index",
        ],
    ],
)


if ISM_REL_COL is None:

    raise RuntimeError(
        "The pre-aligned ISM records do not expose a "
        "relative-position column.\n"
        f"Columns: {df_ism_records.columns.tolist()}"
    )

if ISM_ALT_COL is None:

    raise RuntimeError(
        "The pre-aligned ISM records do not expose "
        "an alternate-base column.\n"
        f"Columns: {df_ism_records.columns.tolist()}"
    )

if ISM_DELTA_COL is None:

    raise RuntimeError(
        "The pre-aligned ISM records do not expose "
        "a mutation-effect column.\n"
        f"Columns: {df_ism_records.columns.tolist()}"
    )

df_ism_work = df_ism_records.copy()

df_ism_work["_relative_position"] = pd.to_numeric(
    df_ism_work[ISM_REL_COL], errors="coerce"
)

df_ism_work["_alt_base"] = df_ism_work[ISM_ALT_COL].astype(str).str.upper().str.strip()

df_ism_work["_delta_model_y"] = pd.to_numeric(
    df_ism_work[ISM_DELTA_COL], errors="coerce"
)

valid_mask = (
    df_ism_work["_relative_position"].notna()
    & df_ism_work["_alt_base"].isin(
        [
            "A",
            "C",
            "G",
            "T",
        ]
    )
    & np.isfinite(df_ism_work["_delta_model_y"].to_numpy(dtype=float))
)

df_ism_work = df_ism_work.loc[valid_mask].copy().reset_index(drop=True)

if len(df_ism_work) == 0:

    raise RuntimeError("No valid aligned ISM records remain.")

# Relative positions should be integer-valued.
relative_values = df_ism_work["_relative_position"].to_numpy(dtype=float)

if not np.allclose(relative_values, np.round(relative_values), atol=1e-8, rtol=0):

    raise RuntimeError("Aligned ISM relative positions are not integer-valued.")

df_ism_work["_relative_position"] = np.round(relative_values).astype(int)

ISM_RELATIVE_WINDOW_MIN = -30

ISM_RELATIVE_WINDOW_MAX = 20

df_ism_aligned = (
    df_ism_work.loc[
        (df_ism_work["_relative_position"] >= ISM_RELATIVE_WINDOW_MIN)
        & (df_ism_work["_relative_position"] <= ISM_RELATIVE_WINDOW_MAX)
    ]
    .copy()
    .reset_index(drop=True)
)

if len(df_ism_aligned) == 0:

    raise RuntimeError(
        "No aligned ISM records remain " "inside the inherited -30:+20 window."
    )

df_ism_agg = (
    df_ism_aligned.groupby(
        [
            "_relative_position",
            "_alt_base",
        ],
        as_index=False,
    )["_delta_model_y"]
    .mean()
    .rename(
        columns={
            "_relative_position": "relative_position",
            "_alt_base": "alt_base",
            "_delta_model_y": "mean_delta_model_y",
        }
    )
)

ISM_LOOKUP = {
    (
        int(row["relative_position"]),
        str(row["alt_base"]),
    ): float(row["mean_delta_model_y"])
    for _, row in df_ism_agg.iterrows()
}

if len(ISM_LOOKUP) == 0:

    raise RuntimeError("ISM mutation-direction lookup is empty.")

df_ism_position_sensitivity = (
    df_ism_aligned.assign(
        absolute_delta_model_y=lambda frame: np.abs(frame["_delta_model_y"])
    )
    .groupby("_relative_position", as_index=False)["absolute_delta_model_y"]
    .mean()
    .rename(
        columns={
            "_relative_position": "relative_position",
            "absolute_delta_model_y": "mean_abs_delta_model_y",
        }
    )
)

ISM_RELATIVE_POSITIONS = sorted(
    int(position)
    for position in df_ism_position_sensitivity["relative_position"].tolist()
)

ISM_POSITION_SENSITIVITY = {
    int(row["relative_position"]): float(row["mean_abs_delta_model_y"])
    for _, row in df_ism_position_sensitivity.iterrows()
}

if (
    len(df_ism_position_sensitivity) > 0
    and float(df_ism_position_sensitivity["mean_abs_delta_model_y"].max()) > 0
):

    normalized_sensitivity = (
        df_ism_position_sensitivity["mean_abs_delta_model_y"]
        / df_ism_position_sensitivity["mean_abs_delta_model_y"].max()
    )

    ISM_GUIDED_EVIDENCE = float(normalized_sensitivity.mean())

else:

    ISM_GUIDED_EVIDENCE = 0.0

SUMMARY_REL_COL = find_col(
    df_ism_summary,
    exact=[
        "relative_position",
        "relative_pos",
        "rel_position",
        "rel_pos",
        "hairpin_relative_position",
    ],
    token_sets=[
        [
            "relative",
            "position",
        ]
    ],
)

SUMMARY_EFFECT_COL = find_col(
    df_ism_summary,
    exact=[
        "mean_abs_delta",
        "mean_abs_delta_model_y",
        "mean_absolute_delta",
        "mean_abs_effect",
        "mean_abs_oof_delta",
    ],
    token_sets=[
        [
            "mean",
            "abs",
            "delta",
        ],
        [
            "mean",
            "absolute",
            "delta",
        ],
    ],
)

SUMMARY_QA_AVAILABLE = SUMMARY_REL_COL is not None


feature_evidence_lookup = df_feature_evidence.set_index("feature")[
    "feature_evidence"
].to_dict()


def mean_feature_evidence(features):

    values = [float(feature_evidence_lookup.get(feature, 0.0)) for feature in features]

    if not values:

        return 0.0

    return float(np.mean(values))


operator_scores = {
    "paired_stem": mean_feature_evidence(
        [
            "stem_length",
            "stem_pairing_score",
            "gu_wobble_count",
            "loop_proximal_gc_pair_fraction",
            "stem_polyT_interaction",
            "compact_gc_hairpin_score",
        ]
    ),
    "loop": mean_feature_evidence(
        [
            "loop_length",
            "loop_proximal_gc_pair_fraction",
        ]
    ),
    "polyT": mean_feature_evidence(
        [
            "freq_T",
            "count_poly_t_4",
            "max_poly_t_run",
            "kmer_TTT",
            "positional_poly_t_score",
            "spacer_length",
            "stem_polyT_coupling",
            "polyT_spacer_proximity",
            "polyT_position_interaction",
        ]
    ),
    "ism_guided": ISM_GUIDED_EVIDENCE,
    "random": mean_feature_evidence(
        FEATURE_GROUPS["Sequence composition"] + FEATURE_GROUPS["3-mer motifs"]
    ),
}

operator_total = sum(max(score, 0.0) for score in operator_scores.values())

if operator_total <= 0:

    raise RuntimeError("Mutation-operator evidence sums to zero.")

OPERATOR_WEIGHTS = {
    operator: max(score, 0.0) / operator_total
    for operator, score in operator_scores.items()
}


aligned_export_columns = [
    "_relative_position",
    "_alt_base",
    "_delta_model_y",
]

df_ism_aligned_export = df_ism_aligned[aligned_export_columns].rename(
    columns={
        "_relative_position": "relative_position",
        "_alt_base": "alt_base",
        "_delta_model_y": "delta_model_y",
    }
)


df_operator_probabilities = pd.DataFrame(
    {
        "operator": list(OPERATOR_WEIGHTS.keys()),
        "raw_evidence_score": [
            operator_scores[operator] for operator in OPERATOR_WEIGHTS.keys()
        ],
        "probability": list(OPERATOR_WEIGHTS.values()),
    }
)


PHASE2_EVIDENCE_PROVENANCE = {
    "individual_pi": str(PI_PATH),
    "group_pi": str(GROUP_PI_PATH),
    "association": str(ASSOC_PATH),
    "raw_oof_ism": (str(RAW_ISM_PATH) if RAW_ISM_PATH is not None else None),
    "hairpin_aligned_ism_records": str(ISM_RECORDS_PATH),
    "hairpin_aligned_ism_summary": str(ISM_SUMMARY_PATH),
    "hairpin_aligned_ism_hotspots": (
        str(ISM_HOTSPOTS_PATH) if ISM_HOTSPOTS_PATH is not None else None
    ),
    "phase3_primary_ism_source": str(ISM_RECORDS_PATH),
    "raw_ism_realigned_in_phase3": False,
    "relative_position_definition": ("0 = last nucleotide of the right / 3' stem"),
    "relative_position_window": [
        ISM_RELATIVE_WINDOW_MIN,
        ISM_RELATIVE_WINDOW_MAX,
    ],
    "new_phase3_hotspot_percentile_threshold": False,
    "phase2_existing_hotspot_file_used_as_primary_guidance": False,
    "phase2_existing_hotspot_file_role": "provenance / QA",
    "full_aligned_ism_used_for_directional_guidance": True,
}


for operator, probability in OPERATOR_WEIGHTS.items():

    pass


import json

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


def coerce_bio_output(output):

    if isinstance(output, tuple) and len(output) > 0:

        return coerce_bio_output(output[0])

    if isinstance(output, pd.Series):

        return output[bio_feature_names].to_numpy(dtype=float)

    if isinstance(output, pd.DataFrame):

        if len(output) != 1:

            raise ValueError("Expected one descriptor row.")

        return output[bio_feature_names].iloc[0].to_numpy(dtype=float)

    if isinstance(output, dict):

        return np.array([output[feature] for feature in bio_feature_names], dtype=float)

    output = np.asarray(output, dtype=float).reshape(-1)

    if output.shape != (29,):

        raise ValueError(f"Unexpected descriptor shape: " f"{output.shape}")

    return output


def extract_one_bio(sequence):

    sequence = str(sequence).upper().strip()

    if hasattr(feature_extractor, "transform"):

        try:

            output = feature_extractor.transform([sequence])

            if isinstance(output, pd.DataFrame):

                return output[bio_feature_names].iloc[0].to_numpy(dtype=float)

            array = np.asarray(output, dtype=float)

            if array.shape == (1, 29):

                return array[0]

        except Exception:

            pass

    if callable(feature_extractor):

        return coerce_bio_output(feature_extractor(sequence))

    if hasattr(feature_extractor, "extract"):

        return coerce_bio_output(feature_extractor.extract(sequence))

    raise TypeError("Unsupported Phase-1 feature extractor.")


def extract_bio_sequences(sequences):

    return np.vstack([extract_one_bio(sequence) for sequence in sequences])


bio_qc_idx = np.unique(
    np.array(
        [
            0,
            len(sequences_train) // 2,
            len(sequences_train) - 1,
        ],
        dtype=int,
    )
)

bio_qc_recomputed = extract_bio_sequences(sequences_train[bio_qc_idx])

bio_qc_reference = X_bio_train[bio_qc_idx]

BIO_EXTRACTOR_MAX_ABS_ERROR = float(
    np.max(np.abs(bio_qc_recomputed - bio_qc_reference))
)


if BIO_EXTRACTOR_MAX_ABS_ERROR > 1e-10:

    raise RuntimeError("Phase-1 29D reconstruction failed.")


def phase1_pair_score(left_base, right_base):

    pair = (left_base, right_base)

    # GC canonical
    if pair in [
        ("G", "C"),
        ("C", "G"),
    ]:

        return 3.0

    # AT canonical
    if pair in [
        ("A", "T"),
        ("T", "A"),
    ]:

        return 2.0

    # GT wobble
    if pair in [
        ("G", "T"),
        ("T", "G"),
    ]:

        return 1.0

    return 0.0


def phase1_hairpin_geometry(sequence):

    sequence = str(sequence).upper().strip()

    best = None

    best_key = None

    for stem_length in range(4, 14):

        for loop_length in range(3, 9):

            window_length = 2 * stem_length + loop_length

            if window_length > len(sequence):

                continue

            for start in range(0, len(sequence) - window_length + 1):

                left_start = start

                left_end = left_start + stem_length

                loop_start = left_end

                loop_end = loop_start + loop_length

                right_start = loop_end

                right_end = right_start + stem_length

                left = sequence[left_start:left_end]

                right = sequence[right_start:right_end]

                pair_scores = np.array(
                    [
                        phase1_pair_score(
                            left[pair_index], right[stem_length - 1 - pair_index]
                        )
                        for pair_index in range(stem_length)
                    ],
                    dtype=float,
                )

                paired_fraction = float(np.mean(pair_scores > 0))

                if paired_fraction < 0.75:

                    continue

                raw_pairing_score = float(pair_scores.sum())

                stem_pairing_score = float(
                    raw_pairing_score - 0.5 * abs(loop_length - 4)
                )

                key = (
                    stem_pairing_score,
                    paired_fraction,
                    stem_length,
                    -abs(loop_length - 4),
                )

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
                        "raw_pairing_score": float(raw_pairing_score),
                        "stem_pairing_score": float(stem_pairing_score),
                        "pairing_score": float(stem_pairing_score),
                        "pair_fraction": float(paired_fraction),
                    }

    return best


BIO_INDEX = {feature: index for index, feature in enumerate(bio_feature_names)}

geometry_mismatches = []

for sample_idx, sequence in enumerate(sequences_train):

    geometry = phase1_hairpin_geometry(sequence)

    expected = (
        float(X_bio_train[sample_idx, BIO_INDEX["stem_length"]]),
        float(X_bio_train[sample_idx, BIO_INDEX["loop_length"]]),
        float(X_bio_train[sample_idx, BIO_INDEX["stem_pairing_score"]]),
    )

    if geometry is None:

        observed = (
            0.0,
            0.0,
            0.0,
        )

    else:

        observed = (
            float(geometry["stem_length"]),
            float(geometry["loop_length"]),
            float(geometry["stem_pairing_score"]),
        )

    if not np.allclose(expected, observed, atol=1e-12, rtol=0):

        geometry_mismatches.append(
            {
                "sample_idx": int(sample_idx),
                "expected_stem_length": expected[0],
                "observed_stem_length": observed[0],
                "expected_loop_length": expected[1],
                "observed_loop_length": observed[1],
                "expected_stem_pairing_score": expected[2],
                "observed_stem_pairing_score": observed[2],
                "sequence": sequence,
            }
        )


if geometry_mismatches:

    display(pd.DataFrame(geometry_mismatches).head(20))

    raise RuntimeError("Hairpin geometry does not reproduce Phase 1.")


def normalize_nt_matrix(X):

    X = np.asarray(X, dtype=float)

    norms = np.linalg.norm(X, axis=1, keepdims=True)

    if (norms <= 0).any():

        raise RuntimeError("Zero-norm NT representation.")

    return X / norms


X_nt_train_norm = normalize_nt_matrix(X_nt_train)

nt_loo_matrix = pairwise_distances(X_nt_train_norm, metric="euclidean")

np.fill_diagonal(nt_loo_matrix, np.inf)

TRAIN_NT_NN_DISTANCE = nt_loo_matrix.min(axis=1)

NT_GLOBAL_SUPPORT_LIMIT = float(np.max(TRAIN_NT_NN_DISTANCE))

BIO_MEDIAN = np.median(X_bio_train, axis=0)

BIO_Q25 = np.percentile(X_bio_train, 25, axis=0)

BIO_Q75 = np.percentile(X_bio_train, 75, axis=0)

BIO_SCALE = BIO_Q75 - BIO_Q25

bio_sd = np.std(X_bio_train, axis=0, ddof=0)

zero_scale = BIO_SCALE <= 1e-12

BIO_SCALE[zero_scale] = bio_sd[zero_scale]

BIO_SCALE[BIO_SCALE <= 1e-12] = 1.0

X_bio_train_scaled = (X_bio_train - BIO_MEDIAN) / BIO_SCALE

bio_loo_matrix = pairwise_distances(X_bio_train_scaled, metric="euclidean")

np.fill_diagonal(bio_loo_matrix, np.inf)

TRAIN_BIO_NN_DISTANCE = bio_loo_matrix.min(axis=1)

BIO_GLOBAL_SUPPORT_LIMIT = float(np.max(TRAIN_BIO_NN_DISTANCE))

NT_SUPPORT_MEDIAN = float(np.median(TRAIN_NT_NN_DISTANCE))

BIO_SUPPORT_MEDIAN = float(np.median(TRAIN_BIO_NN_DISTANCE))

NT_SUPPORT_MAX_TO_MEDIAN = float(
    NT_GLOBAL_SUPPORT_LIMIT / max(NT_SUPPORT_MEDIAN, 1e-12)
)

BIO_SUPPORT_MAX_TO_MEDIAN = float(
    BIO_GLOBAL_SUPPORT_LIMIT / max(BIO_SUPPORT_MEDIAN, 1e-12)
)

TRAIN_SEQUENCE_LENGTHS = np.array(
    [len(sequence) for sequence in sequences_train], dtype=int
)

TRAIN_LENGTH_MIN = int(TRAIN_SEQUENCE_LENGTHS.min())

TRAIN_LENGTH_MAX = int(TRAIN_SEQUENCE_LENGTHS.max())

MODEL_ERROR_FLOOR_MODEL_Y = float(np.median(np.abs(model_y_train - model_y_oof_pred)))

MODEL_ERROR_FLOOR_TE = float(
    np.median(np.abs(te_train - model_y_to_te(model_y_oof_pred)))
)

GLOBAL_SUPPORT_STATE = {
    "nt_global_support_limit": NT_GLOBAL_SUPPORT_LIMIT,
    "bio_global_support_limit": BIO_GLOBAL_SUPPORT_LIMIT,
    "nt_median_training_loo_nn": NT_SUPPORT_MEDIAN,
    "bio_median_training_loo_nn": BIO_SUPPORT_MEDIAN,
    "nt_max_to_median_loo_ratio": NT_SUPPORT_MAX_TO_MEDIAN,
    "bio_max_to_median_loo_ratio": BIO_SUPPORT_MAX_TO_MEDIAN,
    "training_length_min": TRAIN_LENGTH_MIN,
    "training_length_max": TRAIN_LENGTH_MAX,
    "model_error_floor_model_y": MODEL_ERROR_FLOOR_MODEL_Y,
    "model_error_floor_te": MODEL_ERROR_FLOOR_TE,
    "support_definition": (
        "Maximum observed training leave-one-out " "nearest-neighbor distance."
    ),
    "support_caveat": (
        "The maximum leave-one-out nearest-neighbor "
        "threshold can be sensitive to training outliers."
    ),
    "phase1_hairpin_pairing_score_definition": (
        "raw canonical/wobble pair score " "- 0.5 * abs(loop_length - 4)"
    ),
    "phase1_hairpin_ranking_key": (
        "(stem_pairing_score, paired_fraction, "
        "stem_length, -abs(loop_length - 4)); "
        "strict > update; no explicit start-position tie breaker"
    ),
    "raw_pair_score_definition": {
        "GC_or_CG": 3.0,
        "AT_or_TA": 2.0,
        "GT_or_TG": 1.0,
        "other": 0.0,
    },
    "minimum_hairpin_pair_fraction": 0.75,
    "phase1_hairpin_stem_search": [
        4,
        13,
    ],
    "phase1_hairpin_loop_search": [
        3,
        8,
    ],
}


import json

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances


def silverman_bandwidth(values):

    values = np.asarray(values, dtype=float)

    n = len(values)

    sigma = float(np.std(values, ddof=1))

    q25, q75 = np.percentile(
        values,
        [
            25,
            75,
        ],
    )

    robust_sigma = float((q75 - q25) / 1.349)

    valid_scales = [
        value
        for value in [
            sigma,
            robust_sigma,
        ]
        if (np.isfinite(value) and value > 0)
    ]

    if not valid_scales:

        raise RuntimeError("Could not derive Silverman bandwidth.")

    scale = min(valid_scales)

    return float(0.9 * scale * (n ** (-1.0 / 5.0)))


TARGET_BANDWIDTH_MODEL_Y = silverman_bandwidth(model_y_train)


def weighted_median(values, weights):

    values = np.asarray(values, dtype=float)

    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)

    values = values[valid]

    weights = weights[valid]

    if len(values) == 0:

        return np.nan

    if weights.sum() <= 0:

        return float(np.median(values))

    order = np.argsort(values)

    values = values[order]

    weights = weights[order]

    cumulative = np.cumsum(weights)

    threshold = 0.5 * weights.sum()

    index = int(np.searchsorted(cumulative, threshold, side="left"))

    index = min(index, len(values) - 1)

    return float(values[index])


def weighted_mad(values, weights, center):

    return weighted_median(np.abs(np.asarray(values, dtype=float) - center), weights)


feature_evidence_lookup = df_feature_evidence.set_index("feature")[
    "feature_evidence"
].to_dict()

FEATURE_EVIDENCE_VECTOR = np.array(
    [float(feature_evidence_lookup.get(feature, 0.0)) for feature in bio_feature_names],
    dtype=float,
)

if FEATURE_EVIDENCE_VECTOR.sum() <= 0:

    raise RuntimeError("Feature evidence vector sums to zero.")

RULE_WEIGHT_GLOBAL = FEATURE_EVIDENCE_VECTOR / FEATURE_EVIDENCE_VECTOR.sum()


def adaptive_anchor_indices(candidate_indices, target_weights, target_radius):

    candidate_indices = np.asarray(candidate_indices, dtype=int)

    if len(candidate_indices) == 0:

        return []

    if len(candidate_indices) == 1:

        return [int(candidate_indices[0])]

    target_radius = max(float(target_radius), 1e-12)

    first_local = int(np.argmax(target_weights[candidate_indices]))

    selected = [int(candidate_indices[first_local])]

    candidate_X = X_nt_train_norm[candidate_indices]

    while True:

        selected_X = X_nt_train_norm[selected]

        distance_matrix = pairwise_distances(
            candidate_X, selected_X, metric="euclidean"
        )

        nearest_distance = distance_matrix.min(axis=1)

        worst_local = int(np.argmax(nearest_distance))

        worst_distance = float(nearest_distance[worst_local])

        if worst_distance <= target_radius:

            break

        candidate = int(candidate_indices[worst_local])

        if candidate in selected:

            break

        selected.append(candidate)

    return selected


def largest_remainder_allocation(total, weights):

    weights = np.asarray(weights, dtype=float)

    weights = np.clip(weights, 0, None)

    if weights.sum() <= 0:

        weights = np.ones_like(weights)

    probabilities = weights / weights.sum()

    expected = probabilities * int(total)

    allocation = np.floor(expected).astype(int)

    remaining = int(total) - int(allocation.sum())

    if remaining > 0:

        residual = expected - allocation

        order = np.argsort(residual)[::-1]

        for index in order[:remaining]:

            allocation[index] += 1

    if int(allocation.sum()) != int(total):

        raise RuntimeError("Budget allocation failed.")

    return allocation


TRAIN_GEOMETRIES = [phase1_hairpin_geometry(sequence) for sequence in sequences_train]

STRUCTURAL_TRAIN_INDICES = np.array(
    [index for index, geometry in enumerate(TRAIN_GEOMETRIES) if geometry is not None],
    dtype=int,
)

if len(STRUCTURAL_TRAIN_INDICES) == 0:

    raise RuntimeError("No structurally detected training terminators.")

TARGET_STATES = {}

all_plan_parts = []

for target_row_index, target_row in df_targets.iterrows():

    target_id = str(target_row["target_id"])

    target_te = float(target_row["target_te"])

    target_model_y = float(target_row["target_model_y"])

    target_average_strength = float(target_row["target_average_strength"])

    target_dir = TARGET_ROOT / target_id

    raw_weights = np.exp(
        -0.5 * ((model_y_train - target_model_y) / TARGET_BANDWIDTH_MODEL_Y) ** 2
    )

    if raw_weights.sum() <= 0:

        raise RuntimeError(f"Target kernel underflow: " f"{target_id}")

    target_weights = raw_weights / raw_weights.sum()

    target_indices = np.flatnonzero(
        np.abs(model_y_train - target_model_y) <= TARGET_BANDWIDTH_MODEL_Y
    )

    target_neighborhood_fallback = False

    if len(target_indices) == 0:

        nearest_distance = float(np.min(np.abs(model_y_train - target_model_y)))

        target_indices = np.flatnonzero(
            np.isclose(np.abs(model_y_train - target_model_y), nearest_distance)
        )

        target_neighborhood_fallback = True

    target_nt_radius = float(np.median(TRAIN_NT_NN_DISTANCE[target_indices]))

    rule_center = np.zeros(29, dtype=float)

    rule_scale = np.zeros(29, dtype=float)

    for feature_index in range(29):

        values = X_bio_train[:, feature_index]

        center = weighted_median(values, target_weights)

        scale = weighted_mad(values, target_weights, center)

        if not (np.isfinite(scale) and scale > 1e-12):

            global_iqr = float(np.percentile(values, 75) - np.percentile(values, 25))

            if global_iqr > 1e-12:

                scale = global_iqr

            else:

                global_sd = float(np.std(values))

                scale = global_sd if global_sd > 1e-12 else 1.0

        rule_center[feature_index] = center

        rule_scale[feature_index] = scale

    structural_target_indices = np.array(
        [index for index in target_indices if (TRAIN_GEOMETRIES[index] is not None)],
        dtype=int,
    )

    anchor_fallback = False

    if len(structural_target_indices) == 0:

        structural_distances = np.abs(
            model_y_train[STRUCTURAL_TRAIN_INDICES] - target_model_y
        )

        nearest_structural_distance = float(structural_distances.min())

        structural_target_indices = STRUCTURAL_TRAIN_INDICES[
            np.isclose(structural_distances, nearest_structural_distance)
        ]

        anchor_fallback = True

    anchor_indices = adaptive_anchor_indices(
        structural_target_indices, target_weights, target_nt_radius
    )

    if not anchor_indices:

        raise RuntimeError(f"No anchors found for " f"{target_id}")

    anchor_records = []

    for anchor_rank, sample_idx in enumerate(anchor_indices, start=1):

        sequence = sequences_train[sample_idx]

        geometry = TRAIN_GEOMETRIES[sample_idx]

        if geometry is None:

            continue

        prompt = sequence[: geometry["loop_end"]]

        anchor_records.append(
            {
                "anchor_rank": int(anchor_rank),
                "anchor_sample_idx": int(sample_idx),
                "anchor_model_y": float(model_y_train[sample_idx]),
                "anchor_te": float(te_train[sample_idx]),
                "anchor_kernel_weight": float(target_weights[sample_idx]),
                "prompt": prompt,
                "prompt_length": len(prompt),
                "stem_length": int(geometry["stem_length"]),
                "loop_length": int(geometry["loop_length"]),
            }
        )

    df_anchors = pd.DataFrame(anchor_records)

    if len(df_anchors) == 0:

        raise RuntimeError(f"No valid structural prompt for " f"{target_id}")

    allocation = largest_remainder_allocation(
        EVO_SAMPLES_PER_TARGET, df_anchors["anchor_kernel_weight"].to_numpy(dtype=float)
    )

    df_anchors["planned_samples"] = allocation

    rng = np.random.default_rng(GLOBAL_RANDOM_SEED + int(target_row_index) * 10000)

    job_records = []

    for _, anchor in df_anchors.iterrows():

        n_samples = int(anchor["planned_samples"])

        if n_samples <= 0:

            continue

        prompt = str(anchor["prompt"])

        prompt_length = int(anchor["prompt_length"])

        eligible_length_indices = target_indices[
            TRAIN_SEQUENCE_LENGTHS[target_indices] > prompt_length
        ]

        length_fallback = False

        if len(eligible_length_indices) == 0:

            eligible_length_indices = np.flatnonzero(
                TRAIN_SEQUENCE_LENGTHS > prompt_length
            )

            length_fallback = True

        if len(eligible_length_indices) == 0:

            raise RuntimeError(f"No empirical length > prompt " f"for {target_id}")

        length_weights = target_weights[eligible_length_indices]

        if length_weights.sum() <= 0:

            length_weights = np.ones(len(eligible_length_indices), dtype=float)

        length_probabilities = length_weights / length_weights.sum()

        sampled_reference_indices = rng.choice(
            eligible_length_indices,
            size=n_samples,
            replace=True,
            p=length_probabilities,
        )

        sampled_target_lengths = TRAIN_SEQUENCE_LENGTHS[sampled_reference_indices]

        unique_lengths, counts = np.unique(sampled_target_lengths, return_counts=True)

        for target_length, count in zip(unique_lengths, counts):

            target_length = int(target_length)

            n_tokens = target_length - prompt_length

            if n_tokens <= 0:

                continue

            job_records.append(
                {
                    "target_id": target_id,
                    "target_te": target_te,
                    "target_average_strength": target_average_strength,
                    "target_model_y": target_model_y,
                    "design_regime": str(target_row["design_regime"]),
                    "anchor_rank": int(anchor["anchor_rank"]),
                    "anchor_sample_idx": int(anchor["anchor_sample_idx"]),
                    "prompt": prompt,
                    "prompt_length": prompt_length,
                    "target_length": target_length,
                    "n_tokens": int(n_tokens),
                    "n_samples": int(count),
                    "prompt_mode": "through_loop_end",
                    "length_fallback_used": bool(length_fallback),
                }
            )

    df_plan = pd.DataFrame(job_records)

    if int(df_plan["n_samples"].sum()) != EVO_SAMPLES_PER_TARGET:

        raise RuntimeError(f"Evo budget mismatch for " f"{target_id}")

    df_reference = pd.DataFrame(
        {
            "sample_idx": np.arange(n_train, dtype=int),
            "measured_model_y": model_y_train,
            "reconstructed_te": te_train,
            "target_kernel_weight": target_weights,
            "in_target_neighborhood": np.isin(np.arange(n_train), target_indices),
        }
    )

    df_rule_profile = pd.DataFrame(
        {
            "feature": bio_feature_names,
            "target_center": rule_center,
            "target_scale": rule_scale,
            "rule_weight": RULE_WEIGHT_GLOBAL,
            "feature_evidence": FEATURE_EVIDENCE_VECTOR,
        }
    )

    target_state_json = {
        "target_id": target_id,
        "target_te": target_te,
        "target_average_strength": target_average_strength,
        "target_model_y": target_model_y,
        "design_regime": str(target_row["design_regime"]),
        "target_bandwidth_model_y": TARGET_BANDWIDTH_MODEL_Y,
        "target_neighborhood_n": int(len(target_indices)),
        "target_neighborhood_fallback_used": bool(target_neighborhood_fallback),
        "target_nt_radius": target_nt_radius,
        "anchor_fallback_used": bool(anchor_fallback),
        "adaptive_anchor_n": int(len(df_anchors)),
        "planned_evo_samples": int(df_plan["n_samples"].sum()),
        "prompt_mode": "through_loop_end",
    }

    TARGET_STATES[target_id] = {
        **target_state_json,
        "target_indices": np.asarray(target_indices, dtype=int),
        "target_weights": np.asarray(target_weights, dtype=float),
        "rule_center": rule_center,
        "rule_scale": rule_scale,
        "rule_weight": RULE_WEIGHT_GLOBAL.copy(),
    }

    all_plan_parts.append(df_plan)

df_evo_plan_all = pd.concat(all_plan_parts, ignore_index=True)

df_evo_plan_all.insert(0, "job_id", np.arange(len(df_evo_plan_all), dtype=int))


for target_id, group in df_evo_plan_all.groupby("target_id"):

    pass


import json

import numpy as np
import pandas as pd

df_targets = pd.read_csv(TARGET_SPEC_PATH)


def parse_boolean_series(series):

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
            ]
        )
    )


TARGET_STATES = {}

for _, target_row in df_targets.iterrows():

    target_id = str(target_row["target_id"])

    target_dir = TARGET_ROOT / target_id

    reference_path = target_dir / "target_reference.csv"

    rule_path = target_dir / "target_rule_profile.csv"

    state_path = target_dir / "target_state.json"

    for path in [
        reference_path,
        rule_path,
        state_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(path)

    df_reference = pd.read_csv(reference_path)

    df_rule = pd.read_csv(rule_path)

    with open(state_path, "r") as f:

        state_json = json.load(f)

    neighborhood_mask = parse_boolean_series(df_reference["in_target_neighborhood"])

    target_indices = pd.to_numeric(
        df_reference.loc[neighborhood_mask, "sample_idx"], errors="raise"
    ).to_numpy(dtype=int)

    target_weights = pd.to_numeric(
        df_reference["target_kernel_weight"], errors="raise"
    ).to_numpy(dtype=float)

    rule_lookup = df_rule.set_index("feature")

    rule_center = np.array(
        [
            float(rule_lookup.loc[feature, "target_center"])
            for feature in bio_feature_names
        ],
        dtype=float,
    )

    rule_scale = np.array(
        [
            float(rule_lookup.loc[feature, "target_scale"])
            for feature in bio_feature_names
        ],
        dtype=float,
    )

    rule_weight = np.array(
        [
            float(rule_lookup.loc[feature, "rule_weight"])
            for feature in bio_feature_names
        ],
        dtype=float,
    )

    TARGET_STATES[target_id] = {
        **state_json,
        "target_indices": target_indices,
        "target_weights": target_weights,
        "rule_center": rule_center,
        "rule_scale": rule_scale,
        "rule_weight": rule_weight,
    }


if len(TARGET_STATES) != len(df_targets):

    raise RuntimeError("Target-state count mismatch.")


if not EVO_RAW_PATH.exists():

    raise FileNotFoundError(EVO_RAW_PATH)

if not EVO_PLAN_PATH.exists():

    raise FileNotFoundError(EVO_PLAN_PATH)

df_evo_raw = pd.read_csv(EVO_RAW_PATH)

df_evo_plan = pd.read_csv(EVO_PLAN_PATH)

planned_by_target = df_evo_plan.groupby("target_id")["n_samples"].sum()

observed_by_target = df_evo_raw.groupby("target_id").size()

for target_id in df_targets["target_id"].astype(str):

    planned = int(planned_by_target.get(target_id, 0))

    observed = int(observed_by_target.get(target_id, 0))

    if observed != planned:

        raise RuntimeError(
            f"{target_id}: " f"planned={planned}, " f"observed={observed}"
        )

DNA = {
    "A",
    "C",
    "G",
    "T",
}

df_evo_raw["sequence"] = df_evo_raw["sequence"].astype(str).str.upper()

df_evo_raw["prompt"] = df_evo_raw["prompt"].astype(str).str.upper()

df_evo_raw["qc_sequence_length"] = df_evo_raw["sequence"].str.len()

df_evo_raw["qc_valid_sequence_dna"] = df_evo_raw["sequence"].map(
    lambda sequence: (len(sequence) > 0 and set(sequence).issubset(DNA))
)

df_evo_raw["qc_prompt_prefix"] = [
    sequence.startswith(prompt)
    for sequence, prompt in zip(df_evo_raw["sequence"], df_evo_raw["prompt"])
]

inferred_continuations = []

for sequence, prompt in zip(df_evo_raw["sequence"], df_evo_raw["prompt"]):

    if sequence.startswith(prompt):

        inferred_continuations.append(sequence[len(prompt) :])

    else:

        inferred_continuations.append("")

df_evo_raw["qc_inferred_continuation"] = inferred_continuations

df_evo_raw["qc_continuation_length"] = df_evo_raw["qc_inferred_continuation"].str.len()

df_evo_raw["qc_valid_continuation_dna"] = df_evo_raw["qc_inferred_continuation"].map(
    lambda continuation: (len(continuation) > 0 and set(continuation).issubset(DNA))
)

df_evo_raw["qc_target_length_match"] = df_evo_raw[
    "qc_sequence_length"
] == pd.to_numeric(df_evo_raw["target_length"], errors="raise")

df_evo_raw["qc_requested_continuation_length_match"] = df_evo_raw[
    "qc_continuation_length"
] == pd.to_numeric(df_evo_raw["n_tokens"], errors="raise")

df_evo_raw["qc_generation_pass"] = (
    df_evo_raw["qc_valid_sequence_dna"]
    & df_evo_raw["qc_prompt_prefix"]
    & df_evo_raw["qc_valid_continuation_dna"]
    & df_evo_raw["qc_target_length_match"]
    & df_evo_raw["qc_requested_continuation_length_match"]
)


def rejection_reason(row):

    reasons = []

    if not row["qc_valid_sequence_dna"]:

        reasons.append("non_ACGT_sequence")

    if not row["qc_prompt_prefix"]:

        reasons.append("prompt_prefix_mismatch")

    if not row["qc_valid_continuation_dna"]:

        reasons.append("invalid_or_empty_continuation")

    if not row["qc_target_length_match"]:

        reasons.append("target_length_mismatch")

    if not row["qc_requested_continuation_length_match"]:

        reasons.append("continuation_length_mismatch")

    return "PASS" if not reasons else ";".join(reasons)


df_evo_raw["qc_rejection_reason"] = df_evo_raw.apply(rejection_reason, axis=1)

df_evo_valid = (
    df_evo_raw.loc[df_evo_raw["qc_generation_pass"]].copy().reset_index(drop=True)
)

df_evo_rejected = (
    df_evo_raw.loc[~df_evo_raw["qc_generation_pass"]].copy().reset_index(drop=True)
)

df_evo_valid_unique = (
    df_evo_valid.sort_values(
        [
            "target_id",
            "job_id",
            "sample_in_job",
        ]
    )
    .drop_duplicates(
        subset=[
            "target_id",
            "sequence",
        ],
        keep="first",
    )
    .reset_index(drop=True)
)


qc_summary_rows = []

for target_id in df_targets["target_id"].astype(str):

    raw_n = int((df_evo_raw["target_id"] == target_id).sum())

    valid_n = int((df_evo_valid["target_id"] == target_id).sum())

    unique_n = int((df_evo_valid_unique["target_id"] == target_id).sum())

    qc_summary_rows.append(
        {
            "target_id": target_id,
            "attempts": raw_n,
            "format_valid": valid_n,
            "rejected": raw_n - valid_n,
            "valid_unique": unique_n,
        }
    )

df_evo_qc_summary = pd.DataFrame(qc_summary_rows)


display(df_evo_qc_summary)


import gc

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

NT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

NT_BATCH_SIZE = 32

NT_CONSISTENCY_TOL = 1e-5


gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()

nt_tokenizer = AutoTokenizer.from_pretrained(NT_MODEL_NAME, trust_remote_code=True)

nt_mlm_model = (
    AutoModelForMaskedLM.from_pretrained(NT_MODEL_NAME, trust_remote_code=True)
    .to(NT_DEVICE)
    .eval()
)

nt_backbone_model = None

if hasattr(nt_mlm_model, "base_model"):

    candidate_backbone = nt_mlm_model.base_model

    if candidate_backbone is not nt_mlm_model:

        nt_backbone_model = candidate_backbone.to(NT_DEVICE).eval()


def tokenize_nt_batch(sequences):

    sequences = [str(sequence).upper().strip() for sequence in sequences]

    encoded = nt_tokenizer(
        sequences, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )

    input_ids = encoded["input_ids"]

    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for special_id in nt_tokenizer.all_special_ids:

        if special_id is None:

            continue

        special_mask |= input_ids == int(special_id)

    if not (input_ids.shape == attention_mask.shape == special_mask.shape):

        raise RuntimeError("NT token-mask shape mismatch.")

    return (input_ids, attention_mask, special_mask)


def nt_embed_with_runner(sequences, runner, batch_size=NT_BATCH_SIZE):

    sequences = list(sequences)

    all_embeddings = []

    for start in range(0, len(sequences), batch_size):

        batch = sequences[start : start + batch_size]

        (
            input_ids,
            attention_mask,
            special_mask,
        ) = tokenize_nt_batch(batch)

        input_ids = input_ids.to(NT_DEVICE)

        attention_mask = attention_mask.to(NT_DEVICE)

        special_mask = special_mask.to(NT_DEVICE)

        with torch.inference_mode():

            outputs = runner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        if hidden_states is None:

            raise RuntimeError("NT did not return hidden states.")

        if hidden_states[-1].shape[:2] != input_ids.shape:

            raise RuntimeError("NT hidden-state/token shape mismatch.")

        intermediate_repr = torch.stack(
            [hidden_states[int(layer_id)] for layer_id in MID_LAYER_IDS], dim=0
        ).mean(dim=0)

        final_repr = hidden_states[-1]

        valid_mask = attention_mask.bool() & (~special_mask.bool())

        valid_count = valid_mask.sum(dim=1, keepdim=True)

        if (valid_count == 0).any():

            raise RuntimeError("NT sequence has zero valid tokens.")

        mask_float = valid_mask.unsqueeze(-1).to(intermediate_repr.dtype)

        pooled_intermediate = (intermediate_repr * mask_float).sum(
            dim=1
        ) / valid_count.to(intermediate_repr.dtype)

        pooled_final = (final_repr * mask_float).sum(dim=1) / valid_count.to(
            final_repr.dtype
        )

        embedding = torch.cat(
            [
                pooled_intermediate,
                pooled_final,
            ],
            dim=1,
        )

        all_embeddings.append(embedding.float().cpu().numpy())

    X = np.vstack(all_embeddings)

    if X.shape[1] != 2048:

        raise RuntimeError(f"Unexpected NT dimension: " f"{X.shape}")

    return X


nt_qc_indices = np.unique(
    np.array(
        [
            0,
            len(sequences_train) // 2,
            len(sequences_train) - 1,
        ],
        dtype=int,
    )
)

qc_sequences = sequences_train[nt_qc_indices]

qc_reference = X_nt_train[nt_qc_indices]

NT_INFERENCE_MODE = None

NT_CONSISTENCY_MAX_ABS_ERROR = None

if nt_backbone_model is not None:

    try:

        qc_backbone = nt_embed_with_runner(
            qc_sequences, nt_backbone_model, batch_size=len(qc_sequences)
        )

        backbone_error = float(np.max(np.abs(qc_backbone - qc_reference)))

        if backbone_error <= NT_CONSISTENCY_TOL:

            NT_INFERENCE_MODE = "backbone"

            NT_CONSISTENCY_MAX_ABS_ERROR = backbone_error

    except Exception as exc:

        pass

if NT_INFERENCE_MODE is None:

    qc_full = nt_embed_with_runner(
        qc_sequences, nt_mlm_model, batch_size=len(qc_sequences)
    )

    full_error = float(np.max(np.abs(qc_full - qc_reference)))

    if full_error <= NT_CONSISTENCY_TOL:

        NT_INFERENCE_MODE = "full_mlm"

        NT_CONSISTENCY_MAX_ABS_ERROR = full_error

if NT_INFERENCE_MODE is None:

    raise RuntimeError(
        "NT representation consistency failed. " "Do not continue Phase 3."
    )


def nt_embed_sequences(sequences, batch_size=NT_BATCH_SIZE):

    if NT_INFERENCE_MODE == "backbone":

        runner = nt_backbone_model

    else:

        runner = nt_mlm_model

    return nt_embed_with_runner(sequences, runner, batch_size=batch_size)


import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

NT_CACHE = {}

BIO_CACHE = {}

MODEL_Y_PREDICTION_CACHE = {}


def get_nt_cached(sequences):

    sequences = [str(sequence).upper().strip() for sequence in sequences]

    missing = [
        sequence for sequence in dict.fromkeys(sequences) if sequence not in NT_CACHE
    ]

    if missing:

        X_new = nt_embed_sequences(missing)

        for sequence, vector in zip(missing, X_new):

            NT_CACHE[sequence] = np.asarray(vector, dtype=float)

    return np.vstack([NT_CACHE[sequence] for sequence in sequences])


def get_bio_cached(sequences):

    sequences = [str(sequence).upper().strip() for sequence in sequences]

    missing = [
        sequence for sequence in dict.fromkeys(sequences) if sequence not in BIO_CACHE
    ]

    if missing:

        X_new = extract_bio_sequences(missing)

        for sequence, vector in zip(missing, X_new):

            BIO_CACHE[sequence] = np.asarray(vector, dtype=float)

    return np.vstack([BIO_CACHE[sequence] for sequence in sequences])


def get_model_y_prediction_matrix_cached(sequences):

    sequences = [str(sequence).upper().strip() for sequence in sequences]

    missing = [
        sequence
        for sequence in dict.fromkeys(sequences)
        if (sequence not in MODEL_Y_PREDICTION_CACHE)
    ]

    if missing:

        X_nt = get_nt_cached(missing)

        X_bio = get_bio_cached(missing)

        X_raw = np.hstack(
            [
                X_nt,
                X_bio,
            ]
        )

        prediction_matrix = np.vstack(
            [
                np.asarray(model.predict(X_raw), dtype=float).reshape(-1)
                for model in MODELS_50
            ]
        )

        if prediction_matrix.shape != (50, len(missing)):

            raise RuntimeError(
                f"Unexpected prediction matrix: " f"{prediction_matrix.shape}"
            )

        for column_index, sequence in enumerate(missing):

            MODEL_Y_PREDICTION_CACHE[sequence] = prediction_matrix[:, column_index]

    return np.column_stack(
        [MODEL_Y_PREDICTION_CACHE[sequence] for sequence in sequences]
    )


def structure_supported(sequence, bio_vector):

    length_ok = TRAIN_LENGTH_MIN <= len(sequence) <= TRAIN_LENGTH_MAX

    stem_length = float(bio_vector[BIO_INDEX["stem_length"]])

    spacer_length = float(bio_vector[BIO_INDEX["spacer_length"]])

    hairpin_ok = stem_length > 0 and spacer_length != 15

    return bool(length_ok and hairpin_ok)


def evaluate_sequences_for_target(target_id, sequences):

    if target_id not in TARGET_STATES:

        raise KeyError(target_id)

    state = TARGET_STATES[target_id]

    sequences = [str(sequence).upper().strip() for sequence in sequences]

    if len(sequences) == 0:

        return pd.DataFrame()

    for sequence in sequences:

        if len(sequence) == 0 or not set(sequence).issubset(
            {
                "A",
                "C",
                "G",
                "T",
            }
        ):

            raise ValueError("Invalid DNA passed to evaluator.")

    X_nt = get_nt_cached(sequences)

    X_bio = get_bio_cached(sequences)

    model_y_matrix = get_model_y_prediction_matrix_cached(sequences)

    pred_model_y = model_y_matrix.mean(axis=0)

    pred_model_y_sd = model_y_matrix.std(axis=0, ddof=0)

    pred_te = model_y_to_te(pred_model_y)

    te_matrix = model_y_to_te(model_y_matrix)

    pred_te_member_mean = te_matrix.mean(axis=0)

    pred_te_member_sd = te_matrix.std(axis=0, ddof=0)

    target_model_y = float(state["target_model_y"])

    target_te = float(state["target_te"])

    target_error_model_y = np.abs(pred_model_y - target_model_y)

    target_error_te = np.abs(pred_te - target_te)

    X_nt_norm = normalize_nt_matrix(X_nt)

    X_bio_scaled = (X_bio - BIO_MEDIAN) / BIO_SCALE

    nt_global_distance = pairwise_distances(
        X_nt_norm, X_nt_train_norm, metric="euclidean"
    ).min(axis=1)

    bio_global_distance = pairwise_distances(
        X_bio_scaled, X_bio_train_scaled, metric="euclidean"
    ).min(axis=1)

    target_indices = state["target_indices"]

    nt_target_distance = pairwise_distances(
        X_nt_norm, X_nt_train_norm[target_indices], metric="euclidean"
    ).min(axis=1)

    bio_target_distance = pairwise_distances(
        X_bio_scaled, X_bio_train_scaled[target_indices], metric="euclidean"
    ).min(axis=1)

    rule_distance = (
        (np.abs(X_bio - state["rule_center"]) / state["rule_scale"])
        * state["rule_weight"]
    ).sum(axis=1)

    structure_ok = np.array(
        [
            structure_supported(sequence, bio_vector)
            for sequence, bio_vector in zip(sequences, X_bio)
        ],
        dtype=bool,
    )

    nt_supported = nt_global_distance <= NT_GLOBAL_SUPPORT_LIMIT

    bio_supported = bio_global_distance <= BIO_GLOBAL_SUPPORT_LIMIT

    global_support = structure_ok & nt_supported & bio_supported

    return pd.DataFrame(
        {
            "target_id": target_id,
            "target_te": target_te,
            "target_average_strength": float(state["target_average_strength"]),
            "target_model_y": target_model_y,
            "sequence": sequences,
            "sequence_length": [len(sequence) for sequence in sequences],
            "pred_model_y": pred_model_y,
            "pred_model_y_sd": pred_model_y_sd,
            "pred_average_strength": model_y_to_average_strength(pred_model_y),
            "pred_te": pred_te,
            "pred_te_member_mean": pred_te_member_mean,
            "pred_te_member_sd": pred_te_member_sd,
            "target_error_model_y": target_error_model_y,
            "target_error_te": target_error_te,
            "nt_global_distance": nt_global_distance,
            "bio_global_distance": bio_global_distance,
            "nt_target_distance": nt_target_distance,
            "bio_target_distance": bio_target_distance,
            "rule_distance": rule_distance,
            "structure_supported": structure_ok,
            "nt_global_supported": nt_supported,
            "bio_global_supported": bio_supported,
            "global_support": global_support,
        }
    )


smoke_target = None
smoke_sequences = None

for target_id in df_targets["target_id"].astype(str):

    candidates = (
        df_evo_valid_unique.loc[
            df_evo_valid_unique["target_id"] == target_id, "sequence"
        ]
        .head(2)
        .tolist()
    )

    if candidates:

        smoke_target = target_id

        smoke_sequences = candidates

        break

if smoke_target is None:

    raise RuntimeError("No Evo sequence available " "for evaluator smoke test.")

df_smoke = evaluate_sequences_for_target(smoke_target, smoke_sequences)

display(
    df_smoke[
        [
            "target_te",
            "target_model_y",
            "pred_model_y",
            "pred_te",
            "target_error_model_y",
            "target_error_te",
            "global_support",
        ]
    ]
)


import numpy as np
import pandas as pd

PARETO_OBJECTIVES = [
    "target_error_model_y",
    "pred_model_y_sd",
    "nt_target_distance",
    "bio_target_distance",
    "rule_distance",
]


def dominates(a, b):

    return np.all(a <= b) and np.any(a < b)


def nondominated_sort(X):

    X = np.asarray(X, dtype=float)

    n = len(X)

    domination_count = np.zeros(n, dtype=int)

    dominated = [[] for _ in range(n)]

    for p in range(n):

        for q in range(p + 1, n):

            if dominates(X[p], X[q]):

                dominated[p].append(q)

                domination_count[q] += 1

            elif dominates(X[q], X[p]):

                dominated[q].append(p)

                domination_count[p] += 1

    first_front = [index for index in range(n) if (domination_count[index] == 0)]

    fronts = [first_front]

    ranks = np.full(n, -1, dtype=int)

    for index in first_front:

        ranks[index] = 0

    current_front = 0

    while current_front < len(fronts):

        next_front = []

        for p in fronts[current_front]:

            for q in dominated[p]:

                domination_count[q] -= 1

                if domination_count[q] == 0:

                    ranks[q] = current_front + 1

                    next_front.append(q)

        if next_front:

            fronts.append(next_front)

        current_front += 1

    return (ranks, fronts)


def crowding_distance(X, fronts):

    X = np.asarray(X, dtype=float)

    crowding = np.zeros(len(X), dtype=float)

    for front in fronts:

        front = np.asarray(front, dtype=int)

        if len(front) == 0:

            continue

        if len(front) <= 2:

            crowding[front] = np.inf

            continue

        local_X = X[front]

        local_distance = np.zeros(len(front), dtype=float)

        for objective_index in range(X.shape[1]):

            order = np.argsort(local_X[:, objective_index])

            minimum = float(local_X[order[0], objective_index])

            maximum = float(local_X[order[-1], objective_index])

            local_distance[order[0]] = np.inf

            local_distance[order[-1]] = np.inf

            denominator = maximum - minimum

            if denominator <= 0:

                continue

            for k in range(1, len(order) - 1):

                index = order[k]

                if np.isinf(local_distance[index]):

                    continue

                local_distance[index] += (
                    local_X[order[k + 1], objective_index]
                    - local_X[order[k - 1], objective_index]
                ) / denominator

        crowding[front] = local_distance

    return crowding


def annotate_pareto(df):

    result = df.copy().reset_index(drop=True)

    if len(result) == 0:

        result["pareto_rank"] = pd.Series(dtype=int)

        result["crowding_distance"] = pd.Series(dtype=float)

        return result

    X = result[PARETO_OBJECTIVES].to_numpy(dtype=float)

    if not np.all(np.isfinite(X)):

        raise RuntimeError("Non-finite Pareto objective.")

    ranks, fronts = nondominated_sort(X)

    result["pareto_rank"] = ranks

    result["crowding_distance"] = crowding_distance(X, fronts)

    return result


def sort_pareto(df):

    result = annotate_pareto(df)

    if len(result) == 0:

        return result

    return result.sort_values(
        [
            "pareto_rank",
            "crowding_distance",
            "target_error_model_y",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)


def diversity_select_by_nt(df, minimum_distance, max_n):

    if len(df) == 0:

        return df.copy()

    ordered = sort_pareto(df)

    selected_rows = []

    selected_vectors = []

    for _, row in ordered.iterrows():

        sequence = str(row["sequence"])

        vector = normalize_nt_matrix(get_nt_cached([sequence]))[0]

        if not selected_vectors:

            keep = True

        else:

            existing = np.vstack(selected_vectors)

            distance = np.linalg.norm(existing - vector[None, :], axis=1)

            keep = bool(np.all(distance >= minimum_distance))

        if keep:

            selected_rows.append(row.copy())

            selected_vectors.append(vector)

        if len(selected_rows) >= max_n:

            break

    if not selected_rows:

        return ordered.iloc[:0].copy()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


import numpy as np
import pandas as pd

INITIAL_POPULATIONS = {}

EVO_EVALUATED = {}

for target_id in df_targets["target_id"].astype(str):

    ga_dir = GA_ROOT / target_id

    df_target_evo = (
        df_evo_valid_unique.loc[df_evo_valid_unique["target_id"] == target_id]
        .copy()
        .reset_index(drop=True)
    )

    sequences = df_target_evo["sequence"].tolist()

    df_eval = evaluate_sequences_for_target(target_id, sequences)

    df_evaluated = df_eval.merge(
        df_target_evo,
        on=[
            "target_id",
            "sequence",
        ],
        how="left",
        suffixes=("", "_evo"),
    )

    df_evaluated["source"] = "evo"

    df_evaluated["generation"] = 0

    EVO_EVALUATED[target_id] = df_evaluated

    supported = (
        df_evaluated.loc[df_evaluated["global_support"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )

    target_radius = float(TARGET_STATES[target_id]["target_nt_radius"])

    initial = diversity_select_by_nt(
        supported, minimum_distance=target_radius, max_n=MAX_GA_POPULATION
    )

    diversity_fallback_used = False

    if len(initial) < 2:

        state = TARGET_STATES[target_id]

        target_indices = state["target_indices"]

        target_weights = state["target_weights"]

        order = target_indices[np.argsort(target_weights[target_indices])[::-1]]

        natural_sequences = sequences_train[order].tolist()

        natural_eval = evaluate_sequences_for_target(target_id, natural_sequences)

        natural_eval["source"] = "natural_target_reference"

        natural_eval["generation"] = 0

        natural_eval = natural_eval.loc[
            natural_eval["global_support"].astype(bool)
        ].copy()

        combined = pd.concat(
            [
                initial,
                natural_eval,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = combined.drop_duplicates(subset=["sequence"], keep="first")

        initial = diversity_select_by_nt(
            combined, minimum_distance=target_radius, max_n=MAX_GA_POPULATION
        )

        if len(initial) < 2:

            diversity_fallback_used = True

            initial = sort_pareto(combined).head(2).reset_index(drop=True)

    if len(initial) < 2:

        raise RuntimeError(
            f"{target_id}: " "could not construct a viable " "initial GA population."
        )

    initial = sort_pareto(initial)

    initial["generation"] = 0

    initial["initial_diversity_fallback_used"] = diversity_fallback_used

    INITIAL_POPULATIONS[target_id] = initial


import numpy as np

DNA_BASES = np.array(
    [
        "A",
        "C",
        "G",
        "T",
    ]
)

PAIR_OPTIONS = [
    ("G", "C"),
    ("C", "G"),
    ("A", "T"),
    ("T", "A"),
    ("G", "T"),
    ("T", "G"),
]


def target_rule_distance(target_id, sequences):

    state = TARGET_STATES[target_id]

    X = get_bio_cached(sequences)

    return (
        (np.abs(X - state["rule_center"]) / state["rule_scale"]) * state["rule_weight"]
    ).sum(axis=1)


def best_rule_candidate(target_id, candidates):

    candidates = list(dict.fromkeys(candidates))

    if not candidates:

        return None

    distance = target_rule_distance(target_id, candidates)

    return candidates[int(np.argmin(distance))]


def choose_ism_guided_candidate(target_id, records, parent_pred_model_y):

    if not records:

        return None

    target_model_y = float(TARGET_STATES[target_id]["target_model_y"])

    desired_delta = target_model_y - float(parent_pred_model_y)

    scored = []

    for record in records:

        key = (
            int(record["relative_position"]),
            str(record["alt"]).upper(),
        )

        ism_delta = ISM_LOOKUP.get(key)

        if ism_delta is not None:

            scored.append(
                {
                    **record,
                    "ism_delta_model_y": float(ism_delta),
                }
            )

    if scored and not np.isclose(desired_delta, 0):

        desired_sign = np.sign(desired_delta)

        direction_matching = [
            record
            for record in scored
            if (desired_sign * record["ism_delta_model_y"]) > 0
        ]

        if direction_matching:

            strongest = max(
                abs(record["ism_delta_model_y"]) for record in direction_matching
            )

            strongest_sequences = [
                record["sequence"]
                for record in direction_matching
                if np.isclose(abs(record["ism_delta_model_y"]), strongest)
            ]

            return best_rule_candidate(target_id, strongest_sequences)

    return best_rule_candidate(target_id, [record["sequence"] for record in records])


def mutate_paired_stem(target_id, sequence):

    geometry = phase1_hairpin_geometry(sequence)

    if geometry is None:

        return None

    candidates = []

    for pair_index in range(geometry["stem_length"]):

        left_position = geometry["left_start"] + pair_index

        right_position = geometry["right_end"] - 1 - pair_index

        for left_base, right_base in PAIR_OPTIONS:

            if (
                sequence[left_position] == left_base
                and sequence[right_position] == right_base
            ):

                continue

            chars = list(sequence)

            chars[left_position] = left_base

            chars[right_position] = right_base

            candidates.append("".join(chars))

    return best_rule_candidate(target_id, candidates)


def mutate_loop(target_id, sequence):

    geometry = phase1_hairpin_geometry(sequence)

    if geometry is None:

        return None

    candidates = []

    for position in range(geometry["loop_start"], geometry["loop_end"]):

        current = sequence[position]

        for alt in DNA_BASES:

            alt = str(alt)

            if alt == current:

                continue

            chars = list(sequence)

            chars[position] = alt

            candidates.append("".join(chars))

    return best_rule_candidate(target_id, candidates)


def mutate_polyT(target_id, sequence, parent_pred_model_y):

    geometry = phase1_hairpin_geometry(sequence)

    if geometry is None:

        return None

    relative_anchor = geometry["right_end"] - 1

    records = []

    for position in range(geometry["right_end"], len(sequence)):

        relative_position = position - relative_anchor

        current = sequence[position]

        for alt in DNA_BASES:

            alt = str(alt)

            if alt == current:

                continue

            chars = list(sequence)

            chars[position] = alt

            records.append(
                {
                    "sequence": "".join(chars),
                    "relative_position": relative_position,
                    "alt": alt,
                }
            )

    return choose_ism_guided_candidate(target_id, records, parent_pred_model_y)


def mutate_ism_guided(target_id, sequence, parent_pred_model_y):

    geometry = phase1_hairpin_geometry(sequence)

    if geometry is None:

        return None

    relative_anchor = geometry["right_end"] - 1

    records = []

    for relative_position in ISM_RELATIVE_POSITIONS:

        position = relative_anchor + int(relative_position)

        if not (0 <= position < len(sequence)):

            continue

        current = sequence[position]

        for alt in DNA_BASES:

            alt = str(alt)

            if alt == current:

                continue

            chars = list(sequence)

            chars[position] = alt

            records.append(
                {
                    "sequence": "".join(chars),
                    "relative_position": int(relative_position),
                    "alt": alt,
                }
            )

    return choose_ism_guided_candidate(target_id, records, parent_pred_model_y)


def mutate_random(sequence, rng):

    if len(sequence) == 0:

        return None

    position = int(rng.integers(0, len(sequence)))

    current = sequence[position]

    alternatives = [str(base) for base in DNA_BASES if str(base) != current]

    alt = str(rng.choice(alternatives))

    chars = list(sequence)

    chars[position] = alt

    return "".join(chars)


OPERATOR_NAMES = np.array(list(OPERATOR_WEIGHTS.keys()), dtype=object)

OPERATOR_PROBABILITIES = np.array(
    [OPERATOR_WEIGHTS[operator] for operator in OPERATOR_NAMES], dtype=float
)

OPERATOR_PROBABILITIES = OPERATOR_PROBABILITIES / OPERATOR_PROBABILITIES.sum()


def mutate_sequence(target_id, sequence, parent_pred_model_y, rng):

    operator = str(rng.choice(OPERATOR_NAMES, p=OPERATOR_PROBABILITIES))

    if operator == "paired_stem":

        child = mutate_paired_stem(target_id, sequence)

    elif operator == "loop":

        child = mutate_loop(target_id, sequence)

    elif operator == "polyT":

        child = mutate_polyT(target_id, sequence, parent_pred_model_y)

    elif operator == "ism_guided":

        child = mutate_ism_guided(target_id, sequence, parent_pred_model_y)

    elif operator == "random":

        child = mutate_random(sequence, rng)

    else:

        raise RuntimeError(f"Unknown operator: " f"{operator}")

    return (child, operator)


for operator, probability in OPERATOR_WEIGHTS.items():

    pass


def structural_crossover(sequence_a, sequence_b):

    geometry_a = phase1_hairpin_geometry(sequence_a)

    geometry_b = phase1_hairpin_geometry(sequence_b)

    if geometry_a is None or geometry_b is None:

        return None

    child = (
        sequence_a[: geometry_a["right_end"]] + sequence_b[geometry_b["right_end"] :]
    )

    if not (TRAIN_LENGTH_MIN <= len(child) <= TRAIN_LENGTH_MAX):

        return None

    if phase1_hairpin_geometry(child) is None:

        return None

    return child


import numpy as np
import pandas as pd

GA_ARCHIVES = {}

GA_ARCHIVE_LONG = {}

GA_HISTORIES = {}


def run_ga(target_id, initial_population, seed):

    rng = np.random.default_rng(int(seed))

    population = sort_pareto(initial_population).copy().reset_index(drop=True)

    population["run_seed"] = int(seed)

    population_size = len(population)

    if population_size < 2:

        raise RuntimeError("Initial GA population < 2.")

    archive_parts = [population.copy()]

    evaluated_sequences = set(population["sequence"].astype(str))

    history_rows = []

    # Counts newly evaluated GA offspring.
    new_candidate_evaluation_count = 0

    for generation in range(1, MAX_GENERATIONS + 1):

        remaining_budget = GA_EVAL_BUDGET_PER_SEED - new_candidate_evaluation_count

        if remaining_budget <= 0:

            break

        population = sort_pareto(population)

        parent_weights = 1.0 / (population["pareto_rank"].to_numpy(dtype=float) + 1.0)

        parent_probabilities = parent_weights / parent_weights.sum()

        proposal_slots = min(population_size, remaining_budget)

        proposal_records = []

        for _ in range(proposal_slots):

            parent_1_index = int(
                rng.choice(np.arange(population_size), p=parent_probabilities)
            )

            parent_1 = population.iloc[parent_1_index]

            parent_1_sequence = str(parent_1["sequence"])

            working_sequence = parent_1_sequence

            parent_2_sequence = None

            used_crossover = False

            reference_model_y = float(parent_1["pred_model_y"])

            if rng.random() < P_CROSSOVER:

                second_probabilities = parent_probabilities.copy()

                second_probabilities[parent_1_index] = 0

                if second_probabilities.sum() > 0:

                    second_probabilities = (
                        second_probabilities / second_probabilities.sum()
                    )

                    parent_2_index = int(
                        rng.choice(np.arange(population_size), p=second_probabilities)
                    )

                    parent_2 = population.iloc[parent_2_index]

                    parent_2_sequence = str(parent_2["sequence"])

                    crossed = structural_crossover(parent_1_sequence, parent_2_sequence)

                    if crossed is not None:

                        working_sequence = crossed

                        used_crossover = True

                        reference_model_y = float(
                            np.mean(
                                [
                                    parent_1["pred_model_y"],
                                    parent_2["pred_model_y"],
                                ]
                            )
                        )

            child, operator = mutate_sequence(
                target_id, working_sequence, reference_model_y, rng
            )

            if child is None:

                continue

            child = str(child).upper().strip()

            if child in evaluated_sequences:

                continue

            if len(child) == 0 or not set(child).issubset(
                {
                    "A",
                    "C",
                    "G",
                    "T",
                }
            ):

                continue

            child_bio = get_bio_cached([child])[0]

            if not structure_supported(child, child_bio):

                continue

            proposal_records.append(
                {
                    "target_id": target_id,
                    "sequence": child,
                    "generation": generation,
                    "run_seed": int(seed),
                    "mutation_operator": operator,
                    "used_crossover": used_crossover,
                    "parent_1_sequence": parent_1_sequence,
                    "parent_2_sequence": parent_2_sequence,
                }
            )

            evaluated_sequences.add(child)

        if not proposal_records:

            break

        df_meta = (
            pd.DataFrame(proposal_records)
            .drop_duplicates(subset=["sequence"])
            .head(remaining_budget)
            .reset_index(drop=True)
        )

        df_eval = evaluate_sequences_for_target(target_id, df_meta["sequence"].tolist())

        df_offspring = df_eval.merge(
            df_meta,
            on=[
                "target_id",
                "sequence",
            ],
            how="inner",
        )

        df_offspring["source"] = "ga"

        new_candidate_evaluation_count += len(df_offspring)

        archive_parts.append(df_offspring.copy())

        supported_offspring = df_offspring.loc[
            df_offspring["global_support"].astype(bool)
        ].copy()

        combined = pd.concat(
            [
                population,
                supported_offspring,
            ],
            ignore_index=True,
            sort=False,
        )

        combined = combined.drop_duplicates(subset=["sequence"], keep="first")

        combined = sort_pareto(combined)

        population = combined.head(population_size).copy().reset_index(drop=True)

        history_rows.append(
            {
                "target_id": target_id,
                "run_seed": int(seed),
                "generation": generation,
                "cumulative_new_candidate_evaluations": new_candidate_evaluation_count,
                "new_candidates": len(df_offspring),
                "new_global_supported": len(supported_offspring),
                "population_size": len(population),
                "pareto_front_size": int((population["pareto_rank"] == 0).sum()),
                "best_target_error_model_y": float(
                    population["target_error_model_y"].min()
                ),
                "best_target_error_te": float(population["target_error_te"].min()),
            }
        )

    archive = (
        pd.concat(archive_parts, ignore_index=True, sort=False)
        .drop_duplicates(subset=["sequence"], keep="first")
        .reset_index(drop=True)
    )

    history = pd.DataFrame(history_rows)

    return (archive, history, population)


for target_id in df_targets["target_id"].astype(str):

    target_ga_dir = GA_ROOT / target_id

    seed_archives = []

    seed_histories = []

    for seed in GA_SEEDS:

        (
            archive,
            history,
            final_population,
        ) = run_ga(target_id, INITIAL_POPULATIONS[target_id], seed)

        seed_archives.append(archive)

        seed_histories.append(history)

    aggregate_long = pd.concat(seed_archives, ignore_index=True, sort=False)

    seed_presence = (
        aggregate_long.groupby("sequence", as_index=False)["run_seed"]
        .nunique()
        .rename(columns={"run_seed": "ga_seed_presence_n"})
    )

    aggregate_archive = (
        aggregate_long.drop_duplicates(subset=["sequence"], keep="first")
        .merge(seed_presence, on="sequence", how="left")
        .reset_index(drop=True)
    )

    if seed_histories:

        aggregate_history = pd.concat(seed_histories, ignore_index=True, sort=False)

    else:

        aggregate_history = pd.DataFrame()

    GA_ARCHIVE_LONG[target_id] = aggregate_long

    GA_ARCHIVES[target_id] = aggregate_archive

    GA_HISTORIES[target_id] = aggregate_history


import numpy as np
import pandas as pd

FINAL_CANDIDATES = {}

final_summary_rows = []

training_sequence_set = set(sequences_train.tolist())

for _, target_row in df_targets.iterrows():

    target_id = str(target_row["target_id"])

    target_te = float(target_row["target_te"])

    evo_pool = EVO_EVALUATED[target_id].copy()

    ga_pool = GA_ARCHIVES[target_id].copy()

    full_pool = pd.concat(
        [
            evo_pool,
            ga_pool,
        ],
        ignore_index=True,
        sort=False,
    )

    full_pool = full_pool.drop_duplicates(subset=["sequence"], keep="first")

    full_pool = (
        full_pool.loc[full_pool["global_support"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )

    full_pool["is_exact_training_sequence"] = full_pool["sequence"].isin(
        training_sequence_set
    )

    exact_training_removed = int(full_pool["is_exact_training_sequence"].sum())

    novel_pool = (
        full_pool.loc[~full_pool["is_exact_training_sequence"]]
        .copy()
        .reset_index(drop=True)
    )

    target_final_dir = FINAL_ROOT / target_id

    if len(novel_pool) == 0:

        FINAL_CANDIDATES[target_id] = pd.DataFrame()

        final_summary_rows.append(
            {
                "target_id": target_id,
                "target_te": target_te,
                "target_model_y": float(target_row["target_model_y"]),
                "status": "no_supported_novel_candidate",
                "n_final": 0,
            }
        )

        continue

    novel_pool = sort_pareto(novel_pool)

    pareto_front = (
        novel_pool.loc[novel_pool["pareto_rank"] == 0].copy().reset_index(drop=True)
    )

    target_radius = float(TARGET_STATES[target_id]["target_nt_radius"])

    final = diversity_select_by_nt(
        pareto_front, minimum_distance=target_radius, max_n=MAX_FINAL_REPORT
    )

    if len(final) == 0 and len(pareto_front) > 0:

        final = (
            pareto_front.sort_values(
                [
                    "target_error_model_y",
                    "pred_model_y_sd",
                ]
            )
            .head(1)
            .copy()
        )

    final = final.sort_values(
        [
            "target_error_model_y",
            "pred_model_y_sd",
        ]
    ).reset_index(drop=True)

    final["final_report_rank"] = np.arange(1, len(final) + 1)

    final["candidate_label"] = "Pareto-optimal in silico prioritized candidate"

    final["nearest_training_nt_distance"] = final["nt_global_distance"]

    final["nearest_training_bio_distance"] = final["bio_global_distance"]

    FINAL_CANDIDATES[target_id] = final

    best = final.sort_values(
        [
            "target_error_model_y",
            "pred_model_y_sd",
        ]
    ).iloc[0]

    final_summary_rows.append(
        {
            "target_id": target_id,
            "target_te": target_te,
            "target_average_strength": float(target_row["target_average_strength"]),
            "target_model_y": float(target_row["target_model_y"]),
            "design_regime": str(target_row["design_regime"]),
            "status": "candidate_available",
            "supported_novel_pool_n": len(novel_pool),
            "pareto_front_n": len(pareto_front),
            "n_final": len(final),
            "exact_training_removed": exact_training_removed,
            "best_pred_model_y": float(best["pred_model_y"]),
            "best_pred_model_y_sd": float(best["pred_model_y_sd"]),
            "best_pred_te": float(best["pred_te"]),
            "best_target_error_model_y": float(best["target_error_model_y"]),
            "best_target_error_te": float(best["target_error_te"]),
            "best_sequence": str(best["sequence"]),
        }
    )

df_final_summary = pd.DataFrame(final_summary_rows)


display(df_final_summary)


import json

import numpy as np
import pandas as pd

available = (
    df_final_summary.loc[df_final_summary["status"] == "candidate_available"]
    .copy()
    .sort_values("target_te")
    .reset_index(drop=True)
)

if len(available) == 0:

    raise RuntimeError("No final candidate available.")

INSILICO_TE_MAE = float(
    np.mean(np.abs(available["best_pred_te"] - available["target_te"]))
)

INSILICO_TE_RMSE = float(
    np.sqrt(np.mean((available["best_pred_te"] - available["target_te"]) ** 2))
)

INSILICO_TE_MAX_ERROR = float(
    np.max(np.abs(available["best_pred_te"] - available["target_te"]))
)

INSILICO_MODEL_Y_MAE = float(
    np.mean(np.abs(available["best_pred_model_y"] - available["target_model_y"]))
)


plot_y_min = min(-0.02, float(available["best_pred_te"].min()) - 0.02)

plot_y_max = max(1.01, float(available["best_pred_te"].max()) + 0.02)


axis_min = min(
    float(available["target_model_y"].min()),
    float(available["best_pred_model_y"].min()),
)

axis_max = max(
    float(available["target_model_y"].max()),
    float(available["best_pred_model_y"].max()),
)


TARGET_TRACKING_METRICS = {
    "requested_te_values": TARGET_TE_VALUES.tolist(),
    "n_requested_targets": int(len(df_targets)),
    "n_targets_with_final_candidates": int(len(available)),
    "in_silico_te_mae": INSILICO_TE_MAE,
    "in_silico_te_rmse": INSILICO_TE_RMSE,
    "in_silico_te_max_error": INSILICO_TE_MAX_ERROR,
    "in_silico_model_y_mae": INSILICO_MODEL_Y_MAE,
    "training_model_y_range": [
        TRAIN_MODEL_Y_MIN,
        TRAIN_MODEL_Y_MAX,
    ],
    "reconstructed_training_te_range": [
        TRAIN_TE_MIN,
        TRAIN_TE_MAX,
    ],
    "experimental_validation": False,
}


import json
import math

import numpy as np


def json_safe(value):

    if isinstance(value, np.generic):

        value = value.item()

    if isinstance(value, float):

        if not math.isfinite(value):

            return None

    if isinstance(value, np.ndarray):

        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, dict):

        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [json_safe(item) for item in value]

    return value


target_manifests = {}

for target_id in df_targets["target_id"].astype(str):

    state = TARGET_STATES[target_id]

    qc_rows = df_evo_qc_summary.loc[df_evo_qc_summary["target_id"] == target_id]

    if len(qc_rows) == 1:

        qc_row = qc_rows.iloc[0].to_dict()

    else:

        qc_row = {}

    target_manifests[target_id] = {
        "target_te": state["target_te"],
        "target_average_strength": state["target_average_strength"],
        "target_model_y": state["target_model_y"],
        "design_regime": state["design_regime"],
        "target_neighborhood_n": int(len(state["target_indices"])),
        "target_nt_radius": state["target_nt_radius"],
        "adaptive_anchor_n": state["adaptive_anchor_n"],
        "planned_evo_samples": state["planned_evo_samples"],
        "evo_qc": qc_row,
        "final_candidate_n": int(len(FINAL_CANDIDATES[target_id])),
    }


def first_nonmissing(column, default=None):

    if column not in (df_evo_raw.columns):

        return default

    values = df_evo_raw[column].dropna().unique()

    if len(values) == 0:

        return default

    value = values[0]

    if isinstance(value, np.generic):

        value = value.item()

    return value


manifest = {
    "study_stage": "Phase 3",
    "framework": ("Predict -> Explain -> Constrain -> " "Generate -> Optimize"),
    "terminology": {
        "model_y": ("Phase-1 regression target; " "actual = log10(Average Strength)"),
        "conventional_te": (
            "Termination efficiency; "
            "TE = 1 - 1/Average Strength "
            "= 1 - 10^(-model_y)"
        ),
        "target_te": (
            "Requested biological inverse-design query "
            "on the conventional termination-efficiency scale"
        ),
    },
    "target_definition": {
        "requested_te_values": TARGET_TE_VALUES.tolist(),
        "target_model_mapping": ("target_model_y = -log10(1 - target_TE)"),
        "target_average_strength_mapping": ("target_AverageStrength = 1/(1-target_TE)"),
        "exact_te_1_excluded": True,
        "reason": ("TE=1 requires infinite Average Strength " "and infinite model_y."),
        "all_targets_inside_observed_model_y_range": bool(
            df_targets["inside_training_model_y_range"].all()
        ),
    },
    "phase1_target_reconstruction": {
        "source_column": TRAIN_MODEL_Y_COLUMN,
        "source_definition": ("actual = log10(Average Strength)"),
        "train_vs_oof_model_y_max_abs_error": TRAIN_VS_OOF_MODEL_Y_MAX_ERROR,
        "observed_model_y_range": [
            TRAIN_MODEL_Y_MIN,
            TRAIN_MODEL_Y_MAX,
        ],
        "reconstructed_average_strength_range": [
            TRAIN_AVERAGE_STRENGTH_MIN,
            TRAIN_AVERAGE_STRENGTH_MAX,
        ],
        "reconstructed_conventional_te_range": [
            TRAIN_TE_MIN,
            TRAIN_TE_MAX,
        ],
        "exported_original_efficiency_status": SAVED_ORIGINAL_EFFICIENCY_STATUS,
        "exported_original_efficiency_vs_model_y_max_abs_error": SAVED_ORIGINAL_EFFICIENCY_VS_MODEL_Y_MAX_ERROR,
        "exported_original_efficiency_vs_te_max_abs_error": SAVED_ORIGINAL_EFFICIENCY_VS_TE_MAX_ERROR,
        "exported_original_efficiency_used": False,
        "reason_exported_column_not_used": (
            "The exported column named original_efficiency "
            "numerically duplicates or otherwise does not "
            "represent the reconstructed conventional TE. "
            "Phase 3 therefore reconstructs TE directly "
            "from the exact Phase-1 model target."
        ),
        "target_mode_metadata": target_modes,
    },
    "phase1_predictor": {
        "training_n": int(n_train),
        "ensemble": ("50-model repeated-CV stacking ensemble"),
        "raw_input_dimension": 2077,
        "nt_dimension": 2048,
        "descriptor_dimension": 29,
        "median_oof_abs_error_model_y": MODEL_ERROR_FLOOR_MODEL_Y,
        "median_oof_abs_error_te": MODEL_ERROR_FLOOR_TE,
    },
    "nucleotide_transformer": {
        "model": NT_MODEL_NAME,
        "intermediate_layer_ids": MID_LAYER_IDS,
        "representation_dimension": 2048,
        "inference_mode": NT_INFERENCE_MODE,
        "consistency_tolerance": NT_CONSISTENCY_TOL,
        "consistency_max_abs_error": NT_CONSISTENCY_MAX_ABS_ERROR,
    },
    "descriptors": {
        "count": 29,
        "description": ("sequence-derived biophysical " "and heuristic descriptors"),
        "thermodynamic_mfe_features": False,
        "extractor_max_abs_error": BIO_EXTRACTOR_MAX_ABS_ERROR,
        "hairpin_geometry_mismatch_n": len(geometry_mismatches),
    },
    "support": {
        "nt_global_limit": NT_GLOBAL_SUPPORT_LIMIT,
        "bio_global_limit": BIO_GLOBAL_SUPPORT_LIMIT,
        "definition": (
            "maximum observed training leave-one-out " "nearest-neighbor distance"
        ),
        "nt_max_to_median_loo_ratio": NT_SUPPORT_MAX_TO_MEDIAN,
        "bio_max_to_median_loo_ratio": BIO_SUPPORT_MAX_TO_MEDIAN,
        "caveat": (
            "Maximum LOO nearest-neighbor thresholds "
            "can be sensitive to training outliers."
        ),
    },
    "phase2_guidance": {
        "individual_pi_file": str(PI_PATH),
        "group_pi_file": str(GROUP_PI_PATH),
        "association_file": str(ASSOC_PATH),
        "raw_oof_ism_file": (str(RAW_ISM_PATH) if RAW_ISM_PATH is not None else None),
        "primary_hairpin_aligned_ism_records_file": str(ISM_RECORDS_PATH),
        "hairpin_aligned_ism_summary_file": str(ISM_SUMMARY_PATH),
        "hairpin_aligned_ism_hotspots_file": (
            str(ISM_HOTSPOTS_PATH) if ISM_HOTSPOTS_PATH is not None else None
        ),
        "raw_ism_realigned_in_phase3": False,
        "relative_position_definition": ("0 = last nucleotide of the right / 3' stem"),
        "relative_position_window": [
            -30,
            20,
        ],
        "new_phase3_hotspot_percentile_threshold": False,
        "full_aligned_ism_used_for_directional_guidance": True,
        "operator_probabilities": OPERATOR_WEIGHTS,
        "crossover_probability": P_CROSSOVER,
        "interpretation": (
            "PI=model reliance; "
            "actual correlations=observed association; "
            "OOF correlations=association with model predictions; "
            "ISM=model-predicted mutational sensitivity. "
            "No causal biological interpretation is claimed."
        ),
    },
    "evo": {
        "model": first_nonmissing("evo_model_name"),
        "temperature": first_nonmissing("evo_temperature"),
        "top_k": first_nonmissing("evo_top_k"),
        "top_p": first_nonmissing("evo_top_p"),
        "samples_per_target": EVO_SAMPLES_PER_TARGET,
        "total_generation_attempts": int(len(df_evo_raw)),
        "prompt_mode": "through_loop_end",
        "evo_score_role": (
            "generation provenance only; " "never used as terminator function/fitness"
        ),
    },
    "ga": {
        "seeds": GA_SEEDS,
        "maximum_population_cap": MAX_GA_POPULATION,
        "maximum_generations": MAX_GENERATIONS,
        "new_candidate_evaluation_budget_per_target_per_seed": GA_EVAL_BUDGET_PER_SEED,
        "scalar_fitness": False,
        "pareto_objectives": PARETO_OBJECTIVES,
        "mutation_events_per_offspring": 1,
        "parent_selection": (
            "Pareto-rank-weighted selection "
            "using 1/(rank+1); analyst-defined "
            "evolutionary selection mechanism"
        ),
    },
    "final_prioritization": {
        "exact_training_sequences_excluded": True,
        "maximum_reported_candidates_per_target": MAX_FINAL_REPORT,
        "label": ("Pareto-optimal in silico prioritized candidate"),
        "experimental_validation": False,
        "in_silico_te_mae": INSILICO_TE_MAE,
        "in_silico_te_rmse": INSILICO_TE_RMSE,
        "in_silico_te_max_error": INSILICO_TE_MAX_ERROR,
        "in_silico_model_y_mae": INSILICO_MODEL_Y_MAE,
    },
    "targets": target_manifests,
}


import numpy as np
import pandas as pd

qa_failures = []

EXPECTED_TARGET_TE = np.array(
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

if not np.allclose(
    df_targets["target_te"].to_numpy(dtype=float),
    EXPECTED_TARGET_TE,
    atol=1e-12,
    rtol=0,
):

    qa_failures.append("Requested TE grid mismatch.")

roundtrip_te = model_y_to_te(te_to_model_y(EXPECTED_TARGET_TE))

ROUNDTRIP_TE_ERROR = float(np.max(np.abs(roundtrip_te - EXPECTED_TARGET_TE)))

if ROUNDTRIP_TE_ERROR > 1e-12:

    qa_failures.append("TE <-> model_y transform failed.")

if TRAIN_VS_OOF_MODEL_Y_MAX_ERROR > 1e-10:

    qa_failures.append("Train vs OOF model_y mismatch.")

if NT_CONSISTENCY_MAX_ABS_ERROR > NT_CONSISTENCY_TOL:

    qa_failures.append("Nucleotide Transformer consistency failed.")

if BIO_EXTRACTOR_MAX_ABS_ERROR > 1e-10:

    qa_failures.append("29D extractor consistency failed.")

if len(geometry_mismatches) > 0:

    qa_failures.append("Hairpin geometry reconstruction failed.")

if len(ISM_LOOKUP) == 0:

    qa_failures.append("Phase-2 ISM lookup is empty.")

if len(ISM_RELATIVE_POSITIONS) == 0:

    qa_failures.append("Phase-2 ISM relative-position set is empty.")

if min(ISM_RELATIVE_POSITIONS) > -30 or max(ISM_RELATIVE_POSITIONS) < 20:

    qa_failures.append(
        "Expected Phase-2 ISM aligned window -30:+20 " "was not recovered."
    )

for target_id in df_targets["target_id"].astype(str):

    planned = int(
        df_evo_plan.loc[df_evo_plan["target_id"] == target_id, "n_samples"].sum()
    )

    observed = int((df_evo_raw["target_id"] == target_id).sum())

    if planned != observed:

        qa_failures.append(
            (f"{target_id}: " f"Evo attempt count " f"{observed} != {planned}")
        )

candidate_qa_rows = []

training_sequence_set = set(sequences_train.tolist())

for target_id in df_targets["target_id"].astype(str):

    final = FINAL_CANDIDATES[target_id]

    if len(final) == 0:

        candidate_qa_rows.append(
            {
                "target_id": target_id,
                "n_final": 0,
                "all_supported": False,
                "no_exact_training_sequence": True,
                "all_objectives_finite": True,
                "target_id_consistent": True,
            }
        )

        if REQUIRE_FINAL_CANDIDATE_EACH_TARGET:

            qa_failures.append(f"{target_id}: no final candidate.")

        continue

    all_supported = bool(final["global_support"].astype(bool).all())

    no_exact_training_sequence = bool(
        ~final["sequence"].isin(training_sequence_set).any()
    )

    all_objectives_finite = bool(
        np.all(np.isfinite(final[PARETO_OBJECTIVES].to_numpy(dtype=float)))
    )

    target_id_consistent = bool((final["target_id"] == target_id).all())

    candidate_qa_rows.append(
        {
            "target_id": target_id,
            "n_final": len(final),
            "all_supported": all_supported,
            "no_exact_training_sequence": no_exact_training_sequence,
            "all_objectives_finite": all_objectives_finite,
            "target_id_consistent": target_id_consistent,
        }
    )

    if not all_supported:

        qa_failures.append(f"{target_id}: " "unsupported final sequence.")

    if not no_exact_training_sequence:

        qa_failures.append(f"{target_id}: " "exact training sequence remains.")

    if not all_objectives_finite:

        qa_failures.append(f"{target_id}: " "non-finite Pareto objective.")

    if not target_id_consistent:

        qa_failures.append(f"{target_id}: " "target provenance mismatch.")

df_final_hard_qa = pd.DataFrame(candidate_qa_rows)


if not PHASE3_MANIFEST_PATH.exists():

    qa_failures.append("Phase-3 manifest missing.")


display(df_final_hard_qa)

if qa_failures:

    for failure in qa_failures:

        pass

    raise RuntimeError(f"Phase-3 QA failed with " f"{len(qa_failures)} issue(s).")


import pandas as pd

EXPECTED_TARGETS = [
    "te_0p00",
    "te_0p10",
    "te_0p20",
    "te_0p30",
    "te_0p40",
    "te_0p50",
    "te_0p60",
    "te_0p70",
    "te_0p80",
    "te_0p90",
    "te_0p99",
]


records = []

for target_id in EXPECTED_TARGETS:

    target_dir = FINAL_ROOT / target_id

    candidate_file = target_dir / "final_candidates.csv"

    exists = candidate_file.exists()

    if exists:

        df_tmp = pd.read_csv(candidate_file)

        n_rows = len(df_tmp)

        columns = df_tmp.columns.tolist()

    else:

        n_rows = 0
        columns = []

    records.append(
        {
            "target_id": target_id,
            "folder_exists": target_dir.exists(),
            "file_exists": exists,
            "n_candidates": n_rows,
            "file": str(candidate_file),
        }
    )


df_file_check = pd.DataFrame(records)


display(df_file_check)


missing = df_file_check.loc[~df_file_check["file_exists"], "target_id"].tolist()

if missing:

    pass

else:

    pass


import numpy as np
import pandas as pd

EXPECTED_TARGETS = [
    "te_0p00",
    "te_0p10",
    "te_0p20",
    "te_0p30",
    "te_0p40",
    "te_0p50",
    "te_0p60",
    "te_0p70",
    "te_0p80",
    "te_0p90",
    "te_0p99",
]


def target_id_to_te(target_id):

    value = target_id.replace("te_", "").replace("p", ".")

    return float(value)


def normalize_dna(sequence):

    sequence = str(sequence).upper().strip().replace("U", "T")

    if len(sequence) == 0:

        return None

    if not set(sequence).issubset(set("ACGT")):

        return None

    return sequence


all_frames = []

for target_id in EXPECTED_TARGETS:

    path = FINAL_ROOT / target_id / "final_candidates.csv"

    if not path.exists():

        raise FileNotFoundError(f"Missing required file:\n{path}")

    df = pd.read_csv(path)

    sequence_candidates = [
        "sequence",
        "best_sequence",
        "candidate_sequence",
        "seq",
    ]

    sequence_col = next(
        (column for column in sequence_candidates if column in df.columns), None
    )

    if sequence_col is None:

        raise RuntimeError(
            f"{path} does not contain a recognizable sequence column.\n"
            f"Columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # Standardize sequence column
    df["sequence"] = df[sequence_col].map(normalize_dna)

    df["target_id"] = target_id

    df["target_te"] = target_id_to_te(target_id)

    df["target_model_y"] = -np.log10(1.0 - df["target_te"])

    # Provenance
    df["source_target_folder"] = target_id

    df["source_file"] = str(path)

    all_frames.append(df)

df_all_final = pd.concat(all_frames, ignore_index=True, sort=False)


invalid_sequence_mask = df_all_final["sequence"].isna()

n_invalid = int(invalid_sequence_mask.sum())


if n_invalid > 0:

    df_all_final = df_all_final.loc[~invalid_sequence_mask].copy()

duplicate_mask = df_all_final.duplicated(
    subset=[
        "target_id",
        "sequence",
    ],
    keep="first",
)

n_duplicates = int(duplicate_mask.sum())


if n_duplicates > 0:

    df_all_final = df_all_final.loc[~duplicate_mask].copy()

df_all_final = df_all_final.sort_values(
    [
        "target_te",
        "target_id",
    ]
).reset_index(drop=True)

df_all_final["external_validation_candidate_id"] = [
    f"candidate_{i:04d}" for i in range(1, len(df_all_final) + 1)
]

# Put important columns first
priority_columns = [
    "external_validation_candidate_id",
    "target_id",
    "target_te",
    "target_model_y",
    "sequence",
    "source_target_folder",
    "source_file",
]

remaining_columns = [
    column for column in df_all_final.columns if column not in priority_columns
]

df_all_final = df_all_final[priority_columns + remaining_columns]

df_counts = (
    df_all_final.groupby(
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
)


display(df_counts)


observed_targets = set(df_all_final["target_id"].unique())

missing_targets = [
    target for target in EXPECTED_TARGETS if target not in observed_targets
]

if missing_targets:

    raise RuntimeError(f"Missing target candidates after merge: {missing_targets}")


import pandas as pd

df_check = pd.read_csv(OUTPUT_PATH)


display(
    df_check.groupby(
        [
            "target_id",
            "target_te",
        ]
    )
    .size()
    .rename("N")
    .reset_index()
    .sort_values("target_te")
)


display(df_check["sequence"].str.len().describe())


display(
    df_check[
        [
            "external_validation_candidate_id",
            "target_id",
            "target_te",
            "sequence",
        ]
    ].head()
)
