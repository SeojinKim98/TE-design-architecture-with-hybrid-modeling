"""Phase 1 -- hybrid terminator strength predictor.

Encodes each terminator as a Nucleotide Transformer embedding concatenated with 29
sequence-derived descriptors, then trains a leakage-safe stacking ensemble under
10 x 5-fold cross-validation stratified on quintiles of log10 strength.

Outputs: 50 fitted fold models, descriptor matrix, embeddings, out-of-fold
predictions and cross-validation metrics. All consumed by Phase 2.
"""

import re
import time
import random
import warnings

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TARGET_MODE = "as_is"

RT_FLOOR = 1e-4


df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()

if "id" not in df.columns:
    df["id"] = np.arange(len(df)).astype(str)

df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

df = df.dropna(subset=["sequence", "efficiency"]).reset_index(drop=True)

valid_mask = df["sequence"].str.fullmatch(r"[ACGT]+")

n_invalid = (~valid_mask).sum()

if n_invalid > 0:
    pass

df = df.loc[valid_mask].reset_index(drop=True)

if TARGET_MODE == "as_is":

    df["y"] = df["efficiency"].astype(float)

elif TARGET_MODE == "te_percent_to_neglog10_rt":

    if ((df["efficiency"] < 0) | (df["efficiency"] > 100)).any():
        raise ValueError("Termination efficiency must be between 0 and 100.")

    te_fraction = df["efficiency"].values / 100.0
    readthrough = 1.0 - te_fraction

    readthrough = np.clip(readthrough, RT_FLOOR, 1.0)

    df["y"] = -np.log10(readthrough)

else:
    raise ValueError("Unknown TARGET_MODE")


display(df[["id", "sequence", "efficiency", "y"]].head())


dup_stats = (
    df.groupby("sequence")
    .agg(n=("y", "size"), mean_y=("y", "mean"), std_y=("y", "std"))
    .query("n > 1")
)


if len(dup_stats) > 0:
    display(dup_stats.head(10))


CANONICAL_GC = {("G", "C"), ("C", "G")}

CANONICAL_AT = {("A", "T"), ("T", "A")}

WOBBLE_GT = {("G", "T"), ("T", "G")}


def overlapping_count(sequence, motif):
    """
    Overlapping motif count.
    Example:
        TTTT contains two overlapping TTT motifs.
    """
    k = len(motif)

    if len(sequence) < k:
        return 0

    return sum(sequence[i : i + k] == motif for i in range(len(sequence) - k + 1))


def extract_rich_ecoli_features(sequence):
    """
    29 sequence-derived biophysical / heuristic descriptors.

    Note:
    These are NOT direct thermodynamic quantities.
    Hairpin features are heuristic sequence-pairing descriptors.
    """

    sequence = sequence.upper().strip()

    if not re.fullmatch(r"[ACGT]+", sequence):
        raise ValueError("Sequence must contain only A/C/G/T.")

    seq_len = len(sequence)

    if seq_len < 3:
        raise ValueError("Sequence is too short.")

    features = {}

    features["seq_len"] = seq_len

    features["gc_content"] = (sequence.count("G") + sequence.count("C")) / seq_len

    for base in ["A", "C", "G", "T"]:
        features[f"freq_{base}"] = sequence.count(base) / seq_len

    features["count_poly_t_4"] = overlapping_count(sequence, "TTTT")

    t_runs = [len(run) for run in re.findall(r"T+", sequence)]

    features["max_poly_t_run"] = max(t_runs) if t_runs else 0

    key_3mers = ["TTT", "AAA", "GCG", "CGC", "TTG", "TTA", "GCT", "AGC"]

    denom_3mer = seq_len - 3 + 1

    for kmer in key_3mers:

        count = overlapping_count(sequence, kmer)

        features[f"kmer_{kmer}"] = count / denom_3mer

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
                gu_count = 0
                pair_records = []

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
                        gu_count += 1

                    pair_records.append(pair)

                pair_fraction = n_pairs / stem_len

                if pair_fraction < 0.75:
                    continue

                # heuristic loop penalty
                score = pairing_score - 0.5 * abs(loop_len - 4)

                # loop-proximal two base pairs
                proximal_pairs = pair_records[-2:]

                proximal_gc_fraction = sum(
                    p in CANONICAL_GC for p in proximal_pairs
                ) / len(proximal_pairs)

                # deterministic tie-breaking
                ranking_key = (score, pair_fraction, stem_len, -abs(loop_len - 4))

                if best is None or ranking_key > best["ranking_key"]:

                    best = {
                        "ranking_key": ranking_key,
                        "start": start,
                        "stem_len": stem_len,
                        "loop_len": loop_len,
                        "score": score,
                        "gu_count": gu_count,
                        "proximal_gc_fraction": proximal_gc_fraction,
                    }

    if best is None:

        stem_start = None
        stem_len = 0
        loop_len = 0

        stem_score = 0.0
        gu_count = 0
        proximal_gc = 0.0

        upstream_a_richness = 0.0
        positional_t_score = 0.0
        spacer_length = 15

    else:

        stem_start = best["start"]
        stem_len = best["stem_len"]
        loop_len = best["loop_len"]

        stem_score = best["score"]
        gu_count = best["gu_count"]
        proximal_gc = best["proximal_gc_fraction"]

        upstream_region = sequence[max(0, stem_start - 10) : stem_start]

        upstream_a_richness = (
            upstream_region.count("A") / len(upstream_region)
            if len(upstream_region) > 0
            else 0.0
        )

        stem_end = stem_start + 2 * stem_len + loop_len

        downstream_region = sequence[stem_end : stem_end + 15]

        positional_t_score = 0.0
        first_t_position = None

        for pos, base in enumerate(downstream_region):

            if base == "T":

                if first_t_position is None:
                    first_t_position = pos

                positional_t_score += np.exp(-0.2 * pos)

        spacer_length = first_t_position if first_t_position is not None else 15

    features["stem_length"] = stem_len
    features["loop_length"] = loop_len

    features["stem_pairing_score"] = stem_score

    features["gu_wobble_count"] = gu_count

    features["loop_proximal_gc_pair_fraction"] = proximal_gc

    features["upstream_a_richness"] = upstream_a_richness

    features["positional_poly_t_score"] = positional_t_score

    features["spacer_length"] = spacer_length

    features["stem_polyT_coupling"] = (
        stem_score * positional_t_score / (spacer_length + 1.0)
    )

    features["stem_polyT_interaction"] = stem_score * features["max_poly_t_run"]

    features["compact_gc_hairpin_score"] = proximal_gc / (loop_len + 1.0)

    features["polyT_spacer_proximity"] = features["max_poly_t_run"] / (
        spacer_length + 1.0
    )

    features["polyT_position_interaction"] = (
        features["max_poly_t_run"] * positional_t_score
    )

    if len(features) != 29:
        raise RuntimeError(f"Expected 29 features, got {len(features)}")

    return features


df_biophys = pd.DataFrame([extract_rich_ecoli_features(seq) for seq in df["sequence"]])

BIO_FEATURE_NAMES = df_biophys.columns.tolist()

X_bio_raw = df_biophys.to_numpy(dtype=np.float64)


assert X_bio_raw.shape[1] == 29

display(df_biophys.head())


NT_MODEL_NAME = "InstaDeepAI/" "nucleotide-transformer-v2-500m-multi-species"

tokenizer = AutoTokenizer.from_pretrained(NT_MODEL_NAME, trust_remote_code=True)

nt_model = AutoModelForMaskedLM.from_pretrained(
    NT_MODEL_NAME, trust_remote_code=True
).to(device)

nt_model.eval()


MID_LAYER_IDS = (12, 18)
MAX_TOKENS = 2048


def masked_mean_pool(hidden_state, valid_token_mask):
    """
    hidden_state:
        [batch, tokens, hidden]

    valid_token_mask:
        [batch, tokens]
    """

    mask = valid_token_mask.unsqueeze(-1).to(hidden_state.dtype)

    numerator = (hidden_state * mask).sum(dim=1)

    denominator = mask.sum(dim=1).clamp(min=1.0)

    return numerator / denominator


def extract_hierarchical_nt_features(sequences, batch_size=16):

    embeddings = []

    special_ids = set(tokenizer.all_special_ids)

    with torch.no_grad():

        for start in range(0, len(sequences), batch_size):

            batch_seqs = sequences[start : start + batch_size]

            inputs = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = nt_model(**inputs, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states

            expected_num_states = nt_model.config.num_hidden_layers + 1

            assert len(hidden_states) == expected_num_states

            for layer_id in MID_LAYER_IDS:

                if layer_id >= len(hidden_states) - 1:

                    raise ValueError(f"Invalid layer {layer_id}")

            input_ids = inputs["input_ids"]

            valid_mask = inputs["attention_mask"].bool()

            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for token_id in special_ids:

                special_mask |= input_ids == token_id

            valid_mask &= ~special_mask

            intermediate = torch.stack(
                [hidden_states[layer_id] for layer_id in MID_LAYER_IDS], dim=0
            ).mean(dim=0)

            intermediate_pooled = masked_mean_pool(intermediate, valid_mask)

            final_hidden = hidden_states[-1]

            final_pooled = masked_mean_pool(final_hidden, valid_mask)

            fused = torch.cat([intermediate_pooled, final_pooled], dim=-1)

            embeddings.append(fused.cpu().numpy())

    return np.vstack(embeddings)


X_nt = extract_hierarchical_nt_features(df["sequence"].tolist(), batch_size=16)


expected_dim = 2 * nt_model.config.hidden_size

assert X_nt.shape[1] == expected_dim


del nt_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


LLM_DIM = X_nt.shape[1]
BIO_DIM = X_bio_raw.shape[1]

X_raw = np.hstack([X_nt, X_bio_raw])

y = df["y"].to_numpy(dtype=np.float64)


assert BIO_DIM == 29
assert X_raw.shape[1] == 2077


def make_preprocessor(seed):

    llm_selector = SelectFromModel(
        estimator=ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        ),
        threshold="mean",
    )

    llm_pipeline = Pipeline(
        steps=[("select", llm_selector), ("scale", StandardScaler())]
    )

    bio_pipeline = Pipeline(
        steps=[("power", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("llm", llm_pipeline, slice(0, LLM_DIM)),
            ("bio", bio_pipeline, slice(LLM_DIM, LLM_DIM + BIO_DIM)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


class LeakageSafeStackingRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        preprocessor,
        base_estimators,
        final_estimator,
        inner_splits=5,
        random_state=42,
    ):

        self.preprocessor = preprocessor
        self.base_estimators = base_estimators
        self.final_estimator = final_estimator

        self.inner_splits = inner_splits

        self.random_state = random_state

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = len(y)
        n_models = len(self.base_estimators)

        meta_X = np.zeros((n_samples, n_models), dtype=float)

        inner_cv = KFold(
            n_splits=self.inner_splits, shuffle=True, random_state=self.random_state
        )

        for inner_train_idx, inner_val_idx in inner_cv.split(X):

            X_tr = X[inner_train_idx]
            X_va = X[inner_val_idx]

            y_tr = y[inner_train_idx]

            for model_idx, (model_name, estimator) in enumerate(self.base_estimators):

                pipe = Pipeline(
                    steps=[
                        ("preprocess", clone(self.preprocessor)),
                        ("model", clone(estimator)),
                    ]
                )

                pipe.fit(X_tr, y_tr)

                meta_X[inner_val_idx, model_idx] = pipe.predict(X_va)

        self.final_estimator_ = clone(self.final_estimator)

        self.final_estimator_.fit(meta_X, y)

        self.preprocessor_ = clone(self.preprocessor)

        X_transformed = self.preprocessor_.fit_transform(X, y)

        self.base_estimators_ = []

        for model_name, estimator in self.base_estimators:

            fitted_estimator = clone(estimator)

            fitted_estimator.fit(X_transformed, y)

            self.base_estimators_.append((model_name, fitted_estimator))

        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):

        X = np.asarray(X)

        X_transformed = self.preprocessor_.transform(X)

        base_predictions = np.column_stack(
            [estimator.predict(X_transformed) for _, estimator in self.base_estimators_]
        )

        return self.final_estimator_.predict(base_predictions)


def make_hybrid_model(seed):

    preprocessor = make_preprocessor(seed=seed)

    base_estimators = [
        ("svr", SVR(C=20.0, epsilon=0.005, kernel="rbf", gamma="scale")),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            ),
        ),
        (
            "hist_gb",
            HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.03,
                l2_regularization=0.5,
                random_state=seed,
            ),
        ),
    ]

    meta_learner = RidgeCV(alphas=np.logspace(-2, 3, 20))

    model = LeakageSafeStackingRegressor(
        preprocessor=preprocessor,
        base_estimators=base_estimators,
        final_estimator=meta_learner,
        inner_splits=5,
        random_state=seed,
    )

    return model


def safe_pearson(y_true, y_pred):

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return pearsonr(y_true, y_pred)[0]


def safe_spearman(y_true, y_pred):

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return spearmanr(y_true, y_pred)[0]


def regression_metrics(y_true, y_pred):

    return {
        "R2": r2_score(y_true, y_pred),
        "Pearson_r": safe_pearson(y_true, y_pred),
        "Spearman_r": safe_spearman(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def make_regression_bins(y, q=5, n_splits=5):

    y_series = pd.Series(y)

    bins = pd.qcut(y_series, q=q, labels=False, duplicates="drop")

    if bins.nunique() < 2:
        return None

    counts = bins.value_counts()

    if counts.min() < n_splits:
        return None

    return bins.to_numpy()


SEEDS = [42, 100, 2026, 777, 999, 12345, 2002, 2020, 9876, 291]

N_SPLITS = 5
N_BINS = 5


repeat_results = []
fold_results = []

repeat_oof_predictions = []

selected_feature_counts = []

# regression bins
target_bins = make_regression_bins(y, q=N_BINS, n_splits=N_SPLITS)

total_start = time.time()


display(df_repeat)

METRIC_COLUMNS = ["R2", "Pearson_r", "Spearman_r", "RMSE", "MAE"]

summary_rows = []

for metric in METRIC_COLUMNS:

    summary_rows.append(
        {
            "Metric": metric,
            "Mean": df_repeat[metric].mean(),
            "SD": df_repeat[metric].std(ddof=1),
            "Min": df_repeat[metric].min(),
            "Max": df_repeat[metric].max(),
        }
    )

df_cv_summary = pd.DataFrame(summary_rows)


display(df_cv_summary)


oof_matrix = np.vstack(repeat_oof_predictions)

mean_oof_prediction = oof_matrix.mean(axis=0)

oof_prediction_sd = oof_matrix.std(axis=0, ddof=1)

ensemble_oof_metrics = regression_metrics(y, mean_oof_prediction)


oof_output = pd.DataFrame(
    {
        "id": df["id"],
        "sequence": df["sequence"],
        "actual": y,
        "mean_oof_prediction": mean_oof_prediction,
        "oof_prediction_sd": oof_prediction_sd,
    }
)

for i, seed in enumerate(SEEDS):

    oof_output[f"oof_seed_{seed}"] = oof_matrix[i]


selected_feature_counts = np.array(selected_feature_counts)


import numpy as np
import pandas as pd

required_variables = [
    "df",
    "df_biophys",
    "X_nt",
    "y",
    "bio_feature_names" if "bio_feature_names" in globals() else "BIO_FEATURE_NAMES",
]


# Resolve feature-name variable
if "BIO_FEATURE_NAMES" in globals():
    _bio_feature_names = list(BIO_FEATURE_NAMES)

elif "bio_feature_names" in globals():
    _bio_feature_names = list(bio_feature_names)

else:
    # safest fallback if df_biophys exists
    if "df_biophys" in globals():
        _bio_feature_names = df_biophys.columns.tolist()
    else:
        raise NameError(
            "Cannot find BIO_FEATURE_NAMES, bio_feature_names, " "or df_biophys."
        )

# Resolve target
if "y" in globals():
    _y = np.asarray(y, dtype=float)

elif "y_targets" in globals():
    _y = np.asarray(y_targets, dtype=float)

else:
    raise NameError("Cannot find target variable 'y' or 'y_targets'.")

# Resolve NT embeddings
if "X_nt" in globals():
    _X_nt = np.asarray(X_nt)

elif "X_hierarchical_llm" in globals():

    _X_nt = np.asarray(X_hierarchical_llm)

else:
    raise NameError(
        "Cannot find NT embeddings. " "Expected 'X_nt' or 'X_hierarchical_llm'."
    )

# Resolve biophysical dataframe
if "df_biophys" in globals():
    _df_bio = df_biophys.copy()

elif "df_rich_biophys" in globals():
    # fallback for older Phase-1 code
    _df_bio = df_rich_biophys.copy()

else:
    raise NameError(
        "Cannot find biophysical feature dataframe. "
        "Expected 'df_biophys' or 'df_rich_biophys'."
    )


if _X_nt.shape[1] != 2048:
    pass

if _df_bio.shape[1] != 29:
    raise ValueError(
        f"Expected exactly 29 biophysical features, " f"but found {_df_bio.shape[1]}."
    )

if len(_bio_feature_names) != 29:
    raise ValueError(
        f"Expected 29 feature names, " f"but found {len(_bio_feature_names)}."
    )

if len(_X_nt) != len(_df_bio):
    raise ValueError("NT embedding and biophysical feature sample counts differ.")

if len(_X_nt) != len(_y):
    raise ValueError("NT embedding and target sample counts differ.")

# enforce exact column order
_df_bio = _df_bio[_bio_feature_names].copy()


if "df" not in globals():
    raise NameError("'df' is not available. " "Need the cleaned Phase-1 dataframe.")

sequence_df = pd.DataFrame(
    {
        "id": (
            df["id"].astype(str).values
            if "id" in df.columns
            else np.arange(len(df)).astype(str)
        ),
        "sequence": (df["sequence"].astype(str).str.upper().values),
        "actual": _y,
    }
)

# include original efficiency if present
if "efficiency" in df.columns:
    sequence_df["original_efficiency"] = df["efficiency"].values


if "extract_rich_ecoli_features" not in globals():
    raise NameError(
        "extract_rich_ecoli_features() is not defined " "in the current Phase-1 kernel."
    )


# Model information
if "NT_MODEL_NAME" in globals():
    _nt_model_name = NT_MODEL_NAME
else:
    _nt_model_name = "InstaDeepAI/" "nucleotide-transformer-v2-500m-multi-species"

if "MID_LAYER_IDS" in globals():
    _mid_layers = list(MID_LAYER_IDS)
else:
    _mid_layers = [12, 18]

manifest = {
    "n_samples": int(len(_y)),
    "nt_model_name": _nt_model_name,
    "nt_embedding_dim": int(_X_nt.shape[1]),
    "mid_layer_ids": _mid_layers,
    "biophysical_dim": int(_df_bio.shape[1]),
    "biophysical_feature_names": _bio_feature_names,
    "hybrid_raw_dim": int(_X_nt.shape[1] + _df_bio.shape[1]),
    "bio_weight": 1.0,
    "notes": (
        "BIO_WEIGHT was not applied. "
        "Biophysical features are stored in raw form. "
        "Fold-specific PowerTransformer fitting occurs "
        "inside Phase-1 CV models."
    ),
}


if "df_repeat" in globals():

    pass

if "df_fold" in globals():

    pass

if "df_cv_summary" in globals():

    pass

if "oof_output" in globals():

    pass

elif "mean_oof_prediction" in globals() and "oof_prediction_sd" in globals():

    _oof_df = sequence_df.copy()

    _oof_df["mean_oof_prediction"] = mean_oof_prediction

    _oof_df["oof_prediction_sd"] = oof_prediction_sd

    # Save each repetition if available
    if "oof_matrix" in globals():

        for i in range(oof_matrix.shape[0]):

            _oof_df[f"oof_repeat_{i+1:02d}"] = oof_matrix[i]


expected_files = [
    "phase1_biophysical_features.csv",
    "phase1_nt_embeddings.npy",
    "phase1_cleaned_dataset.csv",
    "phase1_feature_extractor.pkl",
    "phase1_manifest.json",
]


all_ok = True

for filename in expected_files:

    exists = os.path.exists(path)

    if not exists:
        all_ok = False

if all_ok:
    pass
else:
    pass


import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

display(df_repeat.head())
display(df_fold.head())
display(df_oof.head())


y_true = df_oof["actual"].to_numpy()

y_pred = df_oof["mean_oof_prediction"].to_numpy()

y_pred_sd = df_oof["oof_prediction_sd"].to_numpy()

ens_r2 = r2_score(y_true, y_pred)

ens_pearson = pearsonr(y_true, y_pred)[0]

ens_spearman = spearmanr(y_true, y_pred)[0]

ens_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

ens_mae = mean_absolute_error(y_true, y_pred)

primary_summary = {}

for metric in ["R2", "Pearson_r", "Spearman_r", "RMSE", "MAE"]:

    primary_summary[metric] = (df_repeat[metric].mean(), df_repeat[metric].std(ddof=1))


for metric, (mean_, sd_) in primary_summary.items():

    pass


x = y_true
yp = y_pred

xy = np.vstack([x, yp])

try:

    density = gaussian_kde(xy)(xy)

except Exception:

    density = np.ones_like(x)

order = density.argsort()

x_sorted = x[order]
yp_sorted = yp[order]
density_sorted = density[order]

sc = ax1.scatter(
    x_sorted,
    yp_sorted,
    c=density_sorted,
    s=28,
    cmap="viridis",
    alpha=0.85,
    edgecolors="none",
)


plot_min = min(x.min(), yp.min())

plot_max = max(x.max(), yp.max())

margin = (plot_max - plot_min) * 0.05

plot_min -= margin
plot_max += margin

ax1.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=1.8,
    color="black",
    label="Identity line",
)

ax1.set_xlim(plot_min, plot_max)

ax1.set_ylim(plot_min, plot_max)

metrics_text = (
    "Repeated-OOF ensemble\n"
    f"$R^2$ = {ens_r2:.3f}\n"
    f"Pearson $r$ = {ens_pearson:.3f}\n"
    f"Spearman $\\rho$ = {ens_spearman:.3f}\n"
    f"RMSE = {ens_rmse:.3f}"
)

ax1.text(
    0.04,
    0.96,
    metrics_text,
    transform=ax1.transAxes,
    va="top",
    ha="left",
    fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
)

ax1.set_xlabel("Measured terminator strength")

ax1.set_ylabel("Predicted terminator strength")

ax1.set_title("A. Repeated-OOF Prediction")

ax1.legend(loc="lower right", frameon=True)

residual = yp - x

ax2.scatter(yp, residual, s=25, alpha=0.55, edgecolors="none")

ax2.axhline(0, linestyle="--", linewidth=1.5, color="black")

# optional trend line
coeff = np.polyfit(yp, residual, deg=1)

trend_x = np.linspace(yp.min(), yp.max(), 100)

trend_y = np.polyval(coeff, trend_x)

ax2.plot(trend_x, trend_y, linewidth=2, label="Linear trend")

ax2.set_xlabel("Predicted terminator strength")

ax2.set_ylabel("Residual (Predicted - Measured)")

ax2.set_title("B. Residual Analysis")

ax2.legend()

performance_metrics = ["R2", "Pearson_r", "Spearman_r"]

positions = np.arange(len(performance_metrics))

data = [df_repeat[m].dropna().values for m in performance_metrics]

bp = ax3.boxplot(
    data, positions=positions, widths=0.55, patch_artist=True, showfliers=False
)

# overlay individual repetitions
rng = np.random.default_rng(42)

for idx, metric in enumerate(performance_metrics):

    values = df_repeat[metric].dropna().values

    jitter = rng.normal(0, 0.045, size=len(values))

    ax3.scatter(
        np.full_like(values, idx, dtype=float) + jitter,
        values,
        s=30,
        alpha=0.75,
        zorder=3,
    )

ax3.set_xticks(positions)

ax3.set_xticklabels(["$R^2$", "Pearson $r$", "Spearman $\\rho$"])

ax3.set_ylabel("Performance")

ax3.set_title("C. Stability Across 10 CV Repetitions")

# annotate mean ± SD
for idx, metric in enumerate(performance_metrics):

    mean_v = df_repeat[metric].mean()

    sd_v = df_repeat[metric].std(ddof=1)

    ax3.text(
        idx,
        ax3.get_ylim()[0],
        f"{mean_v:.3f}\n±{sd_v:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

absolute_error = np.abs(yp - x)

uncertainty_corr = spearmanr(y_pred_sd, absolute_error)[0]

ax4.scatter(y_pred_sd, absolute_error, s=28, alpha=0.6, edgecolors="none")

# linear trend just for visualization
if np.std(y_pred_sd) > 0:

    coeff_u = np.polyfit(y_pred_sd, absolute_error, deg=1)

    ux = np.linspace(y_pred_sd.min(), y_pred_sd.max(), 100)

    uy = np.polyval(coeff_u, ux)

    ax4.plot(ux, uy, linewidth=2)

ax4.text(
    0.05,
    0.94,
    f"Spearman $\\rho$ = {uncertainty_corr:.3f}",
    transform=ax4.transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

ax4.set_xlabel("Prediction SD across 10 repetitions")

ax4.set_ylabel("Absolute prediction error")

ax4.set_title("D. Ensemble Disagreement vs Error")


rng = np.random.default_rng(42)

for metric, ax in metrics:

    values = df_repeat[metric].dropna().values

    jitter = rng.normal(1, 0.035, size=len(values))

    mean_v = values.mean()
    sd_v = values.std(ddof=1)


sort_idx = np.argsort(y_true)

actual_sorted = y_true[sort_idx]

pred_sorted = y_pred[sort_idx]

sd_sorted = y_pred_sd[sort_idx]

sample_rank = np.arange(len(y_true))


metrics = [("R2", "$R^2$"), ("RMSE", "RMSE")]


for ax, (metric, title) in zip(axes, metrics):

    pivot = df_fold.pivot(index="Repeat", columns="Fold", values=metric)

    matrix = pivot.values

    # values in cells
    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            value = matrix[i, j]


if "LLM_features_selected" in df_fold.columns:

    selected_counts = df_fold["LLM_features_selected"].to_numpy()

    model_number = np.arange(1, len(selected_counts) + 1)


else:

    pass


df_diagnostic = df_oof.copy()

df_diagnostic["absolute_error"] = np.abs(
    df_diagnostic["mean_oof_prediction"] - df_diagnostic["actual"]
)


largest_error = df_diagnostic.nlargest(20, "absolute_error").sort_values(
    "absolute_error"
)


if "id" in largest_error.columns:

    labels = largest_error["id"].astype(str)

else:

    labels = largest_error.index.astype(str)


largest_uncertainty = df_diagnostic.nlargest(20, "oof_prediction_sd").sort_values(
    "oof_prediction_sd"
)


if "id" in largest_uncertainty.columns:

    labels = largest_uncertainty["id"].astype(str)

else:

    labels = largest_uncertainty.index.astype(str)


if "LLM_features_selected" in df_fold.columns:

    x_feat = df_fold["LLM_features_selected"].to_numpy()

    y_r2_fold = df_fold["R2"].to_numpy()

    rho, pvalue = spearmanr(x_feat, y_r2_fold)

    if np.std(x_feat) > 0:

        coeff = np.polyfit(x_feat, y_r2_fold, 1)

        xx = np.linspace(x_feat.min(), x_feat.max(), 100)

        yy = np.polyval(coeff, xx)
