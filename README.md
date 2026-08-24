# In silico design of synthetic bacterial terminators with targeted strength

Reference implementation for a three-phase framework in which model interpretation
constrains sequence generation: a hybrid strength predictor is trained, frozen, and
interrogated; the resulting determinants are exported as explicit design rules; and
those rules then set the operators and objective of an evolutionary search that
produces novel terminator sequences at prescribed termination efficiencies.

Seojin Kim and Doheon Lee, Department of Bio and Brain Engineering, KAIST.

## What is here

| Phase | Script | Contents |
|-------|--------|----------|
| 1 | `src/phase1_hybrid_predictor.py` | 29 sequence descriptors, Nucleotide Transformer embedding, leakage-safe stacking ensemble, repeated stratified cross-validation |
| 2 | `src/phase2_interpretation.py` | held-out permutation importance per descriptor and per group, Spearman associations with Benjamini-Hochberg correction, exhaustive in silico mutagenesis and hairpin alignment, design-rule extraction |
| 3a | `src/phase3a_evo_seed_generation.py` | Evo seed generation, prompted from a training terminator so that sampling begins inside terminator-like sequence space |
| 3b | `src/phase3b_constrained_design.py` | five-objective Pareto ranking, rule-derived mutation operators, support and novelty filtering |
| — | `src/external_check_tersp.py` | region annotation into the TerSP schema and scoring of the frozen design set |

Each phase depends only on frozen artifacts from the phase before it. The predictor is
frozen before interpretation and the rules are fixed before generation, so no
information from the design stage can influence either.

## Scope

These files document the analysis, not a packaged tool. Figure generation, artifact
serialisation, checkpointing, progress reporting and the local directory layout have
been removed so that the methodological content is not buried in bookkeeping. What
remains is the feature construction, the models, the attribution and mutagenesis
procedures, the objectives and operators of the search, and the filtering criteria.

As a consequence the scripts are not runnable end to end as published. Restoring the
input and output paths at the top of each file is enough to run them.

## Environments

Three separate environments were used; they are not mutually compatible.

```bash
pip install -r requirements.txt          # Phase 1-2, Python 3.9.12, CUDA 11.8
pip install -r requirements-evo.txt      # Phase 3a, separate env, evo-model pins its own torch
pip install -r requirements-tersp.txt    # external check, Python 3.13
```

Phases 1-2 and 3a were run on a single NVIDIA A100 80 GB GPU. Repeated
cross-validation takes roughly 3.4 h and exhaustive mutagenesis roughly 9 h on that
device. All random seeds are set in the scripts.

## Data

Training data are not redistributed here. They derive from the supplementary tables of

> Chen Y-J, Liu P, Nielsen AAK, et al. Characterization of 582 natural and synthetic
> terminators and quantification of their design constraints.
> *Nature Methods* 2013;10:659-664. doi:10.1038/nmeth.2515

The natural and synthetic tables were concatenated and three columns retained: `Name`,
`Sequence` and `Average Strength`. The only modification applied was a base-10
logarithm of the strength column. No records were excluded and no measurements were
recomputed. Note that the resulting target is a log-strength, not a termination
efficiency; the two are related by `strength = 1 / (1 - TE)`.

The pretrained TerSP model used for the external check is downloaded by
`external_check_tersp.py` from the URL published by its authors.
