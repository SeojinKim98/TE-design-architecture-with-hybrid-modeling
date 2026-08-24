# Results

Artifacts released with the paper.

## `phase3_designed_sequences_330.csv`

The 330 designed terminators: 30 per requested termination efficiency, across eleven
targets from TE = 0.00 to 0.99. One row per candidate, 94 columns, covering the full
provenance of each sequence:

| group | columns | what they record |
|-------|---------|------------------|
| target | `target_id`, `target_te`, `target_model_y` | requested efficiency and its model-space equivalent |
| sequence | `sequence`, `sequence_length` | the designed sequence |
| Phase-1 prediction | `pred_model_y`, `pred_model_y_sd`, `pred_te`, `target_error_te` | ensemble prediction, spread across the 50 fold models, and deviation from target |
| support and novelty | `nt_global_supported`, `bio_global_supported`, `structure_supported`, `nearest_training_nt_distance`, `is_exact_training_sequence` | whether the candidate lies inside the region of feature space populated by the training library, and how far it sits from the nearest training sequence |
| Evo provenance | `prompt`, `anchor_rank`, `evo_*`, `qc_*` | the anchor prompt, sampling settings and the quality-control gate each sample passed |
| search provenance | `pareto_rank`, `crowding_distance`, `mutation_operator`, `used_crossover`, `parent_1_sequence`, `parent_2_sequence`, `run_seed` | how the candidate was produced and ranked |
| external check | `tersp_annotation_status`, `tersp_te`, `tersp_abs_error_te`, `tersp_*` | TerSP region annotation and score, or the reason the candidate was unscorable |

`phase3_designed_sequences_330_summary.csv` is the same 330 rows reduced to the
sequence, the target, the Phase-1 prediction and the TerSP score, for readers who do
not need the full provenance.

## `phase2_design_knowledge.json`

The exported design rules consumed by Phase 3. It records, for each admitted
descriptor, the direction of its association with strength, its effect size, and its
interquartile range among high-strength terminators, alongside the five group
importances and the ISM hotspot positions.

These are model-derived associations and sensitivities, not experimentally established
constraints.
