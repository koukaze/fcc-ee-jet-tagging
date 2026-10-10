# Data specification

## Simulation inputs

Training and checkpoint evaluation accept ROOT files containing the `deepntuplizer/tree` tree, with one row per reconstructed jet. The project uses supplied FCC-ee simulation samples described as `ZH -> qqgg`. Raw simulation files and trained checkpoints are not included.

The original input sequence is `chunk_0.root` through `chunk_9.root`. Preserve the file order and row order when reproducing splits. Training expects the shell to expand file globs; the evaluation commands also accept glob patterns.

| Input group | Tensor shape | Branch-name definition |
| --- | --- | --- |
| Global jet variables | `(n_jets, 12)` | `GLOBAL_BRANCHES` |
| Charged particle-flow candidates | `(n_jets, 25, 19)` | `CPF_BRANCHES` |
| Neutral particle-flow candidates | `(n_jets, 25, 11)` | `NPF_BRANCHES` |
| Secondary vertices | `(n_jets, 4, 11)` | `SV_BRANCHES` |
| V0 candidates | `(n_jets, 4, 11)` | `V0_BRANCHES` |
| Jet truth | `(n_jets, 6)` | `TRUTH_BRANCHES` |

These ordered branch lists are defined in [`src/djt/baseline.py`](../src/djt/baseline.py). The truth fields are `isB_H`, `isC_H`, `isS_H`, `isU_H`, `isD_H`, `isG_H`. Each valid row must represent one true class. Keep the branch definitions, units and preprocessing conventions of the supplied simulation inputs; no unit conversion is performed by the loader.

Jagged object collections are truncated to their configured lengths and padded with zeros. Inputs are converted to float32. The loader replaces non-finite values with zero and reports the count. Model padding masks identify all-zero object feature vectors; each jet needs at least one valid object token.

The split is reconstructed from the concatenated row count, seed and fractions. Saved prediction archives do not contain stable per-jet identifiers, so merging results from independently prepared datasets requires additional alignment information.

## Included prediction archives

Both archives load with `numpy.load(..., allow_pickle=False)`. The six-class order is always `b, c, s, u, d, g`; the specialist order is `s, u, d`.

| File | Main arrays |
| --- | --- |
| `results/final/baseline090_6cls_predictions.npz` | `scores` `(180000, 6)`, `labels` `(180000,)`, `class_names` |
| `results/final/A_no_kaon_baseline_softfusion_predictions.npz` | `baseline_scores` and `fused_scores` `(180000, 6)`, `uds_scores` `(180000, 3)`, `labels`, class orders and pair-feature metadata |

The baseline scores embedded in the fusion archive match the standalone baseline scores element by element, and both label arrays match. The reproduction script verifies those relationships and checks that the fused scores follow the documented probability rule. This checks consistency of the distributed prediction files; it does not independently establish original event identities.

Two text summaries record the associated evaluation and checkpoint metadata. The pair-feature study is stored as `results/studies/pairfeature_ablation_summary.csv`; it summarises seven trained feature-group configurations. Its notebook analyses the recorded table without retraining those configurations.

These are outputs of the FCC-ee simulation study. The random tensors and miniature ROOT files created by software tests are separate synthetic fixtures.

## Generated outputs

`outputs/`, `runs/`, `plots/`, `logs/`, raw ROOT files and model checkpoints are excluded by `.gitignore`. The saved comparison notebooks do not need the large per-jet kinematic metadata archive. No kinematic-dependence claims are made by their figures.
