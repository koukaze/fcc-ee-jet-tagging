# Model and evaluation

## Six-class baseline

The project uses supplied FCC-ee simulation samples described as `ZH -> qqgg`, with jet labels in the order `b, c, s, u, d, g`. It follows the DeepJetTransformer architecture and feature strategy while adapting the task to six classes. The original reference study used a five-quark classification setup at the Z pole; its numerical benchmark is a different task.

Charged particles, neutral particles, secondary vertices and V0 candidates are embedded into a common 128-dimensional token representation. Three Transformer encoder layers with eight attention heads process the tokens. Attention pooling combines the tokens with twelve global jet variables before six-class classification.

The `baseline090` charged-particle input includes `RPj_charged_is_Kaon_smearedUniform090`. Other truth-level particle-identification fields are not input features. Jet truth branches provide the supervised labels.

## Pairwise-attention specialist

The specialist predicts `s`, `u` and `d`. A shared pair-feature MLP generates an additive bias for each attention head:

```text
attention_logit(i, j, head) = Q(i) · K(j) / sqrt(d_head) + B(i, j, head)
```

The pair MLP has hidden width 32; its output layer is initialised to zero. The same generated pair bias is supplied to all encoder layers. Padding masks exclude padded tokens from attention.

The final `no_kaon` configuration contains seventeen ordered-pair inputs:

| Group | Features |
| --- | --- |
| Geometry | Δθ, sin(Δφ), cos(Δφ), log(ΔR) |
| Momentum | log(p_i), log(p_j) |
| Charge | charge product, same nonzero charge, opposite charge |
| Token type | Four-category one-hot representation for each token |

Here ΔR is computed from Δθ and the direct difference Δφ, with the implementation's numerical stabilisation. The sinusoidal features encode angular periodicity, but the direct Δφ inside ΔR is not wrapped. This definition is part of the evaluated model.

`no_kaon` removes explicit kaon-pair features. The original token-level kaon-ID input remains available to the network.

## Soft fusion

For each jet, the six-class baseline sets the total light-flavour probability:

```text
P_light = P6(s) + P6(u) + P6(d)
P_fused(s/u/d) = P_light * P_UDS(s/u/d)
P_fused(b/c/g) = P6(b/c/g)
```

The UDS specialist is evaluated on every jet when constructing the fused six-class output, although it was trained on true UDS jets. The probabilities for b, c and g are unchanged. Their hard-label recalls can nevertheless change because the competing light-flavour scores change.

## Splitting and selection

The data are concatenated in input-file order, shuffled using NumPy `RandomState(1123)`, and split at jet level into 70% training, 15% evaluation and 15% test. The evaluation split selects checkpoints by loss. For the specialist, true UDS jets are selected after the shared full-sample split.

The source sample contains 1,200,000 jets; the saved test predictions contain 180,000. The split is not grouped by event, so the implementation does not enforce event-disjoint samples.

Each pair-feature configuration uses its checkpoint with the lowest validation loss. At those checkpoints, `no_kaon` has the highest validation accuracy among the seven variants (51.09%, compared with 50.83% for `full`), while `full` has slightly lower validation loss (0.955048 versus 0.955294). Checkpoint selection and feature-set selection are distinct decisions.

All seven feature configurations were also compared on the same test sample used for the final comparison. These results do not provide an independent test of feature-set selection. No repeated-seed uncertainty study is included.

## Metrics

Overall accuracy is the fraction of jets assigned the correct class. Per-class recall uses the number of true jets of that class as its denominator. Macro recall is the unweighted mean of the six per-class recalls. Confusion matrices use true classes in rows and predicted classes in columns.

For `s` versus `u/d`, jets are restricted to true UDS labels and ranked by `P(s) / (P(s) + P(u) + P(d))`. For `u` versus `d`, jets are restricted to those two truth labels and ranked by `P(u) / (P(u) + P(d))`.

At a target background efficiency, the threshold is the corresponding quantile of the background scores. Signal efficiency is the fraction of signal scores at or above that threshold. Finite-sample ties can make achieved background efficiency differ slightly from the target.

## Recorded comparison

| Quantity | Baseline | Soft fusion |
| --- | ---: | ---: |
| Overall accuracy | 61.280556% | 62.476111% |
| Macro recall | 61.352958% | 62.527429% |
| b recall | 91.066956% | 91.073635% |
| c recall | 80.185315% | 80.121988% |
| s recall | 64.282116% | 60.379513% |
| u recall | 38.064688% | 35.917295% |
| d recall | 22.745020% | 36.602475% |
| g recall | 71.773655% | 71.069665% |

The larger d-jet recall accompanies reduced s- and u-jet hard-label recall. The no-kaon and full-pair-feature configurations differ by only about 0.027 percentage points in macro recall in the feature study. These data do not establish a robust advantage of no-kaon across seeds, a unique physical mechanism, or an improvement in event-level physics sensitivity.

Use `scripts/reproduce_results.py` to recompute summary tables from the saved arrays. Scientific references and implementation sources are listed in [References](references.md).
