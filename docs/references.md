# References

## Flavour-tagging architecture

F. Blekman, F. Canelli, A. De Moor et al., *Tagging more quark jet flavours at
FCC-ee at 91 GeV with a transformer-based neural network*, European Physical
Journal C **85**, 165 (2025).
[Article](https://doi.org/10.1140/epjc/s10052-025-13785-y) ·
[Preprint](https://arxiv.org/abs/2406.08590)

The six-class baseline follows the DeepJetTransformer architecture and feature
configuration. This project uses supplied FCC-ee simulation samples with
`b/c/s/u/d/g` labels; its numerical comparison concerns that six-class setup.

H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, Proceedings of
the 39th International Conference on Machine Learning, PMLR **162**,
18281–18292 (2022).
[Proceedings](https://proceedings.mlr.press/v162/qu22b.html) ·
[Preprint](https://arxiv.org/abs/2202.03772)

Particle Transformer's use of pairwise information in attention motivates the
UDS specialist's learned corrections to the attention logits.

## Software sources

- [DeepJetFCC](https://github.com/Edler1/DeepJetFCC), revision
  [`63891446`](https://github.com/Edler1/DeepJetFCC/tree/63891446b710cac3f034e264c87acdf90c57bc48):
  baseline architecture, input-feature choices and training recipe.
- [DeepJetCore](https://github.com/DL4Jets/DeepJetCore): reference framework for
  the upstream ROOT-to-data-collection workflow. This package reads ROOT files
  directly with `uproot` and `awkward`.
- [RAdam](https://github.com/LiyuanLucasLiu/RAdam), revision
  [`688cb1ec`](https://github.com/LiyuanLucasLiu/RAdam/tree/688cb1ec99944d52690c1034f6dcfe830b24d3fd):
  optimizer implementation referenced by DeepJetFCC.
- [lookahead_pytorch](https://github.com/lonePatient/lookahead_pytorch), revision
  [`10551280`](https://github.com/lonePatient/lookahead_pytorch/tree/1055128057408fe8533ffa30654551a317f07f0a):
  Lookahead implementation referenced by DeepJetFCC.

Code attribution and license texts are provided in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
