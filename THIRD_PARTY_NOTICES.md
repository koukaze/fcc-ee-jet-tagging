# Third-party notices

The MIT license in [LICENSE](LICENSE) applies to the original contributions
covered by that grant. It does not replace the rights or license terms of
upstream implementations described below.

## DeepJetFCC and DeepJetTransformer

The network architecture, input-feature configuration and training recipe in
`src/djt/baseline.py` and the shared network components in
`src/djt/pair_bias_base.py` follow
[Edler1/DeepJetFCC](https://github.com/Edler1/DeepJetFCC), revision
[`63891446b710cac3f034e264c87acdf90c57bc48`](https://github.com/Edler1/DeepJetFCC/tree/63891446b710cac3f034e264c87acdf90c57bc48),
particularly `pytorch/pytorch_deepjet_transformer_V0.py` and
`pytorch/pytorch_ranger.py`.

The project adds direct ROOT loading, the six-class dataset configuration,
UDS pairwise-attention models and probability-level soft fusion.
DeepJetFCC-derived portions remain subject to their upstream rights.
Scientific references are listed in [docs/references.md](docs/references.md).

## RAdam

The `RAdam` classes in `src/djt/baseline.py` and
`src/djt/pair_bias_base.py` are adapted from the implementation referenced by
DeepJetFCC:

- Source: [LiyuanLucasLiu/RAdam, `radam/radam.py`](https://github.com/LiyuanLucasLiu/RAdam/blob/688cb1ec99944d52690c1034f6dcfe830b24d3fd/radam/radam.py).
- Copyright: 2019 Liyuan Liu.
- License: Apache License 2.0; the upstream text is included in
  [licenses/RAdam-Apache-2.0.txt](licenses/RAdam-Apache-2.0.txt).

The project implementation modifies optimizer integration and simplifies
argument and state handling. The affected classes carry source notices.

## Lookahead

The `Lookahead` classes in `src/djt/baseline.py` and
`src/djt/pair_bias_base.py` are adapted from the implementation referenced by
DeepJetFCC:

- Source: [lonePatient/lookahead_pytorch, `optimizer.py`](https://github.com/lonePatient/lookahead_pytorch/blob/1055128057408fe8533ffa30654551a317f07f0a/optimizer.py).
- Copyright: (c) 2019 lonePatinet, as written in the upstream license.
- License: MIT; the upstream text is included in
  [licenses/Lookahead-MIT.txt](licenses/Lookahead-MIT.txt).

The project implementation modifies optimizer checkpoint state handling and
omits the upstream momentum-pullback options. The affected classes carry source
notices.

## Referenced frameworks

[DeepJetCore](https://github.com/DL4Jets/DeepJetCore) provides the data-collection
workflow used by the upstream project. It is referenced for context and is
not distributed in this repository.
