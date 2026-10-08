# Third-party code

The trajectory teacher is based on [Gerstung Lab / Delphi](https://github.com/gerstung-lab/Delphi), with the local upstream checkout at commit `39b7c5938dd0f5059585a681520ca0f3642e01a5` used as the source of the vendored code.

`Delphi/model.py`, `Delphi/utils.py`, `Delphi/train.py` and `Delphi/configurator.py` retain the upstream code license in `Delphi/LICENSE`, including the Gerstung Lab copyright notice. Local changes include the CN4DP data converter, numerical checks, portable demo configuration, vocabulary inference and disabling upstream vocabulary-specific lifestyle augmentation for generic event labels.

The upstream license distinguishes code (MIT) from upstream model weights (CC BY-NC-ND 4.0). No model weights are included here. That upstream license does not grant a license to a dataset, and does not automatically license the original CN4DP code outside the vendored component. This release does not introduce a new root-level license for the original research code.
