# Trajectory teacher

Vendored and locally adapted code from [Gerstung Lab's Delphi](https://github.com/gerstung-lab/Delphi). See `LICENSE` and the root `THIRD_PARTY.md` for provenance. This directory contains code only; all input binaries, vocabularies and checkpoints are generated locally.

From the repository root:

```bash
python scripts/demo/generate_synthetic.py
python Delphi/data/prepare_cn4dp.py \
  --input-csv data/demo/disease_onsets.csv \
  --exclude-csv data/demo/phenotypes_all.csv \
  --output-dir Delphi/data/demo --val-ratio 0.2
(cd Delphi && python train.py config/train_demo.py)
```

The exclusion table ensures that teacher training uses only the artificial non-imaging subjects. The converter and downstream label loader share the same token convention. The demo checkpoint is saved to `runs/demo/teacher/ckpt.pt`.
