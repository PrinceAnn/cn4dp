from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phenotype_encoder.data import build_preprocess_pipeline, group_feature_columns_by_organ
from phenotype_encoder.model import build_encoder_from_config
from scripts.check_public_release import audit_paths
from scripts.demo.generate_synthetic import generate
from scripts.distill.train_align import PhenotypeTeacherOnlineDataset
from scripts.downstream.utils import datetime_to_year_fraction, roc_auc_score, split_subject_indices


@pytest.fixture
def toy_data(tmp_path):
    generate(tmp_path, teacher_subjects=48, distill_subjects=48, downstream_subjects=96)
    return tmp_path


def test_generator_is_deterministic_and_has_disjoint_cohorts(toy_data, tmp_path):
    second = tmp_path / "second"
    generate(second, teacher_subjects=48, distill_subjects=48, downstream_subjects=96)
    for name in ["disease_onsets.csv", "phenotypes_all.csv", "riskset.csv"]:
        assert (toy_data / name).read_bytes() == (second / name).read_bytes()
    all_ids = set(pd.read_csv(toy_data / "disease_onsets.csv").eid)
    distill = set(pd.read_csv(toy_data / "phenotypes_distill.csv").eid)
    downstream = set(pd.read_csv(toy_data / "phenotypes_downstream.csv").eid)
    imaging = set(pd.read_csv(toy_data / "phenotypes_all.csv").eid)
    teacher = all_ids - imaging
    assert len(teacher) == 48
    assert imaging == distill | downstream
    assert not distill & downstream
    assert not teacher & (distill | downstream)


def test_risksets_are_incident_and_sampled_within_subject_splits(toy_data):
    risk = pd.read_csv(toy_data / "riskset.csv")
    onset = pd.read_csv(toy_data / "disease_onsets.csv").set_index("eid")
    info = pd.read_csv(toy_data / "basic_info.csv").set_index("eid")
    assignments = pd.read_csv(toy_data / "downstream_splits.csv").set_index("eid").split
    for row in risk.itertuples():
        imaging_year = datetime_to_year_fraction(row.imaging_date)
        onset_age = onset.loc[row.eid, "SYN_TARGET"]
        onset_year = onset_age + info.loc[row.eid, "birth_year"]
        assert assignments.loc[row.eid] == assignments.loc[row.matched_to] == row.split
        assert imaging_year < row.case_time
        if row.label:
            assert onset_year > imaging_year
            assert row.delta_time == pytest.approx(onset_year - imaging_year)
        else:
            assert np.isnan(onset_year) or onset_year > row.case_time


def test_repeated_subjects_cannot_cross_partitions(toy_data):
    risk = pd.read_csv(toy_data / "riskset.csv")
    parts = split_subject_indices(risk.eid, seed=42, train_ratio=0.6, val_ratio=0.2,
                                  split_csv=toy_data / "downstream_splits.csv")
    subjects = [set(risk.iloc[part].eid) for part in parts]
    assert all(subjects)
    assert not subjects[0] & subjects[1]
    assert not subjects[0] & subjects[2]
    assert not subjects[1] & subjects[2]
    grouped = split_subject_indices(np.repeat(np.arange(20), 3), seed=42, train_ratio=0.6, val_ratio=0.2)
    assert all(len(part) % 3 == 0 for part in grouped)


def test_missing_split_assignment_fails(toy_data):
    with pytest.raises(ValueError, match="assignment"):
        split_subject_indices([99999], seed=42, train_ratio=0.6, val_ratio=0.2,
                              split_csv=toy_data / "downstream_splits.csv")


def test_distillation_excludes_imaging_boundary_and_future_events(tmp_path):
    pd.DataFrame({"eid": [1], "birth_year": [1980], "imaging_date": ["2020-01-01"]}).to_csv(tmp_path / "basic.csv", index=False)
    pd.DataFrame({"eid": [1], "SYN_PAST": [39.0], "SYN_BOUNDARY": [40.0], "SYN_FUTURE": [41.0]}).to_csv(tmp_path / "onsets.csv", index=False)
    # Converter convention: Padding is the header; row positions start with No event.
    (tmp_path / "labels.csv").write_text("Padding\nNo event\nSYN_PAST\nSYN_BOUNDARY\nSYN_FUTURE\n")
    features = pd.DataFrame({"eid": [1], "brain__feature_001": [0.0]})
    preprocess = build_preprocess_pipeline()
    preprocess.fit(np.array([[0.0]], dtype=np.float32))
    data = PhenotypeTeacherOnlineDataset(phenotype_df=features, feature_names=["brain__feature_001"],
        preprocess=preprocess, disease_onset_csv=tmp_path / "onsets.csv", basic_csv=tmp_path / "basic.csv",
        labels_csv=tmp_path / "labels.csv", eid_col="eid", block_size=16, append_query_token=True,
        traj_mode="pre_imaging")
    assert data[0]["tokens"].tolist() == [2, 1]
    assert data[0]["ages_days"].tolist() == [39 * 365.25, 40 * 365.25]


@pytest.mark.parametrize("aggregator", ["concat", "weighted_pool", "transformer"])
def test_organ_encoders_accept_generic_features(aggregator):
    features = ["brain__feature_001", "heart__feature_001", "heart__feature_002"]
    assert list(group_feature_columns_by_organ(features)) == ["brain", "heart"]
    model, dim = build_encoder_from_config(encoder_cfg={"type": "organ", "aggregator": aggregator,
        "align_dim": 8, "organ_embed_dim": 4, "token_dim": 8, "shared_hidden_dims": [8],
        "nhead": 2, "num_layers": 1}, feature_cols=features)
    output = model(torch.randn(4, 3))
    assert output.shape == (4, dim)
    assert torch.isfinite(output).all()


def test_auc_handles_ties():
    assert roc_auc_score([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)
    assert roc_auc_score([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_release_audit_rejects_data_and_symlinks(tmp_path):
    (tmp_path / "source.py").write_text("print('synthetic example')\n")
    (tmp_path / "weights.pt").write_bytes(b"private artifact")
    (tmp_path / "link.py").symlink_to(tmp_path / "source.py")
    assert not audit_paths(tmp_path, ["source.py"])
    errors = audit_paths(tmp_path, ["weights.pt", "link.py"])
    assert any("unapproved file type" in item for item in errors)
    assert any("symlink" in item for item in errors)
