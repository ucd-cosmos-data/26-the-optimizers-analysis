import json

import numpy as np
import pandas as pd
import pytest

import k_suiter
import personalized_channels_workflow as pc


def _patient(n_events: int = 4) -> pc.PatientFeatureData:
    rng = np.random.default_rng(21)
    rows = []
    channel_map = {
        "A": ["ch::A::relative_delta::mean"],
        "B": ["ch::B::relative_delta::mean"],
        "C": ["ch::C::relative_delta::mean"],
    }
    for event in range(n_events):
        for target in (0, 1):
            for window in range(8):
                rows.append(
                    {
                        "patient_id": "PX",
                        "source_event_id": f"E{event}",
                        "episode_type": "preictal" if target else "interictal",
                        "episode_id": f"E{event}_{'pre' if target else 'inter'}",
                        "recording": f"R{event}",
                        "landmark_seconds": float(window * 5),
                        "landmark_step": window,
                        "time_to_event_seconds": float((8 - window) * 5),
                        "has_event_in_5m": target,
                        # A is strongly useful, B weakly useful, C is noise.
                        "ch::A::relative_delta::mean": (
                            4.0 * target + rng.normal(scale=0.15)
                        ),
                        "ch::B::relative_delta::mean": (
                            0.5 * target + rng.normal(scale=0.8)
                        ),
                        "ch::C::relative_delta::mean": rng.normal(),
                    }
                )
    return pc.PatientFeatureData(
        patient_id="PX",
        frame=pd.DataFrame(rows),
        channel_names=list(channel_map),
        channel_feature_columns=channel_map,
    )


def test_patient_and_k_return_ranked_channels():
    patient = _patient()
    selector = k_suiter.KSuiter(
        pc.PersonalizedConfig(selector_max_iter=500)
    )

    channels = selector.recommend(patient, 2)

    assert channels[0] == "A"
    assert len(channels) == 2
    assert len(set(channels)) == 2
    assert selector.ranking_["rank"].tolist() == [1, 2]
    assert selector.ranking_.iloc[0]["marginal_value"] is None or np.isnan(
        selector.ranking_.iloc[0]["marginal_value"]
    )


def test_ranking_is_nested_and_deterministic():
    patient = _patient()
    top_two = k_suiter.recommend_channels(patient, 2)
    top_three = k_suiter.recommend_channels(patient, 3)

    assert top_three[:2] == top_two


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        k_suiter.recommend_channels(_patient(), k)


def test_rejects_k_larger_than_available_channels():
    with pytest.raises(ValueError, match="only 3 available"):
        k_suiter.recommend_channels(_patient(), 4)


def test_k_finder_handoff_is_loaded_and_saved_without_manual_copy(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    source = results / "sensor_count_step1.json"
    source.write_text(
        json.dumps({"K": 2, "noninferiority_margin": 0.02}),
        encoding="utf-8",
    )

    channels, recommendation_path, ranking_path = (
        k_suiter.recommend_and_save_from_k_finder(
            _patient(), project_dir=tmp_path
        )
    )

    saved = json.loads(recommendation_path.read_text(encoding="utf-8"))
    assert len(channels) == 2
    assert saved["k"] == 2
    assert saved["included_eeg_channels"] == channels
    assert saved["k_finder_metadata"]["noninferiority_margin"] == 0.02
    assert saved["final_held_out_metrics"]["held_out_source_event_id"] == "E3"
    assert saved["development_source_event_ids"] == ["E0", "E1", "E2"]
    assert (recommendation_path.parent / saved["stability_csv"]).exists()
    assert ranking_path.exists()


def test_k_finder_handoff_requires_a_final_untouched_event(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "sensor_count_step1.json").write_text('{"K": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="at least three seizure events"):
        k_suiter.recommend_and_save_from_k_finder(
            _patient(n_events=2), project_dir=tmp_path
        )


def test_k_finder_handoff_rejects_missing_or_invalid_result(tmp_path):
    with pytest.raises(FileNotFoundError, match="K-Finder result is missing"):
        k_suiter.load_k_finder_result(tmp_path)

    results = tmp_path / "results"
    results.mkdir()
    (results / "sensor_count_step1.json").write_text('{"K": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid K"):
        k_suiter.load_k_finder_result(tmp_path)
