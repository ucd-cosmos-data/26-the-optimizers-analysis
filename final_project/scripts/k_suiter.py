"""The K-Suiter: rank the top K EEG channels for one patient.

The selector is intentionally patient-specific.  Candidate channel sets are
evaluated with expanding chronological validation so future seizure episodes
never influence models fitted on earlier episodes.  The value function favors
discrimination on the imbalanced seizure target while mildly penalizing poor
probability calibration:

    value = validation AUPRC - 0.25 * validation Brier score

This is exploratory research code, not a clinical decision system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import pandas as pd

import personalized_channels_workflow as pc


@dataclass(frozen=True)
class ChannelRecommendation:
    """One channel's position and value in the greedy patient-specific ranking."""

    rank: int
    channel: str
    value: float
    marginal_value: float | None
    validation_auprc: float
    validation_brier: float


class KSuiter:
    """Rank channels that best suit a patient's seizure-forecasting history.

    Parameters
    ----------
    config:
        Reuses the personalized workflow's model and reproducibility settings.
        ``k`` and ``swap_refinement`` are overridden by :meth:`rank`; a ranking
        must be nested, so one-for-one swaps are deliberately not used.
    """

    def __init__(self, config: pc.PersonalizedConfig | None = None) -> None:
        self.config = pc.PersonalizedConfig() if config is None else config

    @staticmethod
    def _validate_patient(
        patient: pc.PatientFeatureData,
        k: int,
    ) -> dict[str, list[str]]:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(patient, pc.PatientFeatureData):
            raise TypeError("patient must be a PatientFeatureData instance.")
        frame = patient.frame
        required = {"patient_id", "source_event_id", "has_event_in_5m"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"patient.frame is missing columns: {missing}.")
        ids = frame["patient_id"].astype(str).unique()
        if len(ids) != 1 or ids[0] != str(patient.patient_id):
            raise ValueError(
                "patient.frame must contain exactly the declared patient_id."
            )
        if frame["source_event_id"].astype(str).nunique() < 2:
            raise ValueError(
                "At least two historical seizure events are required for "
                "chronological validation."
            )
        if set(frame["has_event_in_5m"].astype(int).unique()) != {0, 1}:
            raise ValueError("The patient history must contain both target classes.")

        # The full project cache contains a wide, regular feature map; use its
        # prespecified compact subset to control patient-level overfit.  A
        # caller may also supply an already-compact or custom feature map.
        expected_compact_width = (
            len(pc.COMPACT_FEATURE_NAMES) * len(pc.COMPACT_AGGREGATIONS)
        )
        try:
            compact_map = pc.compact_channel_column_map(
                patient.channel_feature_columns
            )
        except (AssertionError, ValueError):
            compact_map = {}
        channel_map = (
            compact_map
            if compact_map
            and all(
                len(columns) == expected_compact_width
                for columns in compact_map.values()
            )
            else {
                channel: list(columns)
                for channel, columns in patient.channel_feature_columns.items()
            }
        )
        if not channel_map or any(not columns for columns in channel_map.values()):
            raise ValueError("Every available channel needs at least one feature.")
        if k > len(channel_map):
            raise ValueError(
                f"Requested k={k}, but patient {patient.patient_id} has only "
                f"{len(channel_map)} available channels."
            )
        absent = sorted(
            {
                column
                for columns in channel_map.values()
                for column in columns
                if column not in frame.columns
            }
        )
        if absent:
            raise ValueError(
                f"patient.frame lacks {len(absent)} mapped feature columns."
            )
        return channel_map

    def rank(
        self,
        patient: pc.PatientFeatureData,
        k: int,
    ) -> list[ChannelRecommendation]:
        """Return the patient's top K channels in greedy conditional-value order.

        Rank 1 is the best channel alone.  Each later rank is the channel that
        adds the most validation value to the channels already selected.
        """

        channel_map = self._validate_patient(patient, k)
        config = replace(self.config, k=int(k), swap_refinement=False)
        config.validate()
        selected, trace, validation_method = pc.select_fixed_k_channels(
            patient.frame,
            channel_map,
            config,
        )
        if validation_method != "expanding_chronological_validation":
            raise RuntimeError(
                "K-Suiter requires at least two events and must not use "
                "training resubstitution."
            )

        values = trace["selection_score"].to_numpy(dtype=float)
        marginal = np.diff(values, prepend=np.nan)
        recommendations = [
            ChannelRecommendation(
                rank=position,
                channel=channel,
                value=float(values[position - 1]),
                marginal_value=(
                    None if position == 1 else float(marginal[position - 1])
                ),
                validation_auprc=float(
                    trace.iloc[position - 1]["validation_auprc"]
                ),
                validation_brier=float(
                    trace.iloc[position - 1]["validation_brier"]
                ),
            )
            for position, channel in enumerate(selected, start=1)
        ]
        self.patient_id_ = str(patient.patient_id)
        self.k_ = int(k)
        self.recommendations_ = recommendations
        self.ranking_ = pd.DataFrame(
            [recommendation.__dict__ for recommendation in recommendations]
        )
        return recommendations

    def recommend(
        self,
        patient: pc.PatientFeatureData,
        k: int,
    ) -> list[str]:
        """Return only the ordered channel names for ``patient`` and ``k``."""

        return [
            recommendation.channel
            for recommendation in self.rank(patient, k)
        ]


def recommend_channels(
    patient: pc.PatientFeatureData,
    k: int,
    *,
    config: pc.PersonalizedConfig | None = None,
) -> list[str]:
    """Convenience interface: ``patient + k -> ordered channel names``."""

    return KSuiter(config=config).recommend(patient, k)
