# The K-Suiter

The K-Suiter answers one question:

> Given one patient's channel-separated history and a fixed `k`, which `k`
> EEG channels best suit that patient?

## Algorithm

For each rank from 1 through `k`, the K-Suiter:

1. tries every remaining channel alongside the channels already chosen;
2. fits the same balanced logistic model on earlier seizure episodes;
3. validates it on the next episode using expanding chronological validation;
4. computes `value = AUPRC - 0.25 × Brier score`; and
5. adds the channel with the highest value.

AUPRC is the primary term because the seizure target is imbalanced. The Brier
term discourages poorly calibrated probabilities. The greedy search costs
`p + (p-1) + ... + (p-k+1)` evaluations rather than checking every one of the
`2^p` channel subsets. Fixed sorting makes ties reproducible.

The output is a nested ranking: asking for `k=4` always begins with the same
channels returned for `k=2`. A channel's value is conditional on channels that
precede it, so a negative marginal value does not mean the channel is harmful
in every possible combination.

## Input and output

```python
from k_suiter import KSuiter, recommend_channels
from personalized_channels_workflow import PatientFeatureData

patient = PatientFeatureData(
    patient_id="PN00",
    frame=patient_landmark_dataframe,
    channel_names=channel_names,
    channel_feature_columns=channel_to_feature_columns,
)

# Minimal patient + k -> channels interface
channels = recommend_channels(patient, k=4)

# Auditable interface
suiter = KSuiter()
recommendations = suiter.rank(patient, k=4)
print(suiter.ranking_)
```

`recommend_channels` returns channel names ordered from rank 1 to rank `k`.
`ranking_` also reports the conditional value, marginal value, validation
AUPRC, and validation Brier score at every rank.

At least two historical seizure events and both target classes are required.
Use only episodes available at recommendation time. This research ranking has
not been clinically validated and must not directly control patient care.
