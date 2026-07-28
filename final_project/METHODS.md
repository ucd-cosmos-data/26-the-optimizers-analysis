# Analysis definitions

This file records the choices that must remain clear when interpreting the
figures.

## Conditions

- **Preictal:** the six non-overlapping 10-second bins from 60 seconds before an
  annotated seizure onset up to, but not including, the onset.
- **Interictal control:** a 60-second block from the same patient that is at
  least 15 minutes from every annotated seizure. Three non-overlapping controls
  are selected per seizure with random seed 42.
- A seizure is excluded if its preictal minute is incomplete, overlaps another
  seizure, or has insufficient patient-matched control time.

Two transparent source-annotation repairs are implemented: a spaced PN10 hour
(`1 6` → `16`) and a PN00 end time that is exactly one hour beyond its EDF
recording (`19` → `18`). The latter correction affects control selection, not
the pre-onset EEG. The event-specific PN00 correction is recorded in
`event_inventory.csv`; the PN10 spacing tolerance is part of the clock parser.

Fifteen minutes is an operational definition imposed by this dataset's
relatively short recordings. A 30- or 60-minute sensitivity analysis is
recommended. With three controls per seizure, either stricter setting retains
42 of the 47 seizures instead of all 47.

## Signal processing

1. Read channels whose EDF labels start with `EEG`.
2. Detrend each 10-second window.
3. Subtract the median across EEG channels at every time point (a robust common
   reference).
4. Reject individual channels that are nearly flat, materially clipped, or
   exceed 1,000 microvolts after detrending.
5. Require at least 10 usable EEG channels in a window.
6. Apply a 60 Hz notch filter to reduce electrical line noise.
7. Estimate power spectral density with four-second Hann-windowed Welch
   segments and 50% overlap.
8. Integrate delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta
   (13–30 Hz), and gamma (30–100 Hz).
9. Divide each band's power by the sum across the five bands.
10. Use the median relative power across usable EEG channels as the scalp-level
    value.

Gamma remains sensitive to muscle and movement artifacts even after these
steps. Gamma findings must be checked against the quality audit and treated
cautiously.

## Random sampled-point analysis

The sampled-point analysis is complementary to the six-bin Welch analysis
above. It estimates time-resolved power rather than treating raw instantaneous
EEG amplitude as frequency power:

1. Read the complete 60-second preictal or interictal block.
2. Apply the same detrending, common-median reference, channel-quality rules,
   and 60 Hz notch filter used by the primary analysis.
3. Apply separate fourth-order zero-phase Butterworth filters for delta
   (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma
   (30–100 Hz).
4. Square each filter's analytic-signal amplitude and smooth it over one
   second to estimate a continuous power envelope.
5. Divide each band's envelope by the sum across all five bands within each
   usable channel, then take the median across channels.
6. Split the preictal minute into ten non-overlapping 6-second bins and,
   using seed 42, select 10 distinct EEG sample indices from every bin.

This produces exactly 100 sampled times per seizure and 4,700 points per band.
The x value is
`log2(preictal relative power / patient interictal mean relative power)`.
Patient-specific interictal means and central 95% reference ranges are
estimated from equivalently stratified samples in all validated matched
control blocks for that patient. Color indicates whether a preictal point lies
inside or outside that patient's range.

For each band, the displayed ordinary least-squares line fits log2 deviation
as a function of seconds relative to onset. The listed `r` is the point-level
Pearson correlation. Because repeated samples from a seizure and repeated
seizures from a patient are not independent, these regressions and p-values
are descriptive and must not be interpreted as independent-observation
inference.

As a frequency-mapping sanity check, synthetic 2, 6, 10, 20, and 50 Hz waves
must be assigned to delta, theta, alpha, beta, and gamma, respectively. The
validation results are recorded in `band_filter_validation.csv`.

## Magnitude and temporal ordering

Each preictal value is normalized to that seizure's matched interictal controls:

```text
log2(preictal relative power / mean matched interictal relative power)
```

Time-course summaries first average seizures within each patient and then
average patients. The confidence intervals bootstrap patients, not individual
windows.

Temporal ordering is assessed from the six band trajectories and their
uncertainty. A visually earlier nonzero bin is descriptive evidence, not proof
of a biological change point.

## Consistency

The consistency analysis uses the final 20 seconds before onset and reports:

- Standard deviation across seizures
- Mean within-patient standard deviation for patients with multiple seizures
- Standard deviation across patient averages
- Percentage of seizures changing in the same direction, balanced by patient
- Percentage of patients changing in the same direction
- Direct beta/gamma versus lower-band variability comparisons at the
  within-patient and across-patient levels

Lower standard deviation and higher direction agreement indicate greater
consistency. These are descriptive effect-size comparisons; the small patient
count does not support strong confirmatory claims.
