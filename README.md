# Signal–Noise Alignment in IBL Brain-Wide Map VISp Data

This repository analyzes how stimulus-related population response differences relate to dominant residual/noise covariance modes in International Brain Laboratory (IBL) Brain-Wide Map Neuropixels recordings. The current analysis focuses on visual cortex, especially VISp, during the IBL visual decision-making task.

The central question is:

> Are visual stimulus coding directions aligned with, or separated from, the dominant low-dimensional modes of trial-to-trial population variability?

The current repository implements data loading, VISp session/insertion selection, firing-rate construction, trial/condition parsing, signal/noise geometry metrics, condition-mean geometry analysis, and a pre-first-stimulus baseline recording-duration check. Decoding, noise ablation, and full Fisher-information analyses are not treated as part of the current main README scope.

---

## 1. Dataset

This project uses the public IBL Brain-Wide Map (BWM) Neuropixels dataset. The associated paper is:

> International Brain Laboratory. **A brain-wide map of neural activity during complex behaviour.** *Nature* (2025).  
> https://www.nature.com/articles/s41586-025-09235-0

The BWM dataset contains large-scale Neuropixels recordings from mice performing a standardized visual decision-making task. The published BWM release reports 621,733 neurons recorded with 699 Neuropixels probes across 139 mice in 12 laboratories, covering 279 brain areas.

Useful IBL resources:

- Brain-Wide Map paper: https://www.nature.com/articles/s41586-025-09235-0
- IBL Brain-Wide Map page: https://www.internationalbrainlab.com/brainwide-map
- ONE quick start: https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html
- Public data download guide: https://docs.internationalbrainlab.org/notebooks_external/data_download.html
- Loading examples: https://docs.internationalbrainlab.org/loading_examples.html

---

## 2. Experimental protocol and behavioral task

The recordings come from the standardized IBL visual decision-making task used in the Brain-Wide Map study.

### Task structure

Mice are head-fixed in front of a screen and use their front paws to turn a wheel. On each trial, a visual stimulus appears in the left or right visual field. The mouse must turn the wheel to move the visual stimulus to the center of the screen. Correct choices are rewarded with water; incorrect choices receive negative feedback such as a white-noise pulse and timeout.

### Stimulus type

The stimulus is a peripheral visual grating. In the IBL task description, mice move a 35° peripheral visual grating to the center of the screen by turning a wheel. The stimulus can appear on the left or right side, and its contrast varies across trials.

In the ALF trial object, the visual stimulus is represented mainly by:

```text
contrastLeft
contrastRight
stimOn_times
stimOff_times
```

For most trials, one side has a nonzero contrast and the other side is zero. The analysis defines a signed contrast as:

```text
signed_contrast = contrastLeft - contrastRight
```

Therefore:

```text
signed_contrast > 0  -> left-side visual stimulus
signed_contrast < 0  -> right-side visual stimulus
signed_contrast = 0  -> zero-contrast / no visual evidence trial
```

The sign convention follows the repository's internal trial-selection code. If a downstream analysis uses right-choice labels or right-stimulus labels, the sign should be checked explicitly.

### Prior-probability blocks

The IBL task includes blockwise stimulus-side priors. The probability that the stimulus appears on the right alternates between biased blocks, commonly 0.2 and 0.8, with an initial unbiased block. This is available in the trial field:

```text
probabilityLeft
```

This repository currently focuses on stimulus-evoked population geometry. Behavioral variables such as choice, feedback, first movement time, and block prior are loaded for context but are not the main target of the current signal/noise geometry analysis.

### Trial fields used in this repository

The main trial fields are:

```text
stimOn_times
stimOff_times
contrastLeft
contrastRight
choice
feedbackType
probabilityLeft
firstMovement_times
intervals
```

The core stimulus-geometry analysis primarily uses:

```text
stimOn_times
stimOff_times
contrastLeft
contrastRight
```

---

## 3. Environment and data access

The repository uses the public OpenAlyx server through the ONE API.

Example setup:

```python
from one.api import ONE

ONE.setup(
    base_url="https://openalyx.internationalbrainlab.org",
    silent=True,
)

one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir="/scratch/midway3/xiaorantu/ONE",
)
```

On Midway, activate the IBL environment with:

```bash
source activate /scratch/midway3/xiaorantu/conda_envs/ibl
```

The cache directory should be placed on scratch rather than in the home directory because spike-sorting files are large.

---

## 4. Current repository organization

The current code is organized around reusable source files and scripts.

```text
signal_noise_alignment/
  README.md
  src/
    ibl_io.py
    firing_rates.py
    trial_selection.py
    alignment_metrics.py
    check_prestim_recording_period.py
  scripts/
    run_condition_geometry.py
  results/
    VISp_subjects_by_lab.json
```

The exact set of files may change as the analysis develops, but the current main pipeline is function-based rather than a single monolithic analyzer class.

### `src/ibl_io.py`

Handles IBL/ONE data access and session-level loading utilities, including:

- ONE setup with OpenAlyx
- reading the current VISp eid list from `results/VISp_subjects_by_lab.json`
- loading trial data
- selecting the best probe insertion for a target visual area
- loading spikes and clusters
- extracting cluster IDs for a target region such as VISp

### `src/firing_rates.py`

Builds trial-by-neuron firing-rate matrices from spike times and trial intervals.

The main static stimulus-response matrix is:

```text
X shape = n_trials × n_units
```

where firing rates are computed over the stimulus presentation window:

```text
stimOn_times -> stimOff_times
```

### `src/trial_selection.py`

Defines stimulus labels and condition masks from trial contrast fields.

The main variables are:

```text
signed_contrast = contrastLeft - contrastRight
high-contrast mask
positive stimulus mask
negative stimulus mask
condition masks for distinct signed contrasts
contrast-pair masks
```

This file determines which trials enter each signal/noise or condition-geometry computation.

### `src/alignment_metrics.py`

Implements the linear algebra used for population geometry analysis, including:

- condition means
- signal axes from mean response differences
- residual responses after condition-mean subtraction
- residual/noise covariance matrices
- eigendecomposition of noise covariance
- dominant noise subspaces
- signal/noise overlap metrics
- condition-mean PCA / condition geometry summaries

### `src/check_prestim_recording_period.py`

Checks whether the current VISp sessions contain neural recording before the first task stimulus onset. This script does not assume official IBL passive-period metadata. It asks a simpler matched-session question:

> For the selected VISp insertion, how much spike recording exists before the first `stimOn_times` event?

The output is saved as:

```text
results/pre_first_stim_recording_check.csv
```

This is useful for deciding whether a matched pre-task baseline covariance can be estimated from the same session and insertion.

---

## 5. Session and insertion selection

The current analysis uses a precomputed VISp session list:

```text
results/VISp_subjects_by_lab.json
```

The JSON is organized by lab and subject and stores the selected VIS eids. Downstream scripts build a flat eid list from this file.

For each eid, the code selects a probe insertion containing visual cortex units. The current target region is VISp or a VIS/VISp prefix depending on the analysis script. The selected insertion is then used to load spikes, clusters, and VISp cluster IDs.

---

## 6. Firing-rate construction

For each selected session and insertion:

1. Load trials.
2. Load spikes and clusters from the selected probe insertion.
3. Select clusters belonging to the target visual area.
4. Count spikes in the stimulus window.
5. Convert spike counts to firing rates.

For the static stimulus-response analysis:

```text
interval_i = [stimOn_times_i, stimOff_times_i]
X_i,n = spike_count_i,n / interval_duration_i
```

This gives:

```text
X ∈ R^{n_trials × n_units}
```

Low-variance or inactive units can be filtered before covariance estimation to avoid numerical degeneracy.

---

## 7. Trial and condition definitions

The main stimulus variable is signed contrast:

```text
signed_contrast = contrastLeft - contrastRight
```

The current analysis separates trials into positive and negative stimulus groups and can also construct condition-specific masks for each signed contrast value.

A typical high-contrast stimulus mask is defined by excluding zero-contrast trials and selecting trials whose absolute signed contrast is above a threshold:

```text
high_mask = (signed_contrast != 0) and (abs(signed_contrast) > threshold)
```

Then:

```text
pos_mask = high_mask and signed_contrast > 0
neg_mask = high_mask and signed_contrast < 0
```

These masks are used to compute stimulus signal axes and residual/noise covariance.

---

## 8. Signal/noise geometry definitions

For one session, define:

```text
X ∈ R^{T × N}
```

where `T` is the number of trials and `N` is the number of selected units.

For two stimulus groups:

```text
μ_pos = mean population response on positive stimulus trials
μ_neg = mean population response on negative stimulus trials
```

The stimulus signal vector is:

```text
Δμ = μ_pos - μ_neg
```

The normalized signal axis is:

```text
u_sig = Δμ / ||Δμ||
```

Residual responses are condition-centered:

```text
R_i = X_i - μ_pos, if trial i is positive
R_i = X_i - μ_neg, if trial i is negative
```

The residual/noise covariance is:

```text
C_noise = Cov(R)
```

After eigendecomposition:

```text
C_noise = U Λ Uᵀ
```

where:

```text
u_1 = first noise eigenvector
U_k = [u_1, ..., u_k]
```

The current repository computes the following core metrics.

### Top-1 signal/noise cosine

```text
top1_cos = |u_sigᵀ u_1|
```

This measures alignment between the stimulus axis and the dominant residual covariance mode.

### Top-k signal/noise overlap

```text
topk_overlap = ||U_kᵀ Δμ||² / ||Δμ||²
```

This measures how much stimulus mean-difference energy lies inside the top-k noise subspace.

### Noise spectrum and effective dimensionality

The repository also summarizes the residual covariance spectrum, including the top eigenvalues and participation ratio:

```text
PR = (sum_i λ_i)^2 / sum_i λ_i^2
```

Small participation ratio indicates that residual variability is concentrated in a low-dimensional subspace.

---

## 9. Condition-mean geometry

In addition to the binary high-contrast signal axis, the repository analyzes geometry across multiple signed contrast conditions.

For each signed contrast condition `c`, compute:

```text
μ_c = mean population response for trials with signed_contrast = c
```

The set of condition means forms a low-dimensional stimulus manifold inside neural population space. The repository analyzes this using condition-mean PCA and contrast-pair axes.

This addresses whether the single binary signal axis is a stable description of visual coding, or whether different contrast pairs induce different local stimulus axes.

Typical quantities include:

```text
condition means μ_c
PCA of condition means
contrast-pair axes Δμ_pair
cosine / alignment between contrast-pair axes
```

This analysis is important because the original high-contrast axis assumes a common left-right visual coding direction. The condition-geometry analysis checks whether that assumption is approximately valid across contrast conditions.

---

## 10. Pre-first-stimulus baseline check

The current BWM eids tested so far do not expose an official `passivePeriods` or `spontaneousActivity` dataset through the simple ONE query:

```python
one.load_dataset(eid, "*passivePeriods*", collection="alf")
```

However, the same sessions can contain spike recording before the first task stimulus onset. The current repository therefore includes a check for the duration of pre-first-stimulus recording in the selected VISp insertion.

This is not called official passive spontaneous activity. It is a matched pre-task baseline interval:

```text
recording_start -> first stimOn
```

Preliminary examples from the current eid list:

```text
b9c205c3-feac-485b-a89d-afc96d9cb280
  first stimOn: 22.443 s
  VISp pre-first-stim duration: 22.37 s
  VISp spikes before first stim: 1249

e1931de1-cf7b-49af-af33-2ade15e8abe7
  first stimOn: 10.544 s
  VISp pre-first-stim duration: 10.54 s
  VISp spikes before first stim: 5052
```

Interpretation:

- The current sessions do contain pre-first-stimulus neural activity.
- The interval is short, around 10–22 seconds in the first checked sessions.
- This interval may be useful as a matched pre-task baseline check.
- It should not be described as the official IBL passive spontaneous protocol unless passive-period metadata are explicitly available for that eid.

---

## 11. Current interpretation

The current repository supports the following analysis scope:

1. Load public IBL BWM task sessions and VISp units.
2. Construct stimulus-evoked population firing-rate matrices.
3. Define signed visual stimulus conditions from `contrastLeft` and `contrastRight`.
4. Compute stimulus signal axes from condition-averaged population responses.
5. Compute residual/noise covariance after subtracting condition means.
6. Compare stimulus axes with dominant residual/noise covariance modes.
7. Analyze whether condition means across contrast levels share a stable low-dimensional geometry.
8. Check whether matched pre-first-stimulus recording exists for baseline/null analyses.

The current working interpretation is:

> Dominant residual covariance modes provide a low-dimensional description of trial-to-trial population variability. The main analysis asks whether visual stimulus coding directions lie inside these dominant variability modes or avoid them. Condition-mean geometry is used to test whether the stimulus axis itself is stable across contrast conditions, rather than assuming one fixed binary signal axis.

The pre-first-stimulus result updates the null-model plan:

> The current task eids do not appear to expose official passive-period metadata through the simple `passivePeriods` object, but they do contain short matched pre-task recording intervals. These intervals can support a conservative pre-task baseline check, but they should be treated separately from a full passive spontaneous-state covariance analysis.

---

## 12. How to run the current checks

Activate the environment:

```bash
source activate /scratch/midway3/xiaorantu/conda_envs/ibl
```

Run the condition-geometry analysis from the repository root or the intended script location:

```bash
python scripts/run_condition_geometry.py
```

Run the pre-first-stimulus recording check:

```bash
cd /home/xiaorantu/signal_noise_alignment/src
python check_prestim_recording_period.py
```

The pre-first-stimulus check writes:

```text
results/pre_first_stim_recording_check.csv
```

For long checks over many sessions, use a Slurm job rather than running on the login node.

---

## 13. Notes on terminology

Use these terms carefully:

```text
residual/noise covariance
```

means covariance of trial-to-trial residuals after subtracting condition-specific evoked means.

```text
pre-first-stimulus baseline
```

means spike activity before the first task stimulus onset in the same session/insertion.

```text
passive spontaneous activity
```

should be reserved for sessions where IBL passive-period metadata are actually present and loadable.

```text
condition geometry
```

means the geometry of condition-averaged population responses across signed contrast values.

---

## 14. References

- International Brain Laboratory. **A brain-wide map of neural activity during complex behaviour.** *Nature* (2025). https://www.nature.com/articles/s41586-025-09235-0
- International Brain Laboratory Brain-Wide Map page. https://www.internationalbrainlab.com/brainwide-map
- Findling et al. **Brain-wide representations of prior information in mouse decision-making.** *Nature* (2025). https://www.nature.com/articles/s41586-025-09226-1
- IBL documentation. https://docs.internationalbrainlab.org/
