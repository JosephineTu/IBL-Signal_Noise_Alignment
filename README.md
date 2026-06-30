# Signal–Noise Alignment in IBL Brain-Wide Map Visual Cortex Data

This repository analyzes the geometry between stimulus signal axes and dominant residual/noise covariance modes in International Brain Laboratory (IBL) Neuropixels recordings, focusing initially on visual cortex populations such as VISp.

The current scope is **alignment baseline only**. Decoding analyses, noise ablation, and signal-to-noise prediction are promising follow-up directions, but they are intentionally not part of the first clean baseline pipeline.

---

## 1. Dataset

We use the public IBL Brain-Wide Map dataset, a large-scale Neuropixels dataset collected across multiple laboratories while mice performed the IBL visual decision-making task.

According to the IBL 2025 Brain Wide Map release page, the released dataset contains hundreds of Neuropixels sessions and probe insertions across many subjects and labs. The release is associated with the public tag:

```text
Brainwidemap
```

Useful IBL documentation:

* Brain Wide Map release page: https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html
* ONE quick start: https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html
* ONE data download / credentials: https://docs.internationalbrainlab.org/notebooks_external/data_download.html
* Loading examples: https://docs.internationalbrainlab.org/loading_examples.html

---

## 2. Environment setup

Create a clean Python environment. Example:

```bash
conda create -n ibl python=3.12
conda activate ibl
```

Install the ONE API and core scientific packages:

```bash
pip install ONE-api
pip install numpy pandas scipy scikit-learn matplotlib iblatlas brainbox
```

Depending on the cluster environment, some IBL packages may already be installed, or you may need to install the full IBL environment. The minimum package needed to access public IBL data is:

```bash
pip install ONE-api
```

---

## 3. Connecting to public IBL data with ONE

IBL public data are accessed through OpenAlyx using the ONE API. For the public server, the password is:

```text
international
```

This is not a private lab API key. It is the public password for the OpenAlyx public data server.

Minimal setup:

```python
from one.api import ONE

ONE.setup(
    base_url="https://openalyx.internationalbrainlab.org",
    silent=True,
)

one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir="/your/path/to/ONE",  # change this
)
```

For local testing, you can use a local cache directory instead:

```python
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    cache_dir="./ONE_cache",
)
```

The cache directory is important because spike-sorting data can be large. On the cluster, prefer scratch storage rather than home directory storage.

---

## 4. Searching for BWM sessions

The Brain-Wide Map release is associated with the tag `Brainwidemap`. A basic search can be done with:

```python
eids = one.search(tag="Brainwidemap")
print(len(eids))
print(eids[:5])
```

To check available search terms:

```python
print(one.search_terms())
```

To search for sessions with spike data:

```python
eids_with_spikes = one.search(
    tag="Brainwidemap",
    datasets="spikes.times.npy",
)
```

IBL sessions are identified by an experiment ID, usually called `eid`.

---

## 5. Loading trial data

For one session:

```python
eid = eids[0]
trials = one.load_object(eid, "trials", collection="alf")
print(trials.keys())
```

Trial fields used in this project usually include:

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

For the alignment baseline, the core variables are:

```python
stim_on = trials["stimOn_times"]
stim_off = trials["stimOff_times"]
contrast_left = trials["contrastLeft"]
contrast_right = trials["contrastRight"]
```

## 6. Loading spikes and clusters

For spike sorting data, use `SpikeSortingLoader`:

```python
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas

atlas = AllenAtlas()

# pid is a probe insertion id
sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)
```

The relevant spike arrays are typically:

```python
spikes["times"]      # spike times in seconds
spikes["clusters"]   # cluster id for each spike
clusters["acronym"]  # brain area acronym for each cluster
clusters["cluster_id"]
```

---

## 7. Selecting VISp recordings

The project currently uses a precomputed session list, generated upstream by `load_subjects_VISp.py` and saved as:

```text
VISp_subjects_by_lab.json
```

The downstream alignment code expects this JSON to contain entries like:

```text
lab -> subject -> VIS_eids
```

For a stricter VISp-only analysis, change the region rule within each session to exact or prefix matching for `VISp`.

---

## 8. Firing-rate matrix construction

For each session and selected probe insertion, the alignment analysis constructs a trial-by-neuron firing rate matrix:

```text
X shape = n_trials × n_units
```

For the static baseline, spikes are counted during the stimulus presentation window:

```text
stimOn_times -> stimOff_times
```

For the time-resolved baseline, spikes are counted in sliding windows relative to `stimOn_times`, for example:

```text
0.00–0.08 s
0.02–0.10 s
0.04–0.12 s
...
```

The time-resolved firing-rate tensor has shape:

```text
n_trials × n_time_bins × n_units
```

---

## 9. Alignment baseline definitions

For one session and one time bin, define:

```text
X: trial-by-neuron firing-rate matrix
T: number of trials
N: number of units
```

Stimulus labels are defined from signed contrast:

```text
y = +1 for one stimulus side
y = -1 for the opposite stimulus side
```

The signal axis is the difference in condition-averaged population responses:

```text
Δμ = μ_pos - μ_neg
u_sig = Δμ / ||Δμ||
```

Noise residuals are condition-centered responses:

```text
R_i = X_i - μ_pos, if trial i is positive
R_i = X_i - μ_neg, if trial i is negative
```

The residual/noise covariance is:

```text
C_noise = Cov(R)
```

We then eigendecompose the noise covariance:

```text
C_noise = U Λ Uᵀ
```

where:

```text
u_1 = top noise eigenvector
U_k = [u_1, ..., u_k]
```

The first baseline alignment metric is top-1 absolute cosine:

```text
top1_cos = |u_sigᵀ u_1|
```

The second metric is top-k signal energy overlap with the dominant noise subspace:

```text
topk_overlap = ||U_kᵀ Δμ||² / ||Δμ||²
```

For `k = 1`, this is equivalent to squared top-1 cosine:

```text
top1_overlap = top1_cos²
```

---

## 10. Random geometric baseline

Raw alignment values should be compared against a random geometric null model. If the signal axis has no privileged orientation relative to a random k-dimensional subspace in N-dimensional neural space, then:

```text
E[topk_overlap] = k / N
```

Therefore, for each session / time bin, report:

```text
topk_overlap_raw
topk_overlap_expected = k / N
topk_overlap_excess = topk_overlap_raw - topk_overlap_expected
topk_overlap_norm = (topk_overlap_raw - topk_overlap_expected) / (1 - topk_overlap_expected)
```

Here, `N` should be the effective number of units used in the covariance after filtering low-variance units, not necessarily the original number of clusters in the session.

A stricter null can be computed by random subspace sampling:

```python
# Pseudocode
for repeat in range(n_null):
    Q = random_orthonormal_matrix(N, k)
    null_overlap[repeat] = np.sum((Q.T @ delta_mu_unit) ** 2)

z = (observed_overlap - null_overlap.mean()) / null_overlap.std()
p = np.mean(null_overlap >= observed_overlap)
```

This answers:

```text
Is the observed signal-noise overlap larger than expected from random high-dimensional geometry after matching N and k?
```

---

## 11. Recommended Day-1 outputs

The first clean pipeline should produce one static baseline CSV and one time-resolved baseline CSV.

Suggested static CSV columns:

```text
eid
subject
lab
pid
n_trials
n_units
n_pos_high
n_neg_high
n_units_cov
sig_norm
noise_top1
noise_PR
top1_cos
top1_overlap
top1_expected
top1_excess
topk_overlap
topk_expected
topk_excess
topk_norm
topk_z
```

Suggested time-resolved CSV columns:

```text
eid
subject
lab
pid
bin_size
step_size
time
n_trials
n_units
n_pos_high
n_neg_high
n_units_cov
sig_norm
noise_top1
noise_PR
top1_cos
top1_overlap
topk_overlap
topk_expected
topk_excess
topk_norm
topk_z
```

Do not include decoding fields in the first baseline CSV. In particular, do not include:

```text
stim_auc
choice_auc
feedback_auc
2d_lda_auc
decoder_weight_frac
ablation_delta_auc
```

Those belong to a later functional/readout analysis stage.

---

## 12. Suggested repository structure

```text
signal_noise_alignment/
  README.md
  docs/
    ibl_data_access.md
    task_and_trial_definitions.md
    alignment_metrics.md
    controls.md
  src/
    data_loading.py
    session_selection.py
    firing_rates.py
    alignment_metrics.py
    null_models.py
  scripts/
    01_build_vis_session_list.py
    02_run_static_alignment_baseline.py
    03_run_time_resolved_alignment_baseline.py
    04_plot_static_alignment_summary.py
    05_plot_time_resolved_alignment.py
  results/
    .gitkeep
  figures/
    .gitkeep
```

---

## 13. Minimal analysis checklist

Before interpreting alignment, check and report:

```text
[ ] eid and pid
[ ] subject and lab
[ ] number of trials
[ ] number of high-contrast positive trials
[ ] number of high-contrast negative trials
[ ] number of VIS / VISp units before covariance filtering
[ ] number of units after covariance filtering
[ ] stimulus time window or time-bin definition
[ ] contrast threshold
[ ] top-k value
[ ] random geometric baseline k / N
```

This checklist is important because alignment values depend strongly on effective dimensionality, trial subset, contrast threshold, and time-bin size.

---
