# Neural representations of reading disorders in children

Secondary analysis of a publicly available fMRI dataset comparing multivariate brain activation
patterns in children with dyslexia, isolated spelling deficit, and typical development.

## Dataset

Data source: OpenNeuro ds003126 v1.3.1
https://openneuro.org/datasets/ds003126/versions/1.3.1

Original study: Banfi, C., Koschutnig, K., Moll, K., Schulte-Korne, G., Fink, A., & Landerl, K. (2021).
Reading-related functional activity in children with isolated spelling deficits and dyslexia.
Language, Cognition and Neuroscience, 36(5), 543-561.
https://doi.org/10.1080/23273798.2020.1859569

58 German-speaking children aged 8-10 years:
- 20 with dyslexia (DL)
- 16 with isolated spelling deficit (SpD)
- 22 typically developing readers (TD)

Task: visual reading-aloud task with real words and pseudohomophones (event-related design,
3 runs x 40 items each).


## What this repository contains

This repository holds the analysis code for a secondary fMRI study applying multivariate
pattern analysis (MVPA) and representational similarity analysis (RSA) to the Banfi et al.
dataset to ask whether dyslexia and isolated spelling deficit are neurally distinguishable.

Main findings:
- MVPA reliably separates DL from TD during pseudohomophone reading (76.2%, p=0.007)
- DL and SpD are neurally indistinguishable (36.1%, p=0.895)
- RSA shows SpD occupies an intermediate representational position between DL and TD
- Cross-contrast transfer is largely contrast-specific


## Repository structure

```
.
├── master_analysis.py          # Run this. Executes all analyses with checkpointing.
│
├── preprocessing/
│   ├── bids_validator.py       # Validate BIDS structure before processing
│   ├── fmriprep_runner.py      # Run fMRIPrep in parallel across subjects
│   └── qc_checker.py          # Quality control checks on fMRIPrep outputs
│
├── glm/
│   └── process_glm.py         # First-level GLM estimation and contrast computation
│
├── mvpa/
│   ├── mvpa_approaches.py      # MVPA with multiple classification strategies
│   ├── mvpa_unbiased.py        # Unbiased MVPA with nested CV (used in final analysis)
│   └── three_way_classification.py  # Three-way and all pairwise group comparisons
│
├── figures/
│   └── generate_figures.py    # Reproduces all manuscript figures and tables
│
├── utils/
│   └── activations.py         # Custom activation functions (experimental, not used in final analysis)
│
├── requirements.txt
└── README.md
```


## How to reproduce the analysis

### 1. Download the data

```bash
# Install openneuro-py if needed
pip install openneuro-py

# Download dataset
openneuro download --dataset ds003126 --tag 1.3.1 /path/to/dataset_root
```

Or download manually from https://openneuro.org/datasets/ds003126/versions/1.3.1

### 2. Set up the environment

```bash
python -m venv fmri_env
source fmri_env/bin/activate
pip install -r requirements.txt
```

### 3. Run preprocessing

Edit the paths at the top of each script before running.

```bash
# Validate BIDS structure
python preprocessing/bids_validator.py

# Run fMRIPrep (requires Docker and a FreeSurfer license)
python preprocessing/fmriprep_runner.py

# Check preprocessing quality
python preprocessing/qc_checker.py
```

fMRIPrep version used: `nipreps/fmriprep:latest` (confirm exact version from your run logs).
A FreeSurfer license is required. See https://surfer.nmr.mgh.harvard.edu/registration.html

### 4. Estimate first-level GLMs

```bash
python glm/process_glm.py
```

This fits a GLM per subject per run, computes five contrast z-maps
(W, PH, Reading, W-PH, PH-W), and averages them across runs.

### 5. Run the main analysis

```bash
python master_analysis.py
```

This runs all analysis sections in order with automatic checkpointing.
Completed sections are skipped on rerun. To force a section to rerun,
delete the corresponding checkpoint file in `analysis_output/master_results/`.

Checkpoint files:
- `ckpt_mvpa.json`
- `ckpt_perm.json`
- `ckpt_rsa.json`
- `ckpt_transfer.json`
- `ckpt_sensitivity.json`

### 6. Generate figures and tables

```bash
python figures/generate_figures.py
```

Outputs saved to `analysis_output/figures/` (PDF, 300 dpi) and
`analysis_output/tables/` (Word .docx).


## Paths configuration

All scripts read paths from variables defined at the top of each file.
Before running, edit these to match your local setup:

```python
BIDS_ROOT   = "/path/to/dataset_root"
GLM_DIR     = "/path/to/analysis_output/first_level_glm"
OUTPUT_ROOT = "/path/to/analysis_output"
```


## Compute environment

Analysis was run on a Linux machine with 256 CPU cores.
Permutation testing is parallelised with joblib (n_jobs=-1).

The pipeline is CPU-only. GPU acceleration was tested but found to be slower
than CPU for this sample size due to data transfer overhead.

Estimated runtimes on a standard workstation (8 cores):
- GLM estimation: ~2-4 hours (all subjects)
- MVPA + permutations (5000): ~3-6 hours
- RSA: ~30 minutes
- Transfer analysis: ~2-4 hours


## Requirements

See `requirements.txt`. Key dependencies:

- Python >= 3.10
- nilearn >= 0.10
- nibabel >= 5.0
- scikit-learn >= 1.3
- scipy >= 1.11
- numpy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7
- joblib >= 1.3
- python-docx >= 1.0

fMRIPrep is run via Docker and is not a Python dependency.


## Notes on the analysis design

A key design principle throughout is the avoidance of data leakage in MVPA.
Feature selection (SelectKBest, k=1000) and signal normalisation (StandardScaler)
are fitted exclusively on training data within each cross-validation fold and applied
without refitting to the test fold. This is enforced in `mvpa_unbiased.py` and
carried through into `master_analysis.py`.

Earlier scripts (`mvpa_approaches.py`) were exploratory and may not enforce this
constraint. The unbiased pipeline in `master_analysis.py` is what is reported in
the manuscript.

Permutation testing shuffles labels before any fitting step, including feature
selection, to ensure the null distribution reflects the full analysis procedure.


## Citation

If you use this code, please cite the original dataset:

Banfi, C., Koschutnig, K., Moll, K., Schulte-Korne, G., Fink, A., & Landerl, K. (2021).
Reading-related functional activity in children with isolated spelling deficits and dyslexia.
Language, Cognition and Neuroscience, 36(5), 543-561.
https://doi.org/10.1080/23273798.2020.1859569

A citation for this analysis code will be added upon publication.

Code repository: https://github.com/satya002u/dyslexia-fmri-mvpa


## License

Code: MIT License (see LICENSE file).
Data: Subject to the OpenNeuro terms of use and the original data sharing agreement.
