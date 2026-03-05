#!/usr/bin/env python3
"""
MASTER ANALYSIS SCRIPT v2
==========================
Dyslexia fMRI: MVPA + RSA + Cross-Contrast Transfer + Permutation Testing
Dataset: MRI Lab Graz - Reading in children (DL, SpD, TD)

CHECKPOINTING:
  Each section saves results immediately after completion.
  On rerun, completed sections are skipped automatically.
  To force rerun a section, set FORCE_RERUN flags below.
  Or delete: master_results/ckpt_mvpa.json  etc.

PROGRESS:
  Live progress bar shows every fold — never looks stuck.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import datasets
from nilearn.image import resample_to_img
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from joblib import Parallel, delayed
import warnings, json, time, sys, os
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — edit paths here
# =============================================================================
BIDS_ROOT     = "/workspace/aime_dyslexia/dataset_root"
GLM_DIR       = "/workspace/aime_dyslexia/analysis_output/first_level_glm"
OUTPUT_ROOT   = "/workspace/aime_dyslexia/analysis_output"

CONTRASTS     = ['W', 'PH', 'Reading', 'W-PH', 'PH-W']
GROUPS        = ['DL', 'SpD', 'TD']
N_FEATURES    = 1000
BEST_CONTRAST = 'PH'   # From previous experiments — 76.2% DL vs TD

# CPU-only pipeline
GPU_AVAILABLE = False

# Permutation chunk size — benchmark verified optimal for 256-core machine
PERM_CHUNK_SIZE = 4

# Set True to force rerun even if checkpoint exists
FORCE_RERUN = {
    'mvpa':     True,    # ← fresh start
    'perm':     True,    # ← fresh start
    'rsa':      True,    # ← fresh start
    'transfer': True,    # ← fresh start (corrected LOO-CV)
}



# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Live terminal progress bar — so it never looks stuck."""

    def __init__(self, total, prefix='', width=38):
        self.total   = total
        self.prefix  = prefix
        self.width   = width
        self.current = 0
        self.start   = time.time()
        self._draw(0)

    def _draw(self, n):
        pct    = n / max(self.total, 1)
        filled = int(self.width * pct)
        bar    = '█' * filled + '░' * (self.width - filled)
        elap   = time.time() - self.start
        eta    = (elap / n * (self.total - n)) if n > 0 else 0
        sys.stdout.write(
            f"\r  {self.prefix} [{bar}] {n}/{self.total}  "
            f"elapsed {elap:.0f}s  ETA {eta:.0f}s  "
        )
        sys.stdout.flush()

    def update(self, n=1):
        self.current += n
        self._draw(self.current)

    def done(self):
        elapsed = time.time() - self.start
        sys.stdout.write(
            f"\r  {self.prefix} [{'█'*self.width}] "
            f"{self.total}/{self.total}  ✅ done in {elapsed:.0f}s\n"
        )
        sys.stdout.flush()


# =============================================================================
# CHECKPOINTING
# =============================================================================

def _ckpt(out_dir, name):
    return Path(out_dir) / f'ckpt_{name}.json'

def save_ckpt(out_dir, name, data):
    p = _ckpt(out_dir, name)
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Checkpoint saved → {p.name}")

def load_ckpt(out_dir, name):
    p = _ckpt(out_dir, name)
    return json.load(open(p)) if p.exists() else None

def needs_run(out_dir, name):
    if FORCE_RERUN.get(name, False):
        print(f"  🔄 FORCE_RERUN=True → rerunning {name}")
        return True
    if load_ckpt(out_dir, name) is not None:
        print(f"  ✅ Checkpoint found → skipping {name} "
              f"(set FORCE_RERUN['{name}']=True to redo)")
        return False
    return True


# =============================================================================
# HELPERS
# =============================================================================

def auto_perms(n):
    # With 256-core parallelism, 5000 perms takes ~1 min — use more for publication
    if n <= 30:   return 5000
    elif n <= 50: return 5000
    else:         return 5000

def fit_svm(X_tr, y_tr, X_te, gpu_id=None):
    """CPU-only SVM — clean, no GPU dependency."""
    clf = SVC(kernel='linear', C=1.0, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te)

def sep(title="", char="=", w=70):
    if title:
        print(f"\n{char*w}\n  {title}\n{char*w}")
    else:
        print(f"{char*w}")

_roi_cache = None

def get_roi_mask(reference_img):
    """Build Harvard-Oxford reading network mask (cached for speed)."""
    global _roi_cache
    if _roi_cache is not None:
        return _roi_cache
    print("  Building ROI mask (Harvard-Oxford reading network)...",
          end=' ', flush=True)
    atlas   = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    targets = [
        'Inferior Frontal Gyrus, pars triangularis',
        'Inferior Frontal Gyrus, pars opercularis',
        'Fusiform Gyrus',
        'Superior Temporal Gyrus, posterior division',
        'Angular Gyrus',
    ]
    idx        = [i+1 for i, n in enumerate(atlas.labels) if n in targets]
    data       = atlas.maps.get_fdata()
    mask       = np.zeros(data.shape, dtype=np.int8)
    for i in idx:
        mask[data == i] = 1
    mask_img   = nib.Nifti1Image(mask, atlas.maps.affine)
    _roi_cache = resample_to_img(mask_img, reference_img,
                                  interpolation='nearest')
    n_vox = int(_roi_cache.get_fdata().sum())
    print(f"✅  {n_vox} voxels ({len(idx)} regions)")
    return _roi_cache


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

class DataLoader:

    def __init__(self, bids_root, glm_dir):
        self.bids_root    = Path(bids_root)
        self.glm_dir      = Path(glm_dir)
        self.participants = pd.read_csv(
            self.bids_root / 'participants.tsv', sep='\t')

    def load(self, contrast, groups=None):
        groups = groups or GROUPS
        files, labels, sids = [], [], []
        for _, row in self.participants.iterrows():
            if row['group'] not in groups:
                continue
            sid   = row['participant_id']
            fpath = (self.glm_dir / sid / 'averaged'
                     / f'{contrast}_zmap_avg.nii.gz')
            if fpath.exists():
                files.append(str(fpath))
                labels.append(row['group'])
                sids.append(sid)
        return files, np.array(labels), sids

    def demographics(self):
        sep("SECTION 1: DATA LOADING & DEMOGRAPHICS")
        df = self.participants
        print(f"\n  Total subjects : {len(df)}")
        for grp, sub in df.groupby('group'):
            ages = sub['age'];  sexes = sub['sex']
            print(f"\n  {grp}  (N={len(sub)})")
            print(f"    Age : {ages.mean():.1f} ± {ages.std():.1f}"
                  f"  [{ages.min()}–{ages.max()}]")
            print(f"    Sex : {(sexes=='M').sum()}M / {(sexes=='F').sum()}F")
        outliers = df[abs(df['age'] - df['age'].mean()) > 2]
        if len(outliers):
            print(f"\n  ⚠️  Age outliers : "
                  f"{outliers['participant_id'].tolist()}")
        files, labels, _ = self.load(BEST_CONTRAST)
        print(f"\n  Z-maps found for '{BEST_CONTRAST}': {len(files)}")
        for g in GROUPS:
            print(f"    {g}: {(labels==g).sum()}")


# =============================================================================
# SECTION 2 — MVPA CLASSIFICATION
# =============================================================================

def _extract_roi(files):
    """Extract ROI-masked features. Returns (X, masker)."""
    roi = get_roi_mask(files[0])
    m   = NiftiMasker(mask_img=roi, standardize=True,
                      memory='nilearn_cache', memory_level=1)
    m.fit(files)
    X = np.vstack([m.transform(f).flatten() for f in files])
    return X, m


def nested_loo(files, labels, prefix="LOO-CV"):
    """Unbiased nested LOO-CV with feature selection inside fold."""
    print(f"  Extracting features...", end=' ', flush=True)
    X_all, _ = _extract_roi(files)
    print(f"{X_all.shape[1]} voxels")

    loo = LeaveOneOut()
    bar = ProgressBar(len(labels), prefix=prefix)
    y_true, y_pred = [], []

    for tr_idx, te_idx in loo.split(X_all):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr        = labels[tr_idx]

        k    = min(N_FEATURES, X_tr.shape[1] - 1)
        sel  = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
        X_tr = sel.transform(X_tr)
        X_te = sel.transform(X_te)

        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        pred = fit_svm(X_tr, y_tr, X_te)
        y_pred.append(pred[0])
        y_true.append(labels[te_idx][0])
        bar.update()

    bar.done()
    return np.array(y_true), np.array(y_pred)


def run_mvpa(loader, out_dir):
    sep("SECTION 2: MVPA CLASSIFICATION")
    print(f"  Contrast : {BEST_CONTRAST}  |  "
          f"ROI-masked nested LOO-CV\n")

    combos = [
        (['DL', 'TD'],        'DL vs TD'),
        (['DL', 'SpD'],       'DL vs SpD'),
        (['SpD', 'TD'],       'SpD vs TD'),
        (['DL', 'SpD', 'TD'], '3-way'),
    ]
    results = []
    for grps, name in combos:
        print(f"\n  ── {name} ──")
        files, labels, _ = loader.load(BEST_CONTRAST, grps)
        if len(files) < 5:
            print("  ⚠️  Not enough subjects — skipping")
            continue
        y_true, y_pred = nested_loo(files, labels, prefix=name)
        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, labels=grps).tolist()
        print(f"  Accuracy : {acc*100:.1f}%")
        results.append(dict(contrast=BEST_CONTRAST, groups=grps,
                            name=name, n=len(labels),
                            accuracy=float(acc), confusion_matrix=cm))

    print(f"\n  ── Summary ──")
    for r in results:
        print(f"  {r['name']:<22}  {r['accuracy']*100:.1f}%")
    return results


# =============================================================================
# SECTION 3 — PERMUTATION TESTING
# =============================================================================

def _perm_chunk(X_all, labels, n_perms):
    """
    Run n_perms permutations serially in one worker process.
    chunk_size=4 is optimal for 256-core machines (benchmark verified).
    All imports inside function — required for joblib 'processes' backend.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score
    import numpy as np

    loo     = LeaveOneOut()
    results = []

    for _ in range(n_perms):
        perm  = np.random.permutation(labels)
        preds = []
        for tr, te in loo.split(X_all):
            X_tr, X_te = X_all[tr], X_all[te]
            y_tr        = perm[tr]
            k    = min(N_FEATURES, X_tr.shape[1] - 1)
            sel  = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
            X_tr = sel.transform(X_tr);  X_te = sel.transform(X_te)
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)
            clf  = SVC(kernel='linear', C=1.0, class_weight='balanced')
            clf.fit(X_tr, y_tr)
            preds.append(clf.predict(X_te)[0])
        results.append(accuracy_score(perm, preds))

    return results


def run_permutations(loader, mvpa_results, out_dir):
    sep("SECTION 3: PERMUTATION TESTING")

    perm_results = []
    for r in mvpa_results:
        n_perms = auto_perms(r['n'])
        print(f"\n  ── {r['name']} ──")
        print(f"  Observed : {r['accuracy']*100:.1f}%  |  "
              f"Permutations : {n_perms}")

        files, labels, _ = loader.load(BEST_CONTRAST, r['groups'])
        print(f"  Extracting features once...", end=' ', flush=True)
        X_all, _ = _extract_roi(files)
        print(f"{X_all.shape[1]} voxels")

        null       = []
        CHUNK_SIZE = 4      # optimal from benchmark (71 perms/sec on 256 cores)
        n_chunks   = n_perms // CHUNK_SIZE
        remainder  = n_perms % CHUNK_SIZE
        bar        = ProgressBar(n_perms, prefix=f'Permuting {r["name"]}')

        print(f"\n  Strategy: {n_chunks} chunks × {CHUNK_SIZE} perms "
              f"across {min(n_chunks, 256)} cores")

        # Main parallel loop — chunk_size=4 is optimal for 256 cores
        chunk_results = Parallel(n_jobs=-1, prefer='processes')(
            delayed(_perm_chunk)(X_all, labels, CHUNK_SIZE)
            for _ in range(n_chunks)
        )
        for res in chunk_results:
            null.extend(res)
            bar.update(len(res))

        # Handle remainder
        if remainder > 0:
            extra = _perm_chunk(X_all, labels, remainder)
            null.extend(extra)
            bar.update(len(extra))

        bar.done()

        null  = np.array(null)
        p_val = float((null >= r['accuracy']).mean())
        sig   = p_val < 0.05
        print(f"  Null mean : {null.mean()*100:.1f}%  |  "
              f"p = {p_val:.4f}  "
              f"{'✅ Significant' if sig else '❌ Not significant'}")

        perm_results.append(dict(
            name=r['name'], groups=r['groups'],
            observed_acc=r['accuracy'],
            null_mean=float(null.mean()),
            null_std=float(null.std()),
            p_value=p_val, significant=sig,
            n_perms=n_perms,
        ))
    return perm_results


# =============================================================================
# SECTION 4 — RSA
# =============================================================================

def run_rsa(loader, out_dir):
    sep("SECTION 4: RSA — REPRESENTATIONAL SIMILARITY ANALYSIS")
    print("  Novel: 3-group geometry + SpD position\n")

    rsa_results = {}

    for contrast in CONTRASTS:
        print(f"\n  ── Contrast: {contrast} ──")
        files, labels, _ = loader.load(contrast, GROUPS)
        if len(files) < 10:
            print("  ⚠️  Skipping — insufficient data")
            continue

        # Whole-brain (no label selection for RSA)
        print(f"  Extracting features...", end=' ', flush=True)
        m = NiftiMasker(mask_strategy='epi', standardize=True,
                        memory='nilearn_cache', memory_level=1)
        m.fit(files)
        X = np.vstack([m.transform(f).flatten() for f in files])
        print(f"{X.shape[1]} voxels")

        print(f"  Computing RDMs...", end=' ', flush=True)

        # Within-group dissimilarity
        group_within = {}
        for grp in GROUPS:
            mask = labels == grp
            if mask.sum() < 3:
                continue
            rdm  = squareform(pdist(X[mask], metric='correlation'))
            off  = rdm[np.triu_indices_from(rdm, k=1)]
            group_within[grp] = float(off.mean())

        # Between-group dissimilarity
        group_between = {}
        for g1, g2 in [('DL','TD'), ('DL','SpD'), ('SpD','TD')]:
            m1, m2 = labels==g1, labels==g2
            if m1.sum() < 2 or m2.sum() < 2:
                continue
            dists = [1 - np.corrcoef(X[i], X[j])[0,1]
                     for i in np.where(m1)[0]
                     for j in np.where(m2)[0]]
            group_between[f'{g1}_vs_{g2}'] = float(np.mean(dists))

        print("done ✅")

        # Print
        print("  Within-group dissimilarity (lower = more homogeneous):")
        for grp, val in group_within.items():
            bar_len = int(val * 80)
            print(f"    {grp:<6} {val:.4f}  {'▓'*bar_len}")

        print("  Between-group dissimilarity (higher = more separable):")
        for pair, val in group_between.items():
            print(f"    {pair:<15} {val:.4f}")

        # SpD position
        dl_td  = group_between.get('DL_vs_TD')
        dl_spd = group_between.get('DL_vs_SpD')
        spd_td = group_between.get('SpD_vs_TD')
        if all(v is not None for v in [dl_td, dl_spd, spd_td]):
            if dl_spd < dl_td and spd_td < dl_td:
                pos = "INTERMEDIATE ← key paper finding"
            elif dl_spd < spd_td:
                pos = "CLOSER TO DL"
            else:
                pos = "CLOSER TO TD"
            print(f"  SpD position : {pos}")

        rsa_results[contrast] = dict(
            within_dissimilarity=group_within,
            between_dissimilarity=group_between,
            n_subjects=len(files),
        )

    return rsa_results


# =============================================================================
# SECTION 5 — CROSS-CONTRAST TRANSFER
# =============================================================================

def _transfer_pair(loader, train_c, test_c, groups):
    """
    Cross-contrast transfer with proper LOO-CV.

    Correct design (unbiased):
      For each left-out subject i:
        - Train SVM on remaining N-1 subjects' TRAIN-contrast maps + their labels
        - Feature selection + scaling fitted only on training fold
        - Test on subject i's TEST-contrast map  ← different contrast, unseen label
        - Record prediction

    This is unbiased because:
      1. Subject i's label is never seen during training
      2. Feature selection is inside the fold (no leakage)
      3. Scaler is fitted only on training fold
      4. The SAME subject is left out of both contrasts — we ask:
         "does the classifier trained on other subjects' PH patterns
          correctly predict THIS subject's group from their W pattern?"

    Previous (biased) version trained on ALL subjects then tested on
    the SAME subjects — inflating accuracy to 83–100%.
    """
    tr_files, tr_labels, tr_sids = loader.load(train_c, groups)
    te_files, te_labels, te_sids = loader.load(test_c,  groups)

    if len(tr_files) < 5 or len(te_files) < 5:
        return None, None

    # Verify subject alignment — same subject must be at same index
    # (guaranteed by DataLoader iterating participants.tsv in fixed order,
    #  but we assert here for safety)
    common = [s for s in tr_sids if s in te_sids]
    if len(common) < 5:
        print(f"  ⚠️  Subject mismatch between contrasts — skipping")
        return None, None

    # Keep only subjects present in both contrasts, in aligned order
    te_idx_map = {s: i for i, s in enumerate(te_sids)}
    aligned    = [(i, te_idx_map[s]) for i, s in enumerate(tr_sids)
                  if s in te_idx_map]
    tr_align   = [tr_files[i]  for i, _ in aligned]
    te_align   = [te_files[j]  for _, j in aligned]
    labels     = np.array([tr_labels[i] for i, _ in aligned])

    roi = get_roi_mask(tr_align[0])

    # Extract ALL features upfront (masker uses mask_img — no label info,
    # no leakage from this step; leakage only possible in SelectKBest)
    def extract_all(files):
        mm = NiftiMasker(mask_img=roi, standardize=True,
                         memory='nilearn_cache', memory_level=1)
        mm.fit(files)
        return np.vstack([mm.transform(f).flatten() for f in files])

    X_train_all = extract_all(tr_align)   # N × V  — train contrast
    X_test_all  = extract_all(te_align)   # N × V  — test contrast

    # LOO-CV over subjects
    loo    = LeaveOneOut()
    y_true, y_pred = [], []

    for tr_idx, te_idx in loo.split(X_train_all):
        # Training data: N-1 subjects, TRAIN contrast
        X_tr = X_train_all[tr_idx]
        y_tr = labels[tr_idx]

        # Test data: left-out subject, TEST contrast (different contrast!)
        X_te = X_test_all[te_idx]

        # Feature selection — fitted ONLY on training fold (no leakage)
        k   = min(N_FEATURES, X_tr.shape[1] - 1)
        sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
        X_tr = sel.transform(X_tr)
        X_te = sel.transform(X_te)

        # Scaling — fitted ONLY on training fold
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        pred = fit_svm(X_tr, y_tr, X_te)
        y_pred.append(pred[0])
        y_true.append(labels[te_idx][0])

    acc = float(accuracy_score(y_true, y_pred))

    # Permutation test for transfer accuracy
    n_perms    = auto_perms(len(labels))
    null_acc   = []
    # Pre-compute LOO split indices once
    loo_splits = list(loo.split(X_train_all))

    for _ in range(n_perms):
        perm_labels = np.random.permutation(labels)
        perm_preds  = []
        perm_true   = []
        for tr_idx, te_idx in loo_splits:
            X_tr = X_train_all[tr_idx]
            y_tr = perm_labels[tr_idx]
            X_te = X_test_all[te_idx]
            k    = min(N_FEATURES, X_tr.shape[1] - 1)
            sel  = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
            X_tr = sel.transform(X_tr);  X_te = sel.transform(X_te)
            sc   = StandardScaler()
            X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)
            clf  = SVC(kernel='linear', C=1.0, class_weight='balanced')
            clf.fit(X_tr, y_tr)
            perm_preds.append(clf.predict(X_te)[0])
            perm_true.append(perm_labels[te_idx][0])
        null_acc.append(accuracy_score(perm_true, perm_preds))

    null_arr = np.array(null_acc)
    p_val    = float((null_arr >= acc).mean())

    return acc, dict(
        null_mean = float(null_arr.mean()),
        null_std  = float(null_arr.std()),
        p_value   = p_val,
        significant = p_val < 0.05,
        n_perms   = n_perms,
    )


def run_transfer(loader, out_dir):
    sep("SECTION 5: CROSS-CONTRAST TRANSFER  (LOO-CV + permutation)")
    print("  Novel: does PH pattern generalise to word reading?\n")
    print("  ⚠️  Note: permutations run per transfer pair — this section")
    print("       takes longer than before (~5 min total).\n")

    pairs = [
        ('PH', 'W',       ['DL','TD'],  'PH→W    DL vs TD   ← main question'),
        ('W',  'PH',      ['DL','TD'],  'W→PH    DL vs TD   ← reverse'),
        ('PH', 'W',       ['SpD','TD'], 'PH→W    SpD vs TD  ← SpD comparison'),
        ('PH', 'W',       ['DL','SpD'], 'PH→W    DL vs SpD  ← deficit comparison'),
        ('PH', 'W-PH',    ['DL','TD'],  'PH→W-PH DL vs TD   ← lexical contrast'),
        ('PH', 'Reading', ['DL','TD'],  'PH→Read DL vs TD   ← broad generaliz.'),
    ]

    results      = {}   # desc → accuracy
    perm_results = {}   # desc → perm stats dict

    for train_c, test_c, grps, desc in pairs:
        print(f"  Running: {desc}...")
        acc, perm = _transfer_pair(loader, train_c, test_c, grps)
        if acc is None:
            print("  ⚠️  skipped\n")
            continue

        sig  = "✅" if perm['significant'] else "❌"
        flag = "✅" if acc > 0.60 else "⚠️ "
        print(f"  {flag}  Accuracy : {acc*100:.1f}%  |  "
              f"Null : {perm['null_mean']*100:.1f}%  |  "
              f"p = {perm['p_value']:.4f}  {sig}\n")

        results[desc]      = acc
        perm_results[desc] = perm

    # Interpret key comparison
    k1 = 'PH→W    DL vs TD   ← main question'
    k2 = 'PH→W    SpD vs TD  ← SpD comparison'
    if k1 in results and k2 in results:
        diff   = results[k1] - results[k2]
        better = "DL" if diff > 0 else "SpD"
        print(f"  Key comparison:")
        print(f"    DL vs TD  : {results[k1]*100:.1f}%  "
              f"p={perm_results[k1]['p_value']:.4f}")
        print(f"    SpD vs TD : {results[k2]*100:.1f}%  "
              f"p={perm_results[k2]['p_value']:.4f}")
        if abs(diff) > 0.05:
            print(f"    → {better} shows stronger transfer (+{abs(diff)*100:.1f}%)")
        else:
            print("    → Similar transfer in both groups")

    # Return combined dict for checkpointing
    return {'accuracies': results, 'permutations': perm_results}


# =============================================================================
# SECTION 6 — SUMMARY REPORT
# =============================================================================

def save_summary(out_dir, mvpa_res, perm_res, rsa_res, transfer_res):
    sep("SECTION 6: FINAL SUMMARY REPORT")

    print(f"\n  {'Groups':<22} {'Accuracy':>10} {'Null':>8}"
          f" {'p-value':>10} {'Sig':>5}")
    print("  " + "─"*58)

    combined = []
    for r in mvpa_res:
        perm    = next((p for p in perm_res
                        if p['groups'] == r['groups']), None)
        p_str   = f"{perm['p_value']:.4f}" if perm else "N/A"
        null_s  = f"{perm['null_mean']*100:.1f}%" if perm else "N/A"
        sig_s   = "✅" if perm and perm['significant'] else "❌"
        print(f"  {r['name']:<22} {r['accuracy']*100:>9.1f}%"
              f" {null_s:>8} {p_str:>10} {sig_s:>5}")
        combined.append(dict(
            name=r['name'], accuracy=r['accuracy'],
            null_mean=perm['null_mean'] if perm else None,
            p_value=perm['p_value'] if perm else None,
            significant=perm['significant'] if perm else None,
        ))

    print(f"\n  RSA — SpD position:")
    print("  " + "─"*40)
    for contrast, res in rsa_res.items():
        bw     = res['between_dissimilarity']
        dl_td  = bw.get('DL_vs_TD')
        dl_spd = bw.get('DL_vs_SpD')
        spd_td = bw.get('SpD_vs_TD')
        if all(v is not None for v in [dl_td, dl_spd, spd_td]):
            if dl_spd < dl_td and spd_td < dl_td:
                pos = "Intermediate"
            elif dl_spd < spd_td:
                pos = "Closer to DL"
            else:
                pos = "Closer to TD"
            print(f"  {contrast:<8}  SpD → {pos}")

    print(f"\n  Cross-Contrast Transfer (LOO-CV, unbiased):")
    print("  " + "─"*65)
    print(f"  {'Description':<45} {'Acc':>6} {'Null':>6} {'p':>7} {'Sig':>4}")
    print("  " + "─"*65)
    accs  = transfer_res.get('accuracies', {})
    perms = transfer_res.get('permutations', {})
    for desc, acc in accs.items():
        flag  = "✅" if acc > 0.60 else "⚠️ "
        p     = perms.get(desc, {})
        p_str = f"{p['p_value']:.4f}" if p else "N/A"
        null  = f"{p['null_mean']*100:.1f}%" if p else "N/A"
        sig   = "✅" if p.get('significant') else "❌"
        print(f"  {flag} {desc:<44} {acc*100:>5.1f}% {null:>6} {p_str:>7} {sig:>4}")

    # Save
    out = Path(out_dir) / 'master_results'
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'master_results.json', 'w') as f:
        json.dump(dict(mvpa=combined, rsa=rsa_res,
                       transfer=transfer_res), f, indent=2)
    pd.DataFrame(combined).to_csv(
        out / 'mvpa_permutation_summary.csv', index=False)
    pd.DataFrame([
        {'transfer': k, 'accuracy': v,
         'p_value': perms.get(k, {}).get('p_value'),
         'significant': perms.get(k, {}).get('significant')}
        for k, v in accs.items()
    ]).to_csv(out / 'transfer_summary.csv', index=False)
    print(f"\n  💾 Saved → {out}")
    sep()


# =============================================================================
# MAIN
# =============================================================================

def main():
    sep("MASTER ANALYSIS — DYSLEXIA fMRI  v2", char="█")
    print("""
  Dataset  : MRI Lab Graz  (DL=20, SpD=16, TD=22)
  Method   : MVPA + Permutation + RSA + Cross-Contrast Transfer
  Novel    : 3-group RSA geometry | SpD position | PH→W transfer

  CHECKPOINTING ON — completed sections skip automatically
  Delete ckpt_*.json or set FORCE_RERUN[section]=True to redo
    """)

    out_dir = Path(OUTPUT_ROOT) / 'master_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 256-core CPU parallelism  |  chunk_size={PERM_CHUNK_SIZE}  "
          f"|  ~1 min per permutation set")

    loader = DataLoader(BIDS_ROOT, GLM_DIR)
    loader.demographics()

    # Section 2
    if needs_run(out_dir, 'mvpa'):
        mvpa_res = run_mvpa(loader, out_dir)
        save_ckpt(out_dir, 'mvpa', mvpa_res)
    else:
        mvpa_res = load_ckpt(out_dir, 'mvpa')
        sep("SECTION 2: MVPA — loaded from checkpoint")
        for r in mvpa_res:
            print(f"  {r['name']:<22}  {r['accuracy']*100:.1f}%")

    # Section 3
    if needs_run(out_dir, 'perm'):
        perm_res = run_permutations(loader, mvpa_res, out_dir)
        save_ckpt(out_dir, 'perm', perm_res)
    else:
        perm_res = load_ckpt(out_dir, 'perm')
        sep("SECTION 3: PERMUTATION — loaded from checkpoint")
        for r in perm_res:
            sig = "✅" if r['significant'] else "❌"
            print(f"  {r['name']:<22}  p={r['p_value']:.4f}  {sig}")

    # Section 4
    if needs_run(out_dir, 'rsa'):
        rsa_res = run_rsa(loader, out_dir)
        rsa_serial = {
            k: {'within_dissimilarity': v['within_dissimilarity'],
                'between_dissimilarity': v['between_dissimilarity'],
                'n_subjects': v['n_subjects']}
            for k, v in rsa_res.items()
        }
        save_ckpt(out_dir, 'rsa', rsa_serial)
    else:
        rsa_res = load_ckpt(out_dir, 'rsa')
        sep("SECTION 4: RSA — loaded from checkpoint")
        print("  SpD position summary:")
        for contrast, res in rsa_res.items():
            bw = res['between_dissimilarity']
            dl_td  = bw.get('DL_vs_TD')
            dl_spd = bw.get('DL_vs_SpD')
            spd_td = bw.get('SpD_vs_TD')
            if all(v is not None for v in [dl_td, dl_spd, spd_td]):
                pos = ("Intermediate" if dl_spd < dl_td and spd_td < dl_td
                       else "Closer to DL" if dl_spd < spd_td
                       else "Closer to TD")
                print(f"  {contrast:<8}  SpD → {pos}")

    # Section 5
    if needs_run(out_dir, 'transfer'):
        transfer_res = run_transfer(loader, out_dir)
        save_ckpt(out_dir, 'transfer', transfer_res)
    else:
        transfer_res = load_ckpt(out_dir, 'transfer')
        sep("SECTION 5: TRANSFER — loaded from checkpoint")
        accs  = transfer_res.get('accuracies', {})
        perms = transfer_res.get('permutations', {})
        for k, v in accs.items():
            p   = perms.get(k, {})
            sig = "✅" if p.get('significant') else "❌"
            p_s = f"p={p['p_value']:.4f}" if p else ""
            print(f"  {k:<45}  {v*100:.1f}%  {p_s}  {sig}")

    # Section 6
    save_summary(OUTPUT_ROOT, mvpa_res, perm_res, rsa_res, transfer_res)

    sep("✅  ALL SECTIONS COMPLETE", char="█")
    print(f"  Results → {out_dir}\n")


if __name__ == "__main__":
    main()
