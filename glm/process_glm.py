#!/usr/bin/env python3
"""
STEP 1: GLM Processing - Run Once
Processes all subjects and saves contrast maps
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import json
import warnings
warnings.filterwarnings('ignore')

class GLMProcessor:
    """Process fMRI data with GLM - run once, save results"""
    
    def __init__(self, bids_root, fmriprep_root, output_root):
        self.bids_root = Path(bids_root)
        self.fmriprep_root = Path(fmriprep_root)
        self.output_root = Path(output_root)
        
        self.glm_dir = self.output_root / 'first_level_glm'
        self.glm_dir.mkdir(parents=True, exist_ok=True)
        
        self.participants = pd.read_csv(self.bids_root / 'participants.tsv', sep='\t')
        print(f" Loaded {len(self.participants)} participants")
        print(f" Groups: {self.participants['group'].value_counts().to_dict()}")
        
        self.tr = self._get_tr()
        
    def _get_tr(self):
        """Get TR from JSON"""
        first_subj = self.participants['participant_id'].iloc[0]
        json_file = list(self.bids_root.glob(f"{first_subj}/ses-1/func/*_bold.json"))[0]
        with open(json_file) as f:
            metadata = json.load(f)
        tr = metadata.get('RepetitionTime', 2.0)
        print(f"   TR: {tr}s")
        return tr
    
    def run_subject_glm(self, subject_id, run='01'):
        """Run GLM for single subject/run"""
        
        # Get files
        bold_file = self.fmriprep_root / subject_id / 'ses-1' / 'func' / \
                   f"{subject_id}_ses-1_task-read_run-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        
        events_file = self.bids_root / subject_id / 'ses-1' / 'func' / \
                     f"{subject_id}_ses-1_task-read_run-{run}_events.tsv"
        
        confounds_file = self.fmriprep_root / subject_id / 'ses-1' / 'func' / \
                        f"{subject_id}_ses-1_task-read_run-{run}_desc-confounds_timeseries.tsv"
        
        if not all([bold_file.exists(), events_file.exists(), confounds_file.exists()]):
            return None
        
        # Load events
        events = pd.read_csv(events_file, sep='\t')
        glm_events = events[events['event_type'].isin(['W', 'PH'])].copy()
        glm_events = glm_events[['onset', 'duration', 'event_type']]
        glm_events.columns = ['onset', 'duration', 'trial_type']
        
        # Load confounds
        confounds_df = pd.read_csv(confounds_file, sep='\t')
        confound_cols = [col for col in confounds_df.columns if 
                        col.startswith(('trans_', 'rot_')) or 
                        col in ['csf', 'white_matter', 'global_signal']]
        confounds = confounds_df[confound_cols].fillna(0)
        
        # Run GLM
        fmri_glm = FirstLevelModel(
            t_r=self.tr,
            noise_model='ar1',
            standardize=True,
            hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            smoothing_fwhm=6.0,
            verbose=0
        )
        
        fmri_glm = fmri_glm.fit(bold_file, events=glm_events, confounds=confounds)
        
        # Compute contrasts
        contrasts = {
            'W': 'W',
            'PH': 'PH',
            'W-PH': 'W - PH',
            'PH-W': 'PH - W',
            'Reading': '0.5*W + 0.5*PH'
        }
        
        contrast_maps = {}
        subj_output_dir = self.glm_dir / subject_id / f'run-{run}'
        subj_output_dir.mkdir(parents=True, exist_ok=True)
        
        for contrast_name, contrast_val in contrasts.items():
            try:
                z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
                output_file = subj_output_dir / f'{contrast_name}_zmap.nii.gz'
                z_map.to_filename(output_file)
                contrast_maps[contrast_name] = str(output_file)
            except:
                pass
        
        return contrast_maps
    
    def process_all(self):
        """Process all subjects"""
        print("\n" + "="*70)
        print("PROCESSING ALL SUBJECTS - GLM")
        print("="*70)
        
        results = []
        
        for idx, row in self.participants.iterrows():
            subject_id = row['participant_id']
            subj_dir = self.fmriprep_root / subject_id
            
            if not subj_dir.exists():
                print(f" Skipping {subject_id} (not preprocessed)")
                continue
            
            print(f"\n {subject_id} ({row['group']})")
            
            for run in ['01', '02', '03']:
                try:
                    maps = self.run_subject_glm(subject_id, run)
                    if maps:
                        print(f" run-{run}: {len(maps)} contrasts")
                        results.append({
                            'subject': subject_id,
                            'run': run,
                            'group': row['group'],
                            'status': 'success'
                        })
                except Exception as e:
                    print(f" run-{run}: {e}")
                    results.append({
                        'subject': subject_id,
                        'run': run,
                        'group': row['group'],
                        'status': f'failed'
                    })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.glm_dir / 'processing_summary.csv', index=False)
        
        print("\n" + "="*70)
        print("GLM PROCESSING COMPLETE")
        print("="*70)
        print(f"Total: {len(results)}")
        print(f"Success: {sum(1 for r in results if r['status']=='success')}")
        
        return results_df
    
    def average_runs(self):
        """Average runs within subjects"""
        print("\n" + "="*70)
        print("AVERAGING RUNS")
        print("="*70)
        
        for idx, row in self.participants.iterrows():
            subject_id = row['participant_id']
            subj_glm_dir = self.glm_dir / subject_id
            
            if not subj_glm_dir.exists():
                continue
            
            print(f" {subject_id}...", end='')
            
            avg_dir = subj_glm_dir / 'averaged'
            avg_dir.mkdir(exist_ok=True)
            
            contrasts = ['W', 'PH', 'W-PH', 'PH-W', 'Reading']
            
            for contrast in contrasts:
                run_maps = []
                for run in ['01', '02', '03']:
                    map_file = subj_glm_dir / f'run-{run}' / f'{contrast}_zmap.nii.gz'
                    if map_file.exists():
                        run_maps.append(nib.load(map_file))
                
                if len(run_maps) >= 2:
                    avg_data = np.mean([img.get_fdata() for img in run_maps], axis=0)
                    avg_img = nib.Nifti1Image(avg_data, run_maps[0].affine)
                    avg_file = avg_dir / f'{contrast}_zmap_avg.nii.gz'
                    avg_img.to_filename(avg_file)
            
            print(" yes")
        
        print("\n Averaging complete")


def main():
    """Main execution"""
    bids_root = "/workspace/aime_dyslexia/dataset_root"
    fmriprep_root = "/workspace/aime_dyslexia/derivatives/fmriprep"
    output_root = "/workspace/aime_dyslexia/analysis_output"
    
    print("\n" + "="*70)
    print("STEP 1: GLM PROCESSING")
    print("="*70)
    
    processor = GLMProcessor(bids_root, fmriprep_root, output_root)
    
    # Check if already processed
    summary_file = processor.glm_dir / 'processing_summary.csv'
    
    if summary_file.exists():
        print("\n⚠️  GLM already processed!")
        response = input("Reprocess? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print(" Using existing GLM results")
            return
    
    # Process
    results = processor.process_all()
    processor.average_runs()
    
    print("\n" + "="*70)
    print(" GLM PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {processor.glm_dir}")
    print("\n➡️  Next: Run MVPA analysis script")


if __name__ == "__main__":
    main()
