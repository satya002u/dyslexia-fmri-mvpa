#!/usr/bin/env python3
"""
UNBIASED MVPA Analysis with Nested Cross-Validation
Fixes data leakage by doing feature selection INSIDE the CV loop

This is the CORRECT way to do MVPA for publication
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import plotting
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# --- NEW IMPORTS FOR ROI MASKING ---
from nilearn import datasets
from nilearn.masking import apply_mask, compute_brain_mask
from nilearn.image import resample_to_img
# -----------------------------------


class UnbiasedMVPA:
    """Unbiased MVPA with proper nested cross-validation"""
    
    def __init__(self, bids_root, glm_dir, output_root):
        self.bids_root = Path(bids_root)
        self.glm_dir = Path(glm_dir)
        self.output_root = Path(output_root)
        
        self.unbiased_dir = self.output_root / 'unbiased_mvpa'
        self.figures_dir = self.unbiased_dir / 'figures'
        
        for d in [self.unbiased_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.participants = pd.read_csv(self.bids_root / 'participants.tsv', sep='\t')
        print(f"✅ Loaded {len(self.participants)} participants")

    # --- NEW FUNCTION FOR ROI MASKING ---
    def create_reading_network_mask(self, reference_img):
        """
        Create a combined mask of reading-specific regions (Left Hemisphere)
        using the Harvard-Oxford cortical atlas.
        """
        print("\n   Creating Reading Network ROI Mask...")
        # Load Harvard-Oxford cortical atlas (max probability, 2mm resolution)
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = atlas.maps
        
        # Region names corresponding to the left-hemisphere reading network
        # These names must match the 'labels' list in the atlas object
        # Approximate names for the key regions:
        # IFG: 'Frontal Orbital Cortex', 'Inferior Frontal Gyrus, pars opercularis', 'Inferior Frontal Gyrus, pars triangularis'
        # VWFA: 'Fusiform Gyrus'
        # STG: 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division'
        # Angular Gyrus: 'Angular Gyrus'

        target_labels = [
            'Inferior Frontal Gyrus, pars triangularis',
            'Inferior Frontal Gyrus, pars opercularis',
            'Fusiform Gyrus', 
            'Superior Temporal Gyrus, posterior division', 
            'Angular Gyrus'
        ]
        
        # Get the corresponding indices from the atlas labels list
        # Note: Index 0 is background and is ignored. Indices are 1-based.
        region_indices = [i + 1 for i, name in enumerate(atlas.labels) if name in target_labels]

        if not region_indices:
            print("   ⚠️  Warning: No matching ROI labels found in the atlas!")
            # Fallback to the default mask if ROIs cannot be created
            return compute_brain_mask(reference_img, threshold=0.5)

        # Create a mask volume (all zeros initially)
        mask_data = np.zeros(atlas_img.get_fdata().shape, dtype=np.int8)

        # Set voxels corresponding to the target indices to 1
        for index in region_indices:
            mask_data[atlas_img.get_fdata() == index] = 1

        # Create the Nifti image for the mask
        mask_img = nib.Nifti1Image(mask_data, atlas_img.affine)

        # Resample the mask to match the resolution of the contrast maps
        # This is a critical step for consistent voxel space
        resampled_mask = resample_to_img(mask_img, reference_img, interpolation='nearest')
        
        print(f"   ✅ Mask created with {len(region_indices)} ROIs.")
        return resampled_mask
    # -----------------------------------
    
    def load_data(self, contrast='Reading', groups=['DL', 'TD']):
        """Load contrast maps and labels"""
        
        image_files = []
        labels = []
        subject_ids = []
        ages = []
        sexes = []
        
        for idx, row in self.participants.iterrows():
            if row['group'] not in groups:
                continue
            
            subject_id = row['participant_id']
            contrast_file = self.glm_dir / subject_id / 'averaged' / f'{contrast}_zmap_avg.nii.gz'
            
            if not contrast_file.exists():
                continue
            
            image_files.append(str(contrast_file))
            labels.append(row['group'])
            subject_ids.append(subject_id)
            ages.append(row['age'])
            sexes.append(row['sex'])
        
        return image_files, np.array(labels), subject_ids, np.array(ages), np.array(sexes)
    
    def check_group_balance(self, labels, ages, sexes, groups):
        """Check for demographic imbalances"""
        print("\n" + "="*70)
        print("DEMOGRAPHIC BALANCE CHECK")
        print("="*70)
        
        for group in groups:
            mask = labels == group
            group_ages = ages[mask]
            group_sexes = sexes[mask]
            
            print(f"\n{group}:")
            print(f"   N: {mask.sum()}")
            print(f"   Age: {group_ages.mean():.1f} ± {group_ages.std():.1f} years")
            print(f"   Sex: {(group_sexes=='M').sum()}M / {(group_sexes=='F').sum()}F")
        
        # Statistical tests
        if len(groups) == 2:
            mask1 = labels == groups[0]
            mask2 = labels == groups[1]
            
            # Age difference
            t_stat, p_age = stats.ttest_ind(ages[mask1], ages[mask2])
            print(f"\n   Age difference: t={t_stat:.2f}, p={p_age:.4f}")
            
            # Sex difference (chi-square)
            sex_table = np.array([
                [(sexes[mask1]=='M').sum(), (sexes[mask1]=='F').sum()],
                [(sexes[mask2]=='M').sum(), (sexes[mask2]=='F').sum()]
            ])
            chi2, p_sex = stats.chi2_contingency(sex_table)[:2]
            print(f"   Sex difference: χ²={chi2:.2f}, p={p_sex:.4f}")
            
            if p_age < 0.05 or p_sex < 0.05:
                print(f"\n   ⚠️  WARNING: Significant demographic differences!")
            else:
                print(f"\n   ✅ Groups are well-matched")
    
    def nested_cv_classification(self, image_files, labels, n_features=1000):
        """
        Nested cross-validation with feature selection INSIDE the loop
        This is the CORRECT unbiased approach
        """
        print("\n" + "="*70)
        print("NESTED CROSS-VALIDATION (Unbiased)")
        print("="*70)
        print(f"   Feature selection: top {n_features} voxels")
        print(f"   CV strategy: Leave-one-out")
        
        # Determine the reference image for mask resampling
        reference_img = image_files[0]
        
        # --- MODIFICATION: CREATE AND USE ROI MASK ---
        roi_mask_img = self.create_reading_network_mask(reference_img)
        
        # Create masker (fitted on all data - this is OK)
        masker = NiftiMasker(
            mask_img=roi_mask_img, # Use the new ROI mask
            standardize=True,
            memory='nilearn_cache',
            memory_level=1
        )
        # Note: We fit the masker only once on the list of image files to ensure 
        # consistent affine/shape handling across all images relative to the mask.
        masker.fit(image_files)
        
        # Extract ALL features for all subjects
        print("\n   Extracting features from all subjects using ROI mask...")
        X_all = []
        for img_file in image_files:
            # The masker handles both the masking and the flattening
            features = masker.transform(img_file)
            X_all.append(features.flatten())
        X_all = np.array(X_all)
        
        print(f"   Total features after ROI masking: {X_all.shape[1]} voxels")
        
        # Nested cross-validation
        print("\n   Running nested CV...")
        loo = LeaveOneOut()
        
        y_true = []
        y_pred = []
        y_scores = []
        
        fold_num = 0
        for train_idx, test_idx in loo.split(X_all):
            fold_num += 1
            
            # Split data
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Feature selection ONLY on training data (NO LEAKAGE!)
            selector = SelectKBest(f_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train classifier
            clf = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True)
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            pred = clf.predict(X_test_scaled)
            # Probability for the second class (groups[1])
            score = clf.predict_proba(X_test_scaled)[0, 1] 
            
            y_true.append(y_test[0])
            y_pred.append(pred[0])
            y_scores.append(score)
            
            if fold_num % 10 == 0:
                print(f"   Processed {fold_num}/{len(labels)} folds...")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        return y_true, y_pred, y_scores, masker
    
    def compute_metrics(self, y_true, y_pred, y_scores, groups):
        """Compute all classification metrics"""
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=groups)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=groups)
        
        # ROC curve (binary only)
        if len(groups) == 2:
            # Assuming groups[1] is the positive class for ROC (e.g., 'TD')
            y_true_binary = np.array([1 if y == groups[1] else 0 for y in y_true])
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None
            fpr, tpr = None, None
        
        # Per-class metrics
        # Assuming groups[0] is the primary negative class (e.g., 'DL')
        # Assuming groups[1] is the primary positive class (e.g., 'TD')
        
        # Sensitivity: True Positive Rate for the second group (groups[1], e.g., TD)
        sensitivity = cm[1, 1] / cm[1].sum()  
        # Specificity: True Negative Rate for the first group (groups[0], e.g., DL)
        specificity = cm[0, 0] / cm[0].sum()  
        
        # PPV: Positive Predictive Value (groups[1])
        # Since we use groups=[DL, TD] in load_data, the confusion matrix structure is:
        # Actual/Predicted | DL (0) | TD (1)
        # -----------------|--------|--------
        # DL (0)          | CM[0,0]| CM[0,1]
        # TD (1)          | CM[1,0]| CM[1,1]
        # PPV for TD (1): CM[1,1] / (CM[0,1] + CM[1,1]) -> Correct predictions of TD / All predicted TD
        ppv = cm[1, 1] / cm[:, 1].sum()       
        # NPV: Negative Predictive Value (groups[0])
        # NPV for DL (0): CM[0,0] / (CM[0,0] + CM[1,0]) -> Correct predictions of DL / All predicted DL
        npv = cm[0, 0] / cm[:, 0].sum()       
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'report': report
        }
    
    def plot_results(self, metrics, contrast_name, groups):
        """Create publication-quality plots"""
        
        groups_str = '_vs_'.join(groups)
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=groups, yticklabels=groups,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix: {contrast_name} ({groups_str})\n'
                    f'Accuracy: {metrics["accuracy"]*100:.1f}% (ROI Masked, Unbiased)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'confusion_roimasked_{contrast_name}_{groups_str}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve (binary only)
        if metrics['roc_auc'] is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=3,
                   label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Chance (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curve: {contrast_name} ({groups_str})\n(ROI Masked, Unbiased Nested CV)',
                        fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.figures_dir / f'roc_roimasked_{contrast_name}_{groups_str}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def check_motion_bias(self, y_pred, labels, subject_ids):
        """Check if misclassifications correlate with motion"""
        print("\n" + "="*70)
        print("MOTION BIAS CHECK")
        print("="*70)
        
        # Load QC data
        qc_file = self.output_root.parent / 'derivatives' / 'qc_report_v2.html'
        
        if not qc_file.exists():
            print("   ⚠️  QC report not found, skipping motion check")
            return
        
        # Try to parse QC data (simplified - would need proper HTML parsing)
        print("   Note: Manual inspection recommended")
        print("   Check if misclassified subjects have high motion")
        
        # Print misclassified subjects
        misclassified = y_pred != labels
        if misclassified.any():
            print(f"\n   Misclassified subjects ({misclassified.sum()}):")
            for subj in np.array(subject_ids)[misclassified]:
                print(f"      {subj}")
        else:
            print("\n   ✅ No misclassifications!")
    
    def analyze_contrast(self, contrast, groups=['DL', 'TD'], n_features=1000):
        """Complete unbiased analysis for one contrast"""
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + f" UNBIASED ANALYSIS: {contrast} ({' vs '.join(groups)}) (ROI MASKED) ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        # Load data
        image_files, labels, subject_ids, ages, sexes = self.load_data(contrast, groups)
        print(f"\n   Loaded {len(subject_ids)} subjects")
        
        # Check demographic balance
        self.check_group_balance(labels, ages, sexes, groups)
        
        # Nested CV classification
        y_true, y_pred, y_scores, masker = self.nested_cv_classification(
            image_files, labels, n_features
        )
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_scores, groups)
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS (UNBIASED, ROI MASKED)")
        print("="*70)
        print(f"\n   Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {metrics['sensitivity']*100:.1f}%")
        print(f"   Specificity: {metrics['specificity']*100:.1f}%")
        print(f"   PPV: {metrics['ppv']*100:.1f}%")
        print(f"   NPV: {metrics['npv']*100:.1f}%")
        if metrics['roc_auc'] is not None:
            print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
        
        print(f"\n   Classification Report:\n{metrics['report']}")
        
        # Plot results
        self.plot_results(metrics, contrast, groups)
        
        # Motion bias check
        self.check_motion_bias(y_pred, labels, subject_ids)
        
        # Save results
        results = {
            'contrast': contrast,
            'groups': groups,
            'n_subjects': len(subject_ids),
            'n_features': n_features,
            'accuracy': float(metrics['accuracy']),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'ppv': float(metrics['ppv']),
            'npv': float(metrics['npv']),
            'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] else None,
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'misclassified_subjects': [s for s, pred, true in zip(subject_ids, y_pred, y_true) if pred != true]
        }
        
        results_file = self.unbiased_dir / f'results_roimasked_{contrast}_{"_".join(groups)}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n   💾 Results saved: {results_file.name}")
        
        return results
    
    def compare_all_contrasts(self):
        """Test all contrasts with unbiased method"""
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " UNBIASED COMPARISON: ALL CONTRASTS (ROI MASKED) ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        # We will use the original single contrasts for testing the masking strategy
        contrasts = ['Reading', 'W', 'PH', 'W-PH', 'PH-W'] 
        
        all_results = []
        
        for contrast in contrasts:
            try:
                # Use the default n_features=1000, which now applies to the masked ROI space
                results = self.analyze_contrast(contrast, groups=['DL', 'TD'], n_features=1000) 
                all_results.append(results)
            except Exception as e:
                print(f"\n   ❌ Error in {contrast} analysis: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        if all_results:
            print("\n" + "="*70)
            print("SUMMARY: ALL CONTRASTS (ROI MASKED, UNBIASED)")
            print("="*70)
            
            summary_df = pd.DataFrame([
                {
                    'Contrast': r['contrast'],
                    'Accuracy': f"{r['accuracy']*100:.1f}%",
                    'Sensitivity': f"{r['sensitivity']*100:.1f}%",
                    'Specificity': f"{r['specificity']*100:.1f}%",
                    'ROC AUC': f"{r['roc_auc']:.3f}" if r['roc_auc'] else 'N/A'
                }
                for r in all_results
            ])
            
            print("\n" + summary_df.to_string(index=False))
            
            summary_df.to_csv(self.unbiased_dir / 'unbiased_summary_roimasked.csv', index=False)
            
            # Best contrast
            try:
                best_idx = np.argmax([r['accuracy'] for r in all_results])
                best = all_results[best_idx]
                
                print("\n" + "="*70)
                print(f"🏆 BEST CONTRAST (ROI MASKED): {best['contrast']} ({best['accuracy']*100:.1f}%)")
                print("="*70)
            except ValueError:
                print("\n   ⚠️ Could not determine best contrast (no successful runs).")

        
        return all_results


def main():
    """Main execution"""
    
    bids_root = "/workspace/aime_dyslexia/dataset_root"
    glm_dir = "/workspace/aime_dyslexia/analysis_output/first_level_glm"
    output_root = "/workspace/aime_dyslexia/analysis_output"
    
    print("\n" + "="*70)
    print("UNBIASED MVPA ANALYSIS")
    print("Nested Cross-Validation (ROI Masking Implemented)")
    print("="*70)
    
    analyzer = UnbiasedMVPA(bids_root, glm_dir, output_root)
    
    # Run all contrasts with the new ROI masking strategy
    results = analyzer.compare_all_contrasts()
    
    print("\n" + "="*70)
    print("✅ UNBIASED ANALYSIS (ROI MASKED) COMPLETE!")
    print("="*70)
    print(f"\nOutputs: {analyzer.unbiased_dir}")
    print(f"Figures: {analyzer.figures_dir}")
    
    print("\n" + "="*70)
    print("These are the TRUE unbiased results for publication (ROI Masked)!")
    print("="*70)


if __name__ == "__main__":
    main()
