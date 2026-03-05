#!/usr/bin/env python3
"""
STEP 2: MVPA Analysis - Multiple Approaches
Uses pre-computed GLM results to test different classification strategies
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import datasets, image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class MVPAAnalyzer:
    """Multiple MVPA approaches using pre-computed GLM results"""
    
    def __init__(self, bids_root, glm_dir, output_root):
        self.bids_root = Path(bids_root)
        self.glm_dir = Path(glm_dir)
        self.output_root = Path(output_root)
        
        self.mvpa_dir = self.output_root / 'mvpa'
        self.figures_dir = self.output_root / 'figures'
        
        for d in [self.mvpa_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.participants = pd.read_csv(self.bids_root / 'participants.tsv', sep='\t')
        
        print(f"✅ Loaded {len(self.participants)} participants")
        print(f"   Groups: {self.participants['group'].value_counts().to_dict()}")
    
    def load_contrast_maps(self, contrast='Reading', groups=None):
        """Load contrast maps for specified groups"""
        
        if groups is None:
            groups = ['DL', 'SpD', 'TD']
        
        image_files = []
        labels = []
        subject_ids = []
        
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
        
        return image_files, np.array(labels), subject_ids
    
    def extract_features_wholebrain(self, image_files):
        """Extract features from whole brain"""
        print("   Strategy: Whole brain")
        
        masker = NiftiMasker(
            mask_strategy='epi',
            standardize=True,
            memory='nilearn_cache',
            memory_level=1
        )
        
        masker.fit(image_files)
        
        feature_data = []
        for img_file in image_files:
            features = masker.transform(img_file)
            feature_data.append(features.flatten())
        
        X = np.array(feature_data)
        print(f"   Features: {X.shape[1]} voxels")
        
        return X, masker
    
    def extract_features_roi(self, image_files):
        """Extract features from reading network ROIs"""
        print("   Strategy: Reading network ROIs")
        
        # Define reading network coordinates (MNI space)
        # These are key regions for reading - well-separated to avoid overlap
        roi_coords = {
            'Left_IFG': (-44, 14, 28),          # Inferior frontal gyrus (Broca's)
            'Left_STG': (-56, -46, 14),         # Superior temporal gyrus
            'Left_Fusiform': (-42, -54, -12),   # Visual word form area
            'Left_IPL': (-48, -40, 40),         # Inferior parietal lobule
        }
        
        # Create sphere ROIs (6mm radius to avoid overlap)
        from nilearn import image as nimg
        from nilearn.maskers import NiftiSpheresMasker
        
        coords = list(roi_coords.values())
        
        masker = NiftiSpheresMasker(
            coords,
            radius=6.0,
            standardize=True,
            memory='nilearn_cache',
            memory_level=1,
            allow_overlap=False
        )
        
        masker.fit(image_files)
        
        feature_data = []
        for img_file in image_files:
            features = masker.transform(img_file)
            feature_data.append(features.flatten())
        
        X = np.array(feature_data)
        print(f"   Features: {X.shape[1]} voxels (from {len(roi_coords)} ROIs)")
        
        return X, masker
    
    def extract_features_selected(self, image_files, labels, n_features=1000):
        """Extract features with feature selection"""
        print(f"   Strategy: Feature selection (top {n_features})")
        
        # First get whole brain features
        masker = NiftiMasker(
            mask_strategy='epi',
            standardize=True,
            memory='nilearn_cache',
            memory_level=1
        )
        
        masker.fit(image_files)
        
        feature_data = []
        for img_file in image_files:
            features = masker.transform(img_file)
            feature_data.append(features.flatten())
        
        X_all = np.array(feature_data)
        
        # Feature selection using ANOVA F-test
        selector = SelectKBest(f_classif, k=n_features)
        X = selector.fit_transform(X_all, labels)
        
        print(f"   Features: {X.shape[1]} selected from {X_all.shape[1]}")
        
        return X, (masker, selector)
    
    def classify(self, X, y, subjects, approach_name):
        """Run classification with leave-one-out CV"""
        print(f"\n   Running classification...")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Classifier
        clf = SVC(kernel='linear', C=1.0, class_weight='balanced')
        
        # Leave-one-out CV
        loo = LeaveOneOut()
        y_pred = []
        y_true = []
        
        for train_idx, test_idx in loo.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            
            y_pred.append(pred[0])
            y_true.append(y_test[0])
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        unique_labels = np.unique(y)
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Save results
        results = {
            'approach': approach_name,
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'labels': unique_labels.tolist(),
            'predictions': list(zip(subjects, y_true.tolist(), y_pred.tolist()))
        }
        
        results_file = self.mvpa_dir / f'{approach_name}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print results
        print(f"\n   ✅ Accuracy: {accuracy*100:.1f}%")
        print(f"\n   Classification Report:\n{report}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, unique_labels, accuracy, approach_name)
        
        return results
    
    def plot_confusion_matrix(self, cm, labels, accuracy, name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'{name}\nAccuracy: {accuracy*100:.1f}%', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.figures_dir / f'{name}_confusion.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved: {output_file.name}")
    
    def run_all_approaches(self, contrast='Reading'):
        """Run all MVPA approaches"""
        print("\n" + "="*70)
        print(f"MVPA ANALYSIS - Contrast: {contrast}")
        print("="*70)
        
        all_results = {}
        
        # ==================================================
        # APPROACH 1: Binary DL vs TD (Whole Brain)
        # ==================================================
        print("\n" + "─"*70)
        print("APPROACH 1: Binary DL vs TD (Whole Brain)")
        print("─"*70)
        
        image_files, labels, subjects = self.load_contrast_maps(contrast, groups=['DL', 'TD'])
        print(f"   Subjects: {len(subjects)} (DL={sum(labels=='DL')}, TD={sum(labels=='TD')})")
        
        X, masker = self.extract_features_wholebrain(image_files)
        results = self.classify(X, labels, subjects, 'Binary_DL_vs_TD_WholeBrain')
        all_results['Binary_DL_vs_TD_WholeBrain'] = results
        
        # ==================================================
        # APPROACH 2: Binary DL vs TD (ROI)
        # ==================================================
        print("\n" + "─"*70)
        print("APPROACH 2: Binary DL vs TD (Reading ROIs)")
        print("─"*70)
        
        X, masker = self.extract_features_roi(image_files)
        results = self.classify(X, labels, subjects, 'Binary_DL_vs_TD_ROI')
        all_results['Binary_DL_vs_TD_ROI'] = results
        
        # ==================================================
        # APPROACH 3: Binary DL vs TD (Feature Selection)
        # ==================================================
        print("\n" + "─"*70)
        print("APPROACH 3: Binary DL vs TD (Feature Selection)")
        print("─"*70)
        
        X, masker = self.extract_features_selected(image_files, labels, n_features=1000)
        results = self.classify(X, labels, subjects, 'Binary_DL_vs_TD_FeatureSelection')
        all_results['Binary_DL_vs_TD_FeatureSelection'] = results
        
        # ==================================================
        # APPROACH 4: Three-way (Feature Selection)
        # ==================================================
        print("\n" + "─"*70)
        print("APPROACH 4: Three-way DL vs SpD vs TD (Feature Selection)")
        print("─"*70)
        
        image_files, labels, subjects = self.load_contrast_maps(contrast, groups=['DL', 'SpD', 'TD'])
        print(f"   Subjects: {len(subjects)} (DL={sum(labels=='DL')}, SpD={sum(labels=='SpD')}, TD={sum(labels=='TD')})")
        
        X, masker = self.extract_features_selected(image_files, labels, n_features=1000)
        results = self.classify(X, labels, subjects, 'ThreeWay_FeatureSelection')
        all_results['ThreeWay_FeatureSelection'] = results
        
        # ==================================================
        # APPROACH 5: Three-way (ROI)
        # ==================================================
        print("\n" + "─"*70)
        print("APPROACH 5: Three-way DL vs SpD vs TD (Reading ROIs)")
        print("─"*70)
        
        X, masker = self.extract_features_roi(image_files)
        results = self.classify(X, labels, subjects, 'ThreeWay_ROI')
        all_results['ThreeWay_ROI'] = results
        
        # ==================================================
        # Summary
        # ==================================================
        print("\n" + "="*70)
        print("SUMMARY OF ALL APPROACHES")
        print("="*70)
        
        summary = []
        for name, res in all_results.items():
            summary.append({
                'Approach': name,
                'Accuracy': f"{res['accuracy']*100:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary)
        print("\n" + summary_df.to_string(index=False))
        
        summary_df.to_csv(self.mvpa_dir / 'summary.csv', index=False)
        
        return all_results


def main():
    """Main execution"""
    bids_root = "/workspace/aime_dyslexia/dataset_root"
    glm_dir = "/workspace/aime_dyslexia/analysis_output/first_level_glm"
    output_root = "/workspace/aime_dyslexia/analysis_output"
    
    # Check if GLM is complete
    if not Path(glm_dir).exists():
        print("❌ GLM not found! Run Step 1 first.")
        return
    
    print("\n" + "="*70)
    print("STEP 2: MVPA ANALYSIS")
    print("="*70)
    
    analyzer = MVPAAnalyzer(bids_root, glm_dir, output_root)
    
    # Ask which contrasts to test
    print("\nAvailable contrasts:")
    print("  1. Reading (default)")
    print("  2. W (Word only)")
    print("  3. PH (Pseudoword only)")
    print("  4. W-PH (Lexical advantage)")
    print("  5. PH-W (Phonological load)")
    
    choice = input("\nSelect contrast [1]: ").strip() or '1'
    
    contrast_map = {
        '1': 'Reading',
        '2': 'W',
        '3': 'PH',
        '4': 'W-PH',
        '5': 'PH-W'
    }
    
    contrast = contrast_map.get(choice, 'Reading')
    
    print(f"\n➡️  Testing with: {contrast}")
    
    # Run all approaches
    results = analyzer.run_all_approaches(contrast)
    
    print("\n" + "="*70)
    print("✅ MVPA ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults: {analyzer.mvpa_dir}")
    print(f"Figures: {analyzer.figures_dir}")


if __name__ == "__main__":
    main()