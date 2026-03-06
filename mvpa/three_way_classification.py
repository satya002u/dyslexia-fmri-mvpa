#!/usr/bin/env python3
"""
Three-Way and Pairwise Classification Analysis
Tests ALL group combinations to leverage SpD controls
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nilearn.maskers import NiftiMasker
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveGroupAnalysis:
    """Test all possible group combinations"""
    
    def __init__(self, bids_root, glm_dir, output_root):
        self.bids_root = Path(bids_root)
        self.glm_dir = Path(glm_dir)
        self.output_root = Path(output_root)
        
        self.group_analysis_dir = self.output_root / 'group_analysis'
        self.figures_dir = self.group_analysis_dir / 'figures'
        
        for d in [self.group_analysis_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.participants = pd.read_csv(self.bids_root / 'participants.tsv', sep='\t')
        
        print(f"Loaded {len(self.participants)} participants")
        print(f" Groups: {self.participants['group'].value_counts().to_dict()}")
    
    def load_data(self, contrast, groups):
        """Load contrast maps for specified groups"""
        
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
    
    def nested_cv_classification(self, image_files, labels, groups, n_features=1000):
        """
        Nested cross-validation classification
        Works for both binary and multi-class
        """
        
        # Create masker
        masker = NiftiMasker(
            mask_strategy='epi',
            standardize=True,
            memory='nilearn_cache',
            memory_level=1
        )
        masker.fit(image_files)
        
        # Extract features
        print(f"   Extracting features...")
        X_all = []
        for img_file in image_files:
            features = masker.transform(img_file)
            X_all.append(features.flatten())
        X_all = np.array(X_all)
        
        print(f"   Total features: {X_all.shape[1]}")
        print(f"   Running nested CV...")
        
        # Nested cross-validation
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        fold_num = 0
        for train_idx, test_idx in loo.split(X_all):
            fold_num += 1
            
            # Split data
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Feature selection (inside CV)
            k = min(n_features, X_train.shape[1] - 1)
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # Train classifier
            clf = SVC(kernel='linear', C=1.0, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            pred = clf.predict(X_test_scaled)
            
            y_true.append(y_test[0])
            y_pred.append(pred[0])
            
            if fold_num % 10 == 0:
                print(f"   Processed {fold_num}/{len(labels)} folds...")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return y_true, y_pred
    
    def compute_metrics(self, y_true, y_pred, groups):
        """Compute classification metrics"""
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=groups)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=groups, output_dict=True)
        
        # Per-class accuracies
        per_class_acc = {}
        for i, group in enumerate(groups):
            mask = y_true == group
            if mask.sum() > 0:
                per_class_acc[group] = (y_pred[mask] == group).sum() / mask.sum()
            else:
                per_class_acc[group] = 0.0
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'report': report
        }
    
    def plot_confusion_matrix(self, cm, groups, title, filename):
        """Plot confusion matrix"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=groups, yticklabels=groups,
                   cbar_kws={'label': 'Count'}, ax=ax,
                   square=True, linewidths=1, linecolor='gray')
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {filename}")
    
    def analyze_group_combination(self, contrast, groups, n_features=1000):
        """Analyze one group combination"""
        
        groups_str = '_vs_'.join(groups)
        
        print("\n" + "="*70)
        print(f"ANALYZING: {contrast} ({groups_str})")
        print("="*70)
        
        # Load data
        image_files, labels, subject_ids = self.load_data(contrast, groups)
        print(f"   Loaded {len(labels)} subjects")
        
        # Show distribution
        for group in groups:
            n = (labels == group).sum()
            print(f"      {group}: {n}")
        
        # Run classification
        y_true, y_pred = self.nested_cv_classification(
            image_files, labels, groups, n_features
        )
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, groups)
        
        # Print results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\n   Overall Accuracy: {metrics['accuracy']*100:.1f}%")
        print(f"\n   Per-Class Accuracy:")
        for group, acc in metrics['per_class_accuracy'].items():
            print(f"      {group}: {acc*100:.1f}%")
        
        print(f"\n   Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Plot confusion matrix
        title = f"{contrast} - {groups_str}\nAccuracy: {metrics['accuracy']*100:.1f}%"
        filename = f'confusion_{contrast}_{groups_str}.png'
        self.plot_confusion_matrix(
            metrics['confusion_matrix'], groups, title, filename
        )
        
        # Save results
        results = {
            'contrast': contrast,
            'groups': groups,
            'n_subjects': len(labels),
            'n_features': n_features,
            'accuracy': float(metrics['accuracy']),
            'per_class_accuracy': {k: float(v) for k, v in metrics['per_class_accuracy'].items()},
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'classification_report': metrics['report']
        }
        
        results_file = self.group_analysis_dir / f'results_{contrast}_{groups_str}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_all_group_combinations(self, contrast='W', n_features=1000):
        """
        Run all possible group combinations
        """
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " COMPREHENSIVE GROUP ANALYSIS ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        print(f"\nContrast: {contrast}")
        print(f"Feature selection: k={n_features}")
        
        # Define all combinations
        combinations = [
            # Three-way (MOST IMPORTANT!)
            (['DL', 'SpD', 'TD'], 'Three-Way'),
            
            # Binary combinations
            (['DL', 'TD'], 'DL vs TD'),
            (['DL', 'SpD'], 'DL vs SpD'),
            (['SpD', 'TD'], 'SpD vs TD'),
        ]
        
        all_results = []
        
        for groups, name in combinations:
            print(f"\n{'='*70}")
            print(f"{name}")
            print(f"{'='*70}")
            
            try:
                results = self.analyze_group_combination(contrast, groups, n_features)
                results['combination_name'] = name
                all_results.append(results)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY: ALL GROUP COMBINATIONS")
        print("="*70)
        
        summary_df = pd.DataFrame([
            {
                'Combination': r['combination_name'],
                'Groups': ' vs '.join(r['groups']),
                'N': r['n_subjects'],
                'Accuracy': f"{r['accuracy']*100:.1f}%"
            }
            for r in all_results
        ])
        
        print("\n" + summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(self.group_analysis_dir / f'summary_{contrast}.csv', index=False)
        
        # Detailed summary with per-class accuracies
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)
        
        for r in all_results:
            print(f"\n{r['combination_name']}:")
            print(f"   Overall: {r['accuracy']*100:.1f}%")
            print(f"   Per-class:")
            for group, acc in r['per_class_accuracy'].items():
                print(f"      {group}: {acc*100:.1f}%")
        
        # Best combination
        best = max(all_results, key=lambda x: x['accuracy'])
        print("\n" + "="*70)
        print(f"BEST COMBINATION: {best['combination_name']}")
        print(f"   Accuracy: {best['accuracy']*100:.1f}%")
        print("="*70)
        
        return all_results
    
    def test_all_contrasts_all_combinations(self):
        """
        Test all contrasts with all group combinations
        COMPREHENSIVE ANALYSIS
        """
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " FULL COMPREHENSIVE ANALYSIS ".center(68) + "║")
        print("║" + " All Contrasts × All Group Combinations ".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        contrasts = ['W', 'PH', 'Reading', 'W-PH', 'PH-W']
        
        all_results = []
        
        for contrast in contrasts:
            print(f"\n\n{'#'*70}")
            print(f"# CONTRAST: {contrast}")
            print(f"{'#'*70}")
            
            try:
                results = self.run_all_group_combinations(contrast, n_features=1000)
                all_results.extend(results)
            except Exception as e:
                print(f"  Error with {contrast}: {e}")
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY: ALL CONTRASTS × ALL COMBINATIONS")
        print("="*70)
        
        final_df = pd.DataFrame([
            {
                'Contrast': r['contrast'],
                'Combination': r['combination_name'],
                'Accuracy': f"{r['accuracy']*100:.1f}%"
            }
            for r in all_results
        ])
        
        # Pivot for better view
        pivot = final_df.pivot(index='Combination', columns='Contrast', values='Accuracy')
        print("\n" + pivot.to_string())
        
        # Save
        final_df.to_csv(self.group_analysis_dir / 'complete_summary.csv', index=False)
        pivot.to_csv(self.group_analysis_dir / 'complete_summary_pivot.csv')
        
        # Best overall
        best = max(all_results, key=lambda x: x['accuracy'])
        print("\n" + "="*70)
        print(f"🏆 BEST OVERALL:")
        print(f"   Contrast: {best['contrast']}")
        print(f"   Groups: {best['combination_name']}")
        print(f"   Accuracy: {best['accuracy']*100:.1f}%")
        print("="*70)
        
        return all_results


def main():
    """Main execution"""
    
    bids_root = "/workspace/aime_dyslexia/dataset_root"
    glm_dir = "/workspace/aime_dyslexia/analysis_output/first_level_glm"
    output_root = "/workspace/aime_dyslexia/analysis_output"
    
    print("\n" + "="*70)
    print("COMPREHENSIVE GROUP ANALYSIS")
    print("Testing: DL vs TD, DL vs SpD, SpD vs TD, Three-Way")
    print("="*70)
    
    analyzer = ComprehensiveGroupAnalysis(bids_root, glm_dir, output_root)
    
    # Option 1: Quick test (just W contrast)
    print("\n" + "="*70)
    print("QUICK TEST: W contrast only")
    print("="*70)
    results = analyzer.run_all_group_combinations(contrast='W', n_features=1000)
    
    # Option 2: Full analysis (all contrasts) - Uncomment to run
    # print("\n\nRunning FULL analysis (all contrasts)...")
    # print("This will take longer...")
    # response = input("\nContinue with full analysis? (yes/no): ").strip().lower()
    # if response in ['yes', 'y']:
    #     results = analyzer.test_all_contrasts_all_combinations()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved in: {analyzer.group_analysis_dir}")
    print(f"Figures saved in: {analyzer.figures_dir}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("\n1. Three-way classification (DL vs SpD vs TD)")
    print("2. Binary: DL vs TD")
    print("3. Binary: DL vs SpD (reading-specific)")
    print("4. Binary: SpD vs TD (spelling-specific)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
