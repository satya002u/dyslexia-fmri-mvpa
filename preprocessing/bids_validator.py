"""
BIDS Dataset Validator
Validates BIDS structure and prepares subject list for preprocessing
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

class BIDSValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.subjects = []
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate_bids_structure(self):
        """Check if dataset follows BIDS structure"""
        print("=" * 70)
        print("BIDS STRUCTURE VALIDATION")
        print("=" * 70 + "\n")
        
        # Check root level files
        required_files = ['dataset_description.json', 'participants.tsv']
        for file in required_files:
            file_path = self.dataset_path / file
            if file_path.exists():
                print(f"✅ Found: {file}")
                self.info.append(f"Found required file: {file}")
            else:
                print(f"❌ Missing: {file}")
                self.errors.append(f"Missing required file: {file}")
        
        # Check for subject directories
        subject_dirs = sorted([d for d in self.dataset_path.iterdir() 
                              if d.is_dir() and d.name.startswith('sub-')])
        
        if subject_dirs:
            print(f"\n✅ Found {len(subject_dirs)} subject directories")
            self.subjects = [d.name for d in subject_dirs]
        else:
            print("\n❌ No subject directories found")
            self.errors.append("No subject directories found")
            return False
        
        return len(self.errors) == 0
    
    def check_subjects(self):
        """Check each subject's data"""
        print("\n" + "=" * 70)
        print("SUBJECT-LEVEL VALIDATION")
        print("=" * 70 + "\n")
        
        subject_status = []
        
        for sub_id in self.subjects:
            sub_path = self.dataset_path / sub_id
            status = {
                'subject': sub_id,
                'has_anat': False,
                'has_func': False,
                'n_anat_files': 0,
                'n_func_files': 0,
                'sessions': []
            }
            
            # Check for session directories
            sessions = sorted([d for d in sub_path.iterdir() 
                             if d.is_dir() and d.name.startswith('ses-')])
            
            if not sessions:
                self.warnings.append(f"{sub_id}: No session directories")
                continue
            
            for ses_dir in sessions:
                ses_id = ses_dir.name
                status['sessions'].append(ses_id)
                
                # Check anatomical
                anat_dir = ses_dir / 'anat'
                if anat_dir.exists():
                    anat_files = list(anat_dir.glob('*.nii.gz'))
                    if anat_files:
                        status['has_anat'] = True
                        status['n_anat_files'] += len(anat_files)
                
                # Check functional
                func_dir = ses_dir / 'func'
                if func_dir.exists():
                    func_files = list(func_dir.glob('*_bold.nii.gz'))
                    if func_files:
                        status['has_func'] = True
                        status['n_func_files'] += len(func_files)
            
            subject_status.append(status)
        
        # Create DataFrame
        df = pd.DataFrame(subject_status)
        
        # Print summary
        complete = df[df['has_anat'] & df['has_func']]
        print(f"Complete subjects (anat + func): {len(complete)}/{len(df)}")
        print(f"Missing anatomical: {len(df[~df['has_anat']])}")
        print(f"Missing functional: {len(df[~df['has_func']])}")
        
        # Show sample
        print("\nFirst 5 subjects:")
        print(df[['subject', 'has_anat', 'has_func', 'n_anat_files', 'n_func_files']].head())
        
        return df
    
    def validate_participants_file(self):
        """Check participants.tsv"""
        print("\n" + "=" * 70)
        print("PARTICIPANTS FILE VALIDATION")
        print("=" * 70 + "\n")
        
        participants_file = self.dataset_path / 'participants.tsv'
        
        if not participants_file.exists():
            print("❌ participants.tsv not found")
            return None
        
        try:
            df = pd.read_csv(participants_file, sep='\t')
            print(f"✅ Loaded participants.tsv: {len(df)} rows")
            print(f"\nColumns: {list(df.columns)}")
            
            # Check for group information
            if 'group' in df.columns:
                print("\n✅ Group information found")
                print(df['group'].value_counts())
            else:
                print("\n⚠️  No 'group' column found")
                self.warnings.append("No group column in participants.tsv")
            
            print("\nFirst 5 rows:")
            print(df.head())
            
            return df
            
        except Exception as e:
            print(f"❌ Error reading participants.tsv: {e}")
            self.errors.append(f"Error reading participants.tsv: {e}")
            return None
    
    def check_functional_runs(self):
        """Check functional runs per subject"""
        print("\n" + "=" * 70)
        print("FUNCTIONAL RUNS CHECK")
        print("=" * 70 + "\n")
        
        runs_per_subject = {}
        
        for sub_id in self.subjects[:5]:  # Check first 5 subjects
            sub_path = self.dataset_path / sub_id
            sessions = [d for d in sub_path.iterdir() 
                       if d.is_dir() and d.name.startswith('ses-')]
            
            for ses_dir in sessions:
                func_dir = ses_dir / 'func'
                if func_dir.exists():
                    bold_files = sorted(func_dir.glob('*_bold.nii.gz'))
                    runs_per_subject[sub_id] = len(bold_files)
                    
                    print(f"{sub_id}:")
                    for bold_file in bold_files:
                        print(f"  - {bold_file.name}")
        
        if runs_per_subject:
            avg_runs = sum(runs_per_subject.values()) / len(runs_per_subject)
            print(f"\nAverage runs per subject: {avg_runs:.1f}")
        
        return runs_per_subject
    
    def create_preprocessing_list(self, output_file='subjects_to_preprocess.txt'):
        """Create list of subjects ready for preprocessing"""
        print("\n" + "=" * 70)
        print("CREATING PREPROCESSING LIST")
        print("=" * 70 + "\n")
        
        # Get subjects with complete data
        df = self.check_subjects()
        complete_subjects = df[df['has_anat'] & df['has_func']]['subject'].tolist()
        
        output_path = self.dataset_path.parent / output_file
        with open(output_path, 'w') as f:
            for sub in complete_subjects:
                # Remove 'sub-' prefix for fMRIPrep
                sub_id = sub.replace('sub-', '')
                f.write(f"{sub_id}\n")
        
        print(f"✅ Created: {output_path}")
        print(f"   {len(complete_subjects)} subjects ready for preprocessing")
        
        return complete_subjects
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70 + "\n")
        
        print(f"Dataset: {self.dataset_path}")
        print(f"Total subjects: {len(self.subjects)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for err in self.errors:
                print(f"  - {err}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        if not self.errors:
            print("\n✅ Dataset is valid and ready for preprocessing!")
        else:
            print("\n❌ Please fix errors before preprocessing")
        
        print("=" * 70)

def main():
    """Main validation function"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "BIDS VALIDATION" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝" + "\n")
    
    # Get dataset path
    dataset_path = input("Enter dataset path (or press Enter for default): ").strip()
    if not dataset_path:
        dataset_path = "/workspace/aime_dyslexia/dataset_root"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        return
    
    # Create validator
    validator = BIDSValidator(dataset_path)
    
    # Run validation
    validator.validate_bids_structure()
    validator.validate_participants_file()
    validator.check_functional_runs()
    
    # Create preprocessing list
    complete_subjects = validator.create_preprocessing_list()
    
    # Generate report
    validator.generate_report()
    
    # Save detailed report
    report_path = Path(dataset_path).parent / 'validation_report.txt'
    with open(report_path, 'w') as f:
        f.write("BIDS VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total subjects: {len(validator.subjects)}\n")
        f.write(f"Complete subjects: {len(complete_subjects)}\n")
        f.write(f"Errors: {len(validator.errors)}\n")
        f.write(f"Warnings: {len(validator.warnings)}\n\n")
        
        if validator.errors:
            f.write("ERRORS:\n")
            for err in validator.errors:
                f.write(f"  - {err}\n")
        
        if validator.warnings:
            f.write("\nWARNINGS:\n")
            for warn in validator.warnings:
                f.write(f"  - {warn}\n")
    
    print(f"\n✅ Detailed report saved: {report_path}")

if __name__ == "__main__":
    main()