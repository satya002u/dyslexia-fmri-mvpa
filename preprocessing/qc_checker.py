"""
fMRIPrep Quality Control Checker

This script runs comprehensive quality checks on fMRIPrep derivatives, including:
1. Output completeness (checks for T1w, functional, confounds files).
2. Motion analysis (max translation and mean framewise displacement).
3. Brain coverage estimation.

It generates a detailed HTML report, a summary CSV, and QC plots.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set plot style for better aesthetics
sns.set_theme(style="whitegrid")

class QualityController:
    """
    Manages the quality control process for an fMRIPrep derivative directory.
    """
    def __init__(self, fmriprep_dir):
        self.fmriprep_dir = Path(fmriprep_dir)
        self.subjects = []
        self.qc_results = []
        
    def find_subjects(self):
        """Find all preprocessed subjects"""
        self.subjects = sorted([d.name for d in self.fmriprep_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('sub-')])
        print(f"Found {len(self.subjects)} preprocessed subjects in {self.fmriprep_dir.name}")
        return self.subjects
        
    def check_motion(self, subject_id, motion_threshold=3.0, fd_threshold=0.5):
        """Check motion parameters for a subject."""
        subject_dir = self.fmriprep_dir / subject_id
        
        # Find confounds files
        confounds_files = list(subject_dir.rglob('*_desc-confounds_timeseries.tsv'))
        
        if not confounds_files:
            return {'subject': subject_id, 'status': 'ERROR', 'max_translation_mm': None, 'mean_fd': None}
        
        motion_data = []
        
        for confounds_file in confounds_files:
            try:
                df = pd.read_csv(confounds_file, sep='\t')
                
                motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
                
                if all(col in df.columns for col in motion_cols):
                    # Max absolute motion
                    max_trans = df[['trans_x', 'trans_y', 'trans_z']].abs().max().max()
                    max_rot = df[['rot_x', 'rot_y', 'rot_z']].abs().max().max()
                    
                    # Framewise displacement
                    mean_fd = df['framewise_displacement'].mean() if 'framewise_displacement' in df.columns else None
                    
                    motion_data.append({
                        'max_trans': max_trans,
                        'max_rot': max_rot,
                        'mean_fd': mean_fd,
                    })
            except Exception:
                pass
        
        if not motion_data:
            return {'subject': subject_id, 'status': 'ERROR', 'max_translation_mm': None, 'mean_fd': None}
        
        # Aggregate across runs
        max_trans = max(d['max_trans'] for d in motion_data)
        valid_fds = [d['mean_fd'] for d in motion_data if d['mean_fd'] is not None]
        mean_fd = np.mean(valid_fds) if valid_fds else None
        
        # Determine status
        if max_trans > motion_threshold:
            status = 'HIGH_MOTION'
        elif mean_fd is not None and mean_fd > fd_threshold:
            status = 'MODERATE_MOTION'
        else:
            status = 'PASS'
        
        return {
            'subject': subject_id,
            'status': status,
            'max_translation_mm': max_trans,
            'mean_fd': mean_fd,
        }
        
    def check_coverage(self, subject_id):
        """Check brain coverage of functional scans"""
        subject_dir = self.fmriprep_dir / subject_id
        
        # Checking for the MNI functional file
        func_files = list(subject_dir.rglob('*space-MNI152NLin2009cAsym*_desc-preproc_bold.nii.gz'))
        
        if not func_files:
            return {'subject': subject_id, 'status': 'ERROR', 'coverage_percent': None}
        
        try:
            img = nib.load(func_files[0])
            data = img.get_fdata()
            mean_img = data.mean(axis=3)
            
            # Simple non-zero voxel coverage as a proxy
            coverage = (mean_img > 0).sum() / mean_img.size * 100
            
            return {
                'subject': subject_id,
                'status': 'PASS' if coverage > 70 else 'LOW_COVERAGE',
                'coverage_percent': coverage,
            }
        
        except Exception:
            return {'subject': subject_id, 'status': 'ERROR', 'coverage_percent': None}
            
    def check_outputs(self, subject_id):
        """
        REVISED: Checks if all expected outputs exist using the likely file patterns
        from the fMRIPrep command you used.
        """
        subject_dir = self.fmriprep_dir / subject_id
        
        # The file patterns are adjusted to be more specific to MNI space outputs
        # which you requested in your runner script (MNI152NLin2009cAsym:res-2).
        expected_outputs = {
            # 1. Anatomical T1w in MNI space (CRITICAL for full completion)
            'anatomical': '*space-MNI152NLin2009cAsym*_desc-preproc_T1w.nii.gz', 
            
            # 2. Functional BOLD file in MNI space
            'functional': '*space-MNI152NLin2009cAsym*_desc-preproc_bold.nii.gz',
            
            # 3. Confounds
            'confounds': '*_desc-confounds_timeseries.tsv',
            
            # 4. HTML Report
            'html_report': '*.html'
        }
        
        found = {}
        for output_type, pattern in expected_outputs.items():
            files = list(subject_dir.rglob(pattern))
            found[output_type] = len(files)
        
        # Determine status: COMPLETE if at least 1 of each critical type is found.
        if (found.get('anatomical', 0) >= 1 and 
            found.get('functional', 0) >= 1 and 
            found.get('confounds', 0) >= 1):
            status = 'COMPLETE'
        elif found.get('html_report', 0) > 0 and (found.get('functional', 0) >= 1 or found.get('anatomical', 0) >= 1):
            status = 'PARTIAL' # Partial success: report exists and at least one core output is present
        else:
            status = 'INCOMPLETE'
        
        return {
            'subject': subject_id,
            'output_status': status,
            'n_t1w_mni': found.get('anatomical', 0),
            'n_functional': found.get('functional', 0),
            'n_confounds': found.get('confounds', 0),
            'has_report': found.get('html_report', 0) > 0,
        }
        
    def run_full_qc(self):
        """Run complete quality control on all subjects"""
        print("\n" + "=" * 70)
        print("RUNNING QUALITY CONTROL (Checking for T1w in MNI space)")
        print("=" * 70 + "\n")
        
        self.find_subjects()
        
        results = []
        
        for i, subject_id in enumerate(self.subjects, 1):
            print(f"[{i}/{len(self.subjects)}] Checking {subject_id}...")
            
            output_check = self.check_outputs(subject_id)
            motion_check = self.check_motion(subject_id)
            coverage_check = self.check_coverage(subject_id)
            
            result = {
                **output_check,
                'motion_status': motion_check.get('status', 'ERROR'),
                'coverage_status': coverage_check.get('status', 'ERROR'),
                'max_translation_mm': motion_check.get('max_translation_mm'),
                'mean_fd': motion_check.get('mean_fd'),
                'coverage_percent': coverage_check.get('coverage_percent'),
                'overall_qc_status': 'PASS' 
            }
            
            # Determine overall status
            if result['output_status'] != 'COMPLETE':
                 result['overall_qc_status'] = 'FAIL (Output)'
            elif result['motion_status'] == 'HIGH_MOTION':
                 result['overall_qc_status'] = 'FAIL (Motion)'
            elif result['coverage_status'] == 'LOW_COVERAGE':
                 result['overall_qc_status'] = 'FAIL (Coverage)'
            elif result['motion_status'] == 'MODERATE_MOTION':
                 result['overall_qc_status'] = 'REVIEW (Motion)'
            
            results.append(result)
            
        self.qc_results = results
        return pd.DataFrame(results)

    def identify_problematic_subjects(self):
        """Prints a summary list of subjects that failed or require review."""
        if not self.qc_results:
            return
            
        df = pd.DataFrame(self.qc_results)
        
        problem_subjects = df[df['overall_qc_status'].str.startswith(('FAIL', 'REVIEW'))]
        
        if not problem_subjects.empty:
            print("\n" + "-" * 40)
            print("Problematic Subjects Summary:")
            print("-" * 40)
            for _, row in problem_subjects.iterrows():
                print(f"| {row['subject']:<20} | Status: {row['overall_qc_status']:<15} |")
            print("-" * 40)
        else:
            print("\n✅ All subjects passed the defined QC thresholds.")

    def generate_qc_report(self, output_file='qc_report_v2.html'):
        """Generate QC report with visualizations"""
        if not self.qc_results:
            print("No QC results available. Run run_full_qc() first.")
            return
            
        df = pd.DataFrame(self.qc_results)
        
        # --- Plotting Section ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('fMRIPrep Quality Control Report', fontsize=16, fontweight='bold')
        
        # Plot 1: Status summary
        ax = axes[0, 0]
        status_counts = df['overall_qc_status'].value_counts()
        color_map = {'PASS': '#28a745', 'REVIEW (Motion)': '#ffc107', 
                     'FAIL (Motion)': '#dc3545', 'FAIL (Output)': '#007bff', 
                     'FAIL (Coverage)': '#dc3545', 'ERROR': '#6c757d'}
        colors = [color_map.get(s, '#6c757d') for s in status_counts.index]
        
        status_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_title('Overall QC Status Summary')
        ax.set_xlabel('Status')
        ax.set_ylabel('Number of Subjects')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Motion distribution
        ax = axes[0, 1]
        if df['max_translation_mm'].notna().any():
            ax.hist(df['max_translation_mm'].dropna(), bins=20, alpha=0.7, color='#007bff')
            ax.axvline(3.0, color='r', linestyle='--', label='Threshold (3mm)')
            ax.set_title('Maximum Translation Distribution')
            ax.set_xlabel('Translation (mm)')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Plot 3: Framewise displacement
        ax = axes[1, 0]
        if df['mean_fd'].notna().any():
            ax.hist(df['mean_fd'].dropna(), bins=20, alpha=0.7, color='#ffc107')
            ax.axvline(0.5, color='r', linestyle='--', label='Threshold (0.5mm)')
            ax.set_title('Mean Framewise Displacement')
            ax.set_xlabel('FD (mm)')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Plot 4: Coverage
        ax = axes[1, 1]
        if df['coverage_percent'].notna().any():
            ax.hist(df['coverage_percent'].dropna(), bins=20, alpha=0.7, color='#28a745')
            ax.axvline(70, color='r', linestyle='--', label='Threshold (70%)')
            ax.set_title('Brain Coverage Distribution')
            ax.set_xlabel('Coverage (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_dir = self.fmriprep_dir.parent / 'qc'
        output_dir.mkdir(exist_ok=True)
        fig_path = output_dir / 'qc_plots.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ QC plots saved: {fig_path}")
        plt.close()
        
        # Generate HTML report
        html_path = output_dir / output_file
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>fMRIPrep QC Report</title>
            <style>
                body {{ font-family: 'Inter', Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #007bff; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .pass_row {{ background-color: #d4edda; color: #155724; }}
                .review_row {{ background-color: #fff3cd; color: #856404; }}
                .fail_row {{ background-color: #f8d7da; color: #721c24; }}
                .fail_output_row {{ background-color: #cfe2ff; color: #075482; }}
                .summary-list {{ list-style: none; padding: 0; }}
                .summary-list li {{ margin-bottom: 5px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 8px; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>fMRIPrep Quality Control Report (v2)</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            <p>Total subjects: {len(df)}</p>
            
            <h2>Summary Statistics</h2>
            <ul class="summary-list">
                <li>✅ Subjects with PASS motion: {len(df[df['motion_status'] == 'PASS'])}</li>
                <li>⚠️ Subjects needing REVIEW (Moderate Motion): {len(df[df['motion_status'] == 'MODERATE_MOTION'])}</li>
                <li>❌ Subjects with HIGH_MOTION: {len(df[df['motion_status'] == 'HIGH_MOTION'])}</li>
                <li>Average Max Translation: {df['max_translation_mm'].mean():.2f} mm</li>
                <li>Average Mean FD: {df['mean_fd'].mean():.2f} mm</li>
            </ul>
            
            <h2>Quality Control Plots</h2>
            <img src="qc_plots.png" alt="QC Plots">
            
            <h2>Individual Subject Results (MNI T1w Check Included)</h2>
        """
        
        def get_row_class(status):
            if status == 'PASS':
                return 'pass_row'
            elif 'REVIEW' in status:
                return 'review_row'
            elif 'FAIL (Output)' in status:
                return 'fail_output_row'
            elif 'FAIL' in status:
                return 'fail_row'
            return ''

        # Use to_html and inject custom classes
        html_table = df.to_html(index=False, classes='dataframe')
        
        # Inject row classes based on overall_qc_status
        rows = html_table.split('<tr>')
        new_rows = [rows[0]] # Header
        for row in rows[1:]:
            subject_id = row.split('<td>')[1].split('</td>')[0]
            try:
                status = df[df['subject'] == subject_id]['overall_qc_status'].iloc[0]
                row_class = get_row_class(status)
            except IndexError:
                row_class = ''
            
            if row_class:
                # Add class to the <tr> tag
                row = f'<tr class="{row_class}">' + row.lstrip('<td>')
            new_rows.append(row)
            
        html_content += "".join(new_rows) + "</body></html>"
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ QC report saved: {html_path}")
        
        # Save CSV
        csv_path = output_dir / 'qc_results_v2.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ QC results CSV: {csv_path}")
        
        return html_path

def main():
    """
    Main execution function for the QC script.
    """
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "fMRIPREP QUALITY CONTROL V2" + " " * 20 + "║")
    print("║" + " " * 12 + "(Checking for MNI T1w and Full Output Completeness)" + " " * 11 + "║")
    print("╚" + "=" * 68 + "╝" + "\n")
    
    # --- Configuration ---
    # Using the paths provided in your parallel runner script
    FMRIPREP_DIR = "/workspace/aime_dyslexia/derivatives/fmriprep"
    
    # NOTE: If your QC script needs to read files created by Docker's root user, 
    # you may need to run this script with 'sudo'.
    
    if not os.path.exists(FMRIPREP_DIR):
        print(f"\n❌ Error: fMRIPrep derivatives directory not found at:")
        print(f"   {FMRIPREP_DIR}")
        print("Please check the path and ensure fMRIPrep has finished running.")
        return

    # Initialize and run QC
    try:
        controller = QualityController(FMRIPREP_DIR)
        
        # 1. Run all checks and get the DataFrame
        results_df = controller.run_full_qc()
        
        if not results_df.empty:
            # 2. Generate report and plots
            html_report_path = controller.generate_qc_report()
            
            # 3. Print summary of problematic subjects
            controller.identify_problematic_subjects()
            
            print("\n" + "=" * 70)
            print("FINISHED.")
            print(f"Review the HTML report: {html_report_path}")
            print(f"You can find the data in the new columns: 'n_t1w_mni' and 'overall_qc_status'")
            print("=" * 70)

        else:
            print("\n⚠️ No QC data generated. Check if subjects were found.")
            
    except Exception as e:
        print(f"\nFATAL ERROR during QC process: {e}")


if __name__ == "__main__":
    # If the fMRIPrep outputs were generated with 'sudo docker run', 
    # you may need to run this script with 'sudo' for file reading permissions.
    main()
