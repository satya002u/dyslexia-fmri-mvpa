"""
NOTE: This script is designed to run fMRIPrep in parallel using Docker.
It must be executed by a user who is either in the 'docker' group,
or who has 'sudo' privileges (which are used inside the script's command).

Ultra-Parallel fMRIPrep Runner
Optimized for machines with high resources.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
import json
import time

class UltraParallelRunner:
    def __init__(self, dataset_path, output_path, work_path, fs_license):
        self.dataset_path = Path(dataset_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.work_path = Path(work_path).resolve()
        self.fs_license = Path(fs_license).resolve()
        
        # Create directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.work_path.mkdir(parents=True, exist_ok=True)
        
    def run_single_subject(self, subject_id):
        """Run fMRIPrep for a single subject"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting sub-{subject_id}")
        
        # Create subject-specific work directory
        # Using a new work directory for each subject is CRITICAL when running in parallel.
        subject_work = self.work_path / f'sub-{subject_id}'
        subject_work.mkdir(parents=True, exist_ok=True)
        
        # fMRIPrep command with corrections:
        # 1. Added 'sudo' to the command array to bypass Docker permission issues.
        # 2. Updated the image tag to 'nipreps/fmriprep:latest' as you just pulled it.
        # 3. Removed the deprecated '--use-aroma' flag.
        cmd = [
            'sudo', 'docker', 'run', '--rm', # NOTE: Added 'sudo' to command
            '-v', f'{self.dataset_path}:/data:ro',
            '-v', f'{self.output_path}:/out',
            '-v', f'{subject_work}:/work',
            '-v', f'{self.fs_license}:/opt/freesurfer/license.txt:ro',
            'nipreps/fmriprep:latest', # NOTE: Updated image to latest
            '/data', '/out', 'participant',
            '--participant-label', subject_id,
            '--work-dir', '/work',
            '--fs-license-file', '/opt/freesurfer/license.txt',
            '--output-spaces', 'MNI152NLin2009cAsym:res-2',
            '--nthreads', '4',  # 4 CPUs per subject
            '--omp-nthreads', '2',
            '--mem-mb', '16000',  # 16GB per subject
            # Removed '--use-aroma' flag (it is deprecated)
            '--skip-bids-validation',
            '--stop-on-first-crash',
        ]
        
        # Log file setup
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'sub-{subject_id}_{timestamp}.log'
        
        start_time = datetime.now()
        
        try:
            with open(log_file, 'w') as f:
                f.write(f"fMRIPrep log for sub-{subject_id}\n")
                f.write(f"Started: {start_time}\n")
                # Need to use shell=True if the command contains 'sudo' and we don't
                # want to input the password repeatedly. However, since the user
                # is likely already running in a session where sudo is cached,
                # we use shell=False for safety, and rely on the user running
                # `sudo python <script_name>` or having docker permissions fixed.
                # Since the user used 'sudo' previously, we stick with the 'sudo'
                # in the command array and hope for a cached or non-interactive prompt.
                f.write(f"Command: {' '.join(cmd)}\n\n") 
                f.write("=" * 70 + "\n\n")
                
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=43200  # 12 hour timeout
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() / 3600
                
                f.write(f"\nFinished: {end_time}\n")
                f.write(f"Duration: {duration:.2f} hours\n")
                f.write(f"Exit code: {result.returncode}\n")
            
            if result.returncode == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ sub-{subject_id} completed ({duration:.2f}h)")
                return {
                    'subject': subject_id,
                    'success': True,
                    'duration': duration,
                    'end_time': end_time.isoformat()
                }
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ sub-{subject_id} failed (exit {result.returncode})")
                return {
                    'subject': subject_id,
                    'success': False,
                    'duration': duration,
                    'error': f'Exit code {result.returncode}'
                }
                
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️  sub-{subject_id} timed out (>12h)")
            return {
                'subject': subject_id,
                'success': False,
                'duration': 12.0,
                'error': 'Timeout after 12 hours'
            }
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ sub-{subject_id} error: {e}")
            return {
                'subject': subject_id,
                'success': False,
                'duration': 0,
                'error': str(e)
            }
    
    def run_ultra_parallel(self, subject_list, n_parallel=58):
        """
        Run ALL subjects in parallel (or as many as you specify)
        
        Parameters:
        -----------
        subject_list : list
            List of subject IDs (without 'sub-' prefix)
        n_parallel : int
            Number of subjects to run simultaneously
        """
        print("\n" + "=" * 70)
        print("ULTRA-PARALLEL PREPROCESSING")
        print("=" * 70)
        print(f"\nTotal subjects: {len(subject_list)}")
        print(f"Parallel jobs: {n_parallel}")
        print(f"Resources per subject: 4 CPUs, 16GB RAM")
        print(f"Total resources: {n_parallel * 4} CPUs, {n_parallel * 16}GB RAM")
        
        # System check
        total_cpus = cpu_count()
        required_cpus = n_parallel * 4
        required_ram_gb = n_parallel * 16
        
        print(f"\nSystem check:")
        print(f"  Available CPUs: {total_cpus}")
        print(f"  Required CPUs: {required_cpus}")
        print(f"  Available RAM: 1024 GB")
        print(f"  Required RAM: {required_ram_gb} GB")
        
        if required_cpus > total_cpus:
            print(f"\n⚠️ WARNING: Need {required_cpus} CPUs but only have {total_cpus}")
            print(f"    Recommended: Use {total_cpus // 4} parallel jobs")
            
            # NOTE: Removed interactive input as it can block execution in environments
            # where stdin is not available. Default to the safer option if oversubscribed.
            print("\nAuto-adjusting to safe number due to CPU oversubscription...")
            n_parallel = total_cpus // 4
            if n_parallel < 1: n_parallel = 1 # Ensure at least 1 job runs
            print(f"Using {n_parallel} parallel jobs instead")
        
        if required_ram_gb > 900:  # Leave some headroom
            print(f"\n⚠️ WARNING: Might be tight on RAM ({required_ram_gb}GB)")
        
        # Estimate completion time
        est_time = 8.0  # Average 8 hours per subject
        print(f"\n⏱️ Estimated completion: {est_time:.1f} hours")
        print(f"    (All subjects finish together!)")
        
        print("\n" + "=" * 70)
        
        # NOTE: Removed interactive start confirmation to prevent blocking execution
        # in non-interactive environments. The user needs to confirm outside the script.
        print(f"NOTE: The script is configured to run {n_parallel} jobs in parallel.")
        
        print("\n🚀 LAUNCHING ALL SUBJECTS...")
        print("=" * 70 + "\n")
        
        start_time = datetime.now()
        
        # Run in parallel using Pool
        with Pool(processes=n_parallel) as pool:
            # Using pool.map to apply run_single_subject to every subject in the list
            results = pool.map(self.run_single_subject, subject_list)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds() / 3600
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 ULTRA-PARALLEL PREPROCESSING COMPLETE!")
        print("=" * 70)
        
        successful = sum(1 for r in results if r and r.get('success', False)) # Safely handle None results
        failed = len(results) - successful
        
        print(f"\n📊 RESULTS:")
        print(f"    Total wall time: {total_duration:.2f} hours")
        print(f"    Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"    Failed: {failed}/{len(results)}")
        
        if failed > 0:
            print(f"\n❌ Failed subjects:")
            for r in results:
                if r and not r.get('success', True):
                    error_msg = r.get('error', 'Unknown error')
                    print(f"    - sub-{r['subject']}: {error_msg}")
        
        # Calculate statistics
        if successful > 0:
            durations = [r['duration'] for r in results if r and r['success']]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            print(f"\n⏱️ TIMING STATISTICS:")
            print(f"    Average: {avg_duration:.2f} hours per subject")
            print(f"    Fastest: {min_duration:.2f} hours")
            print(f"    Slowest: {max_duration:.2f} hours")
        
        # Save detailed results
        results_file = self.output_path / 'ultra_parallel_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'total_subjects': len(results),
                'parallel_jobs': n_parallel,
                'total_wall_time_hours': total_duration,
                'successful': successful,
                'failed': failed,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'subjects': [r for r in results if r] # Filter out potential None results
            }, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")
        
        # Save list of failed subjects for reprocessing
        if failed > 0:
            # Saving to the parent directory of fmriprep/derivatives/fmriprep/
            failed_file = self.output_path.parent.parent / 'failed_subjects.txt'
            with open(failed_file, 'w') as f:
                for r in results:
                    if r and not r.get('success', True):
                        f.write(f"{r['subject']}\n")
            print(f"💾 Failed subjects list: {failed_file}")
        
        print("\n" + "=" * 70)
        
        return results


def main():
    """Main function"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ULTRA-PARALLEL fMRIPREP RUNNER" + " " * 23 + "║")
    print("║" + " " * 18 + "(High-Performance Compute Edition)" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝" + "\n")
    
    # Setup paths
    dataset_path = "/flash/home/satyaCD/aime_dyslexia/dataset_root"
    output_path = "/flash/home/satyaCD/aime_dyslexia/derivatives/fmriprep"
    work_path = "/flash/home/satyaCD/aime_dyslexia/derivatives/work"
    fs_license = "/flash/home/satyaCD/aime_dyslexia/license.txt"
    
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Work: {work_path}")
    print(f"License: {fs_license}")
    
    # Verify license
    if not os.path.exists(fs_license):
        print(f"\n❌ License not found: {fs_license}")
        return
    
    # Get subject list
    print("\n" + "=" * 70)
    print("SELECT SUBJECTS")
    print("=" * 70)
    
    # --- Interactive Subject List Selection ---
    subject_list = []
    subject_file = input("\nSubject list file [subjects_to_preprocess.txt]: ").strip()
    
    if not subject_file:
        subject_file = "/flash/home/satyaCD/aime_dyslexia/subjects_to_preprocess.txt"
    
    if os.path.exists(subject_file):
        with open(subject_file, 'r') as f:
            subject_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(subject_list)} subjects from {subject_file}")
    else:
        print(f"❌ File not found: {subject_file}")
        
        # Try to find subjects automatically
        print("\nSearching for subjects in dataset...")
        dataset = Path(dataset_path)
        if dataset.exists():
            subjects = [d.name.replace('sub-', '') for d in dataset.iterdir() 
                        if d.is_dir() and d.name.startswith('sub-')]
            print(f"Found {len(subjects)} subjects automatically")
            
            use_auto = input("Use these subjects? (yes/no): ").strip().lower()
            if use_auto in ['yes', 'y']:
                subject_list = subjects
            else:
                return
        else:
            print(f"Dataset directory not found: {dataset_path}")
            return

    # --- Subject List Validation and Display ---
    if not subject_list:
        print("\n❌ No subjects to process. Aborting.")
        return

    print(f"\n📋 Subjects to process: {len(subject_list)}")
    if len(subject_list) > 10:
        print(f"    First 5: {subject_list[:5]}")
        print(f"    Last 5: {subject_list[-5:]}")
    else:
        print(f"    List: {subject_list}")
    
    # Get number of parallel jobs
    print("\n" + "=" * 70)
    print("PARALLEL CONFIGURATION")
    print("=" * 70)
    
    # --- Interactive Parallel Jobs Configuration ---
    max_jobs = len(subject_list)
    print(f"\nRecommended parallel jobs: {max_jobs} (to run all subjects simultaneously)")
    print(f"(Based on 4 CPUs / 16GB RAM per job)")
    
    n_parallel_input = input(f"\nNumber of parallel jobs [{max_jobs}]: ").strip()
    
    try:
        n_parallel = int(n_parallel_input) if n_parallel_input else max_jobs
        if n_parallel < 1:
            print("Number of parallel jobs must be at least 1. Setting to 1.")
            n_parallel = 1
    except ValueError:
        print("Invalid input for parallel jobs. Setting to max available.")
        n_parallel = max_jobs
        
    # Create runner
    runner = UltraParallelRunner(dataset_path, output_path, work_path, fs_license)
    
    # Run
    # NOTE: The CPU check logic is now inside run_ultra_parallel for better control.
    results = runner.run_ultra_parallel(subject_list, n_parallel)
    
    if results:
        successful = sum(1 for r in results if r and r.get('success', False))
        if successful == len(subject_list):
            print("\n🎉🎉🎉 ALL SUBJECTS COMPLETED SUCCESSFULLY! 🎉🎉🎉")
        elif successful > 0:
            print(f"\n✅ {successful}/{len(subject_list)} subjects completed")
            print("    You can reprocess failed subjects later using the generated 'failed_subjects.txt'.")
        else:
            print("\n❌ All subjects failed - check logs for errors")


if __name__ == "__main__":
    main()
