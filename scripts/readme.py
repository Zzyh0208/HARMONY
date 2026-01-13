import os
import sys
import subprocess
import time

def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)

    # 定义执行顺序
    pipeline_scripts = [
        "1_train_learner.py",
        "2_build_experts.py",
        "3_evaluate.py"
    ]

    print(f"🚀 Starting HARMONY Pipeline Automation")
    print(f"📂 Project Root: {project_root}")
    print(f"📂 Scripts Dir:  {scripts_dir}")
    print("-" * 60)

    total_start_time = time.time()

    for script_name in pipeline_scripts:
        script_full_path = os.path.join(scripts_dir, script_name)

        # 检查文件是否存在
        if not os.path.exists(script_full_path):
            print(f"❌ Error: Script file not found: {script_name}")
            sys.exit(1)

        print(f"\n▶️  [Step] Running: {script_name}...")
        step_start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, script_full_path],
                cwd=project_root,  
                check=True        
            )
            
            duration = time.time() - step_start_time
            print(f"✅ [Done] {script_name} finished in {duration:.2f}s")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Pipeline Aborted!")
            print(f"   Script '{script_name}' failed with exit code {e.returncode}.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            sys.exit(1)

    total_duration = time.time() - total_start_time
    print("-" * 60)
    print(f"🎉 All pipeline steps completed successfully!")
    print(f"⏱️  Total execution time: {total_duration:.2f}s")

if __name__ == "__main__":
    main()