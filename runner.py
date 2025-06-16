import subprocess
import sys

def main():
    """Run training and start the FastAPI server."""
    # 1) Run the main training script
    print("🛠  Running data preparation, training & evaluation...")
    subprocess.check_call([sys.executable, "main.py"])

    # 2) Start the FastAPI server via uvicorn
    print("🚀  Starting FastAPI server...")
    subprocess.check_call([
        sys.executable, "-m", "uvicorn", 
        "deployment.app:app", "--reload", 
        "--host", "127.0.0.1", "--port", "8000"
    ])

if __name__ == "__main__":
    main()
