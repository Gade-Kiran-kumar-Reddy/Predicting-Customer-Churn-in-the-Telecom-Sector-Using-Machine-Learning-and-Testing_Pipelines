# runner.py
import subprocess
import sys

def main():

    # Run main.py
    print("🛠️  Running data preparation, training & evaluation...")
    subprocess.check_call([sys.executable, "main.py"])

    # Run tests
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"])

    # Generating report
    subprocess.check_call([sys.executable, "generate_report.py"])

    # Starts FastAPI
    print("🚀  Starting FastAPI server...")
    subprocess.check_call([
        sys.executable, "-m", "uvicorn",
        "deployment.app:app", "--reload",
        "--host", "127.0.0.1", "--port", "8000"
    ])

if __name__ == "__main__":
    main()
