# app/scripts/sample_script.py
import sys
import datetime
import os

def main():
    print(f"Hello from sample_script.py! Current time: {datetime.datetime.now()}")

    print(f"Script path: {os.path.abspath(__file__)}")
    print(f"Current working directory: {os.getcwd()}")

    if len(sys.argv) > 1:
        print("Received arguments:")
        for i, arg in enumerate(sys.argv[1:]):
            print(f"  Arg {i+1}: {arg}")
    else:
        print("No arguments received.")

    # Example of accessing an environment variable (if set)
    db_host = os.getenv("DB_HOST", "not_set")
    print(f"DB_HOST environment variable is: {db_host}")

    print("Sample script execution finished.")

if __name__ == "__main__":
    main()
