import requests
import time
import json
import argparse

# Config
EVALUATE_URL = "http://localhost:8000/evaluate"
RESULT_JSONL_PATH = "logs/inference/deepseek-v3_golden.jsonl"  # Replace with your own path


def upload_and_evaluate(llm:str,eval_type:str):
    print(f"[INFO] Reading result file from: {RESULT_JSONL_PATH}")
    try:
        with open(RESULT_JSONL_PATH, "rb") as f:
            files = {"upload_file": (RESULT_JSONL_PATH, f, "application/json")}
            print("[INFO] Sending request to /evaluate endpoint...")
            params = {
                "llm": llm,
                "type": eval_type
            }
            start_time = time.time()

            response = requests.post(EVALUATE_URL, files=files,params=params)

            end_time = time.time()
            elapsed = round(end_time - start_time, 2)
            print(f"[INFO] Request completed in {elapsed} seconds.")

            if response.status_code == 200:
                result = response.json()
                print("\n✅ Evaluation Result:\n")
                metrics = result.get("metrics", {})
                for turn_key in sorted(metrics.keys()):
                    if turn_key == "fully_completed_samples":
                        continue
                    metric = metrics[turn_key]
                    accuracy_percent = metric["accuracy"] * 100
                    print(f"{turn_key}:")
                    print(f"  Total Samples: {metric['total_samples']}")
                    print(f"  Correct:       {metric['correct']}")
                    print(f"  Accuracy:      {accuracy_percent:.2f}%\n")
                print(f"LLM:     {result.get('llm', 'N/A')}")
                print(f"Type:    {result.get('type', 'N/A')}")
                print(f"Report saved to: {result.get('saved_to', 'N/A')}")
                print(f"fully_completed_samples: {str(metrics['fully_completed_samples'])}")
            else:
                print("[ERROR] Failed to evaluate.")
                print("Status code:", response.status_code)
                print("Response:", response.text)

    except FileNotFoundError:
        print(f"[ERROR] File not found: {RESULT_JSONL_PATH}")
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",default=RESULT_JSONL_PATH,type=str)
    parser.add_argument("--llm", type=str, required=True, help="Name of the LLM used (e.g. codellama-7b)")
    parser.add_argument("--type", type=str, required=True, help="Evaluation type (e.g. multi_turn)")
    args = parser.parse_args()

    RESULT_JSONL_PATH = args.path
    upload_and_evaluate(args.llm,args.type)