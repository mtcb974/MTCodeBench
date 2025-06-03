from datasets import Dataset
from typing import List,Dict,Any,TypedDict
import os
import json
from data import SplitBigCodeInstance,BigCodeBenchInstance,MTBigCodeInstance
from data import MTBigCodeInstance,MTResult
from pathlib import Path

def get_bigcode_hard(): 
    return Dataset.from_json("./dataset/bigcodebench_hard_v0_1_4.jsonl")

def load_exsiting_mt_data(path:str)->List[Dict[str,Any]]:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: File {path} is empty or invalid JSON. Starting fresh.")
                return []
    return []

def get_existing_task_ids(existing_data: List[Dict[str, Any]]) -> set:
    """
    Extract the existing set of task_ids.
    """
    return {item["origin_instance"]["task_id"] for item in existing_data}

def filter_unprocessed_instances(
    all_instances: List[BigCodeBenchInstance],
    existing_mt_data: List[MTBigCodeInstance]
) -> List[BigCodeBenchInstance]:
    # Extract the processed task_id
    processed_task_ids = {
        mt_instance["origin_instance"]["task_id"]
        for mt_instance in existing_mt_data
        if "origin_instance" in mt_instance and "task_id" in mt_instance["origin_instance"]
    }

    # Filter unprocessed instances
    unprocessed = [
        instance for instance in all_instances
        if instance["task_id"] not in processed_task_ids
    ]

    return unprocessed

#### Decomposition and generation of the log module #### 

def read_decompose_log(file_path: str = "./logs/decompose.json") -> List[Dict[str,Any]]:
    """
    Read the decompose.json file from the specified path and return the list of existing records.
    If the file does not exist or is empty, return an empty list.
    """
    path = Path(file_path)
    if not path.exists():
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            return []

def append_decompose_log_batch(task_id:str,split_data: List[SplitBigCodeInstance], file_path: str = "./logs/decompose.json"):
    """
    Append and write a new decompose object to the JSON log file
    """
    data = read_decompose_log(file_path)
    data.append({
        task_id: split_data
    })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_testgen_log(gen_base:bool,file_path: str = "./logs/testgen.json") -> List[Dict[str,Any]]:
    """
    Read the testgen.json file from the specified path and return the list of existing records.
    If the file does not exist or is empty, return an empty list.
    """
    if gen_base:
        file_path = "./logs/testgen_base.json"
    path = Path(file_path)
    if not path.exists():
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            return []

def append_testgen_log_batch(task_id:str,success_trajectory: List[Dict[str,Any]],gen_base:bool, file_path: str = "./logs/testgen.json"):
    """
    Append a new decompose object to the JSON log file.
    """
    if gen_base:
        file_path = "./logs/testgen_base.json"
    data = read_testgen_log(gen_base=gen_base,file_path=file_path)
    data.append({
        task_id: success_trajectory
    })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

#### Multi-round Generation Module #### 

def get_multi_turn_dataset(dataset="./dataset/mt_bigcode.json") -> List[MTBigCodeInstance]:
    with open(dataset,'r') as file:
        mt_data = json.load(file)
    return mt_data


def get_log_path(llm:str,golden:bool,qwen3_thinking_mode:bool, prompt_mode:str,run_id:str) -> str:

    golden_str = "_golden" if golden else "_base"
    think_str = "_think" if qwen3_thinking_mode else ""
    prompt_mode = "_"+prompt_mode if prompt_mode != "direct" else ""
    if run_id!="":
        run_id = "_" + run_id
    file_path = f"./logs/inference/{llm}{golden_str}{think_str}{prompt_mode}{run_id}.jsonl"

    return file_path

def read_infer_log(llm:str,golden:bool,qwen3_thinking_mode:bool,prompt_mode:str,run_id:str) -> List[MTResult]:
    file_path = get_log_path(llm,golden,qwen3_thinking_mode,prompt_mode,run_id)
    
    os.makedirs(os.path.dirname(file_path),exist_ok=True)

    infer_logs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    log_data = json.loads(line)
                    infer_logs.append(log_data)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
    except FileNotFoundError:
        with open(file_path, 'w', encoding='utf-8') as file:
            pass
        print(f"The file was not found and a new file has been created: {file_path}")
    return infer_logs

# inference/{model}_{base/golden}_sanitized_calibrated.jsonl
def append_infer_log(llm:str,golden:bool, infer_data:MTResult,qwen3_thinking_mode:bool,prompt_mode:str,run_id:str):
    file_path = get_log_path(llm,golden,qwen3_thinking_mode,prompt_mode,run_id)
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            line = json.dumps(infer_data, ensure_ascii=False) + '\n'
            file.write(line)
    except Exception as e:
        print(f"append_infer_log Error: {e}")

def write_infer_log(llm:str,golden:bool, infer_data_list:List[MTResult],qwen3_thinking_mode:bool,prompt_mode:str,run_id:str):
    file_path = get_log_path(llm,golden,qwen3_thinking_mode,prompt_mode,run_id)

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for infer_data in infer_data_list:
                line = json.dumps(infer_data, ensure_ascii=False) + '\n'
                file.write(line)
    except Exception as e:
        print(f"append_infer_log Error: {e}")