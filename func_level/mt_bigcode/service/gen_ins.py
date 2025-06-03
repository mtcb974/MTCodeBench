from model.prompt import get_decompose_prompts
from model.prompts import PromptEntry
from model.prompt import generate_messages
from model.model import call_llm
from data.loader import read_decompose_log,append_decompose_log_batch
from data.postprocess import postprocess_decompose_result
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict,Any,List
import threading
import traceback
import json

log_lock = threading.Lock()

def gen_decompose_instruction(max_worker:int=8)->List[Dict[str,Any]]:
    print("="*30 + "Decompose start" + "="*30)
    # Read the decomposition log [ {task_id : SplitInstance} ]
    existing_decompose_data:List[Dict[str,Any]] = read_decompose_log()
    
    # Get all prompts [ {bigcode_instance, prompt, system_prompt} ]
    decompose_prompt_entries = get_decompose_prompts()

    # Skip the processed tasks
    existing_task_ids = set()
    for decompose_log in existing_decompose_data:
        existing_task_ids.update(decompose_log.keys())
        
    to_process_prompts = [
        prompt_entry for prompt_entry in decompose_prompt_entries
        if prompt_entry["bigcode_instance"]["task_id"] not in existing_task_ids
    ]
    print(f"Total {len(decompose_prompt_entries)} tasks, {len(existing_task_ids)} tasks are processed, need to process {len(to_process_prompts)} tasks")

    def process_single_attempt(prompt_entry:PromptEntry):
        """Execute a single decomposition attempt"""
        messages = generate_messages(prompt_entry["prompt"], prompt_entry["system_prompt"])
        llm_decompose_output = call_llm(messages)
        split_data_list = postprocess_decompose_result(llm_decompose_output, prompt_entry=prompt_entry)
        
        with log_lock:
            task_id = prompt_entry["bigcode_instance"]["task_id"]
            append_decompose_log_batch(task_id, split_data_list)
        return True

    def process_decompose(prompt_entry:PromptEntry):
        """Process a bigcodebench instance, if there is a JSON parsing error, it will be retried once"""
        task_id = prompt_entry["bigcode_instance"]["task_id"]
        print(f"Processing [decompose_instruction],task_id:{task_id}")
        
        try:
            return process_single_attempt(prompt_entry)
        except json.JSONDecodeError:
            print(f"JSON parsing error, retrying task {task_id}")
            try:
                return process_single_attempt(prompt_entry)
            except Exception as e:
                print(f"Error: =====> Retrying failed, error processing task {task_id}: {e}")
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"Error: =====> Error processing decompose for task ID {task_id}: {e}")
            traceback.print_exc()
            return None

    results = []
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(process_decompose, prompt_entry) for prompt_entry in to_process_prompts]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    print(f"This decompose execution process {len(results)} data. ")
    
    # Re-read the log to ensure the latest data is obtained
    existing_decompose_data = read_decompose_log()
    if len(existing_decompose_data) == len(decompose_prompt_entries):
        print(f"Decompose successfully, total len: {len(existing_decompose_data)}")
    print("="*30 + "Decompose end" + "="*30 + '\n\n')
    return existing_decompose_data
