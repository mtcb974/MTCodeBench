from data.loader import get_multi_turn_dataset,read_infer_log,append_infer_log
from typing import List
from data import MTBigCodeInstance,TurnResult,MTResult
from model.prompt import get_infer_prompt
from model.model import call_llm_inference
from bigcodebench.sanitize import sanitize
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock

log_lock = Lock()

def infer_sing_task(task_data, llm: str, golden: bool,backend:str,qwen3_thinking_mode:bool,prompt_mode:str,run_id:str):
    """
    Single task processing logic, used in threads
    """
    log_result: MTResult = {
        'mt_id': task_data['mt_id'],
        'task_id': task_data['task_id'],
    }
    trajectories: List[TurnResult] = []

    # Process each turn sequentially
    for turn_data in task_data['mt_data']:
        print(f"Start processing task: {task_data['task_id']} in turn {turn_data['turn']}")
        turn_result: TurnResult = {
            'task_id': turn_data['task_id'],
            'instruct_prompt': turn_data['instruct_prompt'],
            'turn': turn_data['turn']
        }

        turn_message = get_infer_prompt(
            trajectories=trajectories,
            instruction_prompt=turn_data['instruct_prompt'],
            golden=golden,
            mt_data=task_data,
            qwen3_thinking_mode=qwen3_thinking_mode,
            prompt_mode=prompt_mode
        )

        turn_raw_solution = call_llm_inference(llm=llm, messages=turn_message,backend=backend)
        turn_result['raw_solution'] = turn_raw_solution
        turn_result['solution'] = sanitize(turn_raw_solution, turn_data['entry_point'])
        print(f"Finish turn {turn_data['turn']} in task: {turn_data['task_id']}!")

        trajectories.append(turn_result)
    log_result['solutions'] = trajectories

    # Write log after locking
    with log_lock:
        append_infer_log(llm=llm, golden=golden, infer_data=log_result,
                         qwen3_thinking_mode=qwen3_thinking_mode,
                         prompt_mode=prompt_mode,run_id=run_id)
        print(f"Successfully processed and logged task: {task_data['task_id']}")

    return True

def multi_turn_inference(
    llm: str,
    dataset: str,
    golden: bool,
    backend: str,
    qwen3_thinking_mode: bool,
    prompt_mode: str,
    run_id:str
):
    # Load multi-turn dataset
    mt_dataset = get_multi_turn_dataset(dataset=dataset)

    # Read log
    infer_logs = read_infer_log(llm=llm,golden=golden,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode,run_id=run_id)

    # Filter out the data that has been generated and execute filtering
    existing_task_ids = set()
    for infer_log in infer_logs:
        if isinstance(infer_log,dict):
            existing_task_ids.add(infer_log['task_id'])
    
    filtered_mt_dataset:List[MTBigCodeInstance] = []
    for mt_data in mt_dataset:
        task_id = mt_data['task_id']
        if task_id not in existing_task_ids:
            filtered_mt_dataset.append(mt_data)
    
    print(f"Total tasks to infer: {len(filtered_mt_dataset)}, skipped {len(mt_dataset) - len(filtered_mt_dataset)} already processed tasks.")

    # Multi-thread execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(infer_sing_task, task_data, llm, golden,backend,qwen3_thinking_mode,prompt_mode,run_id)
            for task_data in filtered_mt_dataset
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred during inference: {e}")

    final_logs = read_infer_log(llm=llm, golden=golden,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode,run_id=run_id)
    final_logs.sort(key=lambda x: x['mt_id'])

    from data.loader import write_infer_log
    write_infer_log(llm=llm, golden=golden, infer_data_list=final_logs,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode,run_id=run_id)
    
    print("All tasks completed.")
