from model.prompts import PromptEntry
from model.prompt import get_testgen_prompt_message
from model.model import call_llm
from data.loader import read_decompose_log,append_testgen_log_batch,get_bigcode_hard,read_testgen_log
from data.postprocess import postprocess_testgen_result
from service.verify import verify_llm_testgen
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict,Any,List
import threading
import traceback
import json

log_lock = threading.Lock()

def gen_test_and_verify_loop(gen_base:bool,max_worker:int=8)->List[Dict[str,Any]]:
    print("="*30 + "Gentest start" + "="*30)

    # Read the original dataset
    bigcode_ds = get_bigcode_hard()
    task_id_to_bigcode_data = {ds["task_id"]: ds for ds in bigcode_ds}

    # Read the decomposition dataset
    decompose_ds = read_decompose_log()

    # Skip the existing dataset
    testgen_logs = read_testgen_log(gen_base)
    existing_task_ids = set()
    for testgen_log in testgen_logs:
        if isinstance(testgen_log,dict):
            existing_task_ids.update(testgen_log.keys())

    # Filter out the processed tasks
    filtered_decompose_ds = []
    for decompose_data in decompose_ds:
        task_id = list(decompose_data.keys())[0]
        if task_id not in existing_task_ids:
            filtered_decompose_ds.append(decompose_data)
    
    print(f"Total tasks to process: {len(filtered_decompose_ds)}, skipped {len(decompose_ds) - len(filtered_decompose_ds)} already processed tasks.")

    def process_testgen(decompose_data:Dict[str,Any],task_id: str,gen_base:bool):
        total_turn = len(decompose_data[task_id])
        print(f"processing task_id: {task_id} , total turn = {total_turn}")
        bigcode_data = task_id_to_bigcode_data[task_id]
        success_trajectory = [] # Success trajectory
        max_retries = 8
        # Process the turn data
        for turn_data in decompose_data[task_id]:
            current_turn = int(turn_data['turn'])
            retry_count = 0
            success = False
            # The current round's attempt record, currently the last feedback is given to LLM
            current_attempts = []
            # Generate-verify loop
            while not success and retry_count < max_retries:
                print(f"processing task_id: {task_id}, turn: {turn_data['turn']}/{total_turn},retry:{retry_count}/{max_retries} ")
                # Prompt: Can be modified at the prompt layer to modify the way to concatenate the attempt record
                turn_prompt =  get_testgen_prompt_message(bigcode_data,turn_data,success_trajectory,current_attempts)
                
                # Call the model
                turn_code, turn_test = postprocess_testgen_result(call_llm(turn_prompt))
                print('\n' + "="*15 + f"llm_output:{task_id}" + "="*15 + "\n")
                print("code:" + '\n' + turn_code)
                print("test:" + '\n' + turn_test)
                print('\n' + "="*60 + "\n")

                # Verify this generation, expected = pass
                print('\n' + "="*15 + f"verify {task_id} , solution_{current_turn} in turn_{current_turn}/{total_turn},retry:{retry_count}/{max_retries} " + "="*15 + "\n")
                status_current,detail_current = verify_llm_testgen(turn_code,turn_test)
                print('\n' + "="*60 + "\n")

                # turn=1, 
                if current_turn == 1:
                    if status_current == 'pass':
                        print(f"\ngreat! task_id: {task_id} pass turn1 test.\n")
                        success = True
                        success_trajectory.append({
                            'turn': current_turn,
                            'code': turn_code,
                            'test': turn_test
                        })
                    else:
                        print(f"\nOh no! task_id: {task_id} fail turn1 test.Detail see above.\n")
                        current_attempts.append({
                            'code': turn_code,
                            'test': turn_test,
                            'feedback': f"**Human Feedback**: Your solution code fails the test case you haved generated.\n**Execution feedback**: status:{str(status_current)},detail: {str(detail_current)}"
                        })
                else: # turn > 1, verify the previous round's code
                    # Only verify the previous round's code if base is not enabled
                    if not gen_base:
                        prev_code = success_trajectory[-1]['code']
                        print('\n' + "="*15 + f"verify {task_id} solution_{current_turn-1} in turn{current_turn}/{total_turn},retry:{retry_count}/{max_retries} " + "="*15 + "\n")
                        status_prev, _ = verify_llm_testgen(prev_code, turn_test)
                        print('\n' + "="*60 + "\n")
                    else:
                        # If base is enabled, set the previous round's code to fail
                        print("User choose --gen_base, set status_prev to fail. ")
                        status_prev = 'fail'
                    
                    if status_current == 'pass' and status_prev == 'fail':
                        print(f"\ngreat! task_id: {task_id} pass turn{current_turn} test and fail turn{current_turn-1} test.\n")
                        success = True
                        success_trajectory.append({
                            'turn': current_turn,
                            'code': turn_code,
                            'test': turn_test
                        })
                    else:
                        print(f"\nOh no! task_id: {task_id} fail turn{current_turn} test or pass turn{current_turn-1} test.\n")
                        feedback_parts = []
                        if status_current != 'pass':
                            feedback_parts.append(f"**Human Feedback**: Your solution code fails the test case you haved generated. **Execution feedback**: status:{str(status_current)} {str(detail_current)}")
                        if status_prev != 'fail':
                            # Reason 1: The previous solution is too perfect
                            # Reason 2: The current test case is not perfect (the framework only considers this case)
                            feedback_parts.append("**Human Feedback**: Previous turn's code passed current test. (This is not desirable, we want the test case to be valid only for the current round and fail for the previous rounds.)")
                        
                        current_attempts.append({
                            'code': turn_code,
                            'test': turn_test,
                            'feedback': 'Another feedback:'.join(feedback_parts)
                        })
                
                retry_count += 1
            
            # If the maximum number of retries is exceeded, record the log
            if not success:
                print("!"*30,f"Attention! over max_retries count,task_id{task_id}, turn={current_turn}/{total_turn}","!"*30)
                with log_lock:
                    err_log_path = './logs/testgen_err.json' if not gen_base else './logs/testgen_err_base.json'
                    with open(err_log_path,'r',encoding='utf-8') as f:
                        data = json.load(f)
                    data.append({
                        "task_id": task_id,
                        "turn": current_turn,
                        "turn_data": turn_data,
                        "current_attempts": current_attempts,
                        "success_trajectory": success_trajectory
                    })
                    with open(err_log_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                break
        # The dataset is constructed successfully
        if len(success_trajectory) == len(decompose_data[task_id]):
            print(f"Wonderful! task_id : {task_id} generate test successfully, appending log...")
            with log_lock:
                append_testgen_log_batch(task_id,success_trajectory,gen_base=gen_base)
            print(f"append log done! task_id:{task_id} ")

    results = []
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = [executor.submit(process_testgen, decompose_data,list(decompose_data.keys())[0],gen_base) for decompose_data in filtered_decompose_ds]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    print(f"gentest execution process {len(results)} data. ")

    # Check the number of processed data
    existing_testgen_data = read_testgen_log(gen_base=gen_base)
    if len(existing_testgen_data) == len(task_id_to_bigcode_data):
        print(f"Decompose successfully, total len: {len(existing_testgen_data)}")

    print("="*30 + "Gentest end" + "="*30)

