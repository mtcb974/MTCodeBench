import argparse
import json
from typing import List,Dict,TypedDict
from model.prompt.shard import DECOMPOSE_INSTRUCTION_PROMPT,DECOMPOSE_OUTPUT_FORMAT
from model.prompt.inference import TEST_GEN_PROMPT,TEST_GEN_FEEDBACK_PROMPT,ASSISTANT_PROMPT_REPLACE,TEST_GEN_FEEDBACK_FORMAT_PROMPT,TEST_GEN_FORMAT_PROMPT
from model.prompt import generate_messages
from model import call_llm
from util import parse_json_output,clean_markdown_code_block
from collections import defaultdict
import statistics
from dataset import DevEvalDependency,DevEvalRequirement,DevEvalInstance,ShardDevEval,PromptEntry,MTDevEvalInstance
from dataset.util import load_dataset,load_context_map
from pathlib import Path
from pass_k import verify_llm_testgen
import os
import ast
import textwrap

### Shard ### 
def get_shard_prompt(args)->List[PromptEntry]:
    """Return the list of prompt entries for the shard"""
    dev_eval_ds = load_dataset(file_path=args.dataset)
    prompt_entrys:List[PromptEntry] = []

    for dev_eval_instance in dev_eval_ds:
        original_requirement = f"{dev_eval_instance['requirement']['Functionality']} {dev_eval_instance['requirement']['Arguments']}"
        decompose_prompt = DECOMPOSE_INSTRUCTION_PROMPT.format(
            original_requirement=original_requirement,
            original_code=dev_eval_instance['gt']
        )
        prompt_entry:PromptEntry = {
            "ds_instance": dev_eval_instance,
            "prompt": decompose_prompt + DECOMPOSE_OUTPUT_FORMAT,
            "system_prompt": None
        }
        prompt_entrys.append(prompt_entry)

    return prompt_entrys

def postprocess_decompose_result(llm_output:str,prompt_entry:PromptEntry):
    """
    llm_output: List[{turn,instruct_prompt}]
    """
    decompose_json_res = parse_json_output(clean_markdown_code_block(llm_output))
    return [
        ShardDevEval(
            turn=turn_data["turn"],
            requirement=turn_data["prompt"],
        ) for turn_data in decompose_json_res
    ]

def shard_instruction(args):
    # Read the completed instances from the file
    completed_instances = set()
    try:
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    instance = json.loads(line)
                    completed_instances.add(instance['namespace'])
    except FileNotFoundError as e:
        print(f"File not found: {e}")

    prompt_list = get_shard_prompt(args)
    print(f"Completed instances: {len(completed_instances)}, Remaining instances: {len(prompt_list) - len(completed_instances)}")
    print("="*30 + "Shard start" + "="*30)

    for prompt_entry in prompt_list:
        if prompt_entry["ds_instance"]["namespace"] in completed_instances:
            continue
            
        messages = generate_messages(prompt=prompt_entry["prompt"], system_message=prompt_entry["system_prompt"])
        llm_shard_output = call_llm(messages)
        shard_data_list = postprocess_decompose_result(llm_shard_output, prompt_entry=prompt_entry)
        
        mt_instance = MTDevEvalInstance(
            namespace=prompt_entry["ds_instance"]["namespace"],
            type=prompt_entry["ds_instance"]["type"],
            project_path=prompt_entry["ds_instance"]["project_path"],
            completion_path=prompt_entry["ds_instance"]["completion_path"],
            signature_position=prompt_entry["ds_instance"]["signature_position"],
            body_position=prompt_entry["ds_instance"]["body_position"],
            dependency=prompt_entry["ds_instance"]["dependency"],
            requirement=prompt_entry["ds_instance"]["requirement"],
            tests=prompt_entry["ds_instance"]["tests"],
            indent=prompt_entry["ds_instance"]["indent"],
            domain=prompt_entry["ds_instance"]["project_path"].split("/")[0],
            gt=prompt_entry["ds_instance"]["gt"],
            context="",
            mt=shard_data_list
        )
        
        with open(args.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(mt_instance) + '\n')
    print("="*30 + "Shard end" + "="*30)

def filter_mt_instances(file_path: str, max_turns: int = 5) -> None:
    """Filter out MT samples with more than the specified number of turns"""
    filtered_instances = []
    removed_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instance = json.loads(line)
                if len(instance['mt']) <= max_turns:
                    filtered_instances.append(instance)
                else:
                    removed_count += 1
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for instance in filtered_instances:
            f.write(json.dumps(instance) + '\n')
            
    print(f"Removed {removed_count} samples with turns > {max_turns}")

def generate_report(file_path: str):
    instances = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    domain_stats = defaultdict(list)
    all_turns = []
    
    for instance in instances:
        turns = len(instance['mt'])
        all_turns.append(turns)
        domain_stats[instance['domain']].append(turns)
    
    report_data = {
        "Overall statistics": {
            "Total samples": len(instances),
            "Minimum turns": min(all_turns),
            "Maximum turns": max(all_turns),
            "Average turns": round(statistics.mean(all_turns), 2),
            "Domain count": len(domain_stats)
        },
        "Domain statistics": {}
    }
    
    # 添加每个domain的统计信息
    for domain, turns in domain_stats.items():
        report_data["Domain statistics"][domain] = {
            "Sample count": len(turns),
            "Minimum turns": min(turns),
            "Maximum turns": max(turns),
            "Average turns": round(statistics.mean(turns), 2)
        }
    
    with open("shard_report.json", 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)

def postprocess_data(args):
    # load bench
    instances_to_update = load_dataset(args.output_file)
    # load ctx
    context_source_file="./Experiments/prompt/LM_prompt_elements.jsonl"
    context_map = load_context_map(context_source_file=context_source_file)
    
    # update
    updated_count = 0
    for instance in instances_to_update:
        namespace = instance.get('namespace')
        if namespace and namespace in context_map:
            instance['context'] = context_map[namespace]
            updated_count += 1
        elif namespace:
            print(f"Warning: Context not found for namespace '{namespace}' in {context_source_file}")

    print(f"Updated {updated_count} instances' context field.")

    # write back
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for instance in instances_to_update:
                f.write(json.dumps(instance, ensure_ascii=False) + '\n')
        print(f"Updated instances written to {args.output_file}")
    except IOError:
         print(f"Error: Failed to write to file {args.output_file}")

def check_test_and_filter(args):
    invalid_syntax_count = 0
    inconsistent_path_count = 0
    inconsistent_class_count = 0
    valid_instances = []
    
    with open(args.output_file, 'r', encoding='utf-8') as f:
        for line in f:
            instance = json.loads(line)
            namespace = instance['namespace']
            tests = instance['tests']
            is_valid = True
            
            # check1: whether all tests use the test selector syntax
            invalid_tests = []
            for test in tests:
                parts = test.split('::')
                if len(parts) < 2:  # At least the file path and method name are required.
                    invalid_tests.append(test)
            if invalid_tests:
                invalid_syntax_count += 1
                print(f"Instance {namespace} contains tests that do not use the test selector syntax: {invalid_tests}")
                is_valid = False
            
            # check2: whether all test selectors have the same file path and class name
            if len(tests) > 1:
                first_test = tests[0]
                first_parts = first_test.split('::')
                first_path = first_parts[0]
                first_class = first_parts[1] if len(first_parts) > 2 else None  # if there are 3 parts, the second is the class name
                
                for test in tests[1:]:
                    current_parts = test.split('::')
                    current_path = current_parts[0]
                    current_class = current_parts[1] if len(current_parts) > 2 else None
                    
                    if current_path != first_path:
                        inconsistent_path_count += 1
                        print(f"Instance {namespace} has inconsistent test file paths: {tests}")
                        is_valid = False
                        break
                    elif first_class and current_class and current_class != first_class:
                        inconsistent_class_count += 1
                        print(f"Instance {namespace} has inconsistent test class names: {tests}")
                        is_valid = False
                        break
            
            if is_valid:
                valid_instances.append(instance)
    
    print(f"\nStatistics:")
    print(f"Invalid test selector syntax count: {invalid_syntax_count}")
    print(f"Inconsistent file path count: {inconsistent_path_count}")
    print(f"Inconsistent class name count: {inconsistent_class_count}")
    print(f"Valid instances count: {len(valid_instances)}")
    
    output_file = "shard_dataset_w_testcheck.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for instance in valid_instances:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')
    print(f"Valid instances written to {output_file}")

def set_test_code(args):
    # read the dataset dictionary
    benchmark_data = {}
    # with open("./example_set_test_code.jsonl", 'r') as f:
    with open("./shard_multi_agent.jsonl", 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    for data_key in benchmark_data.keys():
        data = benchmark_data[data_key]
        tests = data['tests']
        test_codes = []
        
        # construct the path of the test file
        test_path = os.path.join(args.source_code_root, data["project_path"])
        
        for test in tests:
            parts = test.split('::')
            file_path = parts[0]
            full_file_path = os.path.join(test_path, file_path)
            print(f"Processing: {full_file_path}, test: {test}")
            
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # parse the test file content
                if len(parts) == 2:  # file path::method name
                    method_name = parts[1]
                    # use AST to parse the file
                    tree = ast.parse(source_code)
                    
                    # find the target function
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == method_name:
                            # get the start line of the decorator (the smallest line number)
                            decorator_lines = [d.lineno for d in node.decorator_list] if node.decorator_list else []
                            start_line = min(decorator_lines) if decorator_lines else node.lineno
                            
                            # get the end line of the function
                            end_line = node.end_lineno if hasattr(node, 'end_lineno') else None
                            if end_line is None:
                                last_node = node.body[-1] if node.body else node
                                end_line = last_node.lineno if hasattr(last_node, 'lineno') else node.lineno
                            
                            # extract the function from the source code (including multi-line decorators)
                            lines = source_code.splitlines()
                            function_lines = lines[start_line-1:end_line]
                            test_codes.append('\n'.join(function_lines))
                            break
                
                elif len(parts) == 3:  # file path::class name::method name
                    class_name = parts[1]
                    method_name = parts[2]
                    # use AST to parse the file
                    tree = ast.parse(source_code)
                    
                    # find the target class and method
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            # find the target method in the class
                            for class_node in node.body:
                                if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                                    # get the start line of the decorator (the smallest line number)
                                    decorator_lines = [d.lineno for d in class_node.decorator_list] if class_node.decorator_list else []
                                    start_line = min(decorator_lines) if decorator_lines else class_node.lineno
                                    
                                    # get the end line of the method
                                    end_line = class_node.end_lineno if hasattr(class_node, 'end_lineno') else None
                                    if end_line is None:
                                        last_node = class_node.body[-1] if class_node.body else class_node
                                        end_line = last_node.lineno if hasattr(last_node, 'lineno') else class_node.lineno
                                    
                                    # extract the method from the source code (including multi-line decorators)
                                    lines = source_code.splitlines()
                                    method_lines = lines[start_line-1:end_line]
                                    test_codes.append('\n'.join(method_lines))
                                    break
            
            except Exception as e:
                print(f"Error processing test file {full_file_path}: {str(e)}")
                continue
        
        data["test_codes"] = test_codes
        print(f"test_codes: {test_codes}")
    
    output_file_temp = "./settest_shard_multi_agent.jsonl"
    with open(output_file_temp, 'w') as f:
        for namespace, data in benchmark_data.items():
            f.write(json.dumps(data) + '\n')

def set_gt(args):
    benchmark_data = {}
    with open("./fixtest_filter_line10_0523.jsonl", 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    for data_key in benchmark_data.keys():
        data = benchmark_data[data_key]
        completion_path = Path(data['completion_path'])
        completion_path = os.path.join(args.source_code_root, completion_path)
        sos, eos = data['body_position'][0]-1, data['body_position'][1]
        sign_sos = data['signature_position'][0]-1
        
        with open(completion_path,'r') as f:
            file_lines = f.readlines()
            
            signature = ''.join(file_lines[sign_sos:sign_sos+1])
            
            body = ''.join(file_lines[sos:eos])
            
            decorator_start = None
            # check if there are decorators before the signature line
            if sign_sos > 0:
                # find the decorators from the signature line
                current_line = sign_sos - 1
                while current_line >= 0:
                    line_content = file_lines[current_line].strip()
                    if line_content.startswith('@'):  # decorator line
                        decorator_start = current_line
                        current_line -= 1  # continue to find possible multi-layer decorators
                    else:
                        break
            
            # if there are decorators, include them
            if decorator_start is not None:
                decorators = ''.join(file_lines[decorator_start:sign_sos])
                data['gt'] = decorators + signature + body
            else:
                data['gt'] = signature + body

    # sort by the length of the body_position interval
    sorted_data = dict(sorted(benchmark_data.items(), 
                            key=lambda x: x[1]['body_position'][1] - x[1]['body_position'][0]))

    # write to another file as output
    with open('fixgt_line10_0524.json','w') as f:
        json.dump(sorted_data, f, indent=2)
    
    with open('fixgt_line10_0524.jsonl', 'w') as f:
        for namespace, data in sorted_data.items():
            f.write(json.dumps(data) + '\n')


def get_testgen_prompt_message(
        full_instance:MTDevEvalInstance,
        turn_instance:ShardDevEval,
        success_trajectory: List[Dict],
        current_attempts: List[Dict] = []
    ) ->List:
    messages = []
    messages.append({"role": "system", "content": "You are an AI coding assistant.You must only output in the json format wrapped with ```json ```"})
    
    current_turn = turn_instance['turn']
    not_to_impl = ""
    for mt_data in full_instance['mt']:
        if mt_data['turn'] == current_turn:
            continue
        not_to_impl += mt_data['requirement']
        not_to_impl += '\n'
    
    if current_turn > 1 and success_trajectory:
        # turn>1 : add the task description of this round + the successful code of the previous round
        messages.append({
            "role": "user",
            "content": TEST_GEN_PROMPT.format(
                original_solution=full_instance['gt'],
                original_test_case="\n".join(full_instance["test_codes"]),
                not_to_impl=not_to_impl,
                current_requirement=turn_instance["requirement"],
                last_solution=success_trajectory[-1]["code"]
            ) + TEST_GEN_FORMAT_PROMPT
        })
    else:
        # turn=1: only add the task description of this round
        messages.append({
            "role": "user",
            "content": TEST_GEN_PROMPT.format(
                original_solution=full_instance['gt'],
                original_test_case="\n".join(full_instance["test_codes"]),
                not_to_impl=not_to_impl,
                current_requirement=turn_instance["requirement"],
                last_solution="This is the first round, so there are no solutions from the previous round."
            ) + TEST_GEN_FORMAT_PROMPT
        })
    
    # if there are attempts history: add the latest error feedback
    if current_attempts:
        last_attempt = current_attempts[-1]
        messages.append({
            "role": "assistant",
            "content": ASSISTANT_PROMPT_REPLACE.replace("{solution}", last_attempt["code"]).replace("{test}", last_attempt["test"])
        })
        messages.append({
            "role": "user",
            "content": TEST_GEN_FEEDBACK_PROMPT.format(feedback=last_attempt['feedback']) + TEST_GEN_FEEDBACK_FORMAT_PROMPT
        })
    
    return messages

def postprocess_testgen_result(llm_output:str):
    print("postprocessing ... ")
    testgen_outputs = parse_json_output(clean_markdown_code_block(llm_output))
    required_keys = {"solution","test"}
    
    if len(testgen_outputs) != 1 or not required_keys.issubset(testgen_outputs[0].keys()):
        print("!!! Error in postprocess_testgen_result")
        raise ValueError("Error in postprocess_testgen_result")
    
    if "reason" in testgen_outputs[0].keys():
        print(f"reason: {testgen_outputs[0]['reason']}")
    
    return testgen_outputs[0]["solution"],testgen_outputs[0]["test"]

def get_test_func_names(test_code:str):
    tree = ast.parse(test_code)
    # only get the module-level function definitions, ignore nested functions
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) == 0:
        raise ValueError("No function definition found in the code")
    return [fn.name for fn in functions]

def get_test_selector(test_code:str,original_test_selector:str):
    test_code = textwrap.dedent(test_code)
    # selector format: [file name]::[class name]::[function name] or [file name]::[function name]
    test_func_names = get_test_func_names(test_code)
    original_split = original_test_selector.split("::")
    
    # return the list of selectors for all test functions
    selectors = []
    for func_name in test_func_names:
        current_split = original_split.copy()
        current_split[-1] = func_name
        selectors.append("::".join(current_split))
    
    return selectors

import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def generate_test_and_code(args):
    """For each sample in the dataset, generate reference code and test cases for each round of instructions"""
    # read the dataset
    mt_ds:List[MTDevEvalInstance] = load_dataset(args.dataset)
    # save file path
    output_file = args.gentest_output # output file
    doublepass_output_file = "./doublepass_output_file.jsonl" # downgrade output file
    error_file = './claude4_gentest_error_concurrent.jsonl' # error output file
    # maximum retry times
    MAX_RETRY = 10
    # downgrade strategy threshold
    MAX_CONSECUTIVE_PASS = 6

    # read the processed dataset to avoid duplicate processing
    processed_namespaces = set()
    with open(args.gentest_output, 'r', encoding='utf-8') as f:
        for line in f:
            processed_data = json.loads(line)
            processed_namespaces.add(processed_data['namespace'])

    # read the processed samples in the downgrade file
    if os.path.exists(doublepass_output_file):
        with open(doublepass_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                processed_data = json.loads(line)
                processed_namespaces.add(processed_data['namespace'])

    # group and prepare for concurrent processing: group by namespace.split('.')[0]
    grouped_data = defaultdict(list)
    for data in mt_ds:
        namespace = data['namespace']
        if namespace not in processed_namespaces:
            group_key = namespace.split('.')[0]
            grouped_data[group_key].append(data)
    
    file_lock = threading.Lock()

    def process_group(group_key,group_data):
        for data in group_data:
            namespace = data['namespace']
            total_turn = len(data['mt'])
            test_selector = data['tests'][0]
            mt_tests = {}
            success_trajectory = []
            max_retries = MAX_RETRY
            
            for turn_data in data['mt']:
                current_turn = turn_data['turn']
                retry_count = 0
                success = False
                current_attempts = []
                
                while not success and retry_count < max_retries:
                    print(f"processing task_id: {namespace}, turn: {turn_data['turn']}/{total_turn},retry:{retry_count}/{max_retries} ")
                    turn_prompt = get_testgen_prompt_message(data,turn_data,success_trajectory,current_attempts)
                    llm_output = call_llm(turn_prompt)
                    try:
                        turn_code, turn_tests = postprocess_testgen_result(llm_output)
                    except Exception as e:
                        print(f"JSON parsing error, retrying...\nLLM output:\n\n{str(llm_output)}\n\n")
                        retry_count += 2
                        continue
                    # parse the test, if the format is not a function, it will report an error
                    try:
                        turn_test_selector = get_test_selector(test_code=turn_tests,original_test_selector=test_selector)
                    except Exception as e:
                        print(f"An error occurred in get_test_selector. Retrying...\nMessage:\n\n{str(e)}\n\nLLM output:\n\n{str(llm_output)}\n\n")
                        retry_count += 2
                        continue
                    print(f"\n\n LLM output for namespace:{namespace}, turn:{current_turn}/{total_turn}, retry:{retry_count}/{max_retries}\n\n  ==========Current Requirement========== \n {turn_data['requirement']} \n==========Reference Code========== \n  {data['gt']}  \n ==========Reference Test========== \n {''.join(data['test_codes'])} \n ==========code:==========\n{turn_code} \n ==========test:========== \n{turn_tests} \n ==========test_selector:========== \n {turn_test_selector} \n\n Go to verify in this round ...... \n\n")
                    # Verify this turn
                    try:
                        status_current,detail_current = verify_llm_testgen(turn_code,turn_tests,data,turn_test_selector)
                    except Exception as e:
                        print(f"n error occurred in verify_llm_testgen Retrying...\nMessage:\n\n{str(e)}\n\nLLM output:\n\n{str(llm_output)}\n\n")
                        retry_count += 2
                        continue

                    if current_turn == 1:
                        if status_current == 'pass':
                            print(f"\n【great!】 namespace: {namespace} pass turn1 test.\n")
                            success = True
                            success_trajectory.append({
                                'turn': current_turn,
                                'code': turn_code,
                                'test': turn_tests
                            })
                            mt_tests[int(current_turn)] = turn_test_selector
                        else:
                            print(f"\n【Oh no!】 namespace: {namespace} fail turn1 test. \n\n Status: {status_current} \n\nDetail:{str(detail_current)}.\n")
                            current_attempts.append({
                                'code': turn_code,
                                'test': turn_tests,
                                'feedback': f"**Human Feedback**: Your solution code fails the test case you haved generated.\n**Execution feedback**: status:{str(status_current)},detail: {str(detail_current)}"
                            })
                    else:
                        prev_code = success_trajectory[-1]['code']
                        print('\n' + "="*15 + f"verify last round {namespace} solution_{current_turn-1} in turn{current_turn}/{total_turn},retry:{retry_count}/{max_retries} " + "="*15 + "\n")
                        
                        try:
                            status_prev, _ = verify_llm_testgen(prev_code, turn_tests,data,turn_test_selector)
                        except Exception as e:
                            print(f"An error occur during verify_llm_testgen Retrying\nError message:\n\n{str(e)}\n\nLLM Output:\n\n{str(llm_output)}\n\n")
                            retry_count += 2
                            continue

                        if status_current == 'pass' and status_prev != 'pass':
                            print(f"\n【great!】 namespace: {namespace} pass turn{current_turn} test and fail turn{current_turn-1} test.\n")
                            success = True
                            success_trajectory.append({
                                'turn': current_turn,
                                'code': turn_code,
                                'test': turn_tests
                            })
                            mt_tests[int(current_turn)] = turn_test_selector
                        else:
                            # Check if the downgrading conditions are met
                            is_last_turn = current_turn == total_turn
                            consecutive_pass = status_current == 'pass' and status_prev == 'pass'
                            if consecutive_pass and is_last_turn:
                                # Check if the maximum consecutive pass times have been reached
                                if retry_count >= MAX_CONSECUTIVE_PASS - 1:  # because retry_count starts from 0
                                    print(f"\n【Warning - double pass】 namespace: {namespace} meets double pass condition in last turn, applying downgrade strategy.\n")
                                    success = True
                                    success_trajectory.append({
                                        'turn': current_turn,
                                        'code': turn_code,
                                        'test': turn_tests,
                                        'downgraded': True  # Marked as a downgraded sample
                                    })
                                    mt_tests[int(current_turn)] = turn_test_selector
                                    break

                            print(f"\n【Oh no!】 namespace: {namespace} fail turn{current_turn} test or pass turn{current_turn-1} test. status_current:{status_current},status_prev: {status_prev}\n")
                            feedback_parts = []
                            if status_current != 'pass':
                                feedback_parts.append(f"**Human Feedback**: Your solution code fails the test case you haved generated. **Execution feedback**: status:{str(status_current)} {str(detail_current)}")
                            if status_prev == 'pass':
                                feedback_parts.append("**Human Feedback**: Previous turn's code passed current test. (This is not desirable, we want the test case to be valid only for the current round and fail for the previous rounds.)")
                            
                            current_attempts.append({
                                'code': turn_code,
                                'test': turn_tests,
                                'feedback': 'Another feedback:'.join(feedback_parts)
                            })
                    
                    retry_count += 1

                if not success:
                    print("!"*40,'\n\n',f"【QAQ】 Try over max_retries ,task_id{namespace}, turn={current_turn}/{total_turn}","!"*40,'\n\n')
                    error_info = {
                        'namespace': namespace,
                        'turn': current_turn,
                        'total_turn': total_turn,
                        "original_code": data['gt'],
                        "original_test": data['test_codes'],
                        'last_attempt': current_attempts[-1] if current_attempts else None
                    }
                    with file_lock: 
                        with open(error_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_info) + '\n')
                    break

            if len(success_trajectory) == total_turn:

                downgraded = any('downgraded' in item for item in success_trajectory)
                output_target = doublepass_output_file if downgraded else output_file
                if downgraded:
                    print("Successfully processed the downgraded samples!")
                print(f"【Wonderful】!! Namespace : {namespace} generate test successfully, appending log...")

                if "mt_tests" not in data:
                    data["mt_tests"] = {}
                data["mt_tests"] = mt_tests
                for idx,item in enumerate(success_trajectory):
                    data["mt"][idx]["gt"] = item["code"]
                    data["mt"][idx]["test_code"] = item["test"]
                    data["mt"][idx]["tests"] = mt_tests[data["mt"][idx]["turn"]]
                del data["test_list"]
                with file_lock: 
                    with open(output_target, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(data) + '\n')
                    print(f"Append done! Namespace:{namespace}")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for group_key, group_data in grouped_data.items():
            futures.append(executor.submit(process_group, group_key, group_data))
        
        for future in futures:
            future.result()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,help="dataset path",default="./fixturn_shard_multi_agent.jsonl")
    parser.add_argument("--shard",action="store_true",help="whether to shard the instruction")
    parser.add_argument('--source_code_root', type=Path, default=Path('Source_Code'))
    parser.add_argument("--gentest_output",type=str,help="output path of the shard instruction",default="./gentest_0524.jsonl")
    parser.add_argument("--output_file",type=str,help="output path of the shard instruction",default="./shard_0524.jsonl")
    parser.add_argument("--report",action="store_true",help="whether to generate the data report")
    parser.add_argument("--filter",action="store_true",help="whether to filter the samples with turn>5")
    parser.add_argument("--postprocess",action="store_true",help="postprocess, add Context")
    parser.add_argument("--check_test_and_filter",action="store_true",help="check if the test cases are in the same file and filter the samples that do not meet the requirements")
    parser.add_argument("--set_test_code",action="store_true",help="add all the test cases to the samples")
    parser.add_argument("--set_gt",action="store_true",help="reset the ground truth")
    parser.add_argument("--generate_test_and_code",action="store_true",help="generate reference code and test cases for each round of the samples")
    
    args = parser.parse_args()

    # shard the instruction
    if args.shard:
        shard_instruction(args)
    
    # generate report
    if args.report:
        generate_report(args.output_file)
    
    # filter
    if args.filter:
        filter_mt_instances(args.output_file,5)
    
    # postprocess, add related information
    if args.postprocess:
        postprocess_data(args)
    
    # check the test cases
    if args.check_test_and_filter:
        check_test_and_filter(args)
    
    # add test code to the samples
    if args.set_test_code:
        set_test_code(args)
    
    # generate reference code and test cases for each round
    if args.generate_test_and_code:
        generate_test_and_code(args)
    
    # set reliable ground truth, including annotations
    if args.set_gt:
        set_gt(args)