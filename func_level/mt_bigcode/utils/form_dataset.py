import json
from typing import List, TypedDict
import argparse

class TurnDataInstance(TypedDict):
    task_id: str # e.g. 'BigCodeBench/17_1'
    turn: str # e.g. 1
    instruct_prompt: str
    test: str
    code: str
    entry_point: str # always 'task_func'

class MTBigCodeInstance(TypedDict):
    mt_id: int # e.g. 1
    task_id: str # e.g. 'BigCodeBench/17'
    mt_data: List[TurnDataInstance]

def extract_number(task_key: str) -> int:
    return int(task_key.split('/')[-1])

def sort_json_data(data: list) -> list:
    """Sort the data by task_key, for example, 'BigCodeBench/17' is sorted by the numerical part"""
    return sorted(data, key=lambda item: extract_number(next(iter(item))))

def main(testgen_path:str, output:str):
    decompose_log_path = "./logs/decompose.json"
    # Load and sort decompose.json
    with open(decompose_log_path, 'r', encoding='utf-8') as f:
        instruction_data = json.load(f)
    instruction_data = sort_json_data(instruction_data)

    # Load and sort testgen.json
    with open(testgen_path, 'r', encoding='utf-8') as f:
        code_and_test_data = json.load(f)
    code_and_test_data = sort_json_data(code_and_test_data)

    # Ensure the number of tasks is consistent
    assert len(instruction_data) == len(code_and_test_data), \
        "The number of tasks in decompose.json and testgen.json is inconsistent"

    mt_bigcode_dataset: List[MTBigCodeInstance] = []

    for mt_id, (inst_item, code_test_item) in enumerate(zip(instruction_data, code_and_test_data), start=1):
        task_key = next(iter(inst_item))  # Get the current task key, such as BigCodeBench/17

        inst_list = inst_item[task_key]
        code_test_list = code_test_item[task_key]

        assert len(inst_list) == len(code_test_list), \
            f"The number of turns in task {task_key} is inconsistent"

        split_instance: List[TurnDataInstance] = []
        for i in range(len(inst_list)):
            inst = inst_list[i]
            code_test = code_test_list[i]

            merged_item = TurnDataInstance(
                task_id=inst["task_id"],
                turn=str(inst["turn"]),
                instruct_prompt=inst["instruct_prompt"],
                test=code_test["test"],
                code=code_test["code"],
                entry_point=inst["entry_point"]
            )
            split_instance.append(merged_item)

        mt_bigcode_dataset.append({
            "mt_id": mt_id,
            "task_id": task_key,
            "mt_data": split_instance
        })

    # Write to the output file
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(mt_bigcode_dataset, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--testgen_path",type=str,default="./logs/testgen.json",help="The path of the JSON file containing testgen")
    parser.add_argument("--output",type=str,default="mt_bigcode.json",help="The path of the output file")
    args = parser.parse_args()

    main(testgen_path=args.testgen_path,output=args.output)