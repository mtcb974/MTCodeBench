import re
from typing import List,Dict,Any
import json
from data import SplitBigCodeInstance
from model.prompts import PromptEntry

def clean_markdown_code_block(output: str) -> str:
    return re.sub(r'^```json\s*|\s*```$', '', output, flags=re.MULTILINE)

def extract_code_block(text):
    # Use regular expressions to match the content enclosed by ```
    pattern = r'```(?:\w+)?\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

def parse_json_output(output: str) -> List[Dict[str, Any]]:
    try:
        raw_data = json.loads(output)
        if not isinstance(raw_data, list):
            raise ValueError("Model output should be a JSON array.")
        return raw_data
    except json.JSONDecodeError as e:
        print(f"An error occurred during JSON serialization. Model output: \n\n{output}")
        raise e

def postprocess_decompose_result(llm_output:str,prompt_entry:PromptEntry):
    """
    llm_output: List[{turn,instruct_prompt}]
    """
    decompose_json_res = parse_json_output(clean_markdown_code_block(llm_output))

    required_keys = {"turn","instruct_prompt"}
    turn1_data = decompose_json_res[0]
    return [
        SplitBigCodeInstance(
            # The decomposed information output by LLM
            turn=turn_data["turn"],
            instruct_prompt=turn_data["instruct_prompt"],
            # Additional supplementary information
            task_id=prompt_entry["bigcode_instance"]["task_id"]+ "_" + str(turn_data["turn"]),
            entry_point=prompt_entry["bigcode_instance"]["entry_point"]
        ) for turn_data in decompose_json_res if required_keys.issubset(turn_data.keys())
    ]

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
    
