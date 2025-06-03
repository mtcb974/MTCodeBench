from typing import List,Dict,Any,TypedDict,Optional
from model.prompts.ins import DECOMPOSE_INSTRUCTION_PROMPT,DECOMPOSE_OUTPUT_FORMAT,DECOMPOSE_SYSTEM_MESSAGE
from model.prompts.test import TEST_GEN_SYSTEM_PROMPT,TEST_GEN_FEEDBACK_PROMPT,TEST_GEN_FORMAT_PROMPT,TEST_GEN_PROMPT,ASSISTANT_PROMPT_REPLACE,TEST_GEN_FEEDBACK_FORMAT_PROMPT
from model.prompts.inference import INSTRUCTION_PREFIX,EDIT_PROMPT
from data.loader import get_bigcode_hard
from data import BigCodeBenchInstance,SplitBigCodeInstance,TurnResult,MTBigCodeInstance
from model.prompts import PromptEntry

def generate_messages(
    prompt: str,
    system_message: str,
    history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": system_message}
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages

def get_decompose_prompts()->List[PromptEntry]:
    """Return the list of prompt entities for the instruction"""
    bigcode_ds = get_bigcode_hard()
    prompt_entrys:List[PromptEntry] = []

    for bigcode_instance in bigcode_ds:
        decompose_prompt = DECOMPOSE_INSTRUCTION_PROMPT.format(
            original_requirement=bigcode_instance["instruct_prompt"],
            original_code=bigcode_instance["canonical_solution"]
        )
        prompt_entry:PromptEntry = {
            "bigcode_instance": bigcode_instance,
            "prompt": decompose_prompt + DECOMPOSE_OUTPUT_FORMAT,
            "system_prompt": DECOMPOSE_SYSTEM_MESSAGE
        }
        prompt_entrys.append(prompt_entry)

    return prompt_entrys

def get_testgen_prompt_message(
        bigcode_instance:BigCodeBenchInstance,
        split_instance:SplitBigCodeInstance,
        success_trajectory: List[Dict],
        current_attempts: List[Dict] = []
    ) ->PromptEntry:

    messages = [
        {"role": "system", "content": TEST_GEN_SYSTEM_PROMPT}
    ]
    current_turn = split_instance['turn']

    if current_turn > 1 and success_trajectory:
        # turn>1 : add the task description of this turn + the last successful code
        messages.append({
            "role": "user",
            "content": TEST_GEN_PROMPT.format(
                original_requirement=bigcode_instance["instruct_prompt"],
                original_code=bigcode_instance["canonical_solution"],
                original_test=bigcode_instance["test"],
                current_round=split_instance["turn"],
                current_requirement=split_instance["instruct_prompt"],
                last_solution=success_trajectory[-1]["code"]
            ) + TEST_GEN_FORMAT_PROMPT
        })
    else:
        # turn=1: only add the task description of this turn
        messages.append({
            "role": "user",
            "content": TEST_GEN_PROMPT.format(
                original_requirement=bigcode_instance["instruct_prompt"],
                original_code=bigcode_instance["canonical_solution"],
                original_test=bigcode_instance["test"],
                current_round=split_instance["turn"],
                current_requirement=split_instance["instruct_prompt"],
                last_solution="This is the first round, so there are no solutions from the previous round."
            ) + TEST_GEN_FORMAT_PROMPT
        })
    
    # If there is a history of attempts, add the latest error feedback
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

def get_infer_prompt(
        trajectories:List[TurnResult],
        instruction_prompt:str,
        golden:bool,
        mt_data:Optional[MTBigCodeInstance],
        qwen3_thinking_mode:bool,
        prompt_mode:str
    )->List[Dict[str,str]]:

    if golden == True and mt_data is None:
        raise ValueError("If golden is true, then user must provide mt_data")
    
    messages = []

    # Qwen3's thinking mode switch
    if not qwen3_thinking_mode:
        messages.append({"role":"system","content": "/no_think"})

    if prompt_mode == 'direct':
        # direct mode
        task_prompt = f"""{INSTRUCTION_PREFIX}
    {instruction_prompt.strip()}
    """
        # If there is history data, add it in.
        if len(trajectories) > 0:
            for idx,traj in enumerate(trajectories):
                user_msg = {"role": "user","content": traj["instruct_prompt"]}
                if golden:
                    assistant_msg = {"role": "assistant", "content": f"```python\n{mt_data['mt_data'][idx]['code']}```"}
                else:
                    assistant_msg = {"role": "assistant", "content": traj["raw_solution"]}
                messages.append(user_msg)
                messages.append(assistant_msg)
        # Add the current round's message
        messages.append({"role": "user", "content": task_prompt})
    elif prompt_mode == 'edit':
        # Edit mode
        # If it is the first round, use the same prompt as above, and consider whether it is golden
        # If it is not the first round, only use the Edit template
        trajectories_len = len(trajectories)
        if trajectories_len > 0:
            instruction = instruction_prompt.strip()
            if golden:
                previous_code = mt_data['mt_data'][trajectories_len-1]['code']
            else:
                previous_code = trajectories[-1]["solution"]

            task_prompt = EDIT_PROMPT.format(previous_code=previous_code,instruction=instruction)
            messages.append({"role": "user", "content": task_prompt})

        else:
            task_prompt = f"""{INSTRUCTION_PREFIX}
        {instruction_prompt.strip()}
        """
            messages.append({"role": "user", "content": task_prompt})
    elif prompt_mode == 'append':
        # Append mode
        if golden:
            raise ValueError("Append Mode do not support golden")
        
        instructions = ""
        trajectories_len = len(trajectories)
        
        for idx, traj in enumerate(trajectories):
            instructions = instructions + '\n' + traj["instruct_prompt"].strip()
        
        instructions  = instructions + '\n' + instruction_prompt
        task_prompt = f"""{INSTRUCTION_PREFIX}
        {instructions}
        """
        messages.append({"role": "user", "content": task_prompt})
    else:
        raise ValueError("You have provided an unsupported prompt_mode")
    return messages
    


    