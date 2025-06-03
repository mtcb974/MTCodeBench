import json
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from tqdm import tqdm
from typing import List,TypedDict,Dict
import re

class ShardDevEval(TypedDict):
    turn: int
    requirement: str
    test_code: str
    tests: List[str]
    gt: str

class DevEvalDependency(TypedDict):
    intra_class: List[str]
    intra_file: List[str]
    cross_file: List[str]

class DevEvalRequirement(TypedDict):
    Functionality: str
    Arguments: str

class MTDevEvalInstance(TypedDict):
    namespace: str
    type: str
    project_path: str
    completion_path: str
    signature_position: List[int]
    body_position: List[int]
    dependency: DevEvalDependency
    requirement: DevEvalRequirement
    tests: List[str]
    indent: int
    domain: str
    gt: str
    context: str
    test_codes: List[str] # Original test strings
    mt: List[ShardDevEval] # Decomposed dataset, contains: (generated when decomposing)turn,requirement (test case synthesis)test_code,tests[test selector],gt
    mt_tests: Dict[int,List[str]] # Multi-turn test selector field, Key is turn, Value is test selector list


def parse_decomposition_to_mt(decomposed_text: str) -> List[ShardDevEval]:
    """
    Parse the result of multi-turn instruction decomposition into a structured format List[ShardDevEval]
    Example input format:
        Turn1: [basic requirement]
        Turn2: [first clarification]
    """
    pattern = r"Turn(\d+)\s*[:：]\s*(.+)"
    matches = re.findall(pattern, decomposed_text)

    mt_result = []
    for turn_num, req in matches:
        mt_result.append({
            "turn": int(turn_num) - 1,  
            "requirement": req.strip(),
            "gt": "",
            "test_code": "",  
            "tests": [],   
        })
    print(f"Regex processing result:{mt_result}")
    return mt_result


os.environ["OPENAI_API_KEY"] = ""
BASE_URL = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(model="gpt-4.1", temperature=0.5,base_url=BASE_URL)

class State(dict):
    pass

# ---------- Decomposer ----------
def decompose_agent(state: State):
    original_code = state["original_code"]
    original_requirement = state["instruction"]
    feedback = state.get("inspector_feedback", "")
    
    prompt = f"""You are an expert assistant for multi-turn instruction design.

Given a Python function code and its task description, your goal is to convert the task into:
1. A **basic requirement instruction** that describes the user's initial intent
2. Up to **4 clarifying restriction instructions**, representing how a user might refine the requirement in multiple rounds

Constraints:
- Simulate the progressive clarification of vague requirements or add constraint by users
- The number of rounds should be **scientifically determined**, but **not exceed 5**
- Later instructions should not be **easily satisfied by early implementations**
- Each instruction must be **verifiable** and **non-conflicting**
- Order matters: later constraints should depend on earlier steps

--- Python function code ---
{original_code}

--- Function description ---
{original_requirement}

--- Previous Feedback ---
{feedback}

Please format your response as:
Turn1: [basic requirement]
Turn2: [first clarification]
...
"""
    messages = [
        SystemMessage(content="You are a helpful instruction decomposition agent."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    print(f"Decomposer:{response.content}")
    state["decomposed_steps"] = response.content
    return state

# ---------- Verifier ----------
def inspector_agent(state: State):
    code = state["original_code"]
    instruction = state["instruction"]
    decomposition = state["decomposed_steps"]

    prompt = f"""
You are an instruction quality inspector.

Your task is to evaluate whether a given instruction decomposition is **scientific and verifiable**.

You must assess the decomposition based on **three dimensions**:
1. **Distinction**: Can you ensure that the following instructions will not be easily implemented by the previous instructions?
2. **Testability**: Is each step easy to test and complete correctly?
3. **Overall design**: Are there any redundant steps? Is the decomposition logically progressive and coherent?

--- Python function code ---
{code}

--- Original Requirement ---
{instruction}

--- Decomposed Instructions ---
{decomposition}

Respond with:
[Analysis]
...
[Dimension 1 Score (0-10)]:
[Dimension 2 Score (0-10)]:
[Dimension 3 Score (0-10)]:
[Confidence Level (0-100%)]:
[Final Decision: Re-decompose] or [Final Decision: Accept]
"""
    messages = [
        SystemMessage(content="You are a professional code instruction decomposition checker who is able to check whether the decomposition meets the requirements."),
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages)
    feedback = response.content
    print(f"Inspector:{feedback}")
    state["inspector_feedback"] = feedback
    state["inspection_passed"] = "Final Decision: Accept" in feedback
    return state

# ---------- Check if retry ----------
def check_inspection_passed(state: State):
    return "pass" if state.get("inspection_passed") else "retry"

# ---------- Multi-agent LangGraph ----------
graph = StateGraph(State)
graph.add_node("Decomposer", decompose_agent)
graph.add_node("Inspector", inspector_agent)
graph.set_entry_point("Decomposer")
graph.add_edge("Decomposer", "Inspector")
graph.add_conditional_edges("Inspector", check_inspection_passed, {
    "pass": END,
    "retry": "Decomposer"
})
workflow = graph.compile()

# ========== Batch data running ==========
def run_batch(input_file, output_file):
    completed_namespaces = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    instance = json.loads(line)
                    completed_namespaces.add(instance['namespace'])

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(dataset)} data")
    for item in tqdm(dataset):
        if item['namespace'] in completed_namespaces:
            print(f"Skipped completed instance: {item['namespace']}")
            continue

        print(f"\nStart processing instance: {item['namespace']}")
        # Build initial state
        input_state = State({
            "instruction": f"{item['requirement']['Functionality']} \n {item['requirement']['Arguments']}",
            "original_code": item["gt"]
        })

        try:
            final_state = workflow.invoke(input_state)
        except Exception as e:
            print(f"Error processing {item['namespace']}: {e}")
            continue

        decomposed_text = final_state.get("decomposed_steps", "")
        print(f"Final decomposition result: {decomposed_text}")

        # Structured decomposition content
        try:
            item["mt"] = parse_decomposition_to_mt(decomposed_text)
            item["mt_tests"] = {} 
        except Exception as e:
            print(f"Error parsing decomposition for {item['namespace']}: {e}")
            item["mt"] = []
            item["mt_tests"] = {}

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Instance {item['namespace']} processed and saved")

# ========== Main program entry ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    args = parser.parse_args()

    run_batch(args.input_file, args.output_file)
