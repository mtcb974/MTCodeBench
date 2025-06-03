import json
import argparse
import tiktoken
from typing import Dict,List
from modelscope import AutoTokenizer

def load_dataset(file_path:str) -> List:
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def format_openai_messages(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += "<|system|>\n" + msg["content"] + "\n"
        elif msg["role"] == "user":
            prompt += "<|user|>\n" + msg["content"] + "\n"
        elif msg["role"] == "assistant":
            prompt += "<|assistant|>\n" + msg["content"] + "\n"
    prompt += "<|assistant|>\n"
    return prompt

def get_infer_file(llm:str,golden:bool,qwen3_thinking_mode:bool, prompt_mode:str) -> str:

    golden_str = "golden" if golden else "base"
    think_str = "_think" if qwen3_thinking_mode else ""
    prompt_mode = "_"+prompt_mode if prompt_mode != "direct" else ""
    llm = llm.replace("/","_")
    llm = llm.replace(":","_")
    
    return  f"./mt_deveval_result_{llm}_{golden_str}{think_str}{prompt_mode}.jsonl"

def get_output_file(llm:str,golden:bool,qwen3_thinking_mode:bool, prompt_mode:str) -> str:

    golden_str = "golden" if golden else "base"
    think_str = "_think" if qwen3_thinking_mode else ""
    prompt_mode = "_"+prompt_mode if prompt_mode != "direct" else ""
    llm = llm.replace("/","_")
    llm = llm.replace(":","_")
    
    return  f"./token_count/token_{llm}_{golden_str}{think_str}{prompt_mode}.jsonl"

def calculate(llm:str,golden:bool,qwen3_thinking_mode:bool, prompt_mode:str):
    
    infer_path = get_infer_file(llm=llm,golden=golden,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode)
    output_file = get_output_file(llm=llm,golden=golden,qwen3_thinking_mode=qwen3_thinking_mode,prompt_mode=prompt_mode)

    if "Qwen" in llm:
        tokenizer = AutoTokenizer.from_pretrained(llm)
        is_qwen = True
    else:
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise ValueError(f"Failed to load cl100k_base tokenizer for non-Qwen model: {e}")
        is_qwen = False

    all_input_tokens = []
    all_output_tokens = []
    all_total_tokens = []

    tasks_stats = []

    with open(infer_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            task = json.loads(line)
            namespace = task.get("namespace","")
            completions = task.get("completions",[])
            
            input_tokens_sum = 0
            output_tokens_sum = 0

            messages = []

            if not qwen3_thinking_mode and is_qwen:
                messages.append({"role":"system","content": "/no_think"})

            for turn_data in completions:
                messages = turn_data['messages']
                turn = turn_data['turn']
                raw_completion = turn_data['raw_completion']

                if is_qwen:
                    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    input_token_count = len(input_ids)
                    output_token_count = len(tokenizer(raw_completion).input_ids)
                else:
                    input_text = format_openai_messages(messages)
                    input_token_count = len(tokenizer.encode(input_text))
                    output_token_count = len(tokenizer.encode(raw_completion))
                        
                input_tokens_sum += input_token_count
                output_tokens_sum += output_token_count
            
            total_tokens = input_tokens_sum + output_tokens_sum
            all_input_tokens.append(input_tokens_sum)
            all_output_tokens.append(output_tokens_sum)
            all_total_tokens.append(total_tokens)

            tasks_stats.append({
                "namespace": namespace,
                "input_tokens": input_tokens_sum,
                "output_tokens": output_tokens_sum,
                "total_tokens": total_tokens
            })


    avg_input = sum(all_input_tokens) / len(all_input_tokens)
    avg_output = sum(all_output_tokens) / len(all_output_tokens)
    avg_total = sum(all_total_tokens) / len(all_total_tokens)

    final_stats = {
        "tasks": tasks_stats,
        "average_stats": {
            "avg_input_tokens": round(avg_input, 2),
            "avg_output_tokens": round(avg_output, 2),
            "avg_total_tokens": round(avg_total, 2)
        }
    }

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(final_stats, out_f, indent=4, ensure_ascii=False)

    print(f"\nStatistics results have been written to file: {output_file}")
    

if __name__ == '__main__':
    llms = ["gemini-2.5-flash-preview-04-17","anthropic_claude-3.7-sonnet","Qwen/Qwen3-32B"]
    prompt_modes = ["direct","edit","append"]

    for llm in llms:
        for prompt_mode in prompt_modes:
            for golden in [True,False]:
                if prompt_mode == 'append' and golden == True:
                    continue
                if 'Qwen' in llm:
                    for qwen3_thinking_mode in [True,False]:
                        think = qwen3_thinking_mode
                        calculate(llm=llm,prompt_mode=prompt_mode,qwen3_thinking_mode=think,golden=golden)
                else:
                    calculate(llm=llm,prompt_mode=prompt_mode,qwen3_thinking_mode=False,golden=golden)
