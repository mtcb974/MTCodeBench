import argparse
from service.gen_ins import gen_decompose_instruction
from service.gen_test import gen_test_and_verify_loop
from service.gen_infer import multi_turn_inference

def generate_dataset_and_verify(gen_base:bool):
    print("Start generate mt dataset.")
    # Execute decomposition
    gen_decompose_instruction()

    # Generate test cases and code
    gen_test_and_verify_loop(gen_base=gen_base)

def run_inference(
    llm: str,
    golden: bool,
    dataset: str,
    backend: str,
    qwen3_thinking_mode: bool,
    prompt_mode: str,
    run_id:str
):
    multi_turn_inference(
        llm=llm,
        golden=golden,
        dataset=dataset,
        backend=backend,
        qwen3_thinking_mode=qwen3_thinking_mode,
        prompt_mode=prompt_mode,
        run_id=run_id
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="./dataset/mt_bigcode.json",type=str)
    parser.add_argument("--gen_data",action="store_true",help="Start the data construction process")
    parser.add_argument("--gen_base",action="store_true",help="The base method is adopted when building the dataset.")
    parser.add_argument("--inference",action="store_true",help="Start the inference mode")
    parser.add_argument("--llm",default="deepseek-v3",type=str,help="LLM")
    parser.add_argument("--golden",action="store_true",help="Whether to use the correct answer as historical information")
    parser.add_argument("--backend",type=str,help="Select the executed Api backend")
    parser.add_argument("--qwen3_thinking_mode",action="store_true",help="Whether the Qwen3 model has activated the thinking mode")
    parser.add_argument("--prompt_mode",default="direct",choices=["direct","edit","append"],help="prompt mode")
    parser.add_argument("--run_id",type=str,default="",help="Used for the comparative experiment of RQ3")
    args = parser.parse_args()

    if args.gen_data:
        # The pipeline for constructing the dataset
        generate_dataset_and_verify(args.gen_base)
    elif args.inference:
        run_inference(
            llm=args.llm,
            golden=args.golden,
            dataset=args.dataset,
            backend=args.backend,
            qwen3_thinking_mode=args.qwen3_thinking_mode,
            prompt_mode=args.prompt_mode,
            run_id=args.run_id
        )