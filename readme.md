# MTCodeBench

## About

MTCodeBench is a benchmark for evaluating LLMs in progressive multi-turn code generation scenario.

## Function-Level

Function level datasets are in `dataset/function_level.json`, and the code is in `func_level/mt_bigcode`.

### Dataset Format

Data format is as follows:
```
  {
    "mt_id": <dataset ID>,
    "task_id": <dataset ID in bigcodebench>,
    "mt_data": [
      {
        "task_id": <dataset ID with turn>,
        "turn": <turn number>,
        "instruct_prompt": <instruction prompt for LLM>,
        "test": <test case in this round>,
        "code": <reference code in this round>,
        "entry_point": <function entry point in bigcodebench>
      }
      ...
    ]
  },
```

### Inference

Enter the inference dir:
```bash
cd func_level/mt_bigcode
```

Setup api key:
```bash
export OPENAI_API_KEY=<YOU API KEY>
export OPENAI_BASE_URL=<YOU CHOICE OF LLM PLATFORM>
```

For inference, use command below:
```bash
python main.py --inference --llm deepseek-v3 --backend deepseek --prompt_mode direct
```

You can specify params:
- `dataset`: You can specify dataset. Default value is `./dataset/mt_bigcode.json`.
- `llm`: Specify your llm.
- `golden`: Whether to use the correct answer as historical information
- `backend`: Your Api backend
- `qwen3_thinking_mode`: Whether the Qwen3 model has activated the thinking mode
- `prompt_mode`: direct for `Full History`, edit for `Code Edit`, `append` for `Cumulative instruction`
- `run_id`: Used for the comparative experiment of RQ3


### Evaluation

For convenience, the evaluation adopts CS mode. First, you need to open the FastApi service of another terminal.

It is highly recommended to create a separate environment to run the assessment:
```bash
conda create -n mt-bigcode-eval python=3.10
conda activate mt-bigcode-eval
```

Install related dependencies:
```bash
pip install bigcodebench --upgrade
pip install -r ./Requirements/requirements-eval.txt
```

Then you can then run the fastapi program to prepare for subsequent evaluation:
```bash
cd func_level/eval
uvicorn main:app --reload
```

After inference, you can use eval_client to perform evaluation:
```bash
cd func_level/mt_bigcode
python eval_client.py --path YOU_INFERENCE_RESULT_PATH --llm YOU_LLM --type (Identifier that can identify the result of this inference)
```

## Repo-Level

Repo level datasets are in `dataset/repo_level.json`, and the code is in `repo_level/mt_deveval`.

### Dataset Format

The data format is as follows:

```
{
    ## The same as original DevEval
    "namespace": <string, the unique name of the code to be generated, e.g., benedict.utils.type_util.is_bool.>,
    "type": <string, the type of the code to be generated. method means the code is a member function in a class, and function means the code is a individual function.>,
    "project_path": <string, the path of the project, e.g., Text Processing/python-benedict.>,
    "completion_path": <string, the path of the file where the code to be generated is located, e.g., Text Processing/python-benedict/benedict/utils/type_util.py.>,
    "signature_position": <list, the start and end line number of the signature within completion files e.g., [238, 238]. The line number starts from 1.>,
    "body_position": <list, the start and end line number of the reference code within completion files e.g., [239, 254]. The line number starts from 1.>,
    "dependency": <dict, the reference dependency. The keys include intra_class, intra_file, and cross_file. Each key stores a list of strings, which are namespaces of reference dependencies.>,
    "requirement": <dict, the requirement of the code to be generated. The keys include Functionality and Arguments. Functionality is a string describing the functionality of the code to be generated. Arguments is a string explaining the input and output parameters.>,
    "tests": <list, a list of test functions, which are used to evaluate the generated code.>,
    "indent": <int, the indent of the code to be generated.>, 
    
    ## Difference from DevEval
    "domain": <:string, the category of the project e.g., Communications.>,
    "gt": <Corresponding reference code in the original repository>,
    "context": <Context above the function to be completed>,
    "mt": [ <Multi-turn dataset>
        {
            "turn": <int,turn number>,
            "requirement": <string,requirement of this round>, 
            "gt": <string,Reference code for this round>,
            "test_code": <string,Test Code for this round>,
            "tests": <list,tests's selector>
        }, 
    ], 
    "mt_tests": <dict, all tests's selector, key is turn number, value is test selector list.>,
    "function_signature": <The entire function signature, including decorators>,
}
```

### Inference

Same as DevEval,before running the evaluation, researchers need to download the repositories, and dependency data.

The original repositories can be downloaded from [Link](https://zenodo.org/records/15580764). Users need to uncompressed the repositories and put them in the directory (e.g., mt_deveval/Source_Code).

Then, setup api key:
```bash
export OPENAI_API_KEY=<YOU API KEY>
export OPENAI_BASE_URL=<YOU CHOICE OF LLM PLATFORM>
```

Environment Setup:
```bash
conda create --name mt_deveval --file environment.txt
conda activate mt_deveval
pip install -r requirement.txt
# replace the path with your own path
echo "export NLTK_DATA=/home/user/mt_deveval/nltk_data" >> ~/.bashrc
source ~/.bashrc
```

Inference with llm:
```bash
python 03_infer.py --backend deepseek --llm deepseek-v3 --prompt_mode direct
```

Params:
- `dataset`: You also can specify dataset.
- `llm`: Specify your llm.
- `golden`: Whether to use the correct answer as historical information
- `backend`: Your Api backend
- `qwen3_thinking_mode`: Whether the Qwen3 model has activated the thinking mode
- `prompt_mode`: direct for `Full History`, edit for `Code Edit`, `append` for `Cumulative instruction`

### Evaluation

command:
```bash
python pass_k.py --infer_file your_infer_file
```

# 🙏 Acknowledgement

- [BigcodeBench](https://github.com/bigcode-project/bigcodebench)
- [DevEval](https://github.com/seketeam/DevEval?tab=readme-ov-file)