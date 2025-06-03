DECOMPOSE_SYSTEM_MESSAGE = "You are an AI code assistant that is committed to breaking down the user's complex instruction into one basic instruction and multiple restrictive instructions."

DECOMPOSE_INSTRUCTION_PROMPT = """Now I will give you an original complex user instruction with function signature and the corresponding code answer. You need to break it down into a basic instruction and multiple restrictive instructions.

This process simulates the process of users gradually clarifying their originally vague needs when using AI.

Make sure your decomposition follow the rules below:
1. The number of decomposition rounds is scientifically determined based on the complexity of the task and must be between 2-5 rounds.
2. New instructions must clear enough to be verified by test cases.
3. Ensure data diversity as much as possible. Reduce the situation where instructions from later rounds are implemented by earlier rounds.

--- original complex instruction ---
{original_requirement}

--- corresponding code answer ---
{original_code}
"""

DECOMPOSE_OUTPUT_FORMAT = """You must output in the following json format wrapped with ```json ```:
```json
[
    {
        "turn": 1,
        "instruct_prompt": Basic instruction, followed by 'You should write self-contained code starting with:```<code>```',the '<code>' can refer to the original instructions but keep only the code needed for the first round. Do not change function signature `task_func`.
    },
    {
        "turn": k,
        "instruct_prompt": Restrictive instruction in round k, do not add any prefix and suffix
    },
    ...
]
```
"""