TEST_GEN_SYSTEM_PROMPT = "You are the world's most powerful AI programming assistant, able to write solutions and corresponding test cases based on user needs."

TEST_GEN_PROMPT = """You will assist me in completing the code implementation and test case construction of a code requirement. This code requirement is decomposed from the a complex requirement. Your responsibility is to ensure that the generated code is tightly focused on the current requirements without adding many additional optimizations.

I will provide the following information to you:

- **Requirements of this round**: the specific requirement that need to be implemented in this round.
- **Current round**: indicate which round this is. If it is round 1, directly write the solution code and test cases according to the requirements. Otherwise, I will provide the code you generated in the previous round, and please continue to improve it on this basis to meet the new requirements of this round, and add test cases accordingly.
    
Please make sure:
1. The generated code is tightly focused on the current requirements without adding many additional optimizations.
2. Your test program must use the `unittest` library and wrapped by `class TestCases(unittest.TestCase):`.
3. Your test program must be closely focused on the current requirements and should not introduce any extraneous matters.
4. Previous round's code should fail this round of testing!

--- Current round and requirements ---

Current round: {current_round}

Current requirement: {current_requirement}

--- Solution In previous round (If exists) ---
{last_solution}

"""

TEST_GEN_FEEDBACK_PROMPT = """The code and test program are concatenated into a single python file and run by me.

Based on the execution, the solution code or test code you provided seems to have problems. Please rethink the code and test code seriously based on the feedback and refine them to meet the requirements. 

Notice: 
1. Never forget that your test program must use the `unittest` library and wrapped by `class TestCases(unittest.TestCase):`
2. Your solution code should be writen in function `task_func`.
3. Test program should not include `def task_func`, I will concat it by myself!!!

--- feedback from human programer and runtime environment---
{feedback}

"""

ASSISTANT_PROMPT_REPLACE = """
```json
[
    {
        "solution": "{solution}",
        "test": "{test}"
    }
]
```
"""

TEST_GEN_FEEDBACK_FORMAT_PROMPT = """You must only output in the following json format wrapped with ```json ```:
```json
[
    {
        "reason": Why does this happen? Is there a problem with the test or the answer code? Analyze it in detail.
        "plan": Your plan to handle this problem.
        "solution": The solution code you refined.
        "test": The test program you refined, do not include `def task_func`, I will concat it by myself!!!
    }
]
```
"""

TEST_GEN_FORMAT_PROMPT = """You must output in the following json format wrapped with ```json ```:
```json
[
    {
        "solution": The solution code you provided for this round of requirements.
        "test": The test program you wrote to test this round of solution, , do not include `def task_func`, I will concat it by myself!!!
    }
]
```
"""
