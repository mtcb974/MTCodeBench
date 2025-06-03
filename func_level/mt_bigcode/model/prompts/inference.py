INSTRUCTION_PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"

# Edit mode

EDIT_PROMPT = """I will provide you with a code snippet and an edit instruction. Your task is to edit the code to suit the needs.
Previous Code:
```python
{previous_code}
```
Please provide a self-contained Python script that solve the following problem in a markdown code block:
{instruction}
"""