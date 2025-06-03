from typing import TypedDict
from data import BigCodeBenchInstance

class PromptEntry(TypedDict):
    prompt: str
    bigcode_instance: BigCodeBenchInstance
    system_prompt: str

