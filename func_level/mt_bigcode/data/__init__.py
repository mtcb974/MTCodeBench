from typing import TypedDict,List

class BigCodeBenchInstance(TypedDict):
    task_id: str
    complete_prompt: str
    instruct_prompt: str
    canonical_solution: str
    code_prompt: str
    test: str
    entry_point: str
    doc_struct: str
    libs: str

class SplitBigCodeInstance(TypedDict):
    turn: str
    task_id: str
    instruct_prompt: str
    code: str
    test: str
    entry_point: str

class MTBigCodeInstance(TypedDict):
    mt_id :int
    task_id: str
    mt_data: List[SplitBigCodeInstance]


class TurnResult(TypedDict):
    task_id: str
    turn: int
    instruct_prompt: str
    solution: str
    raw_solution: str

class MTResult(TypedDict):
    mt_id: int
    task_id: str
    solutions: List[TurnResult]
