# state.py
from typing import TypedDict, Optional

class AgentState(TypedDict):
    image_path: str
    prompt: Optional[str]

    optimized_prompt: Optional[str]
    negative_prompt: Optional[str]

    mask_path: Optional[str]
    result_path: Optional[str]