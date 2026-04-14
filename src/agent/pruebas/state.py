# state.py
from typing import TypedDict, Optional

class AgentState(TypedDict):
    image_path: str
    mask_path: Optional[str]
    prompt: Optional[str]
    optimized_prompt: Optional[str]
    result_path: Optional[str]