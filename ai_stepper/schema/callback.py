from typing import Optional, Callable
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone

class TokensItem(BaseModel):
    """Token usage information for a completion."""
    model_config = ConfigDict(extra='forbid')
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CodeItem(BaseModel):
    """Code item in a callback."""
    model_config = ConfigDict(extra='forbid')
    content: str
    language: str = "python"

class CallBack(BaseModel):
    """Callback data structure."""
    model_config = ConfigDict(extra='forbid')
    sender: str
    step_name: Optional[str] = None
    object: str
    message: str
    code: Optional[CodeItem] = None
    tokens: Optional[TokensItem] = None
    created: int = int(datetime.now(timezone.utc).timestamp())

# Define callback function type
CallbackFn = Callable[[CallBack], None]

def DEFAULT_CALLBACK(callback_data: CallBack) -> None:
    """Default callback function that does nothing."""
    pass