from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, Union

class InputDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str
    items: Optional[Dict[str, Any]] = None

class Step(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: str
    inputs: Dict[str, Union[str, InputDefinition]]
    outputs: Dict[str, Any]
    max_retries: Optional[int] = 3