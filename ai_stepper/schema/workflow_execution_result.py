from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List

class StepExecutionResult(BaseModel):
    """
    Represents the execution result of a single workflow step.

    Attributes:
        input: Input values provided to the step
        output: Output values produced by the step
        output_schema: JSON schema used for validating the output
        errors: List of errors encountered during execution
        execution_time: Time taken to execute the step in seconds
        tokens: Token usage statistics (prompt, completion, total)
    """
    model_config = ConfigDict(extra='forbid')
    input: Dict[str, Any]
    output: Dict[str, Any]
    output_schema: Dict[str, Any]
    errors: List[str]
    execution_time: float
    tokens: Dict[str, int]

class WorkflowExecutionResult(BaseModel):
    """
    Represents the complete execution result of a workflow.

    Attributes:
        initial_inputs: Input values provided to start the workflow
        steps: List of step execution results
        final_result: Final output of the workflow
        error_count: Total number of errors encountered
        total_execution_time: Total time taken to execute workflow in seconds
        total_tokens: Aggregated token usage across all steps
    """
    model_config = ConfigDict(extra='forbid')
    initial_inputs: Dict[str, Any]
    steps: List[Dict[str, StepExecutionResult]]
    final_result: Dict[str, Any]
    error_count: int
    total_execution_time: float
    total_tokens: Dict[str, int]
