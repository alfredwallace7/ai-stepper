# Standard library imports
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, Type, Callable, List, Tuple

# Third-party imports
from jsonschema import validate, ValidationError
from jsonschema.exceptions import SchemaError
from litellm import completion
from pydantic import BaseModel, ValidationError
from rich import print
from yaml import safe_load

# Local imports
from .schema.callback import CallBack, CallbackFn, DEFAULT_CALLBACK, TokensItem, CodeItem
from .schema.output_validation_error import OutputValidationError
from .schema.step import Step
from .schema.workflow_execution_result import WorkflowExecutionResult
from .utils.retry import retry_with_backoff

class AI_Stepper:
    """
    A class that manages the execution of AI-driven workflow steps.
    
    This class handles loading workflow steps from YAML, executing them through an LLM,
    and managing the flow of data between steps.
    """
    
    def __init__(
        self,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None
    ):
        """Initialize the AI_Stepper."""
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.context = {}
        self._schema_cache = {}  # Initialize schema cache
        
        # Type mapping for schema validation
        self.type_map = {
            "string": str,
            "integer": int,
            "float": float,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "dict": dict,
            "any": object
        }

    def load_steps(self, steps_file: str, initial_inputs: Dict[str, Any]) -> Dict[str, Step]:
        """
        Loads steps from a YAML file and validates their schema structure.

        This method performs several key functions:
        1. Loads the workflow steps from a YAML file
        2. Converts type definitions to JSON Schema format
        3. Validates that all required inputs are available
        4. Ensures step outputs are properly defined

        Args:
            steps_file (str): Path to the YAML file containing step definitions
            initial_inputs (Dict[str, Any]): Initial input values for the workflow

        Returns:
            Dict[str, Step]: Dictionary mapping step names to Step objects

        Raises:
            ValueError: If required inputs are missing or step definitions are invalid
        """
        # JSON Schema type mapping for validation
        type_map = {
            "string": {"type": "string"},
            "integer": {"type": "integer"},
            "float": {"type": "number"},
            "number": {"type": "number"},
            "boolean": {"type": "boolean"},
            "array": {"type": "array", "items": {"type": "any"}},
            "object": {"type": "object"},
            "dict": {"type": "object"},
            "any": {}
        }

        def to_jsonschema(schema):
            """Convert internal schema format to JSON Schema format."""
            if isinstance(schema, str):
                return type_map.get(schema, {})
            
            if isinstance(schema, dict):
                if "type" in schema:
                    # Handle array schema with items
                    if schema["type"] == "array" and "items" in schema:
                        return {
                            "type": "array",
                            "items": to_jsonschema(schema["items"])
                        }
                    # Return schema as-is if already JSON Schema-like
                    return schema
                
                # Handle object properties
                if "properties" in schema:
                    return {
                        "type": "object",
                        "properties": {
                            key: to_jsonschema(value)
                            for key, value in schema["properties"].items()
                        }
                    }
            
            raise ValueError("Invalid schema format. JSON Schema is expected")

        try:
            # Load and parse YAML file
            with open(steps_file, "r", encoding="utf-8") as f:
                steps_dict = safe_load(f)

            # Track available inputs throughout the workflow
            available_inputs = set(initial_inputs.keys())

            # Process and validate each step
            for step_name, step_data in steps_dict.items():
                # Convert step outputs to JSON Schema format
                if "outputs" in step_data:
                    step_data["outputs"] = {
                        key: to_jsonschema(schema)
                        for key, schema in step_data["outputs"].items()
                    }

                # Validate input availability
                if "inputs" in step_data:
                    missing_inputs = [
                        input_key for input_key in step_data["inputs"].keys()
                        if input_key not in available_inputs
                    ]
                    if missing_inputs:
                        raise ValueError(
                            f"Step '{step_name}' requires missing inputs: {', '.join(missing_inputs)}. "
                            f"Ensure these inputs are provided as initial inputs or produced by previous steps."
                        )

                # Add step outputs to available inputs for subsequent steps
                if "outputs" in step_data:
                    available_inputs.update(step_data["outputs"].keys())

            return {name: Step(**data) for name, data in steps_dict.items()}
            
        except (ValidationError, Exception) as e:
            raise Exception(f"Failed to load steps: {e}")

    def format_task(self, task: str, context: Dict[str, Any]) -> str:
        """
        Format a task string by replacing placeholders with context values.

        Args:
            task (str): Task string with placeholders in {key} format
            context (Dict[str, Any]): Dictionary of values to insert into placeholders

        Returns:
            str: Formatted task string with placeholders replaced by values
        """
        try:
            return task.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing context value for key {e}")
        except Exception as e:
            raise ValueError(f"Error formatting task: {e}")

    def validate_output(self, output: Any, schema: Dict[str, Any]) -> Any:
        """
        Validate output against a JSON Schema specification.

        This method ensures that the output from the LLM matches the expected
        schema structure and data types. It uses the jsonschema library for
        validation, which provides detailed error messages when validation fails.

        Args:
            output (Any): The output data to validate
            schema (Dict[str, Any]): JSON Schema specification to validate against

        Returns:
            Any: The validated output (unchanged if validation passes)

        Raises:
            OutputValidationError: If the output fails schema validation or if the schema is invalid
        """
        try:
            validate(instance=output, schema=schema)
            return output
        except SchemaError as e:
            raise OutputValidationError(f"Invalid schema: {e}")
        except ValidationError as e:
            raise OutputValidationError(f"Validation error: {e}")

    def clean_json_response(self, response: str) -> str:
        """
        Clean and extract valid JSON from LLM response.
        Only used for non-GPT models that don't support JSON mode.

        Args:
            response (str): Raw response from the LLM

        Returns:
            str: Cleaned response containing only the relevant JSON data
        """
        # First try to extract JSON from code blocks
        code_block_match = re.search(
            r'```(?:json)?\n?([\s\S]*?)\n?```',
            response,
            flags=re.IGNORECASE
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code blocks, try to find JSON object or array pattern
        json_pattern = (
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
        )
        json_match = re.search(json_pattern, response)
        if json_match:
            return json_match.group(1).strip()

        # If no JSON patterns found, return the stripped response
        return response.strip()

    @retry_with_backoff(max_retries=3, allowed_exceptions=(OutputValidationError,))
    def _execute_llm_query(
        self,
        prompt: str,
        prompt_without_schema: str,
        expected_outputs: Dict[str, Dict],
        step_name: str,
        callback: CallbackFn,
        attempts_history: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Execute a single LLM query with validation.
        
        Args:
            prompt: The prompt to send
            expected_outputs: Expected output schemas
            step_name: Name of the current step
            callback: Callback function for progress updates
            attempts_history: List of previous attempts and errors
            
        Returns:
            Validated outputs
            
        Raises:
            OutputValidationError: If validation fails
        """
        try:
            response, content = self._query_llm_once(
                prompt=prompt,
                prompt_without_schema=prompt_without_schema,
                step_name=step_name,
                attempts_history=attempts_history,
                callback=callback,
                expected_outputs=expected_outputs
            )
            
            # With JSON mode, the response is already parsed JSON
            if "gpt" in self.llm_model_name.lower():
                parsed_output = json.loads(content)
            else:
                # Clean and parse JSON for non-GPT models
                cleaned_content = self.clean_json_response(content)
                try:
                    parsed_output = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Step '{step_name}' - Failed to parse JSON response:\n"
                        f"{str(e)}"
                    )
                    print(f"[bold red]{error_msg}[/bold red]")
                    attempts_history.append((error_msg, cleaned_content))
                    raise OutputValidationError(error_msg)

            # Handle single output case
            if len(expected_outputs) == 1:
                output_name = list(expected_outputs.keys())[0]
                if isinstance(parsed_output, dict):
                    if output_name in parsed_output:
                        parsed_output = {output_name: parsed_output[output_name]}
                    elif "type" in parsed_output and "value" in parsed_output:
                        parsed_output = {output_name: parsed_output["value"]}
                    else:
                        parsed_output = {output_name: parsed_output}
                else:
                    parsed_output = {output_name: parsed_output}

            try:
                # Validate outputs
                return self._validate_and_notify(
                    parsed_output=parsed_output,
                    expected_outputs=expected_outputs,
                    step_name=step_name,
                    callback=callback,
                    response=response,
                    cleaned_content=content,
                    attempts_history=attempts_history
                )
            except OutputValidationError as e:
                # Make sure to re-raise OutputValidationError for retry
                raise e
            except Exception as e:
                # Convert other exceptions to OutputValidationError for retry
                error_msg = f"Validation error in step '{step_name}': {str(e)}"
                attempts_history.append((error_msg, content))
                raise OutputValidationError(error_msg)

        except Exception as e:
            if not isinstance(e, OutputValidationError):
                # Convert any other exception to OutputValidationError for retry
                error_msg = f"Error in step '{step_name}': {str(e)}"
                attempts_history.append((error_msg, ""))
                raise OutputValidationError(error_msg)
            raise

    def _query_llm_once(
        self,
        prompt: str,
        prompt_without_schema: str,
        step_name: str,
        attempts_history: List[Tuple[str, str]] = None,
        callback: CallbackFn = DEFAULT_CALLBACK,
        expected_outputs: Dict[str, Dict] = None
    ) -> Tuple[Any, str]:
        """
        Make a single LLM query attempt.
        
        Args:
            prompt: The prompt to send
            step_name: Name of the current step
            attempts_history: List of previous attempts and errors
            callback: Callback function for progress updates
            expected_outputs: Expected output schemas
            
        Returns:
            Tuple of (LLM response, cleaned content)
            
        Raises:
            OutputValidationError: If JSON parsing fails
        """
        # Add error context for retries
        current_prompt = prompt
        if attempts_history:
            error_context = "\nYour previous attempts failed the validation check:\n"
            for i, (error, response) in enumerate(attempts_history, 1):
                error_context += f"\nAttempt {i}:\n"
                error_context += f"Error: {error.strip()}\n"
                error_context += f"Your previous generation: {response.strip()}\n"
            error_context += "\nPlease fix the output to match the expected schema exactly."
            current_prompt = f"{prompt}{error_context}"
            message = f"Retrying step '{step_name}' with error context:\n{error_context}"
        else:
            message = prompt_without_schema
        
        # Send pre-query callback with attempt information
        callback(CallBack(
            sender="LLM",
            object="input",
            message=message,
            created=int(datetime.now(timezone.utc).timestamp()),
            step_name=step_name,
            code=CodeItem(
                language="json",
                content=json.dumps(expected_outputs, indent=2)
            )
        ))
        
        # Query LLM with prompt and JSON mode enabled
        response = completion(
            model=self.llm_model_name,
            messages=[{"role": "user", "content": current_prompt}],
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            response_format={"type": "json_object"} if "gpt" in self.llm_model_name.lower() else None  # Only use JSON mode with GPT models
        )

        # Get the content
        content = response.choices[0].message.content
        
        # Send post-query callback with response information
        tokens = TokensItem(
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens
        )
        callback(CallBack(
            sender="LLM",
            object="output",
            message=f"Response for step '{step_name}'",
            created=int(datetime.now(timezone.utc).timestamp()),
            step_name=step_name,
            code=CodeItem(
                language="json",
                content=content
            ),
            tokens=tokens
        ))
        
        return response, content

    def _validate_and_notify(
        self,
        parsed_output: Dict[str, Any],
        expected_outputs: Dict[str, Dict],
        step_name: str,
        callback: CallbackFn,
        response: Any,
        cleaned_content: str,
        attempts_history: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Validate the parsed output against expected schemas and notify via callback.
        
        Args:
            parsed_output: The parsed JSON output to validate
            expected_outputs: Dictionary of expected output schemas
            step_name: Name of the current workflow step
            callback: Callback function for progress updates
            response: Raw LLM response object
            cleaned_content: Cleaned response content
            attempts_history: List of previous attempts and errors
            
        Returns:
            Validated outputs
            
        Raises:
            OutputValidationError: If validation fails
        """
        try:
            # Validate each output against its schema
            for output_name, output_schema in expected_outputs.items():
                output_value = parsed_output.get(output_name)
                try:
                    validate(instance=output_value, schema=output_schema)
                except ValidationError as e:
                    error_msg = (
                        f"Step '{step_name}' failed: {str(e)}\n\n"
                        f"Failed validating '{e.validator}' in schema{e.absolute_schema_path}:\n"
                        f"    {e.schema}\n\n"
                        f"On instance{e.absolute_path}:\n"
                        f"    {e.instance}"
                    )
                    print(f"[bold red]{error_msg}[/bold red]")
                    attempts_history.append((error_msg, cleaned_content))
                    raise OutputValidationError(error_msg)

            # If we get here, validation passed
            callback(CallBack(
                sender="AI_Stepper",
                step_name=step_name,
                object="output",
                message=f"Response for step '{step_name}'",
                code=CodeItem(
                    content=str(parsed_output),
                    language="json"
                ),
                created=int(time.time())
            ))
            return parsed_output

        except ValidationError as e:
            error_msg = str(e)
            print(f"[bold red]{error_msg}[/bold red]")
            attempts_history.append((error_msg, cleaned_content))
            raise OutputValidationError(error_msg)

    def query_llm(
        self,
        prompt: str,
        expected_outputs: Dict[str, Dict],
        step_name: str,
        callback: CallbackFn = DEFAULT_CALLBACK,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Query the LLM with enhanced context and error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            expected_outputs: Dictionary of expected output names and their schemas
            step_name: Name of the current workflow step for context
            callback: Callback function for progress updates
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing the validated outputs
            
        Raises:
            OutputValidationError: If output validation fails after all retries
            Exception: For other unexpected errors
        """
        attempts_history: List[Tuple[str, str]] = []
        
        # Add schema to prompt
        prompt_with_schema = self.add_prompt_schema(prompt, expected_outputs)
        
        # Execute query with provided callback
        return self._execute_llm_query(
            prompt=prompt_with_schema,
            prompt_without_schema=prompt,
            expected_outputs=expected_outputs,
            step_name=step_name,
            callback=callback,  # Pass the callback directly
            attempts_history=attempts_history
        )

    def log_error(self, prompt: str, response: str, error: Exception):
        """
        Log error details and debugging information.

        This method provides detailed logging of errors that occur during LLM
        interaction, including the original prompt, raw response, and error details.
        It also suggests potential improvements to the prompt structure.

        Args:
            prompt (str): The original prompt sent to the LLM
            response (str): The raw response received from the LLM
            error (Exception): The error that occurred
        """
        print("[bold red]Failed to parse response.[/bold red]")
        print("[blue]PROMPT:[/blue]", prompt)
        print("[blue]RAW RESPONSE:[/blue]", response)
        print("[bold red]ERROR DETAILS:[/bold red]", error)
        print("[blue]SUGGESTION:[/blue] Review the YAML prompt. Ensure it guides the LLM to produce strictly JSON-formatted output.")

    def add_prompt_schema(self, prompt: str, expected_outputs: dict) -> str:
        """Add schema information to the prompt."""
        # Use cached schema if available
        schema_key = json.dumps(expected_outputs, sort_keys=True)
        if schema_key in self._schema_cache:
            schema_str = self._schema_cache[schema_key]
        else:
            # Get the first output key and its schema
            output_key = next(iter(expected_outputs.keys()))
            schema = expected_outputs[output_key]
            
            # Format the schema structure
            formatted_output = {
                output_key: schema
            }
            schema_str = json.dumps(formatted_output, indent=2)
            self._schema_cache[schema_key] = schema_str

        output_key = next(iter(expected_outputs.keys()))
        schema = expected_outputs[output_key]
        schema_type = schema.get('type', '')
        
        # Build schema path description
        path_desc = f"'{output_key}'"
        if schema_type == 'object' and 'properties' in schema:
            for prop, prop_schema in schema['properties'].items():
                if prop_schema.get('type') == 'array':
                    path_desc += f" → '{prop}' (array)"

        return (
            f"{prompt}\n\n"
            f"IMPORTANT: Your response must be a JSONSchema-compliant object matching this schema EXACTLY:\n"
            f"{schema_str}\n\n"
            f"Response Requirements:\n"
            f"1. Must be a valid JSON object with ONE top-level key: '{output_key}'\n"
            f"2. Must strictly follow the JSONSchema type constraints (string, integer, array, etc.)\n"
            f"3. For complex responses, follow this path: {path_desc}\n"
            f"4. Include ONLY the JSON object, no additional text\n"
        )

    def run(
            self, 
            steps_file: str, 
            initial_inputs: Dict[str, Any], 
            callback: CallbackFn,
        ) -> WorkflowExecutionResult:
        """
        Execute the workflow defined in the steps file.

        This method is the main entry point for running an AI workflow. It:
        1. Loads and validates the workflow steps
        2. Executes each step in sequence
        3. Manages data flow between steps
        4. Handles errors and provides execution logging
        5. Tracks execution time and token usage

        Args:
            steps_file (str): Path to the YAML file containing step definitions
            initial_inputs (Dict[str, Any]): Initial input values for the workflow
            callback (Optional[Callable]): Optional callback function for progress updates

        Returns:
            WorkflowExecutionResult: Complete execution log containing:
                - initial_inputs: The input values provided to start the workflow
                - steps: List of step execution results, each containing:
                    - input: Input values for the step
                    - output: Output values from the step
                    - output_schema: JSON schema used for validation
                    - errors: List of errors encountered
                    - execution_time: Time taken to execute the step
                    - tokens: Token usage statistics
                - final_result: Final output of the workflow
                - error_count: Total number of errors encountered
                - total_execution_time: Total time taken to execute workflow
                - total_tokens: Aggregated token usage across all steps
        """
        # Initialize context with input values
        workflow_start_time = time.time()
        self.context.update(initial_inputs)
        
        # Initialize execution log
        execution_log = {
            "initial_inputs": initial_inputs,
            "steps": [],
            "final_result": {},
            "error_count": 0,
            "total_execution_time": 0,
            "total_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        # Load and validate workflow steps
        steps = self.load_steps(steps_file, initial_inputs)
        print(f"\n[bold blue]Running workflow with {len(steps)} steps...[/bold blue]")

        # Execute each step in sequence
        for step_name, step in steps.items():
            step_start_time = time.time()
            step_log = {
                step_name: {
                    "input": {},
                    "output": None,
                    "output_schema": step.outputs,
                    "errors": [],
                    "execution_time": 0,
                    "tokens": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            }

            try:
                # Validate required inputs
                for input_key in step.inputs.keys():
                    if input_key not in self.context or self.context[input_key] is None:
                        raise ValueError(
                            f"Missing or invalid input '{input_key}' for step '{step_name}'."
                        )
                    step_log[step_name]["input"][input_key] = self.context[input_key]

                # Format and execute the step
                prompt = self.format_task(step.task, self.context)
                print(f"\n[bold blue]Executing step with task:[/bold blue] [black]{prompt}[/black]")

                result = self.query_llm(
                    prompt=prompt,
                    expected_outputs=step.outputs,
                    step_name=step_name,
                    callback=callback,  # Pass the callback directly
                    max_retries=step.max_retries,
                )

                # Update context with step outputs
                for output_name, output_schema in step.outputs.items():
                    if len(step.outputs) == 1:
                        self.context[output_name] = result[output_name]
                    else:
                        self.context[output_name] = result.get(output_name)
                
                step_log[step_name]["output"] = result
                print(f"\n[bold green]Step '{step_name}' completed successfully.[/bold green]")

            except Exception as e:
                # Log error and set outputs to None
                error_detail = {
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'context': {
                        'prompt': step.task,
                        'expected_schema': step.outputs
                    }
                }
                step_log[step_name]['errors'].append(error_detail)
                print(f"\n[bold red]Step '{step_name}' failed: {str(e)}[/bold red]")

                # error callback
                callback(CallBack(
                    sender="AI_Stepper",
                    object="step",
                    message=f"<span style='color: red;'>Step '{step_name}' failed!</span>",
                    step_name=step_name,
                    code=CodeItem(
                        content=error_detail,
                        language="json"
                    ),
                    created=int(time.time())
                ))

                # Use default value None for all defined outputs
                for output_name in step.outputs.keys():
                    self.context[output_name] = None
                step_log[step_name]["output"] = None

            # Record execution time and token usage for the step
            step_end_time = time.time()
            step_log[step_name]["execution_time"] = round(step_end_time - step_start_time, 2)
            
            # Update token counts if available in callback data
            if hasattr(callback, 'last_tokens'):
                tokens = callback.last_tokens
                step_log[step_name]["tokens"] = {
                    "prompt_tokens": tokens.prompt_tokens,
                    "completion_tokens": tokens.completion_tokens,
                    "total_tokens": tokens.total_tokens
                }
                # Update total token counts
                execution_log["total_tokens"]["prompt_tokens"] += tokens.prompt_tokens
                execution_log["total_tokens"]["completion_tokens"] += tokens.completion_tokens
                execution_log["total_tokens"]["total_tokens"] += tokens.total_tokens

            execution_log["steps"].append(step_log)

        # Update final result with last step's output
        if execution_log["steps"]:
            last_step = execution_log["steps"][-1]
            last_step_name = list(last_step.keys())[0]
            last_output = last_step[last_step_name]["output"]
            if last_output:
                execution_log["final_result"] = last_output

        # Calculate total error count
        execution_log["error_count"] = sum(
            1 for step in execution_log["steps"]
            if step[list(step.keys())[0]]["errors"]
        )

        # Record total workflow execution time
        workflow_end_time = time.time()
        execution_log["total_execution_time"] = round(workflow_end_time - workflow_start_time, 2)

        # callback for workflow completion
        callback(CallBack(
            sender="AI_Stepper",
            step_name="workflow",
            object="workflow",
            message=f"<span style='color: green;'>Workflow completed successfully!</span>",
            created=int(time.time())
        ))

        return WorkflowExecutionResult(**execution_log)
