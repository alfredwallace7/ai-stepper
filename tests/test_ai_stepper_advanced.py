import pytest
from unittest.mock import Mock, patch
import yaml
import json
from pathlib import Path
import os
from datetime import datetime
from ai_stepper import AI_Stepper
from ai_stepper.schema.output_validation_error import OutputValidationError
from ai_stepper.schema.callback import CallBack
from ai_stepper.schema.workflow_execution_result import WorkflowExecutionResult
from ai_stepper.utils.retry import retry_with_backoff
from functools import wraps
from dotenv import load_dotenv

load_dotenv(override=True)

@pytest.fixture
def chain_of_thoughts_yaml():
    return Path("yaml/chain_of_thoughts.yaml").absolute()

@pytest.fixture
def stepper():
    return AI_Stepper(
        llm_base_url=os.getenv("OPENAI_API_BASE"),
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model_name="gpt-4"  # Set a default model name for testing
    )

# Test clean_json_response with various formats
def test_clean_json_response_code_block(stepper):
    response = '''Here's the JSON response:
```json
{
    "key": "value"
}
```
'''
    result = stepper.clean_json_response(response)
    assert result == '{\n    "key": "value"\n}'

def test_clean_json_response_raw(stepper):
    response = '{"key": "value"}'
    result = stepper.clean_json_response(response)
    assert result == '{"key": "value"}'

def test_clean_json_response_nested(stepper):
    response = 'Some text {"outer": {"inner": "value"}} more text'
    result = stepper.clean_json_response(response)
    assert result == '{"outer": {"inner": "value"}}'

# Test complex schema validation
def test_complex_schema_validation(stepper):
    schema = {
        "type": "array",
        "items": {
            "type": "string"
        }
    }
    
    value = ["subproblem1", "subproblem2", "subproblem3"]
    result = stepper.validate_output(value, schema)
    assert result == value

# Test callback functionality with different message types
def test_callback_messages(stepper, chain_of_thoughts_yaml):
    messages = []
    
    def test_callback(callback_data: CallBack):
        nonlocal messages
        messages.append(callback_data)
    
    # Mock the completion function to return a proper response
    def mock_completion(*args, **kwargs):
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': json.dumps({
                        "final_answer": "This is a test answer."
                    }),
                    'role': 'assistant'
                })
            })],
            'usage': type('Usage', (), {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            })
        })
    
    with patch('ai_stepper.ai_stepper.completion', side_effect=mock_completion):
        stepper._query_llm_once(
            prompt="Test prompt",
            prompt_without_schema="Test prompt",
            step_name="test_step",
            callback=test_callback,
            expected_outputs={"final_answer": {"type": "string"}}
        )
    
    # Verify different types of callback messages
    assert any(msg.object == "input" for msg in messages), "No input messages found"
    assert any(msg.object == "output" for msg in messages), "No output messages found"
    assert any(msg.tokens is not None for msg in messages), "No token usage info found"

# Test error handling for invalid schema
def test_invalid_schema_handling(stepper):
    invalid_schema = {
        "type": "invalid_type",
        "properties": {}
    }
    
    with pytest.raises(OutputValidationError) as exc_info:
        stepper.validate_output({"test": "value"}, invalid_schema)
    assert "Invalid schema" in str(exc_info.value)

# Test format_task with missing context
def test_format_task_missing_context(stepper):
    task = "Process {count} items from {source}"
    context = {"count": 5}  # Missing 'source'
    
    with pytest.raises(ValueError) as exc_info:
        stepper.format_task(task, context)
    assert "Missing context value" in str(exc_info.value)

# Test LLM retry mechanism with multiple failures
def test_llm_retry_mechanism(stepper, chain_of_thoughts_yaml):
    fail_count = 0
    error_count = 0

    def test_callback(callback_data: CallBack):
        nonlocal error_count
        if (callback_data.object == "input" and
            "Error: Error in step 'direct_answer': Temporary failure" in callback_data.message):
            error_count += 1

    def mock_completion(*args, **kwargs):
        nonlocal fail_count
        if fail_count < 2:
            fail_count += 1
            raise Exception("Temporary failure")

        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': json.dumps({
                        "final_answer": "This is a test answer."
                    })
                })
            })],
            'usage': type('Usage', (), {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            })
        })

    with patch('ai_stepper.ai_stepper.completion', side_effect=mock_completion):
        try:
            stepper.run(
                chain_of_thoughts_yaml,
                {
                    "query": "How to optimize performance?"
                },
                test_callback
            )
        except Exception as e:
            # We expect this to fail due to missing inputs for subsequent steps
            pass

    # We should get one error callback for each retry attempt
    assert error_count == 2, f"Expected 2 error callbacks, got {error_count}"
