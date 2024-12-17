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
from dotenv import load_dotenv
from jsonschema.exceptions import ValidationError

load_dotenv(override=True)

@pytest.fixture
def basic_workflow_yaml():
    return Path("yaml/basic_workflow.yaml").absolute()

@pytest.fixture
def stepper():
    return AI_Stepper(
        llm_base_url=os.getenv("OPENAI_API_BASE"),
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model_name=os.getenv("OPENAI_MODEL_NAME")
    )

def test_initialization():
    """Test AI_Stepper initialization with different configurations"""
    # Test with all parameters
    stepper1 = AI_Stepper(
        llm_base_url="http://test.com",
        llm_api_key="test-key",
        llm_model_name="test-model"
    )
    assert stepper1.llm_base_url == "http://test.com"
    assert stepper1.llm_api_key == "test-key"
    assert stepper1.llm_model_name == "test-model"
    
    # Test with no parameters
    stepper2 = AI_Stepper()
    assert stepper2.llm_base_url is None
    assert stepper2.llm_api_key is None
    assert stepper2.llm_model_name is None

def test_schema_validation_errors(stepper):
    """Test various schema validation scenarios"""
    # Test invalid type
    schema = {"type": "integer"}
    with pytest.raises(ValidationError):
        stepper.validate_output("not an integer", schema)
    
    # Test invalid array items
    schema = {"type": "array", "items": {"type": "integer"}}
    with pytest.raises(ValidationError):
        stepper.validate_output(["1", "2", "3"], schema)
    
    # Test required properties
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    with pytest.raises(ValidationError):
        stepper.validate_output({"name": "John"}, schema)

def test_context_management(stepper):
    """Test context management functionality"""
    # Test context initialization
    assert stepper.context == {}
    
    # Test context update
    test_context = {"key": "value"}
    stepper.context = test_context
    assert stepper.context == test_context
    
    # Test context clearing
    stepper.context = {}
    assert stepper.context == {}

def test_llm_response_parsing(stepper):
    """Test LLM response parsing with different formats"""
    # Test parsing JSON with comments
    response = '''// This is a comment
    {
        "result": "test"
    }'''
    parsed = json.loads(stepper.clean_json_response(response))
    assert parsed == {"result": "test"}
    
    # Test parsing JSON with surrounding text
    response = '''Here's the output:
    {
        "result": "test"
    }
    End of output'''
    parsed = json.loads(stepper.clean_json_response(response))
    assert parsed == {"result": "test"}

def test_error_handling_with_real_llm(stepper):
    """Test error handling with actual LLM connection"""
    # Test with invalid model name
    with pytest.raises(Exception):
        invalid_stepper = AI_Stepper(
            llm_base_url=os.getenv("OPENAI_API_BASE"),
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model_name="invalid-model"
        )
        mock_response = type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': 'Invalid response'
                })
            })]
        })
        with patch('ai_stepper.ai_stepper.completion', return_value=mock_response):
            invalid_stepper.run(
                basic_workflow_yaml,
                {"query": "test query"}
            )

def test_type_validation(stepper):
    """Test basic type validation without conversion"""
    # Test number validation
    schema = {"type": "number"}
    assert stepper.validate_output(42, schema) == 42
    assert stepper.validate_output(42.0, schema) == 42.0
    
    # Test boolean validation
    schema = {"type": "boolean"}
    assert stepper.validate_output(True, schema) is True
    assert stepper.validate_output(False, schema) is False
    
    # Test array validation
    schema = {"type": "array", "items": {"type": "string"}}
    assert stepper.validate_output(["1", "2", "3"], schema) == ["1", "2", "3"]
