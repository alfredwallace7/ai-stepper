# AI Stepper

A lightweight, flexible Python framework for creating step-by-step AI workflows with full LLM prompt control.

## Overview

AI Stepper is a basic sequential AI agent system designed for developers who need:
- Full control over LLM prompts and interactions
- Step-by-step workflow execution
- Strong output validation
- Simple retry mechanisms
- Clear and predictable agent behavior

If you're looking for a straightforward way to create sequential AI workflows without the complexity of full-scale agent frameworks, AI Stepper is the right choice.

## Key Features

- **LLM Agnostic**: Works with any LLM through litellm annotations (OpenAI, Anthropic, local models, etc.)
- **Wide Compatibility**: Compatible with LLMs that don't support function calling.
- **Full Prompt Control**: Define exactly how your LLM should behave at each step
- **Sequential Execution**: Each step's output is automatically available as input for subsequent steps
- **YAML-Driven Workflows**: Define your entire workflow in a simple YAML file
- **Schema Compatibility**: Supports both JSON Schema and YAML annotations for input/output validation
- **Strong Validation**: Validate LLM outputs against predefined schemas
- **Rich Logging**: Built-in Markdown-formatted logging with step context
- **Flexible Callbacks**: Custom callback system for monitoring and debugging

## Installation

```bash
pip install -U ai-stepper
```

## Usage

### 1. Environment Setup
Create a `.env` file with your LLM configuration:

```env
OPENAI_API_BASE=your_llm_api_base
OPENAI_API_KEY=your_llm_api_key
OPENAI_MODEL_NAME=your_model_name
```

### 2. Create a Workflow
Define your workflow steps in a YAML file (e.g., `chain_of_thoughts.yaml`):

```yaml
direct_answer:
  task: >
    {query}
  inputs:
    query:
      type: string
  outputs:
    final_answer:
      type: string

define_problem:
  task: >
    Break down the problem statement {query} into its core components. Identify what needs to be solved or answered and
    list these subproblems explicitly.
  inputs:
    query:
      type: string
  outputs:
    subproblems:
      type: array
      items:
        type: string

generate_subsolutions:
  task: >
    Solve each subproblem {subproblems} step by step. For each subproblem, provide a concise solution or analysis.
    If the subproblem cannot be solved directly, explain why.
  inputs:
    subproblems:
      type: array
      items:
        type: string
  outputs:
    subsolutions:
      type: array
      items:
        type: object
        properties:
          subproblem:
            type: string
          solution:
            type: string

combine_results:
  task: >
    Combine the solutions to the subproblems {subsolutions} into a coherent final answer to the original query.
    Ensure the reasoning flows logically, and the answer directly addresses the query.
  inputs:
    subsolutions:
      type: array
      items:
        type: object
        properties:
          subproblem:
            type: string
          solution:
            type: string
  outputs:
    final_answer:
      type: string
```

### 3. Implement the Runner
Create a Python script to run your workflow:

```python
from ai_stepper import AI_Stepper
import os
from dotenv import load_dotenv
from rich import print
from typing import Optional
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

def agent_logger(message: str, step_name: Optional[str] = None):
    """Log agent actions and responses."""
    if step_name:
        print(f"\n### {step_name.upper()}\n{message}\n")
    else:
        print(f"\n{message}\n")

def run_workflow(stepper: AI_Stepper, name: str, yaml_file: str, inputs: dict) -> None:
    """Run a single workflow and handle any errors."""
    try:
        print(f"\n[bold blue]Running {name} workflow...[/bold blue]")
        result = stepper.run(
            steps_file=yaml_file,
            initial_inputs=inputs,
            callback=agent_logger
        )
        print(f"[green]Result:[/green]", result)
    except Exception as e:
        print(f"[bold red]Error in {name} workflow:[/bold red] {str(e)}")

def main():
    """Main execution function."""
    try:
        # Initialize the AI_Stepper
        stepper = AI_Stepper(
            llm_base_url=os.getenv("OPENAI_API_BASE"),
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model_name=os.getenv("OPENAI_MODEL_NAME")
        )

        # Define and run workflows
        workflows = [
            ("Chain of Thoughts", "yaml/chain_of_thoughts.yaml", {
                "query": "A farmer has 20 apples. He gives 5 apples to his neighbor and then splits the remaining apples equally among his 3 children. How many apples does each child get?"
            }),
        ]
    
        for name, yaml_file, inputs in workflows:
            run_workflow(stepper, name, yaml_file, inputs)

    except Exception as e:
        print(f"[bold red]Critical error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
```

This example demonstrates a workflow that:
1. Breaks down the question into subproblems
2. Solves each subproblem step by step
3. Combines the solutions into a final answer

The framework handles:
- Environment configuration
- YAML workflow loading
- Step execution
- Error handling
- Logging
- Output validation

For more examples, check the `yaml/` directory in the repository.

## Logging and Output

### Markdown Logger

AI Stepper includes a built-in markdown logger utility that creates well-formatted logs of your workflow execution. The logger supports:

- Timestamped entries with timezone support
- Structured message formatting
- Code block formatting with syntax highlighting
- Token usage tracking
- JSON pretty-printing
- File output capabilities

Example usage:

```python
from ai_stepper.utils.logger import markdown_logger
from ai_stepper.schema.callback import CallBack

# Create a callback
callback = CallBack(
    sender="LLM",
    step_name="analyze_sentiment",
    object="output",
    message="Response for sentiment analysis",
    created=int(datetime.utcnow().timestamp()),
    code=CodeItem(
        language="json",
        content={"sentiment_analysis": {"product_sentiments": [...]}}
    )
)

# Log to file
markdown_logger(callback, "log.md")
```

### Schema Validation

AI Stepper enforces strict schema validation for LLM outputs. Each step's output must conform to the schema defined in your YAML workflow:

```yaml
analyze_sentiment:
  task: >
    Perform sentiment analysis on the feedback {feedback}.
    Determine whether each theme has a positive, negative, or neutral sentiment.
  inputs:
    feedback:
      type: array
  outputs:
    sentiment_analysis:
      type: object
      properties:
        product_sentiments:
          type: array
          items:
            type: object
            properties:
              product:
                type: string
              theme_sentiments:
                type: array
                items:
                  type: object
                  properties:
                    theme:
                      type: string
                    sentiment:
                      type: string
```

The framework will:
1. Validate all outputs against their schemas
2. Provide clear error messages for validation failures
3. Support retry mechanisms for failed validations
4. Log validation results in markdown format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
