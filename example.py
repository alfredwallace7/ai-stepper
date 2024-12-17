#!/usr/bin/env python3
"""
AI Stepper Example Runner
This script demonstrates various workflow complexities using the AI Stepper framework.
"""

from ai_stepper import AI_Stepper
import os
from dotenv import load_dotenv
from rich import print
from typing import Optional
from datetime import datetime
import sys
from ai_stepper.schema.callback import CallBack
from ai_stepper.utils.logger import markdown_logger

# Load environment variables from .env file
load_dotenv(override=True)

# Configuration
LOG_FILE = "log.md"
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS

def setup_logging():
    """Initialize logging setup and clear previous log file."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    print(f"\n[bold green]Starting new test run at {RUN_TIMESTAMP}[/bold green]\n")

def agent_logger(callBack: CallBack, step_name: Optional[str] = None):
    """
    Callback function to log messages from the agent.
    
    Args:
        callBack (CallBack): Callback object containing message details
        step_name (Optional[str]): Optional step name for context
    """
    # Ensure step_name is consistent
    if not callBack.step_name and step_name:
        callBack.step_name = step_name

    # Generate markdown log
    markdown_logger(callBack, log_file=LOG_FILE)

def run_workflow(stepper: AI_Stepper, name: str, yaml_file: str, inputs: dict) -> None:
    """
    Run a single workflow and handle any errors.
    
    Args:
        stepper (AI_Stepper): The AI Stepper instance
        name (str): Name of the workflow
        yaml_file (str): Path to the YAML workflow file
        inputs (dict): Initial inputs for the workflow
    """
    try:
        print(f"\n[bold blue]Running '{name}' workflow...[/bold blue]")
        result = stepper.run(
            steps_file=yaml_file,
            initial_inputs=inputs,
            callback=agent_logger
        )
        print(f"[green]Result:[/green]", result)
    except Exception as e:
        print(f"[bold red]Error in '{name}' workflow:[/bold red] {str(e)}")

def main():
    """Main execution function."""
    try:
        # Initialize logging
        setup_logging()
        
        # Validate environment variables
        required_env = ["OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_MODEL_NAME"]
        missing_env = [env for env in required_env if not os.getenv(env)]
        if missing_env:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_env)}")

        # Initialize the AI_Stepper
        stepper = AI_Stepper(
            llm_base_url=os.getenv("OPENAI_API_BASE"),
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model_name=os.getenv("OPENAI_MODEL_NAME")
        )

        # Run test workflows
        workflows = [
            ("chain of thoughts", "yaml/chain_of_thoughts.yaml", {"query": "I need to plan a trip from Boston to Rockport. The train ticket costs $50, and a bus ticket costs $30 but takes 2 hours longer. If your budget is $100 and you need to arrive as soon as possible, what option should I choose? How much money would I have left?"}),
        ]
    
        for name, yaml_file, inputs in workflows:
            run_workflow(stepper, name, yaml_file, inputs)

        print(f"Check markdown log: {LOG_FILE}")

    except Exception as e:
        print(f"[bold red]Critical error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()