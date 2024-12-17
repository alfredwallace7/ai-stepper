#!/usr/bin/env python3
"""
AI Stepper Example 2: Retry Mechanism Demonstration
This script demonstrates the retry mechanism when dealing with complex schema validation.
"""

from ai_stepper import AI_Stepper
import os
from dotenv import load_dotenv
from rich import print
from datetime import datetime
from ai_stepper.schema.callback import CallBack
import sys

# Load environment variables
load_dotenv(override=True)

def custom_callback(callback_data: CallBack):
    """
    Custom callback function to track retries and validation errors.
    Prints detailed information about each attempt and any validation failures.
    """
    if callback_data.object == "input":
        if "Retrying" in callback_data.message:
            print(f"\n[yellow]===== RETRY ATTEMPT =====[/yellow]")
            print(f"[yellow]Previous error:[/yellow] {callback_data.message}")
        else:
            print(f"\n[blue]Input:[/blue] {callback_data.message}")
    elif callback_data.object == "step" and "failed" in callback_data.message:
        print(f"\n[red]Step Failed:[/red] {callback_data.message}")
        if hasattr(callback_data, 'code') and callback_data.code:
            print(f"[red]Error details:[/red] {callback_data.code.content}")
    elif callback_data.object == "output":
        print(f"\n[green]Success:[/green] {callback_data.message}")
        if hasattr(callback_data, 'code') and callback_data.code:
            print(f"[green]Output:[/green] {callback_data.code.content}")

def main():
    """Main execution function demonstrating retry mechanism."""
    try:
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

        # Sample product metrics that will require complex analysis
        sample_metrics = """
        {
            "revenue": 1500000,
            "users": 50000,
            "churn_rate": 0.15
        }
        """

        print("\n[bold blue]Running Complex Schema Retry Example...[/bold blue]")
        print("[bold]This example demonstrates the retry mechanism with a complex schema[/bold]")
        print("Watch for retry attempts if the LLM's output doesn't match the schema...\n")

        # Run the workflow with our custom callback to track retries
        result = stepper.run(
            steps_file="yaml/complex_schema_retry.yaml",
            initial_inputs={"metrics": sample_metrics},
            callback=custom_callback
        )

        # Display final result
        print("\n[bold green]Final Result:[/bold green]")
        print(result)

    except Exception as e:
        print(f"[bold red]Critical error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
