from typing import Optional
import os
import pytz
import json
from ai_stepper.schema.callback import CallBack
from datetime import datetime, timezone

def markdown_logger(callback: CallBack, log_file: str = None) -> str:
    """
    Generate a Markdown-friendly log message from a CallBack object and optionally save to a file.

    Args:
        callback (CallBack): The callback object to log.
        log_file (str, optional): File path to save the log. Raises an error if the file cannot be created.

    Returns:
        str: A Markdown-formatted log string.
    """
    # Convert timestamp to specified timezone
    local_tz = pytz.timezone(os.getenv('TIMEZONE', 'UTC'))
    created_at = datetime.fromtimestamp(callback.created, tz=timezone.utc).astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')

    # Initialize log string
    md_log = ""

    # Workflow Step Details in table format
    md_log += (
        f"| **Sender** | **Step Name**       | **Object** | **Created ({local_tz})** |\n"
        f"|------------|---------------------|------------|-------------------------|\n"
        f"| {callback.sender} | {callback.step_name} | {callback.object} | {created_at} |\n\n"
    )

    # Message
    md_log += f"#### Message\n*{callback.message}*\n\n"

    # Code block (even if content is missing)
    if callback.code:
        code_title = "Code"
        if callback.object == "input":
            code_title = "Expected output"
        elif callback.object == "output":
            code_title = callback.code.language

        md_log += f"#### {code_title}\n"
        md_log += f"```{callback.code.language}\n"
        if isinstance(callback.code.content, (dict, list)):
            md_log += json.dumps(callback.code.content, indent=2)
        else:
            md_log += str(callback.code.content)
        md_log += "\n```\n\n"

    # Tokens table (even if data is missing)
    if callback.tokens:
        md_log += "#### Tokens\n\n"
        md_log += (
            f"| **Prompt Tokens** | **Completion Tokens** | **Total Tokens** |\n"
            f"|--------------------|-----------------------|------------------|\n"
            f"| {callback.tokens.prompt_tokens} | {callback.tokens.completion_tokens} | {callback.tokens.total_tokens} |\n\n"
        )

    # Footer separator
    md_log += "---\n\n"

    # Write to file if log_file is provided
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(md_log)
        except Exception as e:
            raise IOError(f"Could not create log file '{log_file}': {e}")

    return md_log
