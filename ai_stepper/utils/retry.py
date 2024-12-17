from typing import Callable, TypeVar, Any
import time
import logging
from functools import wraps

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    logger: logging.Logger = None,
    allowed_exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that implements retry logic with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Base delay for exponential backoff in seconds
        logger (logging.Logger): Logger instance for logging retry attempts
        allowed_exceptions (tuple): Tuple of exceptions that trigger retries
        
    Returns:
        Callable: Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3)
        def my_function():
            # Function implementation
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Changed to exponential backoff
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. "
                                f"Retrying in {delay:.1f}s. Error: {str(e)}"
                            )
                        else:
                            print(f"[yellow]Retry attempt {attempt + 1}/{max_retries} for {func.__name__}[/yellow]")
                            print(f"[yellow]Error: {str(e)}[/yellow]")
                            print(f"[yellow]Retrying in {delay:.1f}s...[/yellow]")
                        time.sleep(delay)
                    else:
                        print(f"[red]All {max_retries} retry attempts failed for {func.__name__}[/red]")
                        raise
            
            raise last_exception  # type: ignore
            
        return wrapper
    return decorator
