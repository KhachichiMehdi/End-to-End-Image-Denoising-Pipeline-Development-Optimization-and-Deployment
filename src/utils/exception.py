import sys
from src.logger import logging

def error_message_detail(error: Exception, error_detail: type) -> str:
    """
    Constructs a detailed error message including the file name, line number, and error description.

    Args:
        error (Exception): The exception that was raised.
        error_detail (type): The type of the exception details, typically provided by sys.

    Returns:
        str: A formatted string containing the error details.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract the traceback information
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the exception occurred
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class that provides detailed error messages.

    This class extends the base Exception class and uses the `error_message_detail` function to
    generate a comprehensive error message, including the file name and line number where the
    error occurred.

    Attributes:
        error_message (str): The formatted error message.
    """

    def __init__(self, error_message: str, error_detail: type):
        """
        Initialize the CustomException instance.

        Args:
            error_message (str): The message describing the error.
            error_detail (type): The type of the exception details, typically provided by sys.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        """
        Return the string representation of the error message.

        Returns:
            str: The detailed error message.
        """
        return self.error_message
