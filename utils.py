import logging
import time
import traceback
import uuid
from datetime import datetime
from typing import Callable


def retry(
    function_to_retry: Callable,
    function_args,
    max_retries=3,
    delay_in_seconds=5,
    error_msg="",
):
    """
    Wrapper for functions that interact with DOT/311 systems. Retries the input
    function in case of failure; if failures > max_retries, logs error.

    Args:
        function_to_retry (Callable): The function to retry.
        function_args: Arguments to pass to the function.
        max_retries (int): Maximum number of retry attempts.
        delay_in_seconds (int): Initial delay between retries in seconds.
        error_msg (str): Error message to log if the function fails.

    Returns:
        The result of the function call if successful, None otherwise.
    """

    retries = 0
    while retries < max_retries:
        try:
            result = function_to_retry(function_args)
            return result
        except Exception as e:
            logging.error(error_msg)
            logging.error(e)
            logging.error(function_args)
            logging.error(f"{datetime.now()}")
            logging.error(traceback.format_exc())
            retries += 1
            logging.error(f"Retrying attempt {retries}.")
            time.sleep(delay_in_seconds)

    logging.error("Maximum number of retries reached.")
    return


class CentralIDGenerator:
    """
    Generates unique IDs used to track vehicles.

    Attributes:
        _used_ids (set): Set of previously generated IDs.
    """

    def __init__(self):
        self._used_ids = set()

    def generate_id(self):
        """
        Generates a unique UUID.

        Returns:
            uuid.UUID: A unique identifier.
        """

        unique_id = uuid.uuid4()
        while unique_id in self._used_ids:
            unique_id = uuid.uuid4()
        self._used_ids.add(unique_id)
        return unique_id
