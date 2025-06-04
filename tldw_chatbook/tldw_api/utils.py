# tldw_chatbook/tldw_api/utils.py
#
#
# Imports
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, IO, Tuple
import mimetypes
#
# 3rd-party Libraries
from pydantic import BaseModel
#
#######################################################################################################################
#
# Functions:

def model_to_form_data(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic model instance into a dictionary suitable for
    FastAPI Form data submission (handles None, bool, list to string).
    """
    form_data = {}
    for field_name, field_value in model_instance.model_dump(exclude_none=True).items():
        if field_name == "keywords" and isinstance(field_value, list):
            form_data[field_name] = ",".join(field_value) # Server expects comma-separated string for "keywords"
        elif isinstance(field_value, bool):
            form_data[field_name] = str(field_value).lower() # FastAPI Form booleans
        elif isinstance(field_value, list):
            # For lists other than keywords (e.g., urls), httpx handles them correctly
            # if the server endpoint expects multiple values for the same form field name.
            form_data[field_name] = field_value
        elif field_value is not None:
            form_data[field_name] = str(field_value) # Most other fields as strings
    return form_data

def prepare_files_for_httpx(
    file_paths: Optional[List[str]],
    upload_field_name: str = "files"
) -> Optional[List[Tuple[str, Tuple[str, IO[bytes], Optional[str]]]]]:
    """
    Prepares a list of file paths for httpx multipart upload.

    Args:
        file_paths: A list of string paths to local files.
        upload_field_name: The name of the field for file uploads (FastAPI often uses 'files').

    Returns:
        A list of tuples formatted for httpx's `files` argument, or None.
        Example: [('files', ('filename.mp4', <file_obj>, 'video/mp4')), ...]
    """
    if not file_paths:
        return None

    httpx_files_list = []
    for file_path_str in file_paths:
        try:
            file_path_obj = Path(file_path_str)
            if not file_path_obj.is_file():
                # Or raise an error, or log and skip
                # Consider using logging module here instead of logging.info for a library
                logging.warning(f"Warning: File not found or not a file: {file_path_str}")
                continue

            file_obj = open(file_path_obj, "rb")

            mime_type, _ = mimetypes.guess_type(file_path_obj.name) # Use filename for guessing

            if mime_type is None:
                # If the type can't be guessed, you can fallback to a generic MIME type
                # 'application/octet-stream' is a common default for unknown binary data.
                mime_type = 'application/octet-stream'
                logging.warning(f"Could not guess MIME type for {file_path_obj.name}. Defaulting to {mime_type}.")
                logging.info(f"Warning: Could not guess MIME type for {file_path_obj.name}. Defaulting to {mime_type}.")

            httpx_files_list.append(
                (upload_field_name, (file_path_obj.name, file_obj, mime_type))
            )
        except Exception as e:
            # Consider using logging module here
            logging.error(f"Error preparing file {file_path_str} for upload: {e}")
            # Handle error, e.g., skip this file or raise
            # If you skip, ensure file_obj is closed if it was opened.
            # However, in this structure, if open() fails, the exception occurs before append.
            # If an error occurs after open() but before append, the file might not be closed.
            # Using a try/finally for file_obj.close() or opening file_obj within a
            # `with open(...) as file_obj:` block inside the `prepare_files_for_httpx`
            # is safer if you add logic between open() and append() that could fail.
            # For now, httpx will manage the file objects passed to it.
    return httpx_files_list if httpx_files_list else None

#
# End of utils.py
#######################################################################################################################
