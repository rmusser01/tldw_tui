# tldw_chatbook/tldw_api/utils.py
#
#
# Imports
from pathlib import Path
from typing import Dict, Any, Optional, List, IO, Tuple
#
# 3rd-party Libraries
from pydantic import BaseModel
import httpx
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
                print(f"Warning: File not found or not a file: {file_path_str}")
                continue

            file_obj = open(file_path_obj, "rb")
            # Basic MIME type guessing, can be improved with `mimetypes` library
            mime_type = None
            if file_path_obj.suffix.lower() == ".mp4":
                mime_type = "video/mp4"
            elif file_path_obj.suffix.lower() == ".mp3":
                mime_type = "audio/mpeg"
            # Add more MIME types as needed

            httpx_files_list.append(
                (upload_field_name, (file_path_obj.name, file_obj, mime_type))
            )
        except Exception as e:
            print(f"Error preparing file {file_path_str} for upload: {e}")
            # Handle error, e.g., skip this file or raise
    return httpx_files_list if httpx_files_list else None

#
# End of utils.py
#######################################################################################################################
