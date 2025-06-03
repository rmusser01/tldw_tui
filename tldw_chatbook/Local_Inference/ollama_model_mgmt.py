# ollama_model_mgmt.py
#
# Imports
#
# Third-Party Libraries
#
# Local Imports
#
#######################################################################################################################
# Ollama Management API Calls
#######################################################################################################################
import logging
import json as json_parser
from typing import Optional, Callable, Tuple, Any, Dict, List

import requests
from textual.widgets import RichLog


# Note: TypingDict is used as an alias for Dict in type hints for Ollama functions
# to avoid any potential (though unlikely) conflict if 'Dict' itself were a parameter name.
# `logging` refers to the project's logger, already configured.
# `requests` is imported at the top.
# `json_parser` is an alias for the standard `json` module.
# `RichLog` is imported from `textual.widgets`.
# `Callable`, `Tuple`, `Optional`, `Any`, `List`, `Union` are from `typing`.


def _ollama_request(
        method: str,
        base_url: str,
        endpoint: str,
        log_widget: Optional[RichLog] = None,
        stream_log_callback: Optional[Callable[[str], None]] = None,
        **kwargs: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Helper function to make HTTP requests to the Ollama API.
    """
    if not base_url.startswith(("http://", "https://")):
        err_msg = f"Invalid base_url scheme: {base_url}. Must be http or https."
        logging.error(err_msg)
        if log_widget:
            log_widget.write(f"[bold red]Error:[/bold red] {err_msg}")
        return None, err_msg

    full_url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    if stream_log_callback:
        if 'json' in kwargs and isinstance(kwargs['json'], dict):
            kwargs['json']['stream'] = True

    logging.debug(f"Ollama Request: {method} {full_url} | Payload: {kwargs.get('json', kwargs.get('params', 'None'))}")

    try:
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.request(method, full_url, stream=bool(stream_log_callback), timeout=300, **kwargs)

            if stream_log_callback:
                last_line_json: Optional[Dict[str, Any]] = None
                accumulated_error_messages: List[str] = []

                if response.status_code >= 400:  # Error before streaming (e.g. 404)
                    try:
                        error_detail = response.json().get("error", response.text)
                    except requests.exceptions.JSONDecodeError:
                        error_detail = response.text
                    err_msg = f"Error: {response.status_code} - {error_detail}"
                    stream_log_callback(f"[bold red]{err_msg}[/bold red]\n")
                    logging.error(f"Ollama stream initiation error: {err_msg} for {method} {full_url}")
                    return None, err_msg

                for line_bytes in response.iter_lines():
                    if line_bytes:
                        line = line_bytes.decode('utf-8')
                        stream_log_callback(line + "\n")
                        try:
                            json_line = json_parser.loads(line)
                            if isinstance(json_line, dict):
                                last_line_json = json_line
                                if "error" in json_line:  # Check for errors within the stream content
                                    accumulated_error_messages.append(str(json_line["error"]))
                        except json_parser.JSONDecodeError:
                            pass  # Not all lines are JSON

                response.raise_for_status()  # Check status after stream for non-2xx

                if accumulated_error_messages:
                    final_error_message = "; ".join(accumulated_error_messages)
                    logging.error(
                        f"Ollama stream contained error messages: {final_error_message} for {method} {full_url}")
                    return last_line_json, final_error_message

                return last_line_json, None

            else:  # Not streaming
                response.raise_for_status()
                if response.content and "application/json" in response.headers.get("Content-Type", ""):
                    return response.json(), None
                elif response.status_code == 200 and not response.content:  # Successful DELETE
                    return {"status": "success",
                            "message": f"Operation {method} on {endpoint} successful (200 OK, No Content)."}, None
                elif response.status_code == 200:  # Successful but not JSON
                    logging.warning(
                        f"Ollama request to {full_url} successful (200) but response is not JSON. Body: {response.text[:100]}")
                    return {"status": "success",
                            "message": f"Operation successful (200), non-JSON response: {response.text[:100]}"}, None
                else:  # Should be caught by raise_for_status or other conditions
                    err_msg = f"Unexpected non-JSON response with status {response.status_code}: {response.text[:200]}"
                    logging.error(err_msg)
                    if log_widget: log_widget.write(f"[bold red]API Error:[/bold red] {err_msg}")
                    return None, err_msg

    except requests.exceptions.HTTPError as e:
        err_msg_detail = e.response.text
        try:
            error_json = e.response.json()
            err_msg_detail = error_json.get('error', e.response.text)
        except json_parser.JSONDecodeError:
            pass
        err_msg = f"HTTP Error: {e.response.status_code} - {err_msg_detail}"
        logging.error(f"Ollama request HTTP error for {method} {full_url}: {err_msg}", exc_info=False)
        if log_widget: log_widget.write(f"[bold red]API Error:[/bold red] {err_msg}")
        return None, err_msg
    except requests.exceptions.ConnectionError:
        err_msg = f"Connection Error: Failed to connect to Ollama server at {base_url}. Ensure Ollama is running and accessible."
        logging.error(f"Ollama connection error for {method} {full_url}: {err_msg}", exc_info=False)
        if log_widget: log_widget.write(f"[bold red]Connection Error:[/bold red] {err_msg}")
        return None, err_msg
    except requests.exceptions.Timeout:
        err_msg = f"Timeout: The request to {full_url} timed out."
        logging.error(err_msg, exc_info=False)
        if log_widget: log_widget.write(f"[bold red]Timeout Error:[/bold red] {err_msg}")
        return None, err_msg
    except requests.exceptions.RequestException as e:
        err_msg = f"Request Exception: An unexpected error occurred: {e}"
        logging.error(f"Ollama generic request error for {method} {full_url}: {err_msg}", exc_info=True)
        if log_widget: log_widget.write(f"[bold red]Request Error:[/bold red] {err_msg}")
        return None, err_msg
    except json_parser.JSONDecodeError as e:  # Fallback for non-streaming JSON decode issues
        err_msg = f"JSON Decode Error: Failed to parse response from {full_url} as JSON. Error: {e}. Response text: {response.text[:200]}"
        logging.error(err_msg, exc_info=True)
        if log_widget: log_widget.write(f"[bold red]Response Error:[/bold red] {err_msg}")
        return None, err_msg


def ollama_list_local_models(base_url: str, log_widget: Optional[RichLog] = None) -> Tuple[
    Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_list_local_models: Requesting /api/tags from {base_url}")
    return _ollama_request("GET", base_url, "/api/tags", log_widget=log_widget)


def ollama_model_info(base_url: str, model_name: str, log_widget: Optional[RichLog] = None) -> Tuple[
    Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_model_info: Requesting /api/show for model '{model_name}' from {base_url}")
    return _ollama_request("POST", base_url, "/api/show", log_widget=log_widget, json={"name": model_name})


def ollama_copy_model(base_url: str, source: str, destination: str, log_widget: Optional[RichLog] = None) -> Tuple[
    Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_copy_model: Requesting /api/copy from '{source}' to '{destination}' at {base_url}")
    return _ollama_request("POST", base_url, "/api/copy", log_widget=log_widget,
                           json={"source": source, "destination": destination})


def ollama_delete_model(base_url: str, model_name: str, stream_log_callback: Optional[Callable[[str], None]] = None,
                        log_widget: Optional[RichLog] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_delete_model: Requesting /api/delete for model '{model_name}' from {base_url}")
    return _ollama_request("DELETE", base_url, "/api/delete", stream_log_callback=stream_log_callback,
                           log_widget=log_widget, json={"name": model_name})


def ollama_pull_model(base_url: str, model_name: str, insecure: bool = False,
                      stream_log_callback: Optional[Callable[[str], None]] = None,
                      log_widget: Optional[RichLog] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(
        f"ollama_pull_model: Requesting /api/pull for model '{model_name}' from {base_url}, insecure: {insecure}")
    payload = {"name": model_name, "insecure": insecure}
    return _ollama_request("POST", base_url, "/api/pull", stream_log_callback=stream_log_callback,
                           log_widget=log_widget, json=payload)


def ollama_create_model(base_url: str, model_name: str, path: str,
                        stream_log_callback: Optional[Callable[[str], None]] = None,
                        log_widget: Optional[RichLog] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(
        f"ollama_create_model: Requesting /api/create for model '{model_name}' with path '{path}' from {base_url}")
    payload = {"name": model_name, "path": path}
    return _ollama_request("POST", base_url, "/api/create", stream_log_callback=stream_log_callback,
                           log_widget=log_widget, json=payload)


def ollama_push_model(base_url: str, model_name: str, insecure: bool = False,
                      stream_log_callback: Optional[Callable[[str], None]] = None,
                      log_widget: Optional[RichLog] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(
        f"ollama_push_model: Requesting /api/push for model '{model_name}' from {base_url}, insecure: {insecure}")
    payload = {"name": model_name, "insecure": insecure}
    return _ollama_request("POST", base_url, "/api/push", stream_log_callback=stream_log_callback,
                           log_widget=log_widget, json=payload)


def ollama_generate_embeddings(base_url: str, model_name: str, prompt: str, options: Optional[Dict[str, Any]] = None,
                               log_widget: Optional[RichLog] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_generate_embeddings: Requesting /api/embeddings for model '{model_name}' from {base_url}")
    payload: Dict[str, Any] = {"model": model_name, "prompt": prompt}
    if options:
        payload["options"] = options
    return _ollama_request("POST", base_url, "/api/embeddings", log_widget=log_widget, json=payload)


def ollama_list_running_models(base_url: str, log_widget: Optional[RichLog] = None) -> Tuple[
    Optional[Dict[str, Any]], Optional[str]]:
    logging.debug(f"ollama_list_running_models: Requesting /api/ps from {base_url}")
    return _ollama_request("GET", base_url, "/api/ps", log_widget=log_widget)