# tldw_chatbook/Event_Handlers/ingest_events.py
#
#
# Imports
import logging
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Callable

from loguru import logger
#
# 3rd-party Libraries
from textual.widgets import Select, Input, TextArea, Checkbox, RadioSet, RadioButton, Label, Static, Markdown, ListItem, \
    ListView
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll

from ..Prompt_Management.Prompts_Interop import parse_yaml_prompts_from_content, parse_json_prompts_from_content, \
    parse_markdown_prompts_from_content, parse_txt_prompts_from_content, is_initialized, import_prompts_from_files, \
    _get_file_type
from ..Third_Party.textual_fspicker import Filters, FileOpen
#
# Local Imports
from ..tldw_api import (
    TLDWAPIClient, ProcessVideoRequest, ProcessAudioRequest, ProcessPDFRequest,
    ProcessEbookRequest, ProcessDocumentRequest, ProcessXMLRequest, ProcessMediaWikiRequest,
    APIConnectionError, APIRequestError, APIResponseError, AuthenticationError,
    MediaItemProcessResult, ProcessedMediaWikiPage  # Assuming BatchMediaProcessResponse contains this
)
from ..DB.Client_Media_DB_v2 import MediaDatabase # For type hinting
#
if TYPE_CHECKING:
    from ..app import TldwCli
########################################################################################################################
#
# Functions:


# --- Prompt Ingest Constants ---
MAX_PROMPT_PREVIEWS = 10
PROMPT_FILE_FILTERS = Filters(
    ("Markdown", lambda p: p.suffix.lower() == ".md"),
    ("JSON", lambda p: p.suffix.lower() == ".json"),
    ("YAML", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Text", lambda p: p.suffix.lower() == ".txt"),
    ("All Supported", lambda p: p.suffix.lower() in (".md", ".json", ".yaml", ".yml", ".txt")),
    ("All Files", lambda _: True),
)


def _truncate_text(text: Optional[str], max_len: int) -> str:
    """
    Truncates a string to a maximum length, adding ellipsis if truncated.
    Returns 'N/A' if the input text is None or empty.
    """
    if not text: # Handles None or empty string
        return "N/A"
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


async def _update_prompt_preview_display(app: 'TldwCli') -> None:
    """Updates the prompt preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_prompts_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no prompts found.", id="ingest-prompts-preview-placeholder"))
            return

        num_to_display = len(app.parsed_prompts_for_preview)
        prompts_to_show = app.parsed_prompts_for_preview[:MAX_PROMPT_PREVIEWS]

        for idx, prompt_data in enumerate(prompts_to_show):
            name = prompt_data.get("name", f"Unnamed Prompt {idx + 1}")
            author = prompt_data.get("author", "N/A")
            details = _truncate_text(prompt_data.get("details"), 150)
            system_prompt = _truncate_text(prompt_data.get("system_prompt"), 200)
            user_prompt = _truncate_text(prompt_data.get("user_prompt"), 200)
            keywords_list = prompt_data.get("keywords", [])
            keywords = ", ".join(keywords_list) if keywords_list else "N/A"

            md_content = f"""### {name}
**Author:** {author}
**Keywords:** {keywords}

**Details:**
```text
{details}
```

**System Prompt:**
```text
{system_prompt}
```

**User Prompt:**
```text
{user_prompt}
```
---
"""
            await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

        if num_to_display > MAX_PROMPT_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_PROMPT_PREVIEWS} more prompts loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for prompt preview update: {e}")
        app.notify("Error updating prompt preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating prompt preview: {e}", exc_info=True)
        app.notify("Unexpected error during preview update.", severity="error")


def _parse_single_prompt_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """Parses a single prompt file and returns a list of prompt data dicts."""
    file_type = _get_file_type(file_path)  # Use helper from interop
    if not file_type:
        logger.warning(f"Unsupported file type for preview: {file_path}")
        return [{"name": f"Error: Unsupported type {file_path.name}",
                 "details": "File type not recognized for prompt import."}]

    parser_map: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
        "json": parse_json_prompts_from_content,
        "yaml": parse_yaml_prompts_from_content,
        "markdown": parse_markdown_prompts_from_content,
        "txt": parse_txt_prompts_from_content,
    }
    parser = parser_map.get(file_type)
    if not parser:
        logger.error(f"No parser found for file type {file_type} (preview)")
        return [
            {"name": f"Error: No parser for {file_path.name}", "details": f"File type '{file_type}' has no parser."}]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = parser(content)
        if not parsed:  # If parser returns empty list (e.g. empty file or no valid prompts)
            logger.info(f"No prompts found in {file_path.name} by parser for preview.")
            # Not necessarily an error, could be an empty file.
            # Return an empty list, or a specific message if preferred.
            return []
        return parsed
    except RuntimeError as e:
        logger.error(f"Parser dependency missing for {file_path}: {e}")
        app_ref.notify(f"Cannot preview {file_path.name}: Required library missing ({e}).", severity="error", timeout=7)
        return [{"name": f"Error processing {file_path.name}", "details": str(e)}]
    except ValueError as e:
        logger.error(f"Failed to parse {file_path} for preview: {e}")
        app_ref.notify(f"Error parsing {file_path.name}: Invalid format.", severity="warning", timeout=7)
        return [{"name": f"Error parsing {file_path.name}", "details": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error reading {file_path.name}.", severity="error", timeout=7)
        return [{"name": f"Error reading {file_path.name}", "details": str(e)}]


async def _handle_prompt_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """
    Callback function executed after the FileOpen dialog for prompt selection returns.
    This function is wrapped by app.call_after_refresh in the calling code.
    """
    if selected_path:
        logger.info(f"Prompt file selected via dialog: {selected_path}")
        if selected_path in app.selected_prompt_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the selection.", severity="warning")
            return

        app.selected_prompt_files_for_import.append(selected_path)
        app.last_prompt_import_dir = selected_path.parent  # Remember this directory

        # Update selected files list UI
        try:
            list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)
            # Clear "No files selected" placeholder if it's the only item
            if list_view._nodes and isinstance(list_view._nodes[0], ListItem) and \
                    isinstance(list_view._nodes[0].children[0], Label) and \
                    list_view._nodes[0].children[0].renderable == "No files selected.":
                await list_view.clear()
            await list_view.append(ListItem(Label(str(selected_path))))
        except QueryError:
            logger.error("Could not find #ingest-prompts-selected-files-list ListView to update.")

        # Parse this file and add to overall preview list
        parsed_prompts_from_file = _parse_single_prompt_file_for_preview(selected_path, app)
        app.parsed_prompts_for_preview.extend(parsed_prompts_from_file)

        await _update_prompt_preview_display(app)  # Update the preview display
    else:
        logger.info("Prompt file selection cancelled by user.")
        app.notify("File selection cancelled.")


##################################################################################
# THIS IS THE FUNCTION YOU WERE ASKING ABOUT - ITS DEFINITION
##################################################################################
async def handle_ingest_prompts_select_file_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Select Prompt File(s)' button press."""
    logger.debug("Select Prompt File(s) button pressed. Opening file dialog.")
    current_dir = app.last_prompt_import_dir or Path(".")  # Start in last used dir or current dir

    # The FileOpen dialog handles one file selection.
    # The callback _handle_prompt_file_selected_callback will be called after the dialog closes.
    # We use app.call_after_refresh to ensure the callback runs safely after screen changes.
    app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Prompt File (.md, .json, .yaml, .txt)",
            filters=PROMPT_FILE_FILTERS
        ),
        # The callback to FileOpen's push_screen receives the path.
        # We then schedule _handle_prompt_file_selected_callback to run.
        lambda path: app.call_after_refresh(lambda: _handle_prompt_file_selected_callback(app, path))
    )


##################################################################################
# END OF THE FUNCTION DEFINITION
##################################################################################

async def handle_ingest_prompts_clear_files_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Clear Selection' button press for prompt import."""
    logger.info("Clearing selected prompt files and preview.")
    app.selected_prompt_files_for_import.clear()
    app.parsed_prompts_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))

        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.clear()
        app.notify("Selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing prompt selection: {e}")
        app.notify("Error clearing UI.", severity="error")


async def handle_ingest_prompts_import_now_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Import Selected Files Now' button press."""
    logger.info("Import Selected Prompt Files Now button pressed.")

    if not app.selected_prompt_files_for_import:
        app.notify("No prompt files selected to import.", severity="warning")
        return

    if not is_initialized():
        msg = "Prompts database is not initialized. Cannot import."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting import.")
        return

    status_area = app.query_one("#prompt-import-status-area", TextArea)
    status_area.clear()
    status_area.write("Starting import process...\n")
    app.notify("Importing prompts... This may take a moment.")

    # --- Worker for actual import ---
    async def import_worker():
        # Pass the list of Path objects
        return import_prompts_from_files(app.selected_prompt_files_for_import)

    def on_import_success(results: List[Dict[str, Any]]):
        log_text = ["Import process finished.\n\nResults:\n"]
        successful_imports = 0
        failed_imports = 0
        for res in results:
            status = res.get("status", "unknown")
            file_path_str = res.get("file_path", "N/A")
            prompt_name = res.get("prompt_name", "N/A")
            message = res.get("message", "")

            log_text.append(f"File: {Path(file_path_str).name}\n")  # Use Path().name for just filename
            if prompt_name and prompt_name != "N/A":
                log_text.append(f"  Prompt: '{prompt_name}'\n")
            log_text.append(f"  Status: {status.upper()}\n")
            if message:
                log_text.append(f"  Message: {message}\n")
            log_text.append("-" * 30 + "\n")

            if status == "success":
                successful_imports += 1
            else:
                failed_imports += 1

        summary = f"\nSummary: {successful_imports} prompts imported successfully, {failed_imports} failed."
        log_text.append(summary)

        status_area.load_text("".join(log_text))  # Use load_text for full replacement
        app.notify(f"Prompt import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(summary)

        # Optionally, you might want to call the clear files function here:
        # app.call_later(handle_ingest_prompts_clear_files_button_pressed, app) # Or run it directly if safe

    def on_import_failure(error: Exception):
        logger.error(f"Prompt import worker failed: {error}", exc_info=True)
        status_area.write(f"\nImport process failed critically: {error}\nCheck logs for details.\n")
        app.notify(f"Prompt import failed: {error}", severity="error", timeout=10)

    app.run_worker(
        import_worker,
        on_success=on_import_success,
        on_failure=on_import_failure,
        name="prompt_import_worker",
        group="file_operations",
        description="Importing selected prompt files."
    )


# --- TLDW API Form Handlers ---
async def handle_tldw_api_auth_method_changed(app: 'TldwCli', event_value: str) -> None:
    # ... (implementation as you provided)
    logger.debug(f"TLDW API Auth method changed to: {event_value}")
    try:
        custom_token_input = app.query_one("#tldw-api-custom-token", Input)
        custom_token_label = app.query_one("#tldw-api-custom-token-label", Label)
        if event_value == "custom_token":
            custom_token_input.display = True
            custom_token_label.display = True
            custom_token_input.focus()
        else:
            custom_token_input.display = False
            custom_token_label.display = False
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth method change: {e}")

# --- TLDW API Form Specific Option Containers (IDs) ---
TLDW_API_VIDEO_OPTIONS_ID = "tldw-api-video-options"
TLDW_API_AUDIO_OPTIONS_ID = "tldw-api-audio-options"
TLDW_API_PDF_OPTIONS_ID = "tldw-api-pdf-options"
TLDW_API_EBOOK_OPTIONS_ID = "tldw-api-ebook-options"
TLDW_API_DOCUMENT_OPTIONS_ID = "tldw-api-document-options"
TLDW_API_XML_OPTIONS_ID = "tldw-api-xml-options"
TLDW_API_MEDIAWIKI_OPTIONS_ID = "tldw-api-mediawiki-options"

ALL_TLDW_API_OPTION_CONTAINERS = [
    TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID,
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID,
    TLDW_API_MEDIAWIKI_OPTIONS_ID
]

async def handle_tldw_api_media_type_changed(app: 'TldwCli', event_value: str) -> None:
    """Shows/hides media type specific option containers."""
    logger.debug(f"TLDW API Media Type changed to: {event_value}")
    try:
        # Hide all specific option containers first
        for container_id in ALL_TLDW_API_OPTION_CONTAINERS:
            try:
                app.query_one(f"#{container_id}", Container).display = False
            except QueryError:
                pass # Container might not exist if not composed yet, or for all types

        # Show the relevant one
        target_container_id = f"tldw-api-{event_value.lower().replace('_', '-')}-options"
        if event_value: # If a type is selected
            try:
                container_to_show = app.query_one(f"#{target_container_id}", Container)
                container_to_show.display = True
                logger.info(f"Displaying options container: {target_container_id}")
            except QueryError:
                logger.warning(f"Options container #{target_container_id} not found for media type {event_value}.")

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API media type change: {e}")
    except Exception as ex:
        logger.error(f"Unexpected error handling media type change: {ex}", exc_info=True)


def _collect_common_form_data(app: 'TldwCli') -> Dict[str, Any]:
    """Collects common data fields from the TLDW API form."""
    data = {}
    try:
        data["urls"] = [url.strip() for url in app.query_one("#tldw-api-urls", TextArea).text.splitlines() if url.strip()]
        data["local_files"] = [fp.strip() for fp in app.query_one("#tldw-api-local-files", TextArea).text.splitlines() if fp.strip()] # New field for local paths
        data["title"] = app.query_one("#tldw-api-title", Input).value or None
        data["author"] = app.query_one("#tldw-api-author", Input).value or None
        data["keywords_str"] = app.query_one("#tldw-api-keywords", TextArea).text # API client schema expects list, will be parsed there
        data["custom_prompt"] = app.query_one("#tldw-api-custom-prompt", TextArea).text or None
        data["system_prompt"] = app.query_one("#tldw-api-system-prompt", TextArea).text or None
        data["perform_analysis"] = app.query_one("#tldw-api-perform-analysis", Checkbox).value
        data["overwrite_existing_db"] = app.query_one("#tldw-api-overwrite-db", Checkbox).value # For local DB

        # Chunking Options (Common)
        data["perform_chunking"] = app.query_one("#tldw-api-perform-chunking", Checkbox).value
        chunk_method_select = app.query_one("#tldw-api-chunk-method", Select)
        data["chunk_method"] = chunk_method_select.value if chunk_method_select.value != Select.BLANK else None
        data["chunk_size"] = int(app.query_one("#tldw-api-chunk-size", Input).value or "500")
        data["chunk_overlap"] = int(app.query_one("#tldw-api-chunk-overlap", Input).value or "200")
        data["chunk_language"] = app.query_one("#tldw-api-chunk-lang", Input).value or None
        data["use_adaptive_chunking"] = app.query_one("#tldw-api-adaptive-chunking", Checkbox).value
        data["use_multi_level_chunking"] = app.query_one("#tldw-api-multi-level-chunking", Checkbox).value
        data["custom_chapter_pattern"] = app.query_one("#tldw-api-custom-chapter-pattern", Input).value or None

        # Analysis Options
        analysis_api_select = app.query_one("#tldw-api-analysis-api-name", Select)
        data["api_name"] = analysis_api_select.value if analysis_api_select.value != Select.BLANK else None
        # data["api_key"] will be handled separately by auth logic
        data["summarize_recursively"] = app.query_one("#tldw-api-summarize-recursively", Checkbox).value
        data["perform_rolling_summarization"] = app.query_one("#tldw-api-perform-rolling-summarization", Checkbox).value


    except QueryError as e:
        logger.error(f"Error querying common TLDW API form field: {e}")
        app.notify(f"Error: Missing common form field: {e.widget.id if e.widget else 'Unknown'}", severity="error")
        raise
    except ValueError as e:
        logger.error(f"Error converting TLDW API form field value: {e}")
        app.notify(f"Error: Invalid value in common form field. Check numbers.", severity="error")
        raise
    return data

def _collect_video_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessVideoRequest:
    try:
        common_data["transcription_model"] = app.query_one("#tldw-api-video-transcription-model", Input).value or "deepdml/faster-whisper-large-v3-turbo-ct2"
        common_data["transcription_language"] = app.query_one("#tldw-api-video-transcription-language", Input).value or "en"
        common_data["diarize"] = app.query_one("#tldw-api-video-diarize", Checkbox).value
        common_data["timestamp_option"] = app.query_one("#tldw-api-video-timestamp", Checkbox).value
        common_data["vad_use"] = app.query_one("#tldw-api-video-vad", Checkbox).value
        common_data["perform_confabulation_check_of_analysis"] = app.query_one("#tldw-api-video-confab-check", Checkbox).value
        common_data["start_time"] = app.query_one("#tldw-api-video-start-time", Input).value or None
        common_data["end_time"] = app.query_one("#tldw-api-video-end-time", Input).value or None
        # Convert keywords_str to list of strings for Pydantic model
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]

        return ProcessVideoRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying video-specific TLDW API form field: {e}")
        app.notify(f"Error: Missing video form field: {e.widget.id if e.widget else 'Unknown'}", severity="error")
        raise
    except ValueError as e: # For int/float conversions if any added later
        logger.error(f"Error converting video-specific TLDW API form field value: {e}")
        app.notify(f"Error: Invalid value in video form field.", severity="error")
        raise

def _collect_audio_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessAudioRequest:
    try:
        common_data["transcription_model"] = app.query_one("#tldw-api-audio-transcription-model", Input).value or "deepdml/faster-distil-whisper-large-v3.5"
        # other audio specific fields...
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessAudioRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying audio-specific TLDW API form field: {e}")
        app.notify(f"Error: Missing audio form field: {e.widget.id if e.widget else 'Unknown'}", severity="error")
        raise


async def handle_tldw_api_submit_button_pressed(app: 'TldwCli') -> None:
    logger.info("TLDW API Submit button pressed.")
    app.notify("Processing request via tldw API...")

    # 1. Get Endpoint URL and Auth
    try:
        endpoint_url_input = app.query_one("#tldw-api-endpoint-url", Input)
        auth_method_select = app.query_one("#tldw-api-auth-method", Select)
        media_type_select = app.query_one("#tldw-api-media-type", Select)

        endpoint_url = endpoint_url_input.value.strip()
        auth_method = auth_method_select.value
        selected_media_type = media_type_select.value

        if not endpoint_url:
            app.notify("API Endpoint URL is required.", severity="error")
            endpoint_url_input.focus()
            return
        if auth_method == Select.BLANK:
            app.notify("Please select an Authentication Method.", severity="error")
            auth_method_select.focus()
            return
        if selected_media_type == Select.BLANK:
            app.notify("Please select a Media Type to process.", severity="error")
            media_type_select.focus()
            return

        auth_token: Optional[str] = None
        if auth_method == "custom_token":
            custom_token_input = app.query_one("#tldw-api-custom-token", Input)
            auth_token = custom_token_input.value.strip()
            if not auth_token:
                app.notify("Custom Auth Token is required for selected method.", severity="error")
                custom_token_input.focus()
                return
        elif auth_method == "config_token":
            auth_token = app.app_config.get("tldw_api", {}).get("auth_token_config")
            if not auth_token:
                app.notify("Auth Token not found in tldw_api.auth_token_config. Please configure or use custom.", severity="error")
                return
        # Add more auth methods like ENV VAR here if needed

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API submission: {e}")
        app.notify(f"Error: Missing required UI field: {e.widget.id if e.widget else 'Unknown'}", severity="error")
        return

    # 2. Collect Form Data and Create Request Model
    request_model: Optional[Any] = None
    local_file_paths: Optional[List[str]] = None
    try:
        common_data = _collect_common_form_data(app)
        local_file_paths = common_data.pop("local_files", []) # Extract local files
        common_data["api_key"] = auth_token # Pass the resolved token as api_key for the request model

        if selected_media_type == "video":
            request_model = _collect_video_specific_data(app, common_data)
        elif selected_media_type == "audio":
            request_model = _collect_audio_specific_data(app, common_data)
        # Add elif for ProcessPDFRequest, ProcessEbookRequest, etc.
        # Example for PDF:
        # elif selected_media_type == "pdf":
        #     specific_pdf_data = {} # Collect PDF specific fields
        #     request_model = ProcessPDFRequest(**common_data, **specific_pdf_data)
        else:
            app.notify(f"Media type '{selected_media_type}' not yet supported by this client form.", severity="warning")
            return

    except QueryError: # Already handled by app.notify in collectors
        return
    except ValueError: # Already handled
        return
    except Exception as e:
        logger.error(f"Error preparing request model for TLDW API: {e}", exc_info=True)
        app.notify("Error: Could not prepare data for API request.", severity="error")
        return

    if not request_model:
        app.notify("Failed to create request model.", severity="error")
        return

    # Ensure URLs and local_file_paths are not both empty if they are the primary inputs
    if not request_model.urls and not local_file_paths:
        app.notify("Please provide at least one URL or one local file path.", severity="warning")
        try:
            app.query_one("#tldw-api-urls", TextArea).focus()
        except QueryError: pass
        return


    # 3. Initialize API Client and Run Worker
    api_client = TLDWAPIClient(base_url=endpoint_url, token=auth_token) # Token for client, api_key in model for server
    overwrite_db = common_data.get("overwrite_existing_db", False) # Get the DB overwrite flag

    async def process_media_worker():
        nonlocal request_model # Allow modification for XML/MediaWiki
        try:
            if selected_media_type == "video":
                return await api_client.process_video(request_model, local_file_paths)
            elif selected_media_type == "audio":
                return await api_client.process_audio(request_model, local_file_paths)
            # Add elif for other types...
            # elif selected_media_type == "xml":
            #    if not local_file_paths: raise ValueError("XML processing requires a local file path.")
            #    return await api_client.process_xml(request_model, local_file_paths[0]) # XML takes single path
            # elif selected_media_type == "mediawiki_dump":
            #    if not local_file_paths: raise ValueError("MediaWiki processing requires a local file path.")
            #    # For streaming, the worker should yield, not return directly.
            #    # This example shows how to initiate and collect, actual handling of stream in on_success would differ.
            #    results = []
            #    async for item in api_client.process_mediawiki_dump(request_model, local_file_paths[0]):
            #        results.append(item) # Collect all streamed items
            #    return results # Return collected list for on_success
            else:
                raise NotImplementedError(f"Client-side processing for {selected_media_type} not implemented.")
        finally:
            await api_client.close()

    def on_worker_success(response_data: Any): # Type hint can be Union of BatchMediaProcessResponse, etc.
        app.notify("TLDW API request successful. Ingesting results...", timeout=3)
        logger.info(f"TLDW API Response: {response_data}")

        if not app.media_db:
            logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
            app.notify("Error: Local media database not available.", severity="error")
            return

        processed_count = 0
        error_count = 0

        # Handle different response types
        results_to_ingest: List[MediaItemProcessResult] = []
        if isinstance(response_data, dict) and "results" in response_data and "processed_count" in response_data: # Standard BatchMediaProcessResponse
            typed_response = response_data # It's already a dict here, Pydantic parsing happened in client
            results_to_ingest = [MediaItemProcessResult(**item) for item in typed_response.get("results", [])]

        elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
            # MediaWiki dump (if collected into a list by worker)
            for mw_page in response_data:
                if mw_page.status == "Error":
                    error_count +=1
                    logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                    continue
                # Adapt ProcessedMediaWikiPage to MediaItemProcessResult structure for ingestion
                results_to_ingest.append(MediaItemProcessResult(
                    status="Success", # Assume success if no error status
                    input_ref=mw_page.input_ref or mw_page.title,
                    processing_source=mw_page.title, # or another identifier
                    media_type="mediawiki_article", # or "mediawiki_page"
                    metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace, "revision_id": mw_page.revision_id, "timestamp": mw_page.timestamp},
                    content=mw_page.content,
                    chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if mw_page.chunks else None, # Simplified chunk adaptation
                    # analysis, summary, etc. might not be directly available from MediaWiki processing
                ))
        # Add elif for XML if it returns a single ProcessXMLResponseItem or similar

        for item_result in results_to_ingest:
            if item_result.status == "Success":
                try:
                    # Prepare data for add_media_with_keywords
                    # Keywords: API response might not have 'keywords'. Use originally submitted ones if available.
                    # For simplicity, let's assume API response's metadata *might* have keywords.
                    api_keywords = item_result.metadata.get("keywords", []) if item_result.metadata else []
                    if isinstance(api_keywords, str): # If server returns comma-sep string
                        api_keywords = [k.strip() for k in api_keywords.split(',') if k.strip()]

                    # Chunks for UnvectorizedMediaChunks
                    # Ensure chunks are in the format: [{'text': str, 'start_char': int, ...}, ...]
                    unvectorized_chunks_to_save = None
                    if item_result.chunks:
                        unvectorized_chunks_to_save = []
                        for idx, chunk_item in enumerate(item_result.chunks):
                            if isinstance(chunk_item, dict) and "text" in chunk_item:
                                unvectorized_chunks_to_save.append({
                                    "text": chunk_item.get("text"),
                                    "start_char": chunk_item.get("metadata", {}).get("start_char"), # Assuming metadata structure
                                    "end_char": chunk_item.get("metadata", {}).get("end_char"),
                                    "chunk_type": chunk_item.get("metadata", {}).get("type", selected_media_type),
                                    "metadata": chunk_item.get("metadata", {}) # Store full chunk metadata
                                })
                            elif isinstance(chunk_item, str): # If chunks are just strings
                                 unvectorized_chunks_to_save.append({"text": chunk_item})
                            else:
                                logger.warning(f"Skipping malformed chunk item: {chunk_item}")


                    media_id, media_uuid, msg = app.media_db.add_media_with_keywords(
                        url=item_result.input_ref, # Original URL/filename
                        title=item_result.metadata.get("title", item_result.input_ref) if item_result.metadata else item_result.input_ref,
                        media_type=item_result.media_type,
                        content=item_result.content or item_result.transcript, # Use transcript if content is empty
                        keywords=api_keywords or (request_model.keywords if hasattr(request_model, "keywords") else []), # Fallback to request
                        prompt=request_model.custom_prompt if hasattr(request_model, "custom_prompt") else None, # From original request
                        analysis_content=item_result.analysis or item_result.summary,
                        transcription_model=item_result.analysis_details.get("transcription_model") if item_result.analysis_details else (request_model.transcription_model if hasattr(request_model, "transcription_model") else None),
                        author=item_result.metadata.get("author") if item_result.metadata else (request_model.author if hasattr(request_model, "author") else None),
                        # ingestion_date: use current time,
                        overwrite=overwrite_db, # Use the specific DB overwrite flag
                        chunks=unvectorized_chunks_to_save # Pass prepared chunks
                    )
                    if media_id:
                        logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB. Media ID: {media_id}. Message: {msg}")
                        processed_count += 1
                    else:
                        logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB. Message: {msg}")
                        error_count += 1
                except Exception as e_ingest:
                    logger.error(f"Error ingesting item '{item_result.input_ref}' into local DB: {e_ingest}", exc_info=True)
                    error_count += 1
            else:
                logger.error(f"API processing error for '{item_result.input_ref}': {item_result.error}")
                error_count += 1

        final_msg = f"Ingestion complete. Processed: {processed_count}, Errors: {error_count}."
        app.notify(final_msg, severity="information" if error_count == 0 else "warning", timeout=5)

    def on_worker_failure(error: Exception):
        logger.error(f"TLDW API request worker failed: {error}", exc_info=True)
        if isinstance(error, APIResponseError):
            app.notify(f"API Error {error.status_code}: {str(error)[:200]}", severity="error", timeout=8)
        elif isinstance(error, (APIConnectionError, APIRequestError, AuthenticationError)):
            app.notify(f"API Client Error: {str(error)[:200]}", severity="error", timeout=8)
        else:
            app.notify(f"TLDW API processing failed: {str(error)[:200]}", severity="error", timeout=8)

    app.run_worker(
        process_media_worker,
        on_success=on_worker_success,
        on_failure=on_worker_failure,
        name="tldw_api_media_processing",
        group="api_calls",
        description="Processing media via TLDW API"
    )


#
# End of tldw_chatbook/Event_Handlers/ingest_events.py
#######################################################################################################################