# tldw_chatbook/Event_Handlers/ingest_events.py
#
#
# Imports
import logging
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict

from loguru import logger
#
# 3rd-party Libraries
from textual.widgets import Select, Input, TextArea, Checkbox, RadioSet, RadioButton, Label
from textual.css.query import QueryError
from textual.containers import Container
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


async def handle_tldw_api_auth_method_changed(app: 'TldwCli', event_value: str) -> None:
    """Shows/hides the custom token input based on auth method selection."""
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