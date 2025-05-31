# worker_events.py
# Description:
#
# Imports
import asyncio
import logging
import json # Added for SSE JSON parsing
from typing import TYPE_CHECKING, Generator, Any, Union
#
# 3rd-Party Imports
from loguru import logger as loguru_logger  # If used directly here
from rich.text import Text
from rich.markup import escape as escape_markup
from textual.message import Message
from textual.worker import Worker, WorkerState
from textual.widgets import Static, TextArea, Label  # Added TextArea
from textual.containers import VerticalScroll  # Added VerticalScroll
from textual.css.query import QueryError  # Added QueryError
#
# Local Imports
from ..Widgets.chat_message import ChatMessage
from ..Utils.Emoji_Handling import get_char, EMOJI_THINKING, FALLBACK_THINKING
# Import the actual chat function if it's to be called from chat_wrapper_function
from ..Chat.Chat_Functions import chat as core_chat_function
from ..Character_Chat import Character_Chat_Lib as ccl # For saving AI messages
from ..DB.ChaChaNotes_DB import CharactersRAGDBError, InputError  # For specific error handling
#
if TYPE_CHECKING:
    from ..app import TldwCli

# Custom Messages for Streaming
class StreamingChunk(Message):
    """Custom message to send a piece of streamed text."""
    def __init__(self, text_chunk: str) -> None:
        super().__init__()
        self.text_chunk = text_chunk

class StreamDone(Message):
    """Custom message to signal the end of a stream."""
    def __init__(self, full_text: str, error: Union[str, None] = None) -> None: # Added Optional error
        super().__init__()
        self.full_text = full_text
        self.error = error # Store error
#
########################################################################################################################
#
# Functions:

########################################################################################################################
#
# Event Handlers (called by app.py's on_worker_state_changed)
#
########################################################################################################################

async def handle_api_call_worker_state_changed(app: 'TldwCli', event: Worker.StateChanged) -> None:
    """Handles completion/failure of background API workers (chat calls)."""
    logger = getattr(app, 'loguru_logger', logging)
    worker_name = event.worker.name or "Unknown Worker"
    logger.debug(f"Worker '{worker_name}' state changed to {event.state}")

    # ---- batching UI updates ----
    # Define how often to update the UI
    UI_UPDATE_CHUNK_INTERVAL = 5  # Update UI every 5 text chunks
    # Alternatively, or in addition, a time interval:
    # UI_UPDATE_TIME_INTERVAL_SEC = 0.1 # Update UI at least every 0.1 seconds
    # last_ui_update_time = time.monotonic()
    # ------------------------------------------

    if not worker_name.startswith("API_Call_"):
        logger.debug(f"Ignoring worker state change for non-API call worker: {worker_name}")
        return

    prefix_parts = worker_name.replace("API_Call_", "").split('_regenerate')
    prefix = prefix_parts[0] # Should be "chat" for the main chat tab

    ai_message_widget = app.current_ai_message_widget # This is the placeholder ChatMessage

    if ai_message_widget is None or not ai_message_widget.is_mounted:
        logger.warning(
            f"Worker '{worker_name}' finished, but its AI placeholder widget is missing or not mounted. "
            f"Current placeholder ref ID: {getattr(app.current_ai_message_widget, 'id', 'N/A') if app.current_ai_message_widget else 'N/A'}"
        )
        # ... (fallback error reporting as before) ...
        try:
            chat_container_fallback: VerticalScroll = app.query_one(f"#{prefix}-log", VerticalScroll)
            error_msg_text = Text.from_markup(
                f"[bold red]Error:[/]\nAI response for worker '{worker_name}' received, but its display widget was missing.")
            # Use plain text for ChatMessage content if it doesn't support Text directly
            await chat_container_fallback.mount(ChatMessage(message=error_msg_text.plain, role="System", classes="-error"))
        except QueryError:
            logger.error(f"Fallback: Could not find chat container #{prefix}-log.")
        app.current_ai_message_widget = None
        return

    try:
        chat_container: VerticalScroll = app.query_one(f"#{prefix}-log", VerticalScroll)
        static_text_widget_in_ai_msg = ai_message_widget.query_one(".message-text", Static)

        if event.state is WorkerState.SUCCESS:
            result = event.worker.result
            is_streaming_result = isinstance(result, Generator)

            ai_sender_name_for_db = "AI"
            if not app.current_chat_is_ephemeral and app.current_chat_conversation_id:
                # For persistent chats, get the character associated with THIS conversation
                if app.chachanotes_db:
                    conv_char_name = ccl.get_character_name_for_conversation(app.chachanotes_db, app.current_chat_conversation_id)
                    if conv_char_name: ai_sender_name_for_db = conv_char_name
            elif app.current_chat_active_character_data:
                ai_sender_name_for_db = app.current_chat_active_character_data.get('name', 'AI')

            # Set the role of the placeholder AI widget to the determined sender name
            # This ensures the header of the AI message bubble shows the correct character name
            ai_message_widget.role = ai_sender_name_for_db
            # Update the header label directly if role reactive doesn't auto-update it
            try:
                header_label = ai_message_widget.query_one(".message-header", Label)
                header_label.update(ai_sender_name_for_db)
            except QueryError:
                logger.warning("Could not update AI message header label with character name.")

            current_thinking_text_pattern = f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}"
            current_thinking_fallback_pattern = FALLBACK_THINKING # Simpler check
            current_message_text_strip = ai_message_widget.message_text.strip()

            if current_message_text_strip.startswith(current_thinking_text_pattern) or \
               current_message_text_strip.endswith(current_thinking_fallback_pattern) or \
               current_message_text_strip == current_thinking_fallback_pattern: # Exact match for fallback
                logger.debug(f"Clearing thinking placeholder: '{current_message_text_strip}'")
                ai_message_widget.message_text = ""
                static_text_widget_in_ai_msg.update("")
            else:
                logger.debug(f"Thinking placeholder not found or already cleared. Current text: '{current_message_text_strip}'")


            if is_streaming_result:
                logger.info(
                    f"API call ({prefix}) for worker '{worker_name}' returned a sync Generator – processing as stream.")
                full_original_text_streamed = "" # Initialize accumulator for the full text
                current_batch_text_for_ui_update = ""

                if not ai_message_widget or not ai_message_widget.is_mounted:
                    logger.warning(
                        f"Stream processing for '{prefix}' (worker '{worker_name}') aborted before start: AI widget no longer mounted.")
                    app.current_ai_message_widget = None
                    return

                try:  # Inner try for the loop itself
                    logger.debug(f"Starting stream iteration for worker '{worker_name}'...")
                    for chunk_idx, chunk_raw in enumerate(result):  # type: ignore[misc]
                        if not ai_message_widget or not ai_message_widget.is_mounted:
                            logger.warning(
                                f"Stream processing for '{prefix}' (worker '{worker_name}') aborted mid-stream (chunk {chunk_idx}): AI widget became unmounted.")
                            break

                        chunk = str(chunk_raw)
                        logger.trace(f"Stream chunk {chunk_idx} RAW from generator for '{worker_name}': >>{chunk}<<")

                        if not chunk.strip():
                            logger.trace(f"Stream chunk {chunk_idx} SKIPPED (empty or whitespace)")
                            continue

                        if chunk.startswith("data: "):
                            json_str = chunk[len("data: "):].strip()
                            logger.debug(f"Stream chunk {chunk_idx} JSON part for '{worker_name}': >>{json_str}<<")

                            if json_str == "[DONE]":
                                logger.info(f"Stream chunk {chunk_idx}: Received [DONE] signal for '{worker_name}'.")
                                if current_batch_text_for_ui_update:  # Process final batch before breaking
                                    logger.debug(f"Updating UI with final batch before [DONE] for '{worker_name}'.")
                                    ai_message_widget.message_text += current_batch_text_for_ui_update
                                    static_text_widget_in_ai_msg.update(escape_markup(ai_message_widget.message_text))
                                    if chat_container.is_mounted:
                                        chat_container.scroll_end(animate=False, duration=0.05)
                                    await asyncio.sleep(0.01)
                                    current_batch_text_for_ui_update = ""
                                break

                            if not json_str:
                                logger.trace(
                                    f"Stream chunk {chunk_idx} SKIPPED (empty JSON part after 'data: ' stripping).")
                                continue

                            # Initialize extracted_text for this chunk before the try-except for json.loads
                            extracted_text = None
                            finish_reason = None

                            try:  # Innermost try for JSON processing of a single chunk
                                json_data = json.loads(json_str)
                                logger.trace(f"Stream chunk {chunk_idx} PARSED JSON for '{worker_name}': {json_data}")

                                choices = json_data.get("choices")
                                if choices and isinstance(choices, list) and len(choices) > 0:
                                    delta = choices[0].get("delta", {})
                                    extracted_text = delta.get("content")
                                    finish_reason = choices[0].get("finish_reason")
                                    if "tool_calls" in delta:
                                        logger.debug(
                                            f"Stream chunk {chunk_idx}: Delta contains tool_calls: {delta['tool_calls']}")

                            except json.JSONDecodeError as e_json:
                                logger.warning(
                                    f"Stream chunk {chunk_idx}: JSON parsing error for '{worker_name}': {e_json}. JSON string was: >>{json_str}<<")
                                # Do not try to process extracted_text if JSON parsing failed for this chunk
                                continue  # Move to the next chunk
                            except Exception as e_inner_parse:  # Catch other errors during parsing this specific chunk
                                logger.error(
                                    f"Stream chunk {chunk_idx}: Error processing inner JSON data for '{worker_name}' (JSON: >>{json_str}<<): {e_inner_parse}",
                                    exc_info=True)
                                # Do not try to process extracted_text if an error occurred
                                continue  # Move to the next chunk

                            # This block now only runs if JSON parsing for the current chunk was successful
                            if extracted_text is not None:
                                logger.debug(
                                    f"Stream chunk {chunk_idx}: Extracted text for '{worker_name}': >>{extracted_text}<<")
                                full_original_text_streamed += extracted_text
                                current_batch_text_for_ui_update += extracted_text
                                text_chunks_since_last_ui_update += 1

                                if text_chunks_since_last_ui_update >= UI_UPDATE_CHUNK_INTERVAL:
                                    logger.trace(
                                        f"UI Update Triggered (chunk interval) for '{worker_name}'. Batch: >>{current_batch_text_for_ui_update}<<")
                                    ai_message_widget.message_text += current_batch_text_for_ui_update
                                    static_text_widget_in_ai_msg.update(escape_markup(ai_message_widget.message_text))
                                    if chat_container.is_mounted:
                                        chat_container.scroll_end(animate=False, duration=0.05)
                                    current_batch_text_for_ui_update = ""
                                    text_chunks_since_last_ui_update = 0
                                    await asyncio.sleep(0.01)
                            elif finish_reason:
                                logger.info(
                                    f"Stream chunk {chunk_idx}: Received finish_reason '{finish_reason}' for '{worker_name}'.")
                            else:
                                logger.trace(
                                    f"Stream chunk {chunk_idx}: No text content or finish_reason in choices for '{worker_name}'.")
                        else:
                            logger.warning(
                                f"Stream chunk {chunk_idx}: Received non-SSE chunk or unexpected format for '{worker_name}': >>{chunk[:200]}...<<")
                    # End of for loop

                    # After the loop, update with any remaining text in the batch
                    if current_batch_text_for_ui_update:
                        logger.debug(
                            f"Updating UI with final remaining batch after loop for '{worker_name}'. Batch: >>{current_batch_text_for_ui_update}<<")
                        ai_message_widget.message_text += current_batch_text_for_ui_update
                        static_text_widget_in_ai_msg.update(escape_markup(ai_message_widget.message_text))
                        if chat_container.is_mounted:
                            chat_container.scroll_end(animate=False, duration=0.05)
                        await asyncio.sleep(0)

                    if ai_message_widget and ai_message_widget.is_mounted:  # Check again before marking complete
                        ai_message_widget.mark_generation_complete()
                    logger.info(
                        f"Stream fully processed for '{prefix}' (worker '{worker_name}'). Final original length: {len(full_original_text_streamed)} chars.")

                    # Save streamed AI message to DB if chat is persistent
                    # Ensure that full_original_text_streamed is not empty before saving
                    if app.chachanotes_db and app.current_chat_conversation_id and \
                       not app.current_chat_is_ephemeral and full_original_text_streamed.strip():
                        try:
                            ai_msg_db_id_version = ccl.add_message_to_conversation(
                                app.chachanotes_db, app.current_chat_conversation_id,
                                ai_sender_name_for_db, # Use determined sender name
                                full_original_text_streamed
                            )
                            if ai_msg_db_id_version: # This is just the ID
                                saved_ai_msg_details = app.chachanotes_db.get_message_by_id(ai_msg_db_id_version)
                                if saved_ai_msg_details:
                                    ai_message_widget.message_id_internal = saved_ai_msg_details.get('id')
                                    ai_message_widget.message_version_internal = saved_ai_msg_details.get('version')
                                    logger.debug(
                                        f"Streamed AI message saved to DB. ConvID: {app.current_chat_conversation_id}, "
                                        f"MsgID: {saved_ai_msg_details.get('id')}, Version: {saved_ai_msg_details.get('version')}")
                                else:
                                    logger.error(f"Failed to retrieve saved streamed AI message details from DB for ID {ai_msg_db_id_version}.")
                            else:
                                logger.error(f"Failed to save streamed AI message to DB (no ID returned).")
                        except (CharactersRAGDBError, InputError) as e_save_ai_stream:
                            logger.error(f"Failed to save streamed AI message to DB: {e_save_ai_stream}", exc_info=True)
                    elif not full_original_text_streamed.strip():
                        logger.info(f"Stream for '{prefix}' (worker '{worker_name}') resulted in empty content. Not saving to DB.")
                    else:
                         logger.debug(f"Streamed AI message for '{prefix}' (worker '{worker_name}') not saved to DB (chat is ephemeral or other condition not met).")


                except Exception as exc_stream_outer:
                    logger.error(f"Outer error during stream processing for worker '{worker_name}'. Exception type: {type(exc_stream_outer).__name__}, Message: {exc_stream_outer}", exc_info=True)
                    if ai_message_widget and ai_message_widget.is_mounted:
                        current_text_plain = ai_message_widget.message_text  # Get what was streamed so far
                        # Ensure exc_stream_outer is converted to string for display
                        error_message_display = Text.from_markup(
                            escape_markup(
                                current_text_plain) + f"\n[bold red]Error during stream processing:\n{escape_markup(str(exc_stream_outer))}[/]")
                        try:
                            static_text_widget_in_ai_msg.update(error_message_display)
                        except QueryError:
                            logger.error(
                                "Failed to update AI widget with stream processing error message after outer exception.")
                        if hasattr(ai_message_widget, 'mark_generation_complete'):  # Check before calling
                            ai_message_widget.mark_generation_complete()
                finally:
                    logger.debug(f"Stream processing 'finally' block reached for worker '{worker_name}'.")
                    app.current_ai_message_widget = None  # Crucial
                    if app.is_mounted:
                        try:
                            # Try to focus input only if no critical error has occurred that might prevent UI interaction
                            if event.state is WorkerState.SUCCESS or (ai_message_widget and ai_message_widget.is_mounted):
                                app.query_one(f"#{prefix}-input", TextArea).focus()
                        except QueryError:
                            logger.debug(f"Could not focus input #{prefix}-input after stream, widget might not exist.")
                        except Exception as e_focus:
                            logger.error(f"Error focusing input after stream: {e_focus}", exc_info=True)
                        pass
            else:  # Non-streaming result
                worker_result_content = result
                logger.debug(
                    f"NON-STREAMING RESULT for '{prefix}' (worker '{worker_name}'): {str(worker_result_content)[:200]}...")

                final_display_text_obj: Text
                original_text_for_storage: str = ""

                if isinstance(worker_result_content, dict) and 'choices' in worker_result_content:
                    # ... (same parsing logic as before) ...
                    try:
                        original_text_for_storage = worker_result_content['choices'][0]['message']['content']
                        final_display_text_obj = Text(original_text_for_storage)
                    except (KeyError, IndexError, TypeError) as e_parse:
                        logger.error(f"Error parsing non-streaming dict result: {e_parse}. Resp: {worker_result_content}", exc_info=True)
                        original_text_for_storage = "[AI: Error parsing successful response structure.]"
                        final_display_text_obj = Text.from_markup(f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                elif isinstance(worker_result_content, str):
                    # ... (same parsing logic as before) ...
                    original_text_for_storage = worker_result_content
                    if worker_result_content.startswith(("[bold red]Error during chat processing (worker target):[/]",
                                                         "[bold red]Error during chat processing:[/]",
                                                         "An error occurred in the chat function:")):
                        final_display_text_obj = Text.from_markup(worker_result_content)
                    else:
                        final_display_text_obj = Text(worker_result_content)
                elif worker_result_content is None:
                    # ... (same parsing logic as before) ...
                    original_text_for_storage = "[AI: Error – No response received from API (Result was None).]"
                    final_display_text_obj = Text.from_markup(f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                else:
                    logger.error(f"Unexpected result type from API via worker: {type(worker_result_content)}. Content: {str(worker_result_content)[:200]}...")
                    original_text_for_storage = f"[Error: Unexpected result type ({type(worker_result_content).__name__}) from API worker.]"
                    final_display_text_obj = Text.from_markup(f"[bold red]{escape_markup(original_text_for_storage)}[/]")

                ai_message_widget.message_text = original_text_for_storage
                static_text_widget_in_ai_msg.update(final_display_text_obj)
                ai_message_widget.mark_generation_complete()

                is_error_message = original_text_for_storage.startswith(("[AI: Error", "[Error:", "[bold red]Error"))
                if app.chachanotes_db and app.current_chat_conversation_id and \
                        not app.current_chat_is_ephemeral and original_text_for_storage and not is_error_message:
                    try:
                        ai_msg_db_id_ns_version = ccl.add_message_to_conversation(
                            app.chachanotes_db, app.current_chat_conversation_id,
                            ai_sender_name_for_db, # Use determined sender name
                            original_text_for_storage
                        )
                        if ai_msg_db_id_ns_version: # This is just the ID
                            saved_ai_msg_details_ns = app.chachanotes_db.get_message_by_id(ai_msg_db_id_ns_version)
                            if saved_ai_msg_details_ns:
                                ai_message_widget.message_id_internal = saved_ai_msg_details_ns.get('id')
                                ai_message_widget.message_version_internal = saved_ai_msg_details_ns.get('version')
                                logger.debug(
                                    f"Non-streamed AI message saved to DB. ConvID: {app.current_chat_conversation_id}, "
                                    f"MsgID: {saved_ai_msg_details_ns.get('id')}, Version: {saved_ai_msg_details_ns.get('version')}")
                            else:
                                logger.error(f"Failed to retrieve saved non-streamed AI message details from DB for ID {ai_msg_db_id_ns_version}.")
                        else:
                            logger.error(f"Failed to save non-streamed AI message to DB (no ID returned).")
                    except (CharactersRAGDBError, InputError) as e_save_ai_ns:
                        logger.error(f"Failed to save non-streamed AI message to DB: {e_save_ai_ns}", exc_info=True)

                app.current_ai_message_widget = None
                pass

        elif event.state is WorkerState.ERROR:
            # ... (same error handling as before) ...
            error_from_worker = event.worker.error
            logger.error(f"Worker '{worker_name}' failed with an unhandled exception in worker target function.",
                         exc_info=error_from_worker)
            error_message_str = f"AI System Error: Worker failed unexpectedly.\nDetails: {str(error_from_worker)}"
            escaped_error_for_display = escape_markup(error_message_str)
            ai_message_widget.message_text = error_message_str # Store raw error
            ai_message_widget.role = "System" # Display as System error
            # Update header label as well for consistency
            try:
                header_label = ai_message_widget.query_one(".message-header", Label)
                header_label.update("System Error")
            except QueryError:
                logger.warning("Could not update AI message header label for worker error state.")

            static_text_widget_in_ai_msg.update(Text.from_markup(f"[bold red]{escaped_error_for_display}[/]"))
            ai_message_widget.mark_generation_complete()
            app.current_ai_message_widget = None

        if chat_container.is_mounted:
            chat_container.scroll_end(animate=True)
        if app.is_mounted and app.current_ai_message_widget is None: # Focus input if AI turn is fully complete
            try:
                app.query_one(f"#{prefix}-input", TextArea).focus()
            except QueryError:
                logger.debug(f"Could not focus input #{prefix}-input after API call, widget might not exist.")
            except Exception as e_final_focus:
                logger.error(f"Error focusing input after API call processing: {e_final_focus}", exc_info=True)

    except QueryError as qe_outer:
        logger.error(
            f"QueryError in handle_api_call_worker_state_changed for '{worker_name}': {qe_outer}. Widget might have been removed.",
            exc_info=True)
        if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
            try:
                await app.current_ai_message_widget.remove()
            except Exception as e_remove_final:
                logger.error(f"Error removing AI widget during outer QueryError handling: {e_remove_final}")
        app.current_ai_message_widget = None
    except Exception as exc_outer:
        logger.exception( # Use logger.exception to include stack trace for unexpected errors
            f"Unexpected outer error in handle_api_call_worker_state_changed for worker '{worker_name}': {exc_outer}")
        if ai_message_widget and ai_message_widget.is_mounted:
            try:
                static_widget_unexp_err = ai_message_widget.query_one(".message-text", Static)
                error_update_text_unexp = Text.from_markup(
                    f"[bold red]Internal error handling AI response:[/]\n{escape_markup(str(exc_outer))}")
                static_widget_unexp_err.update(error_update_text_unexp)
                ai_message_widget.role = "System"
                try:
                    header_label = ai_message_widget.query_one(".message-header", Label)
                    header_label.update("System Error")
                except QueryError: pass
                ai_message_widget.mark_generation_complete()
            except Exception as e_unexp_final_update:
                logger.error(f"Further error updating AI widget during outer unexpected error: {e_unexp_final_update}")
        app.current_ai_message_widget = None


#
########################################################################################################################
#
# Worker Target Function
#
########################################################################################################################

def chat_wrapper_function(app_instance: 'TldwCli', **kwargs: Any) -> Any:
    """
    This function is the target for the worker. It calls the core_chat_function.
    Since core_chat_function is synchronous, this wrapper is also synchronous.
    If core_chat_function yields for streaming, this wrapper will return a sync Generator.
    """
    logger = getattr(app_instance, 'loguru_logger', logging)
    api_endpoint = kwargs.get('api_endpoint')
    model_name = kwargs.get('model')
    streaming_requested = kwargs.get('streaming', False)  # Get the streaming flag
    logger.debug(
        f"chat_wrapper_function (sync worker target) executing for endpoint '{api_endpoint}', model '{model_name}', streaming: {streaming_requested}")

    try:
        # core_chat_function is your synchronous `Chat.Chat_Functions.chat`
        result = core_chat_function(**kwargs)

        if isinstance(result, Generator):  # Check if it returned a sync generator
            logger.debug(
                f"chat_wrapper_function for '{api_endpoint}' (model '{model_name}') is returning a sync Generator (streaming).")
        else:
            logger.debug(
                f"chat_wrapper_function for '{api_endpoint}' (model '{model_name}') is returning a direct result (type: {type(result)}).")
        return result

    except Exception as e:
        logger.exception(
            f"Error inside chat_wrapper_function (sync worker target) for endpoint {api_endpoint} (model {model_name}): {e}")
        # Return a string that ChatMessage can display as an error
        return f"[bold red]Error during chat processing (worker target):[/]\n{escape_markup(str(e))}"

#
# End of worker_events.py
########################################################################################################################
