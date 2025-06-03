# worker_events.py
# Description:
#
# Imports
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
            #app.notify(f"Error: AI response for worker '{worker_name}' received, but its display widget was missing.", severity="error", timeout=3)
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
                
                # Check if streaming was handled by events by examining the result
                is_streaming_handled_by_events = (result == "STREAMING_HANDLED_BY_EVENTS")

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


                if is_streaming_handled_by_events:
                    logger.info(
                        f"API call ({prefix}) for worker '{worker_name}' SUCCESS was handled with streaming events. "
                        f"The StreamDone message will finalize UI updates, DB saving, and clear current_ai_message_widget. "
                        f"No further action for this worker result in this function.")
                    # Note: The placeholder thinking message was already cleared.
                    # The ai_message_widget.role and header were already updated.
                    # We intentionally do NOT set app.current_ai_message_widget = None here for streaming success.
                    # That's the responsibility of the StreamDone message handler (e.g., on_stream_done in app.py)
                    # after the full message is processed or if an error occurred during streaming.
                else:  # Non-streaming result (result contains the actual data)
                    worker_result_content = result # This is the actual data, not "STREAMING_HANDLED_BY_EVENTS"
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

            if app.current_chat_is_streaming:
                logger.info(f"Worker '{worker_name}' failed during an active stream. Posting StreamDone with error.")
                # Post StreamDone event with the error details
                app.post_message(StreamDone(full_text=ai_message_widget.message_text, error=str(error_from_worker)))
                # DO NOT set app.current_ai_message_widget = None here.
                # The handle_stream_done handler will be responsible for this.
            else:
                logger.info(f"Worker '{worker_name}' failed (non-streaming). Clearing current_ai_message_widget.")
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

        elif event.worker.name == "respond_for_me_worker":
            if event.state is WorkerState.SUCCESS:
                suggestion_result = event.worker.result
                logger.debug(f"Respond_for_me_worker raw result: {str(suggestion_result)[:1000]}") # Log a larger snippet
                suggested_text = ""
                if isinstance(suggestion_result, dict) and suggestion_result.get("choices"):
                    try:
                        suggested_text = suggestion_result["choices"][0].get("message", {}).get("content", "")
                    except (IndexError, KeyError, AttributeError):
                        logger.error(f"Error parsing suggestion_result dict: {suggestion_result}")
                        suggested_text = "" # Fallback to empty if parsing fails
                elif isinstance(suggestion_result, str):
                    suggested_text = suggestion_result
                else:
                    logger.warning(f"Unexpected result type from 'respond_for_me_worker': {type(suggestion_result)}. Content: {str(suggestion_result)[:200]}")

                logger.debug(f"Respond_for_me_worker initial extracted suggested_text: '{suggested_text[:500]}...'")

                if suggested_text:
                    if suggested_text: # Only log if there's something to clean
                        logger.debug(f"Respond_for_me_worker suggested_text BEFORE cleaning: '{suggested_text[:500]}...'")
                    else:
                        logger.debug("Respond_for_me_worker suggested_text is empty or None BEFORE cleaning.")
                    # Clean the text
                    cleaned_suggested_text = suggested_text.strip()
                    logger.debug(f"Respond_for_me_worker cleaned_suggested_text: '{cleaned_suggested_text[:500]}...'")
                    common_fillers = [
                        "Sure, here's a suggestion:", "Here's a possible response:", "You could say:", "How about this:",
                        "Okay, here's a suggestion:", "Here is a suggestion:", "Here's a suggestion for your response:"
                    ]
                    for filler in common_fillers:
                        if cleaned_suggested_text.lower().startswith(filler.lower()):
                            cleaned_suggested_text = cleaned_suggested_text[len(filler):].strip()
                    if (cleaned_suggested_text.startswith('"') and cleaned_suggested_text.endswith('"')) or \
                       (cleaned_suggested_text.startswith("'") and cleaned_suggested_text.endswith("'")):
                        cleaned_suggested_text = cleaned_suggested_text[1:-1]

                    logger.debug(f"Respond_for_me_worker cleaned_suggested_text after further cleaning: '{cleaned_suggested_text[:500]}...'")

                    try:
                        chat_input_widget = app.query_one("#chat-input", TextArea)
                        chat_input_widget.text = cleaned_suggested_text
                        chat_input_widget.focus()
                        try:
                            app.notify("Suggestion populated in the input field.", severity="information", timeout=3)
                        except Exception as e_notify:
                            logger.error(f"Respond_for_me_worker: Error during app.notify call: {e_notify}", exc_info=True)
                        logger.info(f"Suggestion populated from worker: '{cleaned_suggested_text[:100]}...'")
                    except QueryError:
                        logger.error("Respond_for_me_worker: Failed to query #chat-input to populate suggestion.")
                        try:
                            app.notify("Error populating suggestion (UI element missing).", severity="error")
                        except Exception as e_notify:
                            logger.error(f"Respond_for_me_worker: Error during app.notify call: {e_notify}", exc_info=True)
                else:
                    logger.warning("'respond_for_me_worker' succeeded but returned empty suggestion.")
                    try:
                        app.notify("AI returned an empty suggestion.", severity="warning")
                    except Exception as e_notify:
                        logger.error(f"Respond_for_me_worker: Error during app.notify call: {e_notify}", exc_info=True)

            elif event.state is WorkerState.ERROR:
                logger.error(f"Worker 'respond_for_me_worker' failed: {event.worker.error}", exc_info=event.worker.error)
                try:
                    app.notify(f"Failed to generate suggestion: {str(event.worker.error)[:100]}", severity="error", timeout=5)
                except Exception as e_notify:
                    logger.error(f"Respond_for_me_worker: Error during app.notify call: {e_notify}", exc_info=True)
            # Button re-enabling is handled by the finally block in handle_respond_for_me_button_pressed
        else:
            logger.debug(f"Ignoring worker state change for unhandled worker: {worker_name}")


#
########################################################################################################################
#
# Worker Target Function
#
########################################################################################################################

def chat_wrapper_function(app_instance: 'TldwCli', strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
    """
    This function is the target for the worker. It calls the core_chat_function.
    If core_chat_function returns a generator (for streaming), this function consumes it,
    posts StreamingChunk and StreamDone messages, and returns a specific string value.
    If core_chat_function returns a direct result (non-streaming), this function returns it as is.
    """
    logger = getattr(app_instance, 'loguru_logger', logging)
    api_endpoint = kwargs.get('api_endpoint')
    model_name = kwargs.get('model')
    # streaming_requested flag from kwargs is implicitly handled by core_chat_function's return type.
    logger.debug(
        f"chat_wrapper_function executing for endpoint '{api_endpoint}', model '{model_name}'")

    try:
        # core_chat_function is your synchronous `Chat.Chat_Functions.chat`
        result = core_chat_function(strip_thinking_tags=strip_thinking_tags, **kwargs)

        if isinstance(result, Generator):  # Streaming case
            logger.info(f"Core chat function returned a generator for '{api_endpoint}' (model '{model_name}', strip_tags={strip_thinking_tags}). Processing stream in worker.")
            accumulated_full_text = ""
            error_message_if_any = None
            try:
                for chunk_raw in result:
                    # Check for worker cancellation at the beginning of each iteration
                    if app_instance.current_chat_worker and app_instance.current_chat_worker.is_cancelled:
                        app_instance.loguru_logger.info("Chat worker cancelled by user during stream processing in chat_wrapper_function.")
                        if hasattr(result, 'close'):
                            result.close()
                            app_instance.loguru_logger.debug("Closed response_gen (result).")
                        # Post a StreamDone event indicating cancellation
                        # Use accumulated_full_text if it contains partial data, or a specific message
                        cancellation_message = "Streaming cancelled by user."
                        # If accumulated_full_text is meaningful, you could prepend/append the cancellation reason.
                        # For now, we assume the UI will primarily use the 'error' field from StreamDone.
                        app_instance.post_message(StreamDone(full_text=accumulated_full_text, error=cancellation_message))
                        return "STREAMING_HANDLED_BY_EVENTS" # Exit the worker function

                    # Process each raw chunk from the generator (expected to be SSE lines)
                    line = str(chunk_raw).strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        if json_str == "[DONE]":
                            logger.info(f"SSE Stream: Received [DONE] for '{api_endpoint}', model '{model_name}'.")
                            break  # End of stream
                        if not json_str:
                            continue
                        try:
                            json_data = json.loads(json_str)
                            actual_text_chunk = ""
                            # Standard OpenAI SSE structure, adapt if providers differ or if pre-parsed objects are yielded
                            choices = json_data.get("choices")
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                if "content" in delta and delta["content"] is not None:
                                    actual_text_chunk = delta["content"]

                            if actual_text_chunk:
                                app_instance.post_message(StreamingChunk(actual_text_chunk))
                                accumulated_full_text += actual_text_chunk
                            # else:
                            #     logger.trace(f"SSE Stream: No text content in data chunk: {json_str[:100]}")

                        except json.JSONDecodeError as e_json:
                            logger.warning(f"SSE Stream: JSON parsing error for chunk in '{api_endpoint}', model '{model_name}': {e_json}. Chunk: >>{json_str[:100]}<<")
                        except Exception as e_parse:
                            logger.error(f"SSE Stream: Error processing JSON data in '{api_endpoint}', model '{model_name}': {e_parse}. Data: >>{json_str[:100]}<<", exc_info=True)

                    elif line.startswith("event:"):
                        event_type = line[len("event:"):].strip()
                        logger.debug(f"SSE Stream: Received event type '{event_type}' for '{api_endpoint}', model '{model_name}'.")
                        # Handle specific events if necessary (e.g., error events from provider)
                        if event_type == "error": # Example for a hypothetical provider error event
                            logger.error(f"SSE Stream: Provider indicated an error event for '{api_endpoint}', model '{model_name}'. Line: {line}")
                            error_message_if_any = f"Provider error event: {event_type}" # Potentially parse more details
                            # Depending on severity, might want to break here.
                    # else:
                        # logger.trace(f"SSE Stream: Ignoring non-data/non-event line: {line[:100]}")

            except Exception as e_stream_loop:
                logger.exception(
                    f"Error during stream processing loop for '{api_endpoint}', model '{model_name}': {e_stream_loop}")
                error_message_if_any = f"Error during streaming: {str(e_stream_loop)}"
            finally:
                logger.info(f"SSE Stream: Loop finished for '{api_endpoint}', model '{model_name}'. Posting StreamDone. Total length: {len(accumulated_full_text)}. Error: {error_message_if_any}")
                app_instance.post_message(StreamDone(full_text=accumulated_full_text, error=error_message_if_any))

            return "STREAMING_HANDLED_BY_EVENTS"  # Signal that streaming was handled via events

        else:  # Non-streaming case
            logger.debug(
                f"chat_wrapper_function for '{api_endpoint}' (model '{model_name}', strip_tags={strip_thinking_tags}) is returning a direct result (type: {type(result)}).")
            return result  # Return the complete response directly

    except Exception as e:
        # This catches errors from core_chat_function if it fails *before* returning a generator,
        # or other unexpected errors within this wrapper function itself (outside the stream loop).
        logger.exception(
            f"Error in chat_wrapper_function for '{api_endpoint}', model '{model_name}' (potentially pre-stream or non-stream path): {e}")
        # To ensure consistent error handling through StreamDone for the UI:
        app_instance.post_message(StreamDone(full_text="", error=f"Chat wrapper error: {str(e)}"))
        return "STREAMING_HANDLED_BY_EVENTS" # Signal that this path also used an event to report error
        # Alternative for WorkerState.ERROR: raise e

#
# End of worker_events.py
########################################################################################################################
