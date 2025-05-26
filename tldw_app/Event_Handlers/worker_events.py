# worker_events.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING, Generator, Any, Union, Dict, List  # Added Union, Dict, List
#
# 3rd-Party Imports
from loguru import logger as loguru_logger  # If used directly here
from rich.text import Text
from rich.markup import escape as escape_markup
from textual.worker import Worker, WorkerState
from textual.widgets import Static, TextArea  # Added TextArea
from textual.containers import VerticalScroll  # Added VerticalScroll
from textual.css.query import QueryError  # Added QueryError
#
# Local Imports
from ..Widgets.chat_message import ChatMessage
from ..Utils.Emoji_Handling import get_char, EMOJI_THINKING, FALLBACK_THINKING
# Import the actual chat function if it's to be called from chat_wrapper_function
from ..Chat.Chat_Functions import chat as core_chat_function
from ..Character_Chat import Character_Chat_Lib as ccl # For saving AI messages
from ..DB.ChaChaNotes_DB import CharactersRAGDBError # For specific error handling
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

########################################################################################################################
#
# Event Handlers (called by app.py)
#
########################################################################################################################

async def handle_api_call_worker_state_changed(app: 'TldwCli', event: Worker.StateChanged) -> None:
    """Handles completion/failure of background API workers (chat calls)."""
    worker_name = event.worker.name or "Unknown Worker"
    logging.debug(f"Worker '{worker_name}' state changed to {event.state}")

    if not worker_name.startswith("API_Call_"):
        logging.debug(f"Ignoring worker state change for non-API call worker: {worker_name}")
        return

    # Extract prefix, handling potential "_regenerate" suffix
    prefix_parts = worker_name.replace("API_Call_", "").split('_regenerate')
    prefix = prefix_parts[0]  # Should always exist, e.g., "chat" or "ccp"

    ai_message_widget = app.current_ai_message_widget

    if ai_message_widget is None or not ai_message_widget.is_mounted:
        logging.warning(
            f"Worker '{worker_name}' finished, but its AI placeholder widget is missing or not mounted. ID: {getattr(app.current_ai_message_widget, 'id', 'N/A') if app.current_ai_message_widget else 'N/A'}")
        try:
            chat_container_fallback: VerticalScroll = app.query_one(f"#{prefix}-log", VerticalScroll)
            error_msg_text = f"[bold red]Error:[/]\nAI response for worker '{worker_name}' received, but its display widget was missing. Check logs."
            error_widget_fallback = ChatMessage(Text.from_markup(error_msg_text), role="System", classes="-error")
            await chat_container_fallback.mount(error_widget_fallback)
            chat_container_fallback.scroll_end(animate=False)
        except QueryError:
            logging.error(f"Fallback: Could not find chat container #{prefix}-log to report missing AI placeholder.")
        except Exception as e_fallback:
            logging.error(f"Fallback: Error reporting missing AI placeholder: {e_fallback}", exc_info=True)
        app.current_ai_message_widget = None
        return

    try:
        chat_container: VerticalScroll = app.query_one(f"#{prefix}-log", VerticalScroll)
        static_text_widget_in_ai_msg = ai_message_widget.query_one(".message-text", Static)

        if event.state is WorkerState.SUCCESS:
            result = event.worker.result
            is_streaming_result = isinstance(result, Generator)

            current_thinking_text = f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}"
            if ai_message_widget.message_text.strip().startswith(current_thinking_text):
                ai_message_widget.message_text = ""  # Clear internal state
                static_text_widget_in_ai_msg.update("")  # Clear UI part

            if is_streaming_result:
                logging.info(f"API call ({prefix}) returned a generator – streaming.")

                # Define an inner async function to process the stream
                async def process_stream_chunks() -> None:
                    full_original_text_streamed = ""
                    if not ai_message_widget or not ai_message_widget.is_mounted:  # Re-check widget
                        logging.warning(f"Stream processing for '{prefix}' aborted: AI widget no longer mounted.")
                        return
                    try:
                        # Re-query static text widget inside stream processor for safety
                        stream_static_text_widget = ai_message_widget.query_one(".message-text", Static)
                    except QueryError:
                        logging.error(f"Stream proc for '{prefix}' aborted: .message-text Static widget not found.")
                        if ai_message_widget.is_mounted: ai_message_widget.mark_generation_complete()
                        app.current_ai_message_widget = None
                        return

                    try:
                        async for chunk in result:  # type: ignore[misc] # result is known to be Generator here
                            text_chunk_original = str(chunk)
                            full_original_text_streamed += text_chunk_original
                            # logging.debug(f"STREAM CHUNK for '{prefix}': {text_chunk_original!r}") # Can be very verbose

                            # Append UNESCAPED chunk to internal message_text
                            ai_message_widget.message_text += text_chunk_original
                            # Update display by rendering the full (cumulative) ESCAPED message_text
                            stream_static_text_widget.update(escape_markup(ai_message_widget.message_text))

                            if chat_container.is_mounted:
                                chat_container.scroll_end(animate=False, duration=0.05)

                        ai_message_widget.mark_generation_complete()
                        logging.info(
                            f"Stream finished for '{prefix}'. Final original length: {len(full_original_text_streamed)} chars.")
                        # Save AI message to DB after stream completion
                        if app.notes_service and app.current_chat_conversation_id and not app.current_chat_is_ephemeral:
                            db_for_ai_msg = app.notes_service._get_db(app.notes_user_id)
                            try:
                                ai_msg_id = ccl.add_message_to_conversation(
                                    db_for_ai_msg, app.current_chat_conversation_id, "AI", full_original_text_streamed
                                )
                                if ai_msg_id: ai_message_widget.message_id_internal = ai_msg_id
                                loguru_logger.debug(
                                    f"Streamed AI message saved to DB (ConvID: {app.current_chat_conversation_id}, MsgID: {ai_msg_id})")
                            except Exception as e_save_ai:
                                loguru_logger.error(f"Failed to save streamed AI message to DB: {e_save_ai}",
                                                    exc_info=True)


                    except Exception as exc_stream_inner:
                        logging.exception(f"Stream failure for worker '{worker_name}': {exc_stream_inner}")
                        if ai_message_widget.is_mounted and stream_static_text_widget.is_mounted:
                            current_text_plain = ai_message_widget.message_text  # Get internally stored original text
                            error_message_display = Text.from_markup(
                                escape_markup(current_text_plain) + "\n[bold red]Error during stream.[/]")
                            stream_static_text_widget.update(error_message_display)
                        if ai_message_widget.is_mounted: ai_message_widget.mark_generation_complete()
                    finally:
                        app.current_ai_message_widget = None  # Clear app's reference
                        if app.is_mounted:  # Ensure app is still there
                            try:
                                app.query_one(f"#{prefix}-input", TextArea).focus()
                            except QueryError:
                                pass  # Input might be gone

                app.run_task(process_stream_chunks(), name=f"stream_processor_{prefix}", group="streams")

            else:  # Non-streaming result
                worker_result_str = str(result) if result is not None else ""
                logging.debug(f"NON-STREAMING RESULT for '{prefix}': {worker_result_str!r}")

                final_display_text_obj: Text
                original_text_for_storage: str = ""

                if isinstance(result, dict) and 'choices' in result:  # OpenAI-like successful response
                    try:
                        original_text_for_storage = result['choices'][0]['message']['content']
                        final_display_text_obj = Text(original_text_for_storage)  # Assumed plain text
                    except (KeyError, IndexError, TypeError) as e_parse:
                        logging.error(f"Error parsing non-streaming dict result: {e_parse}. Resp: {result}",
                                      exc_info=True)
                        original_text_for_storage = "[AI: Error parsing successful response structure.]"
                        final_display_text_obj = Text.from_markup(
                            f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                elif isinstance(result, str):
                    original_text_for_storage = result
                    if result.startswith(
                            ("[bold red]Error during chat processing:[/]", "An error occurred in the chat function:")):
                        final_display_text_obj = Text.from_markup(result) if result.startswith("[bold red]") else Text(
                            result)
                    else:  # Assume other strings are plain text from API
                        final_display_text_obj = Text(result)
                elif result is None:
                    original_text_for_storage = "[AI: Error – No response received from API.]"
                    final_display_text_obj = Text.from_markup(
                        f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                else:
                    logging.error(f"Unexpected result type from API: {type(result)}. Content: {result!r}")
                    original_text_for_storage = f"[Error: Unexpected result type ({type(result).__name__}) from API.]"
                    final_display_text_obj = Text.from_markup(
                        f"[bold red]{escape_markup(original_text_for_storage)}[/]")

                ai_message_widget.message_text = original_text_for_storage  # Store raw text
                static_text_widget_in_ai_msg.update(final_display_text_obj)  # Update display
                ai_message_widget.mark_generation_complete()

                # Save non-streamed AI message to DB
                if app.notes_service and app.current_chat_conversation_id and not app.current_chat_is_ephemeral and original_text_for_storage and not original_text_for_storage.startswith(
                        "["):  # Avoid saving error messages
                    db_for_ai_msg_ns = app.notes_service._get_db(app.notes_user_id)
                    try:
                        ai_msg_id_ns = ccl.add_message_to_conversation(
                            db_for_ai_msg_ns, app.current_chat_conversation_id, "AI", original_text_for_storage
                        )
                        if ai_msg_id_ns: ai_message_widget.message_id_internal = ai_msg_id_ns
                        loguru_logger.debug(
                            f"Non-streamed AI message saved to DB (ConvID: {app.current_chat_conversation_id}, MsgID: {ai_msg_id_ns})")
                    except Exception as e_save_ai_ns:
                        loguru_logger.error(f"Failed to save non-streamed AI message to DB: {e_save_ai_ns}",
                                            exc_info=True)

                app.current_ai_message_widget = None  # Clear reference

        elif event.state is WorkerState.ERROR:
            error_from_worker = event.worker.error
            logging.error(f"Worker '{worker_name}' failed.", exc_info=error_from_worker)
            error_message_str = f"AI Error: Processing failed.\nDetails: {str(error_from_worker)}"
            escaped_error_for_display = escape_markup(error_message_str)

            ai_message_widget.message_text = error_message_str  # Store original error
            static_text_widget_in_ai_msg.update(Text.from_markup(f"[bold red]{escaped_error_for_display}[/]"))
            ai_message_widget.mark_generation_complete()
            app.current_ai_message_widget = None

        # Common cleanup actions
        if chat_container.is_mounted:
            chat_container.scroll_end(animate=True)
        if app.is_mounted and app.current_ai_message_widget is None:  # Only focus if processing finished
            try:
                app.query_one(f"#{prefix}-input", TextArea).focus()
            except QueryError:
                pass  # Input area might be gone or on different tab

    except QueryError as qe_outer:
        logging.error(
            f"QueryError in on_worker_state_changed for '{worker_name}': {qe_outer}. Widget might have been removed.",
            exc_info=True)
        if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
            try:
                await app.current_ai_message_widget.remove()  # Attempt to clean up if possible
            except Exception as e_remove_final:
                logging.error(f"Error removing AI widget during outer QueryError: {e_remove_final}")
        app.current_ai_message_widget = None
    except Exception as exc_outer:
        logging.exception(f"Unexpected error in on_worker_state_changed for worker '{worker_name}': {exc_outer}")
        if ai_message_widget and ai_message_widget.is_mounted:
            try:
                static_widget_unexp_err = ai_message_widget.query_one(".message-text", Static)
                error_update_text_unexp = Text.from_markup(
                    f"[bold red]Internal error handling AI response:[/]\n{escape_markup(str(exc_outer))}")
                static_widget_unexp_err.update(error_update_text_unexp)
                ai_message_widget.mark_generation_complete()
            except Exception as e_unexp_final_update:
                logging.error(f"Further error updating AI widget during outer unexp error: {e_unexp_final_update}")
        app.current_ai_message_widget = None


#
########################################################################################################################
#
# Worker Target Function (called by app.py's worker creation)
#
########################################################################################################################

def chat_wrapper_function(app_instance: 'TldwCli', **kwargs: Any) -> Any:
    """
    This function is the target for the worker. It calls the core chat logic.
    The `app_instance` is passed to allow access to config or other app-level state if needed,
    though ideally, all necessary parameters are passed via `kwargs`.
    """
    api_endpoint = kwargs.get('api_endpoint')
    logging.debug(f"chat_wrapper_function (worker target) executing for endpoint '{api_endpoint}'")
    try:
        # All necessary parameters for core_chat_function should be in kwargs
        # core_chat_function is imported from ..Chat.Chat_Functions
        result = core_chat_function(**kwargs)
        logging.debug(f"chat_wrapper_function finished for '{api_endpoint}'. Result type: {type(result)}")
        return result
    except Exception as e:
        logging.exception(f"Error inside chat_wrapper_function (worker target) for endpoint {api_endpoint}: {e}")
        # Return a formatted error string that handle_api_call_worker_state_changed can display
        return f"[bold red]Error during chat processing (worker target):[/]\n{escape_markup(str(e))}"

#
# End of worker_events.py
########################################################################################################################
