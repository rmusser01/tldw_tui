# worker_events.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING, Generator, Any, Union, Dict, List, AsyncGenerator  # Added Union, Dict, List
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
    prefix = prefix_parts[0]

    ai_message_widget = app.current_ai_message_widget

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
            await chat_container_fallback.mount(ChatMessage(error_msg_text, role="System", classes="-error"))
        except QueryError:
            logger.error(f"Fallback: Could not find chat container #{prefix}-log.")
        app.current_ai_message_widget = None
        return

    try:
        chat_container: VerticalScroll = app.query_one(f"#{prefix}-log", VerticalScroll)
        static_text_widget_in_ai_msg = ai_message_widget.query_one(".message-text", Static)

        if event.state is WorkerState.SUCCESS:
            result = event.worker.result  # This is what chat_wrapper_function returned

            # Your chat function is synchronous. If streaming, it returns a sync Generator.
            is_streaming_result = isinstance(result, Generator)

            current_thinking_text_pattern = f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}"
            if ai_message_widget.message_text.strip().startswith(current_thinking_text_pattern):
                ai_message_widget.message_text = ""
                static_text_widget_in_ai_msg.update("")

            if is_streaming_result:
                logger.info(
                    f"API call ({prefix}) for worker '{worker_name}' returned a sync Generator – processing as stream.")
                full_original_text_streamed = ""

                if not ai_message_widget or not ai_message_widget.is_mounted:
                    logger.warning(
                        f"Stream processing for '{prefix}' (worker '{worker_name}') aborted before start: AI widget no longer mounted.")
                    app.current_ai_message_widget = None
                    return

                try:
                    # Iterate over the synchronous generator provided by the worker's result
                    # This loop will run in the main (UI) thread because on_worker_state_changed is async
                    # but the generator itself is sync. This is okay for Textual as long as
                    # each chunk processing is fast.
                    for chunk in result:  # type: ignore[misc] # result is Generator here
                        if not ai_message_widget or not ai_message_widget.is_mounted:
                            logger.warning(
                                f"Stream processing for '{prefix}' (worker '{worker_name}') aborted mid-stream: AI widget became unmounted.")
                            break

                        text_chunk_original = str(chunk)
                        full_original_text_streamed += text_chunk_original

                        ai_message_widget.message_text += text_chunk_original
                        static_text_widget_in_ai_msg.update(escape_markup(ai_message_widget.message_text))

                        if chat_container.is_mounted:
                            # For sync generator, direct call to scroll_end is fine.
                            # If it were an async generator, self.call_later or scheduling might be needed
                            # if updates were too frequent and blocking. But Textual handles this well.
                            chat_container.scroll_end(animate=False, duration=0.05)

                            # After the loop finishes (or breaks)
                    if ai_message_widget and ai_message_widget.is_mounted:
                        ai_message_widget.mark_generation_complete()
                        logger.info(
                            f"Stream finished for '{prefix}' (worker '{worker_name}'). Final original length: {len(full_original_text_streamed)} chars.")

                        if app.notes_service and app.current_chat_conversation_id and not app.current_chat_is_ephemeral:
                            db_for_ai_msg_stream = app.notes_service._get_db(app.notes_user_id)
                            try:
                                ai_msg_db_id = ccl.add_message_to_conversation(
                                    db_for_ai_msg_stream, app.current_chat_conversation_id, "AI",
                                    full_original_text_streamed
                                )
                                if ai_msg_db_id: ai_message_widget.message_id_internal = ai_msg_db_id
                                logger.debug(
                                    f"Streamed AI message saved to DB (ConvID: {app.current_chat_conversation_id}, MsgID: {ai_msg_db_id})")
                            except Exception as e_save_ai_stream:
                                logger.error(f"Failed to save streamed AI message to DB: {e_save_ai_stream}",
                                             exc_info=True)

                except Exception as exc_stream_outer:
                    logger.exception(
                        f"Error during sync stream iteration for worker '{worker_name}': {exc_stream_outer}")
                    if ai_message_widget and ai_message_widget.is_mounted:
                        current_text_plain = ai_message_widget.message_text
                        error_message_display = Text.from_markup(
                            escape_markup(current_text_plain) + "\n[bold red]Error during stream processing.[/]")
                        try:
                            static_text_widget_in_ai_msg.update(error_message_display)
                        except QueryError:
                            logger.error("Failed to update AI widget with stream processing error message.")
                        ai_message_widget.mark_generation_complete()
                finally:
                    app.current_ai_message_widget = None
                    if app.is_mounted:
                        try:
                            app.query_one(f"#{prefix}-input", TextArea).focus()
                        except QueryError:
                            pass

            else:  # Non-streaming result (direct value, not a generator)
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
                        logger.error(
                            f"Error parsing non-streaming dict result: {e_parse}. Resp: {worker_result_content}",
                            exc_info=True)
                        original_text_for_storage = "[AI: Error parsing successful response structure.]"
                        final_display_text_obj = Text.from_markup(
                            f"[bold red]{escape_markup(original_text_for_storage)}[/]")
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
                    final_display_text_obj = Text.from_markup(
                        f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                else:
                    # ... (same parsing logic as before) ...
                    logger.error(
                        f"Unexpected result type from API via worker: {type(worker_result_content)}. Content: {str(worker_result_content)[:200]}...")
                    original_text_for_storage = f"[Error: Unexpected result type ({type(worker_result_content).__name__}) from API worker.]"
                    final_display_text_obj = Text.from_markup(
                        f"[bold red]{escape_markup(original_text_for_storage)}[/]")

                ai_message_widget.message_text = original_text_for_storage
                static_text_widget_in_ai_msg.update(final_display_text_obj)
                ai_message_widget.mark_generation_complete()

                is_error_message = original_text_for_storage.startswith(("[AI: Error", "[Error:", "[bold red]Error"))
                if app.notes_service and app.current_chat_conversation_id and \
                        not app.current_chat_is_ephemeral and original_text_for_storage and not is_error_message:
                    db_for_ai_msg_non_stream = app.notes_service._get_db(app.notes_user_id)
                    try:
                        ai_msg_db_id_ns = ccl.add_message_to_conversation(
                            db_for_ai_msg_non_stream, app.current_chat_conversation_id, "AI", original_text_for_storage
                        )
                        if ai_msg_db_id_ns: ai_message_widget.message_id_internal = ai_msg_db_id_ns
                        logger.debug(
                            f"Non-streamed AI message saved to DB (ConvID: {app.current_chat_conversation_id}, MsgID: {ai_msg_db_id_ns})")
                    except Exception as e_save_ai_ns:
                        logger.error(f"Failed to save non-streamed AI message to DB: {e_save_ai_ns}", exc_info=True)

                app.current_ai_message_widget = None

        elif event.state is WorkerState.ERROR:
            # ... (same error handling as before) ...
            error_from_worker = event.worker.error
            logger.error(f"Worker '{worker_name}' failed with an unhandled exception in worker target function.",
                         exc_info=error_from_worker)
            error_message_str = f"AI System Error: Worker failed unexpectedly.\nDetails: {str(error_from_worker)}"
            escaped_error_for_display = escape_markup(error_message_str)
            ai_message_widget.message_text = error_message_str
            static_text_widget_in_ai_msg.update(Text.from_markup(f"[bold red]{escaped_error_for_display}[/]"))
            ai_message_widget.mark_generation_complete()
            app.current_ai_message_widget = None

        if chat_container.is_mounted:
            chat_container.scroll_end(animate=True)
        if app.is_mounted and app.current_ai_message_widget is None:
            try:
                app.query_one(f"#{prefix}-input", TextArea).focus()
            except QueryError:
                pass

    except QueryError as qe_outer:
        # ... (same outer QueryError handling as before) ...
        logger.error(
            f"QueryError in handle_api_call_worker_state_changed for '{worker_name}': {qe_outer}. Widget might have been removed.",
            exc_info=True)
        if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
            try:
                await app.current_ai_message_widget.remove()
            except Exception as e_remove_final:
                logger.error(f"Error removing AI widget during outer QueryError: {e_remove_final}")
        app.current_ai_message_widget = None
    except Exception as exc_outer:
        # ... (same outer Exception handling as before) ...
        logger.exception(
            f"Unexpected error in handle_api_call_worker_state_changed for worker '{worker_name}': {exc_outer}")
        if ai_message_widget and ai_message_widget.is_mounted:
            try:
                static_widget_unexp_err = ai_message_widget.query_one(".message-text", Static)
                error_update_text_unexp = Text.from_markup(
                    f"[bold red]Internal error handling AI response:[/]\n{escape_markup(str(exc_outer))}")
                static_widget_unexp_err.update(error_update_text_unexp)
                ai_message_widget.mark_generation_complete()
            except Exception as e_unexp_final_update:
                logger.error(f"Further error updating AI widget during outer unexp error: {e_unexp_final_update}")
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
        return f"[bold red]Error during chat processing (worker target):[/]\n{escape_markup(str(e))}"

#
# End of worker_events.py
########################################################################################################################
