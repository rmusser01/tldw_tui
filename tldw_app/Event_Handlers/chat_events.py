# tldw_app/Event_Handlers/chat_events.py
# Description:
#
# Imports
import logging
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Dict, Any, Optional
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from rich.text import Text
from rich.markup import escape as escape_markup
from textual.widgets import (
    Button, Input, TextArea, Static, Select, Checkbox, ListView, ListItem, Label
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError

from ..Utils.Utils import safe_float, safe_int
#
# Local Imports
from ..Widgets.chat_message import ChatMessage
from ..Widgets.titlebar import TitleBar
from ..Utils.Emoji_Handling import (
    get_char, EMOJI_THINKING, FALLBACK_THINKING, EMOJI_EDIT, FALLBACK_EDIT,
    EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT, EMOJI_COPIED, FALLBACK_COPIED, EMOJI_COPY, FALLBACK_COPY
)
from ..Character_Chat import Character_Chat_Lib as ccl
from ..DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError # Import specific DB errors
#
if TYPE_CHECKING:
    from ..app import TldwCli, API_IMPORTS_SUCCESSFUL


#
########################################################################################################################
#
# Functions:

async def handle_chat_send_button_pressed(app: 'TldwCli', prefix: str) -> None:
    """Handles the send button press for the main chat tab."""
    loguru_logger.info(f"Send button pressed for '{prefix}' (main chat)") # Use loguru_logger consistently

    # --- 1. Query UI Widgets ---
    try:
        text_area = app.query_one(f"#{prefix}-input", TextArea)
        chat_container = app.query_one(f"#{prefix}-log", VerticalScroll)
        provider_widget = app.query_one(f"#{prefix}-api-provider", Select)
        model_widget = app.query_one(f"#{prefix}-api-model", Select)
        system_prompt_widget = app.query_one(f"#{prefix}-system-prompt", TextArea)
        temp_widget = app.query_one(f"#{prefix}-temperature", Input)
        top_p_widget = app.query_one(f"#{prefix}-top-p", Input)
        min_p_widget = app.query_one(f"#{prefix}-min-p", Input)
        top_k_widget = app.query_one(f"#{prefix}-top-k", Input)

        llm_max_tokens_widget = app.query_one(f"#{prefix}-llm-max-tokens", Input)
        llm_seed_widget = app.query_one(f"#{prefix}-llm-seed", Input)
        llm_stop_widget = app.query_one(f"#{prefix}-llm-stop", Input)
        llm_response_format_widget = app.query_one(f"#{prefix}-llm-response-format", Select)
        llm_n_widget = app.query_one(f"#{prefix}-llm-n", Input)
        llm_user_identifier_widget = app.query_one(f"#{prefix}-llm-user-identifier", Input)
        llm_logprobs_widget = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
        llm_top_logprobs_widget = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
        llm_logit_bias_widget = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
        llm_presence_penalty_widget = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
        llm_frequency_penalty_widget = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
        llm_tools_widget = app.query_one(f"#{prefix}-llm-tools", TextArea)
        llm_tool_choice_widget = app.query_one(f"#{prefix}-llm-tool-choice", Input)

    except QueryError as e:
        loguru_logger.error(f"Send Button: Could not find UI widgets for '{prefix}': {e}")
        try:
            container_for_error = chat_container if 'chat_container' in locals() and chat_container.is_mounted else app.query_one(
                f"#{prefix}-log", VerticalScroll) # Re-query if initial one failed
            await container_for_error.mount(
                ChatMessage(Text.from_markup(f"[bold red]Internal Error:[/]\nMissing UI elements for {prefix}."), role="System", classes="-error"))
        except QueryError:
            loguru_logger.error(f"Send Button: Critical - could not even find chat container #{prefix}-log to display error.")
        return

    # --- 2. Get Message and Parameters from UI ---
    message_text_from_input = text_area.text.strip()
    reuse_last_user_bubble = False

    if not message_text_from_input: # Try to reuse last user message if input is empty
        try:
            last_msg_widget: Optional[ChatMessage] = None
            # Iterate over a materialized list to avoid issues if querying during modification (though less likely here)
            for widget in reversed(list(chat_container.query(ChatMessage))):
                if widget.role == "User": # Only reuse User messages
                    last_msg_widget = widget
                    break
            if last_msg_widget:
                message_text_from_input = last_msg_widget.message_text
                reuse_last_user_bubble = True
                loguru_logger.debug("Reusing last user message as input is empty.")
        except Exception as exc:
            loguru_logger.error("Failed to inspect last message for reuse: %s", exc, exc_info=True)

    if not message_text_from_input:
        loguru_logger.debug("Send Button: Empty message and no reusable user bubble in '%s'. Focusing input.", prefix)
        text_area.focus()
        return

    selected_provider = str(provider_widget.value) if provider_widget.value != Select.BLANK else None
    selected_model = str(model_widget.value) if model_widget.value != Select.BLANK else None
    system_prompt = system_prompt_widget.text
    temperature = safe_float(temp_widget.value, 0.7, "temperature") # Use imported safe_float
    top_p = safe_float(top_p_widget.value, 0.95, "top_p")
    min_p = safe_float(min_p_widget.value, 0.05, "min_p")
    top_k = safe_int(top_k_widget.value, 50, "top_k") # Use imported safe_int
    custom_prompt = ""  # Assuming this isn't used directly in chat send, but passed
    should_stream = False # Defaulting to False as per original logic

    llm_max_tokens_value = safe_int(llm_max_tokens_widget.value, 1024, "llm_max_tokens")
    llm_seed_value = safe_int(llm_seed_widget.value, None, "llm_seed") # None is a valid default
    llm_stop_value = [s.strip() for s in llm_stop_widget.value.split(',') if s.strip()] if llm_stop_widget.value.strip() else None
    llm_response_format_value = {"type": str(llm_response_format_widget.value)} if llm_response_format_widget.value != Select.BLANK else {"type": "text"}
    llm_n_value = safe_int(llm_n_widget.value, 1, "llm_n")
    llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
    llm_logprobs_value = llm_logprobs_widget.value
    llm_top_logprobs_value = safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs") if llm_logprobs_value else 0
    llm_presence_penalty_value = safe_float(llm_presence_penalty_widget.value, 0.0, "llm_presence_penalty")
    llm_frequency_penalty_value = safe_float(llm_frequency_penalty_widget.value, 0.0, "llm_frequency_penalty")
    llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None

    try:
        llm_logit_bias_text = llm_logit_bias_widget.text.strip()
        llm_logit_bias_value = json.loads(llm_logit_bias_text) if llm_logit_bias_text and llm_logit_bias_text != "{}" else None
    except json.JSONDecodeError:
        loguru_logger.warning(f"Invalid JSON in llm_logit_bias: '{llm_logit_bias_widget.text}'")
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Invalid JSON in LLM Logit Bias. Parameter not used."), role="System", classes="-error"))
        llm_logit_bias_value = None
    try:
        llm_tools_text = llm_tools_widget.text.strip()
        llm_tools_value = json.loads(llm_tools_text) if llm_tools_text and llm_tools_text != "[]" else None
    except json.JSONDecodeError:
        loguru_logger.warning(f"Invalid JSON in llm_tools: '{llm_tools_widget.text}'")
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Invalid JSON in LLM Tools. Parameter not used."), role="System", classes="-error"))
        llm_tools_value = None

    # --- 3. Basic Validation ---
    if not selected_provider:
        await chat_container.mount(ChatMessage(Text.from_markup("Please select an API Provider."), role="System", classes="-error")); return
    if not selected_model:
        await chat_container.mount(ChatMessage(Text.from_markup("Please select a Model."), role="System", classes="-error")); return
    if not app.API_IMPORTS_SUCCESSFUL: # Access as app attribute
        await chat_container.mount(ChatMessage(Text.from_markup("Error: Core API functions failed to load."), role="System", classes="-error"))
        loguru_logger.error("Attempted to send message, but API imports failed.")
        return

    # --- 4. Build Chat History for API ---
    # History should contain messages *before* the current user's input.
    # The current user's input (`message_text_from_input`) will be passed as the `message` param to `app.chat_wrapper`.
    chat_history_for_api: List[Dict[str, str]] = []
    try:
        # Iterate through all messages currently in the UI
        all_ui_messages = list(chat_container.query(ChatMessage))

        # Determine how many messages to actually include in history sent to API
        # (e.g., based on token limits or a fixed number)
        # For now, let's take all completed User/AI messages *before* any reused bubble

        messages_to_process_for_history = all_ui_messages
        if reuse_last_user_bubble and all_ui_messages:
            # If we are reusing the last bubble, it means it's already in the UI.
            # The history should include everything *before* that reused bubble.
            # Find the index of the last_msg_widget (which is the one being reused)
            try:
                # 'last_msg_widget' would have been set if reuse_last_user_bubble is True
                # This assumes last_msg_widget is still a valid reference from the reuse logic block
                idx_of_reused_msg = -1
                # Search for the widget instance if `last_msg_widget` is not directly available
                # or if we need to be more robust:
                temp_last_user_msg_widget = None
                for widget in reversed(all_ui_messages):
                    if widget.role == "User":
                        temp_last_user_msg_widget = widget
                        break
                if temp_last_user_msg_widget:
                    idx_of_reused_msg = all_ui_messages.index(temp_last_user_msg_widget)

                if idx_of_reused_msg != -1:
                    messages_to_process_for_history = all_ui_messages[:idx_of_reused_msg]
            except (ValueError, NameError): # NameError if last_msg_widget wasn't set, ValueError if not found
                 loguru_logger.warning("Could not definitively exclude reused message from history; sending full history.")
                 # Fallback: send all current UI messages as history; API might get duplicate of last user msg.
                 # `app.chat_wrapper` or `chat()` would need to handle this.
                 pass


        for msg_widget in messages_to_process_for_history:
            if msg_widget.role in ("User", "AI") and msg_widget.generation_complete:
                role_for_api = "assistant" if msg_widget.role == "AI" else "user"
                chat_history_for_api.append({"role": role_for_api, "content": msg_widget.message_text})

        loguru_logger.debug(f"Built chat history for API with {len(chat_history_for_api)} messages.")

    except Exception as e:
        loguru_logger.error(f"Failed to build chat history for API: {e}", exc_info=True)
        await chat_container.mount(ChatMessage(Text.from_markup("Internal Error: Could not retrieve chat history."), role="System", classes="-error"))
        return

    # --- 5. DB and Conversation ID Setup ---
    active_conversation_id = app.current_chat_conversation_id
    db = app.notes_service._get_db(app.notes_user_id) if app.notes_service else None
    user_msg_widget_instance: Optional[ChatMessage] = None # To hold the instance of the user message widget

    # --- 6. Mount User Message to UI ---
    if not reuse_last_user_bubble:
        user_msg_widget_instance = ChatMessage(message_text_from_input, role="User")
        await chat_container.mount(user_msg_widget_instance)
        loguru_logger.debug(f"Mounted new user message to UI: '{message_text_from_input[:50]}...'")

    # --- 7. Save User Message to DB (IF CHAT IS ALREADY PERSISTENT) ---
    if not app.current_chat_is_ephemeral and active_conversation_id and db:
        if not reuse_last_user_bubble and user_msg_widget_instance: # Only save if it's a newly added UI message
            try:
                loguru_logger.debug(f"Chat is persistent (ID: {active_conversation_id}). Saving user message to DB.")
                # Assuming no image for user messages sent via text input for now
                user_message_db_id = ccl.add_message_to_conversation(
                    db, conversation_id=active_conversation_id, sender="User", content=message_text_from_input,
                    image_data=None, image_mime_type=None # Placeholder for potential future image uploads
                )
                if user_message_db_id:
                    user_msg_widget_instance.message_id_internal = user_message_db_id
                    loguru_logger.debug(f"User message saved to DB with ID: {user_message_db_id}")
                else:
                    loguru_logger.error(f"Failed to save user message to DB for conversation {active_conversation_id}.")
            except Exception as e_add_msg:
                loguru_logger.error(f"Error saving user message to DB: {e_add_msg}", exc_info=True)
    elif app.current_chat_is_ephemeral:
        loguru_logger.debug("Chat is ephemeral. User message not saved to DB at this stage.")


    # --- 8. UI Updates (Clear input, scroll, focus) ---
    chat_container.scroll_end(animate=True) # Scroll after mounting user message
    text_area.clear()
    text_area.focus()

    # --- 9. API Key Fetching ---
    api_key_for_call = None
    if selected_provider: # Should always be true due to earlier check
        provider_settings_key = selected_provider.lower()
        provider_config_settings = app.app_config.get("api_settings", {}).get(provider_settings_key, {})

        if "api_key" in provider_config_settings:
            direct_config_key_checked = True
            config_api_key = provider_config_settings.get("api_key", "").strip()
            if config_api_key:
                api_key_for_call = config_api_key
                loguru_logger.debug(f"Using API key for '{selected_provider}' from config file field.")

        if not api_key_for_call: # If not found in direct 'api_key' field or it was empty
            env_var_name = provider_config_settings.get("api_key_env_var", "").strip()
            if env_var_name:
                env_api_key = os.environ.get(env_var_name, "").strip()
                if env_api_key:
                    api_key_for_call = env_api_key
                    loguru_logger.debug(f"Using API key for '{selected_provider}' from ENV var '{env_var_name}'.")
                else:
                    loguru_logger.debug(f"ENV var '{env_var_name}' for '{selected_provider}' not found or empty.")
            else:
                loguru_logger.debug(f"No 'api_key_env_var' specified for '{selected_provider}' in config.")

    providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]
    if selected_provider in providers_requiring_key and not api_key_for_call:
        loguru_logger.error(f"API Key for '{selected_provider}' is missing and required.")
        error_message_markup = (
            f"API Key for {selected_provider} is missing.\n\n"
            "Please add it to your config file under:\n"
            f"\\[api_settings.{selected_provider.lower()}\\]\n" # Ensure key matches config
            "api_key = \"YOUR_KEY\"\n\n"
            "Or set the environment variable specified by 'api_key_env_var' in the config for this provider."
        )
        await chat_container.mount(ChatMessage(Text.from_markup(error_message_markup), role="System", classes="-error"))
        # Clean up placeholder if one was accidentally about to be made or if logic changes
        if app.current_ai_message_widget and app.current_ai_message_widget.is_mounted:
            await app.current_ai_message_widget.remove()
            app.current_ai_message_widget = None
        return

    # --- 10. Mount Placeholder AI Message ---
    ai_placeholder_widget = ChatMessage(
        message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}",
        role="AI", generation_complete=False
    )
    await chat_container.mount(ai_placeholder_widget)
    chat_container.scroll_end(animate=False) # Scroll after mounting placeholder
    app.current_ai_message_widget = ai_placeholder_widget

    # --- 11. Prepare and Dispatch API Call via Worker ---
    loguru_logger.debug(f"Dispatching API call to worker. Current message: '{message_text_from_input[:50]}...', History items: {len(chat_history_for_api)}")
    worker_target = lambda: app.chat_wrapper(
        message=message_text_from_input, # Current user utterance
        history=chat_history_for_api,    # History *before* current utterance
        api_endpoint=selected_provider,
        api_key=api_key_for_call,
        custom_prompt=custom_prompt,
        temperature=temperature,
        system_message=system_prompt,
        streaming=should_stream,
        minp=min_p,
        model=selected_model,
        topp=top_p,
        topk=top_k,
        llm_max_tokens=llm_max_tokens_value,
        llm_seed=llm_seed_value,
        llm_stop=llm_stop_value,
        llm_response_format=llm_response_format_value,
        llm_n=llm_n_value,
        llm_user_identifier=llm_user_identifier_value,
        llm_logprobs=llm_logprobs_value,
        llm_top_logprobs=llm_top_logprobs_value,
        llm_logit_bias=llm_logit_bias_value,
        llm_presence_penalty=llm_presence_penalty_value,
        llm_frequency_penalty=llm_frequency_penalty_value,
        llm_tools=llm_tools_value,
        llm_tool_choice=llm_tool_choice_value,
        media_content={}, # Placeholder for now
        selected_parts=[], # Placeholder for now
        chatdict_entries=None, # Placeholder for now
        max_tokens=500, # This is the existing chatdict max_tokens, distinct from llm_max_tokens
        strategy="sorted_evenly" # Default or get from config/UI
    )
    app.run_worker(worker_target, name=f"API_Call_{prefix}", group="api_calls", thread=True,
                   description=f"Calling {selected_provider}")


async def handle_chat_action_button_pressed(app: 'TldwCli', button: Button, action_widget: ChatMessage) -> None:
    button_classes = button.classes
    message_text = action_widget.message_text  # This is the raw, unescaped text
    message_role = action_widget.role
    db = app.notes_service._get_db(app.notes_user_id) if app.notes_service else None

    if "edit-button" in button_classes:
        logging.info("Action: Edit clicked for %s message: '%s...'", message_role, message_text[:50])
        is_editing = getattr(action_widget, "_editing", False)
        static_text_widget: Static = action_widget.query_one(".message-text", Static)

        if not is_editing:  # Start editing
            current_text_for_editing = message_text  # Use the internally stored raw text
            static_text_widget.display = False
            editor = TextArea(text=current_text_for_editing, id="edit-area", classes="edit-area")
            editor.styles.width = "100%"
            await action_widget.mount(editor, before=static_text_widget)
            editor.focus()
            action_widget._editing = True
            button.label = get_char(EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT)
            logging.debug("Editing started.")
        else:  # Stop editing and save
            try:
                editor: TextArea = action_widget.query_one("#edit-area", TextArea)
                new_text = editor.text  # This is plain text from TextArea
                await editor.remove()

                action_widget.message_text = new_text  # Update internal raw text
                static_text_widget.update(escape_markup(new_text))  # Update display with escaped text
                static_text_widget.display = True
                action_widget._editing = False
                button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)  # Reset to Edit icon
                logging.debug("Editing finished. New length: %d", len(new_text))

                # Persist edit to DB if message has an ID
                if db and hasattr(action_widget, 'message_id_internal') and action_widget.message_id_internal:
                    try:
                        db.update_message_content(action_widget.message_id_internal, new_text)
                        loguru_logger.info(f"Message ID {action_widget.message_id_internal} content updated in DB.")
                        app.notify("Message edit saved to DB.", severity="information", timeout=2)
                    except Exception as e_db_update:
                        loguru_logger.error(
                            f"Failed to update message {action_widget.message_id_internal} in DB: {e_db_update}",
                            exc_info=True)
                        app.notify("Failed to save edit to DB.", severity="error")

            except QueryError:
                logging.error("Edit TextArea not found when stopping edit. Restoring original.")
                static_text_widget.update(escape_markup(message_text))  # Restore original escaped text
                static_text_widget.display = True
                action_widget._editing = False
                button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)
            except Exception as e_edit_stop:
                logging.error(f"Error stopping edit: {e_edit_stop}", exc_info=True)
                # Attempt to restore original state
                if 'static_text_widget' in locals():  # Check if queried
                    static_text_widget.update(escape_markup(message_text))
                    static_text_widget.display = True
                if hasattr(action_widget, '_editing'): action_widget._editing = False
                if 'button' in locals(): button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)


    elif "copy-button" in button_classes:
        logging.info("Action: Copy clicked for %s message: '%s...'", message_role, message_text[:50])
        app.copy_to_clipboard(message_text)  # message_text is already the raw, unescaped version
        button.label = get_char(EMOJI_COPIED, FALLBACK_COPIED) + "Copied"
        app.set_timer(1.5, lambda: setattr(button, "label", get_char(EMOJI_COPY, FALLBACK_COPY)))

    elif "speak-button" in button_classes:
        logging.info(f"Action: Speak clicked for {message_role} message: '{message_text[:50]}...'")
        # Actual TTS would go here. For UI feedback:
        try:
            text_widget = action_widget.query_one(".message-text", Static)
            original_display = text_widget.renderable  # Store to restore
            text_widget.update(Text.from_markup(f"[italic]Speaking: {escape_markup(message_text)}[/]"))
            # After TTS simulation/actual call:
            # app.set_timer(3, lambda: text_widget.update(original_display)) # Example restore
        except QueryError:
            logging.error("Could not find .message-text Static for speak action.")


    elif "thumb-up-button" in button_classes:
        logging.info(f"Action: Thumb Up clicked for {message_role} message.")
        button.label = "👍(OK)"
        # Add DB interaction for feedback if needed

    elif "thumb-down-button" in button_classes:
        logging.info(f"Action: Thumb Down clicked for {message_role} message.")
        button.label = "👎(OK)"
        # Add DB interaction for feedback if needed

    elif "delete-button" in button_classes:
        logging.info("Action: Delete clicked for %s message: '%s...'", message_role, message_text[:50])
        message_id_to_delete = getattr(action_widget, 'message_id_internal', None)
        try:
            await action_widget.remove()
            if action_widget is app.current_ai_message_widget:
                app.current_ai_message_widget = None

            if db and message_id_to_delete:
                try:
                    db.soft_delete_message(message_id_to_delete)  # Assuming soft delete
                    loguru_logger.info(f"Message ID {message_id_to_delete} soft-deleted from DB.")
                    app.notify("Message deleted.", severity="information", timeout=2)
                except Exception as e_db_delete:
                    loguru_logger.error(f"Failed to delete message {message_id_to_delete} from DB: {e_db_delete}",
                                        exc_info=True)
                    app.notify("Failed to delete message from DB.", severity="error")
        except Exception as exc:
            logging.error("Failed to delete message widget: %s", exc, exc_info=True)

    elif "regenerate-button" in button_classes and message_role == "AI":
        loguru_logger.info(
            f"Action: Regenerate clicked for AI message ID: {getattr(action_widget, 'message_id_internal', 'N/A')}")
        prefix = "chat"  # Assuming regeneration only happens in the main chat tab
        try:
            chat_container = app.query_one(f"#{prefix}-log", VerticalScroll)
        except QueryError:
            loguru_logger.error(f"Regenerate: Could not find chat container #{prefix}-log. Aborting.")
            app.notify("Error: Chat log not found for regeneration.", severity="error")
            return

        history_for_regeneration = []
        widgets_to_remove_after_regen_source = []  # Renamed for clarity
        found_target_ai_message_for_regen = False

        all_message_widgets_in_log = list(chat_container.query(ChatMessage))

        for msg_widget_iter in all_message_widgets_in_log:
            if msg_widget_iter is action_widget:  # This is the AI message we're regenerating
                found_target_ai_message_for_regen = True
                widgets_to_remove_after_regen_source.append(msg_widget_iter)
                # Don't add this AI message to history_for_regeneration
                continue

            if found_target_ai_message_for_regen:
                # All messages *after* the AI message being regenerated should also be removed
                widgets_to_remove_after_regen_source.append(msg_widget_iter)
            else:
                # This message is *before* the one we're regenerating
                if msg_widget_iter.role in ("User", "AI") and msg_widget_iter.generation_complete:
                    role_for_api = "assistant" if msg_widget_iter.role == "AI" else "user"
                    history_for_regeneration.append({"role": role_for_api, "content": msg_widget_iter.message_text})

        if not history_for_regeneration:
            loguru_logger.warning("Regenerate: No history found before the target AI message. Cannot regenerate.")
            app.notify("Cannot regenerate: No preceding messages found.", severity="warning")
            return

        loguru_logger.debug(
            f"Regenerate: History for regeneration ({len(history_for_regeneration)} messages): {history_for_regeneration}")

        for widget_to_remove_iter in widgets_to_remove_after_regen_source:
            loguru_logger.debug(
                f"Regenerate: Removing widget {widget_to_remove_iter.id} (Role: {widget_to_remove_iter.role}) from UI.")
            await widget_to_remove_iter.remove()

        if app.current_ai_message_widget in widgets_to_remove_after_regen_source:
            app.current_ai_message_widget = None

        # Fetch current chat settings (same as send-chat button logic)
        try:
            provider_widget_regen = app.query_one(f"#{prefix}-api-provider", Select)
            model_widget_regen = app.query_one(f"#{prefix}-api-model", Select)
            system_prompt_widget_regen = app.query_one(f"#{prefix}-system-prompt", TextArea)
            temp_widget_regen = app.query_one(f"#{prefix}-temperature", Input)
            top_p_widget_regen = app.query_one(f"#{prefix}-top-p", Input)
            min_p_widget_regen = app.query_one(f"#{prefix}-min-p", Input)
            top_k_widget_regen = app.query_one(f"#{prefix}-top-k", Input)
            # Full chat settings
            llm_max_tokens_widget_regen = app.query_one(f"#{prefix}-llm-max-tokens", Input)
            llm_seed_widget_regen = app.query_one(f"#{prefix}-llm-seed", Input)
            llm_stop_widget_regen = app.query_one(f"#{prefix}-llm-stop", Input)
            llm_response_format_widget_regen = app.query_one(f"#{prefix}-llm-response-format", Select)
            llm_n_widget_regen = app.query_one(f"#{prefix}-llm-n", Input)
            llm_user_identifier_widget_regen = app.query_one(f"#{prefix}-llm-user-identifier", Input)
            llm_logprobs_widget_regen = app.query_one(f"#{prefix}-llm-logprobs", Checkbox)
            llm_top_logprobs_widget_regen = app.query_one(f"#{prefix}-llm-top-logprobs", Input)
            llm_logit_bias_widget_regen = app.query_one(f"#{prefix}-llm-logit-bias", TextArea)
            llm_presence_penalty_widget_regen = app.query_one(f"#{prefix}-llm-presence-penalty", Input)
            llm_frequency_penalty_widget_regen = app.query_one(f"#{prefix}-llm-frequency-penalty", Input)
            llm_tools_widget_regen = app.query_one(f"#{prefix}-llm-tools", TextArea)
            llm_tool_choice_widget_regen = app.query_one(f"#{prefix}-llm-tool-choice", Input)
        except QueryError as e_query_regen:
            loguru_logger.error(f"Regenerate: Could not find UI settings widgets for '{prefix}': {e_query_regen}")
            await chat_container.mount(
                ChatMessage(Text.from_markup("[bold red]Internal Error:[/]\nMissing UI settings for regeneration."),
                            role="System", classes="-error"))
            return

        selected_provider_regen = str(
            provider_widget_regen.value) if provider_widget_regen.value != Select.BLANK else None
        selected_model_regen = str(model_widget_regen.value) if model_widget_regen.value != Select.BLANK else None
        system_prompt_regen = system_prompt_widget_regen.text
        temperature_regen = safe_float(temp_widget_regen.value, 0.7, "temperature")
        top_p_regen = safe_float(top_p_widget_regen.value, 0.95, "top_p")
        min_p_regen = safe_float(min_p_widget_regen.value, 0.05, "min_p")
        top_k_regen = app._safe_int(top_k_widget_regen.value, 50, "top_k")

        llm_max_tokens_value_regen = app._safe_int(llm_max_tokens_widget_regen.value, 1024, "llm_max_tokens")
        llm_seed_value_regen = app._safe_int(llm_seed_widget_regen.value, None, "llm_seed")
        llm_stop_value_regen = [s.strip() for s in
                                llm_stop_widget_regen.value.split(',')] if llm_stop_widget_regen.value.strip() else None
        llm_response_format_value_regen = {"type": str(
            llm_response_format_widget_regen.value)} if llm_response_format_widget_regen.value != Select.BLANK else {
            "type": "text"}
        llm_n_value_regen = app._safe_int(llm_n_widget_regen.value, 1, "llm_n")
        llm_user_identifier_value_regen = llm_user_identifier_widget_regen.value.strip() or None
        llm_logprobs_value_regen = llm_logprobs_widget_regen.value
        llm_top_logprobs_value_regen = app._safe_int(llm_top_logprobs_widget_regen.value, 0,
                                                     "llm_top_logprobs") if llm_logprobs_value_regen else 0
        llm_presence_penalty_value_regen = safe_float(llm_presence_penalty_widget_regen.value, 0.0,
                                                           "llm_presence_penalty")
        llm_frequency_penalty_value_regen = safe_float(llm_frequency_penalty_widget_regen.value, 0.0,
                                                            "llm_frequency_penalty")
        llm_tool_choice_value_regen = llm_tool_choice_widget_regen.value.strip() or None
        try:
            llm_logit_bias_text_regen = llm_logit_bias_widget_regen.text.strip()
            llm_logit_bias_value_regen = json.loads(
                llm_logit_bias_text_regen) if llm_logit_bias_text_regen and llm_logit_bias_text_regen != "{}" else None
        except json.JSONDecodeError:
            llm_logit_bias_value_regen = None
        try:
            llm_tools_text_regen = llm_tools_widget_regen.text.strip()
            llm_tools_value_regen = json.loads(
                llm_tools_text_regen) if llm_tools_text_regen and llm_tools_text_regen != "[]" else None
        except json.JSONDecodeError:
            llm_tools_value_regen = None

        if not selected_provider_regen or not selected_model_regen:
            loguru_logger.warning("Regenerate: Provider or model not selected.")
            await chat_container.mount(
                ChatMessage(Text.from_markup("[bold red]Error:[/]\nPlease select provider and model for regeneration."),
                            role="System", classes="-error"))
            return

        api_key_for_regen = None  # API Key fetching logic (same as send-chat)
        provider_settings_key_regen = selected_provider_regen.lower()
        provider_config_settings_regen = app.app_config.get("api_settings", {}).get(provider_settings_key_regen, {})
        if provider_config_settings_regen.get("api_key"):
            api_key_for_regen = provider_config_settings_regen["api_key"]
        elif provider_config_settings_regen.get("api_key_env_var"):
            api_key_for_regen = os.environ.get(provider_config_settings_regen["api_key_env_var"])

        providers_requiring_key_regen = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter",
                                         "HuggingFace", "DeepSeek"]
        if selected_provider_regen in providers_requiring_key_regen and not api_key_for_regen:
            loguru_logger.error(
                f"Regenerate aborted: API Key for required provider '{selected_provider_regen}' is missing.")
            await chat_container.mount(ChatMessage(
                Text.from_markup(f"[bold red]API Key for {selected_provider_regen} is missing for regeneration.[/]"),
                role="System", classes="-error"))
            return

        ai_placeholder_widget_regen = ChatMessage(
            message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)} (Regenerating...)",
            role="AI", generation_complete=False
        )
        await chat_container.mount(ai_placeholder_widget_regen)
        chat_container.scroll_end(animate=False)
        app.current_ai_message_widget = ai_placeholder_widget_regen

        # The "message" to chat_wrapper is empty because we're using the history
        worker_target_regen = lambda: app.chat_wrapper(
            message="", history=history_for_regeneration, api_endpoint=selected_provider_regen,
            api_key=api_key_for_regen,
            custom_prompt="", temperature=temperature_regen, system_message=system_prompt_regen, streaming=False,
            # Or get from UI
            minp=min_p_regen, model=selected_model_regen, topp=top_p_regen, topk=top_k_regen,
            llm_max_tokens=llm_max_tokens_value_regen, llm_seed=llm_seed_value_regen, llm_stop=llm_stop_value_regen,
            llm_response_format=llm_response_format_value_regen, llm_n=llm_n_value_regen,
            llm_user_identifier=llm_user_identifier_value_regen, llm_logprobs=llm_logprobs_value_regen,
            llm_top_logprobs=llm_top_logprobs_value_regen, llm_logit_bias=llm_logit_bias_value_regen,
            llm_presence_penalty=llm_presence_penalty_value_regen,
            llm_frequency_penalty=llm_frequency_penalty_value_regen,
            llm_tools=llm_tools_value_regen, llm_tool_choice=llm_tool_choice_value_regen,
            media_content={}, selected_parts=[], chatdict_entries=None, max_tokens=500, strategy="sorted_evenly"
        )
        app.run_worker(worker_target_regen, name=f"API_Call_{prefix}_regenerate", group="api_calls", thread=True,
                       description=f"Regenerating for {selected_provider_regen}")


async def handle_chat_new_conversation_button_pressed(app: 'TldwCli') -> None:
    loguru_logger.info("New Chat button pressed.")
    try:
        chat_log_widget = app.query_one("#chat-log", VerticalScroll)
        await chat_log_widget.remove_children()
    except QueryError:
        loguru_logger.error("Failed to find #chat-log to clear.")

    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True  # This triggers watcher to update UI elements

    try:
        # Watcher should handle most of this, but explicit clearing is safer
        app.query_one("#chat-conversation-title-input", Input).value = ""
        app.query_one("#chat-conversation-keywords-input", TextArea).text = ""
        app.query_one("#chat-conversation-uuid-display", Input).value = "Ephemeral Chat"
        app.query_one(TitleBar).reset_title()
        app.query_one("#chat-input", TextArea).focus()
    except QueryError as e:
        loguru_logger.error(f"UI component not found during new chat setup: {e}")


async def handle_chat_save_current_chat_button_pressed(app: 'TldwCli') -> None:
    loguru_logger.info("Save Current Chat button pressed.")
    if not (app.current_chat_is_ephemeral and app.current_chat_conversation_id is None):
        loguru_logger.warning("Chat not eligible for saving (not ephemeral or already has ID).")
        app.notify("This chat is already saved or cannot be saved in its current state.", severity="warning")
        return

    if not app.notes_service:
        app.notify("Database service not available.", severity="error")
        loguru_logger.error("Notes service not available for saving chat.")
        return

    db = app.notes_service._get_db(app.notes_user_id)
    try:
        chat_log_widget = app.query_one("#chat-log", VerticalScroll)
    except QueryError:
        app.notify("Chat log not found, cannot save.", severity="error")
        return

    messages_in_log = list(chat_log_widget.query(ChatMessage))

    if not messages_in_log:
        app.notify("Nothing to save in an empty chat.", severity="warning")
        return

    ui_messages_to_save: List[Dict[str, Any]] = []
    for msg_widget in messages_in_log:
        if msg_widget.generation_complete:
            ui_messages_to_save.append({
                'sender': msg_widget.role,
                'content': msg_widget.message_text,
                'image_data': getattr(msg_widget, 'image_data', None),
                'image_mime_type': getattr(msg_widget, 'image_mime_type', None),
                'timestamp': getattr(msg_widget, 'timestamp', datetime.now().isoformat())  # Add timestamp if available
            })

    default_title = f"Saved Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if ui_messages_to_save and ui_messages_to_save[0]['sender'] == "User":
        content_preview = ui_messages_to_save[0]['content'][:30].strip()
        if content_preview: default_title = f"Chat: {content_preview}..."

    try:
        new_conv_id = ccl.create_conversation(
            db, title=default_title, character_id=ccl.DEFAULT_CHARACTER_ID,  # For regular chats
            initial_messages=ui_messages_to_save, system_keywords=["__regular_chat", "__saved_ephemeral"]
        )

        if new_conv_id:
            app.current_chat_conversation_id = new_conv_id
            app.current_chat_is_ephemeral = False  # Now it's saved, triggers watcher
            app.notify("Chat saved successfully!", severity="information")

            # Watcher for current_chat_is_ephemeral will enable/disable buttons.
            # We need to populate the fields for the newly saved chat.
            try:
                app.query_one("#chat-conversation-title-input", Input).value = default_title
                app.query_one("#chat-conversation-uuid-display", Input).value = new_conv_id
                # Keywords are not set automatically on first save, user can add them.
                app.query_one("#chat-conversation-keywords-input", TextArea).text = ""
                app.query_one(TitleBar).update_title(f"Chat - {default_title}")

                # Update message_id_internal for all messages in the chat log from the DB
                # This is important if the user wants to edit/delete them later from this session.
                # This requires create_conversation to return message IDs or a way to fetch them.
                # Assuming ccl.create_conversation handles setting message IDs if it saves them.
                # If not, a follow-up fetch would be needed. For now, we assume it's handled or IDs are set upon message creation.

            except QueryError as e:
                loguru_logger.error(f"Error updating UI after saving chat: {e}")
        else:
            app.notify("Failed to save chat (no ID returned).", severity="error")
    except Exception as e_save_chat:
        loguru_logger.error(f"Exception while saving chat: {e_save_chat}", exc_info=True)
        app.notify(f"Error saving chat: {e_save_chat}", severity="error")


async def handle_chat_save_details_button_pressed(app: 'TldwCli') -> None:
    loguru_logger.info("Save conversation details button pressed.")
    if app.current_chat_is_ephemeral or not app.current_chat_conversation_id:
        loguru_logger.warning("Cannot save details for an ephemeral or non-existent chat.")
        app.notify("No active saved conversation to update details for.", severity="warning")
        return

    if not app.notes_service:
        loguru_logger.error("Notes service not available for saving chat details.")
        app.notify("Database service not available.", severity="error")
        return

    conversation_id = app.current_chat_conversation_id
    db = app.notes_service._get_db(app.notes_user_id)

    try:
        title_input = app.query_one("#chat-conversation-title-input", Input)
        keywords_input_widget = app.query_one("#chat-conversation-keywords-input", TextArea)

        new_title = title_input.value.strip()
        new_keywords_str = keywords_input_widget.text.strip()

        conv_details = db.get_conversation_by_id(conversation_id)
        if not conv_details:
            loguru_logger.error(f"Conversation {conversation_id} not found in DB for saving details.")
            app.notify("Error: Conversation not found in database.", severity="error")
            return

        current_version = conv_details.get('version')
        if current_version is None:
            loguru_logger.error(f"Conversation {conversation_id} is missing version information.")
            app.notify("Error: Conversation version information is missing.", severity="error")
            return

        title_changed = False
        if new_title != conv_details.get('title', ''):  # Compare with empty string if title is None
            db.update_conversation(conversation_id, {'title': new_title}, current_version)
            current_version += 1  # Version is now incremented for the conversation row
            title_changed = True
            loguru_logger.info(f"Title updated for conversation {conversation_id}. New version: {current_version}")
            try:
                app.query_one(TitleBar).update_title(f"Chat - {new_title}")
            except QueryError:
                loguru_logger.error("Failed to update TitleBar after title save.")

        # Keywords Update (from app.py, adapted)
        all_db_keywords_list = db.get_keywords_for_conversation(conversation_id)
        db_user_keywords_map = {kw['keyword']: kw['id'] for kw in all_db_keywords_list if
                                not kw['keyword'].startswith("__")}
        db_user_keywords_set = set(db_user_keywords_map.keys())
        ui_user_keywords_set = {kw.strip() for kw in new_keywords_str.split(',') if
                                kw.strip() and not kw.strip().startswith("__")}

        keywords_to_add = ui_user_keywords_set - db_user_keywords_set
        keywords_to_remove_text = db_user_keywords_set - ui_user_keywords_set
        keywords_changed = False

        for keyword_text_add in keywords_to_add:
            keyword_detail_add = db.get_keyword_by_text(keyword_text_add)  # Does not take user_id
            keyword_id_to_link = None
            if not keyword_detail_add:  # Keyword doesn't exist globally
                added_kw_id = db.add_keyword(keyword_text_add)  # Takes no user_id, returns int ID
                if isinstance(added_kw_id, int):
                    keyword_id_to_link = added_kw_id
                else:
                    logging.error(f"Failed to add keyword '{keyword_text_add}', received: {added_kw_id}"); continue
            else:
                keyword_id_to_link = keyword_detail_add['id']

            if keyword_id_to_link:
                db.link_conversation_to_keyword(conversation_id, keyword_id_to_link)
                keywords_changed = True

        for keyword_text_remove in keywords_to_remove_text:
            keyword_id_to_unlink = db_user_keywords_map.get(keyword_text_remove)
            if keyword_id_to_unlink:
                db.unlink_conversation_from_keyword(conversation_id, keyword_id_to_unlink)
                keywords_changed = True

        if title_changed or keywords_changed:
            app.notify("Conversation details saved!", severity="information", timeout=3)
            # Refresh keywords in UI to reflect any changes
            final_db_keywords_after_save = db.get_keywords_for_conversation(conversation_id)
            final_user_keywords_after_save = [kw['keyword'] for kw in final_db_keywords_after_save if
                                              not kw['keyword'].startswith("__")]
            keywords_input_widget.text = ", ".join(final_user_keywords_after_save)
        else:
            app.notify("No changes to save.", severity="information", timeout=2)

    except QueryError as e_query:
        loguru_logger.error(f"Save Conversation Details: UI component not found: {e_query}", exc_info=True)
        app.notify("Error accessing UI fields.", severity="error", timeout=3)
    except ConflictError as e_conflict:
        loguru_logger.error(f"Conflict saving conversation details for {conversation_id}: {e_conflict}", exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload.", severity="error", timeout=5)
    except CharactersRAGDBError as e_db:  # More generic DB error
        loguru_logger.error(f"DB error saving conversation details for {conversation_id}: {e_db}", exc_info=True)
        app.notify("Database error saving details.", severity="error", timeout=3)
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error saving conversation details for {conversation_id}: {e_unexp}",
                            exc_info=True)
        app.notify("Unexpected error saving details.", severity="error", timeout=3)


async def handle_chat_load_selected_button_pressed(app: 'TldwCli') -> None:
    loguru_logger.info("Load selected chat button pressed.")
    try:
        results_list_view = app.query_one("#chat-conversation-search-results-list", ListView)
        highlighted_item = results_list_view.highlighted_child
        if not (highlighted_item and hasattr(highlighted_item, 'conversation_id')):
            app.notify("No chat selected to load.", severity="warning")
            loguru_logger.info("No conversation selected in the list to load.")
            return

        loaded_conversation_id = highlighted_item.conversation_id
        loguru_logger.info(f"Attempting to load and display conversation ID: {loaded_conversation_id}")

        # _display_conversation_in_chat_tab handles UI updates and history loading
        await display_conversation_in_chat_tab_ui(app, loaded_conversation_id)
        app.current_chat_is_ephemeral = False  # A loaded chat is persistent

        # Notification is usually handled by _display_conversation_in_chat_tab or its callees
        # app.notify(f"Chat '{getattr(highlighted_item, 'conversation_title', 'Untitled')}' loaded.", severity="information")

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found for loading chat: {e_query}", exc_info=True)
        app.notify("Error accessing UI for loading chat.", severity="error")
    except CharactersRAGDBError as e_db:
        loguru_logger.error(f"Database error loading chat: {e_db}", exc_info=True)
        app.notify("Database error loading chat.", severity="error")
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error loading chat: {e_unexp}", exc_info=True)
        app.notify("Unexpected error loading chat.", severity="error")


async def perform_chat_conversation_search(app: 'TldwCli') -> None:
    loguru_logger.debug("Performing chat conversation search...")
    try:
        search_bar = app.query_one("#chat-conversation-search-bar", Input)
        search_term = search_bar.value.strip()

        include_char_chats_checkbox = app.query_one("#chat-conversation-search-include-character-checkbox", Checkbox)
        include_character_chats = include_char_chats_checkbox.value  # Currently unused in DB query, filtered client side

        all_chars_checkbox = app.query_one("#chat-conversation-search-all-characters-checkbox", Checkbox)
        search_all_characters = all_chars_checkbox.value

        char_filter_select = app.query_one("#chat-conversation-search-character-filter-select", Select)
        selected_character_id_filter = char_filter_select.value if not char_filter_select.disabled and char_filter_select.value != Select.BLANK else None

        results_list_view = app.query_one("#chat-conversation-search-results-list", ListView)
        await results_list_view.clear()

        if not app.notes_service:
            loguru_logger.error("Notes service not available for conversation search.")
            await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
            return

        db = app.notes_service._get_db(app.notes_user_id)
        conversations: List[Dict[str, Any]] = []

        # The DB query `search_conversations_by_title` uses character_id=None for "all characters"
        # or a specific character_id. It doesn't have a separate flag for "regular chats only".
        # The `include_character_chats` checkbox logic might need client-side filtering if
        # "regular chats" mean `character_id IS NULL` specifically.

        # Simplified logic: if a specific character is chosen, filter by it. Otherwise, search all.
        # The `include_character_chats` primarily acts as a conceptual filter for now,
        # unless `search_conversations_by_title` is enhanced.

        effective_character_id_for_search = None
        if include_character_chats:
            if not search_all_characters and selected_character_id_filter:
                effective_character_id_for_search = selected_character_id_filter
            # if search_all_characters or selected_character_id_filter is None, effective_character_id_for_search remains None (search all)
        else:  # Only "regular" (non-character) chats. This requires DB support or client filter.
            loguru_logger.info(
                "Searching for regular chats only (character_id is NULL). This may require DB method enhancement or client-side filter.")
            # For now, let's assume search_conversations_by_title with character_id=ccl.DEFAULT_CHARACTER_ID
            # or some other marker for "regular" if your DB is structured that way.
            # Or if DEFAULT_CHARACTER_ID implies non-specific character chats
            effective_character_id_for_search = ccl.DEFAULT_CHARACTER_ID  # Placeholder assumption
            # A more robust way for "character_id IS NULL" would be a dedicated DB method or flag.

        loguru_logger.debug(
            f"Searching conversations. Term: '{search_term}', CharID for DB: {effective_character_id_for_search}, IncludeCharFlag: {include_character_chats}")
        conversations = db.search_conversations_by_title(
            title_query=search_term,
            character_id=effective_character_id_for_search,  # This will be None if searching all/all_chars checked
            limit=100
        )

        # If include_character_chats is False, and the DB query couldn't filter by "IS NULL"
        # we might need to filter here:
        if not include_character_chats and conversations:
            # This assumes regular chats are those associated with DEFAULT_CHARACTER_ID
            # or if your DB uses NULL for truly "no character" chats. Adjust as needed.
            # conversations = [conv for conv in conversations if conv.get('character_id') == ccl.DEFAULT_CHARACTER_ID or conv.get('character_id') is None]
            # For now, we assume the DB call with DEFAULT_CHARACTER_ID handled this.
            pass

        if not conversations:
            await results_list_view.append(ListItem(Label("No conversations found.")))
        else:
            for conv_data in conversations:
                title_str = conv_data.get('title') or f"Chat ID: {conv_data['id'][:8]}..."
                # Optionally, prefix with character name if not already part of title logic
                # char_id_of_conv = conv_data.get('character_id')
                # if char_id_of_conv and char_id_of_conv != ccl.DEFAULT_CHARACTER_ID: # Example: don't prefix for default
                #     char_info = db.get_character_card_by_id(char_id_of_conv)
                #     if char_info and char_info.get('name'):
                #         title_str = f"[{char_info['name']}] {title_str}"

                item = ListItem(Label(title_str))
                item.conversation_id = conv_data['id']
                item.conversation_title = conv_data.get('title')  # Store for potential use
                # item.conversation_keywords = conv_data.get('keywords') # Not directly available from search_conversations_by_title
                await results_list_view.append(item)
        loguru_logger.info(f"Conversation search yielded {len(conversations)} results for display.")

    except QueryError as e_query:
        loguru_logger.error(f"UI component not found during conversation search: {e_query}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: UI component missing.")))
            except:
                pass
    except CharactersRAGDBError as e_db:
        loguru_logger.error(f"Database error during conversation search: {e_db}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: Database search failed.")))
            except:
                pass
    except Exception as e_unexp:
        loguru_logger.error(f"Unexpected error during conversation search: {e_unexp}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            try:
                await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))
            except:
                pass


async def handle_chat_conversation_search_bar_changed(app: 'TldwCli', event_value: str) -> None:
    if app._conversation_search_timer:
        app._conversation_search_timer.stop()  # Corrected: Use stop()
    app._conversation_search_timer = app.set_timer(
        0.5,
        lambda: perform_chat_conversation_search(app)
    )


async def handle_chat_search_checkbox_changed(app: 'TldwCli', checkbox_id: str, value: bool) -> None:
    
    loguru_logger.debug(f"Chat search checkbox '{checkbox_id}' changed to {value}")

    if checkbox_id == "chat-conversation-search-all-characters-checkbox":
        try:
            char_filter_select = app.query_one("#chat-conversation-search-character-filter-select", Select)
            char_filter_select.disabled = value
            if value:
                char_filter_select.value = Select.BLANK  # Clear selection when "All" is checked
        except QueryError as e:
            loguru_logger.error(f"Error accessing character filter select: {e}", exc_info=True)

    # Trigger a new search based on any checkbox change that affects the filter
    await perform_chat_conversation_search(app)


async def display_conversation_in_chat_tab_ui(app: 'TldwCli', conversation_id: str):
    if not app.notes_service:
        logging.error("Notes service unavailable, cannot display conversation in chat tab.")
        # Potentially update UI to show an error
        return

    db = app.notes_service._get_db(app.notes_user_id)
    conv_details_disp = db.get_conversation_by_id(conversation_id)

    if not conv_details_disp:
        logging.error(f"Cannot display conversation: Details for ID {conversation_id} not found.")
        app.notify(f"Error: Could not load chat {conversation_id}.", severity="error")
        # Update UI to reflect error state
        try:
            app.query_one("#chat-conversation-title-input", Input).value = "Error: Not Found"
            app.query_one("#chat-conversation-keywords-input", TextArea).text = ""
            app.query_one("#chat-conversation-uuid-display", Input).value = conversation_id
            app.query_one(TitleBar).update_title(f"Chat - Error Loading")
            chat_log_err = app.query_one("#chat-log", VerticalScroll)
            await chat_log_err.remove_children()
            await chat_log_err.mount(ChatMessage(Text.from_markup("[bold red]Failed to load conversation details.[/]"), role="System", classes="-error"))
        except QueryError as qe_err_disp: logging.error(f"UI component missing during error display for conv {conversation_id}: {qe_err_disp}")
        return

    app.current_chat_conversation_id = conversation_id # Ensure reactive is set for context
    app.current_chat_is_ephemeral = False # Loaded chats are not ephemeral

    try:
        app.query_one("#chat-conversation-title-input", Input).value = conv_details_disp.get('title', '')
        app.query_one("#chat-conversation-uuid-display", Input).value = conversation_id

        keywords_input_disp = app.query_one("#chat-conversation-keywords-input", TextArea)
        all_keywords_list_disp = db.get_keywords_for_conversation(conversation_id)
        visible_keywords_disp = [kw['keyword'] for kw in all_keywords_list_disp if not kw['keyword'].startswith("__")]
        keywords_input_disp.text = ", ".join(visible_keywords_disp)

        app.query_one(TitleBar).update_title(f"Chat - {conv_details_disp.get('title', 'Untitled Conversation')}")

        chat_log_widget_disp = app.query_one("#chat-log", VerticalScroll)
        await app._load_branched_conversation_history(conversation_id, chat_log_widget_disp)

        app.query_one("#chat-input", TextArea).focus()
        app.notify(f"Chat '{conv_details_disp.get('title', 'Untitled')}' loaded.", severity="information", timeout=3)
    except QueryError as qe_disp_main:
        logging.error(f"UI component missing during display_conversation for {conversation_id}: {qe_disp_main}")
        app.notify("Error updating UI for loaded chat.", severity="error")
    logging.info(f"Displayed conversation '{conv_details_disp.get('title', 'Untitled')}' (ID: {conversation_id}) in chat tab.")

    logging.info(
        f"Displayed conversation '{conv_details_disp.get('title', 'Untitled')}' (ID: {conversation_id}) in chat tab.")


async def load_branched_conversation_history_ui(app: 'TldwCli', target_conversation_id: str, chat_log_widget: VerticalScroll):
    """
    Loads the complete message history for a given conversation_id,
    tracing back through parent branches to the root if necessary.
    """
    if not app.notes_service:
        logging.error("Notes service not available for loading branched history.")
        await chat_log_widget.mount(
            ChatMessage("Error: Notes service unavailable.", role="System", classes="-error"))
        return

    db = app.notes_service._get_db(app.notes_user_id)
    await chat_log_widget.remove_children()
    logging.debug(f"Loading branched history for target_conversation_id: {target_conversation_id}")

    # 1. Trace path from target_conversation_id up to its root,
    #    collecting (conversation_id, fork_message_id_in_parent_that_started_this_segment)
    #    The 'fork_message_id_in_parent' is what we need to stop at when loading the parent's messages.
    path_segments_info = []  # Stores (conv_id, fork_msg_id_in_parent)

    current_conv_id_for_path = target_conversation_id
    while current_conv_id_for_path:
        conv_details = db.get_conversation_by_id(current_conv_id_for_path)
        if not conv_details:
            logging.error(f"Path tracing failed: Conversation {current_conv_id_for_path} not found.")
            await chat_log_widget.mount(
                ChatMessage(f"Error: Conversation segment {current_conv_id_for_path} not found.", role="System",
                            classes="-error"))
            return  # Stop if a segment is missing

        path_segments_info.append({
            "id": conv_details['id'],
            "forked_from_message_id": conv_details.get('forked_from_message_id'),
            # ID of message in PARENT where THIS conv started
            "parent_conversation_id": conv_details.get('parent_conversation_id')
        })
        current_conv_id_for_path = conv_details.get('parent_conversation_id')

    path_segments_info.reverse()  # Now path_segments_info is from root-most to target_conversation_id

    all_messages_to_display = []
    for i, segment_info in enumerate(path_segments_info):
        segment_conv_id = segment_info['id']

        # Get all messages belonging to this specific segment_conv_id
        messages_this_segment = db.get_messages_for_conversation(
            segment_conv_id,
            order_by_timestamp="ASC",
            limit=10000  # Effectively all messages for this segment
        )

        # If this segment is NOT the last one in the path, it means it was forked FROM.
        # We need to know where the NEXT segment (its child) forked from THIS segment.
        # The 'forked_from_message_id' of the *next* segment is the message_id in *this* segment.
        stop_at_message_id_for_this_segment = None
        if (i + 1) < len(path_segments_info):  # If there is a next segment
            next_segment_info = path_segments_info[i + 1]
            # next_segment_info['forked_from_message_id'] is the message in current segment_conv_id
            # from which the next_segment_info['id'] was forked.
            stop_at_message_id_for_this_segment = next_segment_info['forked_from_message_id']

        for msg_data in messages_this_segment:
            all_messages_to_display.append(msg_data)
            if stop_at_message_id_for_this_segment and msg_data['id'] == stop_at_message_id_for_this_segment:
                logging.debug(f"Stopping message load for segment {segment_conv_id} at fork point {msg_data['id']}")
                break  # Stop adding messages from this segment, as the next segment takes over

    # Now mount all collected messages
    logging.debug(f"Total messages collected for display: {len(all_messages_to_display)}")
    for msg_data in all_messages_to_display:
        image_data_for_widget = msg_data.get('image_data')
        chat_message_widget = ChatMessage(
            message=msg_data['content'],
            role=msg_data['sender'],
            timestamp=msg_data.get('timestamp'),
            image_data=image_data_for_widget,
            image_mime_type=msg_data.get('image_mime_type'),
            message_id=msg_data['id']
        )
        await chat_log_widget.mount(chat_message_widget)

    if chat_log_widget.is_mounted:
        chat_log_widget.scroll_end(animate=False)
    logging.info(
        f"Loaded {len(all_messages_to_display)} messages for conversation {target_conversation_id} (including history).")

#
# End of chat_events.py
########################################################################################################################
