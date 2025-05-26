# conv_char_events.py
# Description:
#
# Imports
import json  # For export
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Any, List, Dict, Union
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from textual.widgets import (
    Input, ListView, TextArea, Label, Collapsible, Select, Button, Static, ListItem
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError
from rich.text import Text # For displaying messages if needed
#
# Local Imports
from ..Widgets.chat_message import ChatMessage # If CCP tab displays ChatMessage widgets
from ..Character_Chat import Character_Chat_Lib as ccl
from ..Prompt_Management import Prompts_Interop as prompts_interop
from ..DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError # For specific error handling
#
if TYPE_CHECKING:
    from ..app import TldwCli
    # from ..app import API_IMPORTS_SUCCESSFUL # If needed directly
#
########################################################################################################################
#
# Functions:

########################################################################################################################
#
# Helper Functions (specific to CCP tab logic, moved from app.py)
#
########################################################################################################################

########################################################################################################################
#
# Helper Functions (specific to CCP tab logic, moved from app.py)
#
########################################################################################################################

async def populate_ccp_character_select(app: 'TldwCli') -> None:
    """Populates the character selection dropdown in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Attempting to populate #conv-char-character-select dropdown.")
    if not app.notes_service:
        logger.error("Notes service not available, cannot populate character select for CCP tab.")
        try:
            char_select_widget = app.query_one("#conv-char-character-select", Select)
            char_select_widget.set_options([("Service Unavailable", Select.BLANK)])
            char_select_widget.value = Select.BLANK
            char_select_widget.prompt = "Service Unavailable"
        except QueryError:
            logger.error("Failed to find #conv-char-character-select to show service error.")
        return

    try:
        db = app.notes_service._get_db(app.notes_user_id)
        character_cards = db.list_character_cards(limit=1000)

        options = []
        if character_cards:
            options = [(char['name'], char['id']) for char in character_cards if
                       char.get('name') and char.get('id') is not None]

        char_select_widget = app.query_one("#conv-char-character-select", Select)

        if options:
            char_select_widget.set_options(options)
            char_select_widget.prompt = "Select Character..."
            logger.info(f"Populated #conv-char-character-select with {len(options)} characters.")
        else:
            char_select_widget.set_options([("No characters found", Select.BLANK)])
            char_select_widget.value = Select.BLANK
            char_select_widget.prompt = "No characters available"
            logger.info("No characters found to populate #conv-char-character-select.")

    except QueryError as e_query:
        logger.error(f"Failed to find #conv-char-character-select widget: {e_query}", exc_info=True)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error populating #conv-char-character-select: {e_db}", exc_info=True)
        try:
            char_select_widget_err = app.query_one("#conv-char-character-select", Select)
            char_select_widget_err.set_options([("Error Loading Characters", Select.BLANK)])
            char_select_widget_err.prompt = "Error Loading"
        except QueryError:
            pass
    except Exception as e_unexp:
        logger.error(f"Unexpected error populating #conv-char-character-select: {e_unexp}", exc_info=True)
        try:
            char_select_widget_unexp = app.query_one("#conv-char-character-select", Select)
            char_select_widget_unexp.set_options([("Error Loading (Unexpected)", Select.BLANK)])
            char_select_widget_unexp.prompt = "Error Loading"
        except QueryError:
            pass


async def populate_ccp_prompts_list_view(app: 'TldwCli', search_term: Optional[str] = None) -> None:
    """Populates the prompts list view in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    if not app.prompts_service_initialized:
        try:
            list_view_prompt_err = app.query_one("#ccp-prompts-listview", ListView)
            await list_view_prompt_err.clear()
            await list_view_prompt_err.append(ListItem(Label("Prompts service not available.")))
        except QueryError:
            logger.error("Failed to find #ccp-prompts-listview to show service error.")
        return

    try:
        list_view_prompt = app.query_one("#ccp-prompts-listview", ListView)
        await list_view_prompt.clear()
        results: List[Dict[str, Any]] = []

        if search_term:
            results_tuple = prompts_interop.search_prompts(
                search_query=search_term,
                search_fields=["name", "details", "keywords"],
                page=1, results_per_page=100,
                include_deleted=False
            )
            results = results_tuple[0] if results_tuple else []
        else:
            results_tuple = prompts_interop.list_prompts(page=1, per_page=100, include_deleted=False)
            results = results_tuple[0] if results_tuple else []

        if not results:
            await list_view_prompt.append(ListItem(Label("No prompts found.")))
        else:
            for prompt_data in results:
                display_name = prompt_data.get('name', 'Unnamed Prompt')
                item = ListItem(Label(display_name))
                item.prompt_id = prompt_data.get('id')
                item.prompt_uuid = prompt_data.get('uuid')
                await list_view_prompt.append(item)
        logger.info(f"Populated CCP prompts list. Search: '{search_term}', Found: {len(results)}")
    except QueryError as e_query:
        logger.error(f"UI component error populating CCP prompts list: {e_query}", exc_info=True)
    except (prompts_interop.DatabaseError, RuntimeError) as e_prompt_service:
        logger.error(f"Error populating CCP prompts list: {e_prompt_service}", exc_info=True)
        try:
            list_view_err_detail = app.query_one("#ccp-prompts-listview", ListView)
            await list_view_err_detail.clear()
            await list_view_err_detail.append(
                ListItem(Label(f"Error loading prompts: {type(e_prompt_service).__name__}")))
        except QueryError:
            pass
    except Exception as e_unexp:
        logger.error(f"Unexpected error populating CCP prompts list: {e_unexp}", exc_info=True)


def clear_ccp_prompt_fields(app: 'TldwCli') -> None:
    """Clears prompt input fields in the CCP right pane."""
    logger = getattr(app, 'loguru_logger', logging)
    try:
        app.query_one("#ccp-prompt-name-input", Input).value = ""
        app.query_one("#ccp-prompt-author-input", Input).value = ""
        app.query_one("#ccp-prompt-description-textarea", TextArea).text = ""
        app.query_one("#ccp-prompt-system-textarea", TextArea).text = ""
        app.query_one("#ccp-prompt-user-textarea", TextArea).text = ""
        app.query_one("#ccp-prompt-keywords-textarea", TextArea).text = ""

        app.current_prompt_id = None
        app.current_prompt_uuid = None
        app.current_prompt_name = None  # Reactive will become None
        app.current_prompt_author = None
        app.current_prompt_details = None
        app.current_prompt_system = None
        app.current_prompt_user = None
        app.current_prompt_keywords_str = ""  # Reactive will become empty string
        app.current_prompt_version = None
    except QueryError as e:
        logger.error(f"Error clearing CCP prompt fields: {e}", exc_info=True)


async def load_ccp_prompt_for_editing(app: 'TldwCli', prompt_id: Optional[int] = None,
                                      prompt_uuid: Optional[str] = None) -> None:
    """Loads prompt details into the CCP right pane for editing."""
    logger = getattr(app, 'loguru_logger', logging)
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    identifier_to_fetch: Union[int, str, None] = None
    if prompt_id is not None:
        identifier_to_fetch = prompt_id
    elif prompt_uuid is not None:
        identifier_to_fetch = prompt_uuid
    else:
        logger.warning("load_ccp_prompt_for_editing called with no ID or UUID.")
        clear_ccp_prompt_fields(app)
        return

    try:
        prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)

        if prompt_details:
            app.current_prompt_id = prompt_details.get('id')
            app.current_prompt_uuid = prompt_details.get('uuid')
            app.current_prompt_name = prompt_details.get('name', '')
            app.current_prompt_author = prompt_details.get('author', '')
            app.current_prompt_details = prompt_details.get('details', '')
            app.current_prompt_system = prompt_details.get('system_prompt', '')
            app.current_prompt_user = prompt_details.get('user_prompt', '')
            # Ensure keywords is a list before joining, default to empty list if None or not found
            keywords_list_from_db = prompt_details.get('keywords', [])
            app.current_prompt_keywords_str = ", ".join(keywords_list_from_db if keywords_list_from_db else [])
            app.current_prompt_version = prompt_details.get('version')

            app.query_one("#ccp-prompt-name-input", Input).value = app.current_prompt_name or ""
            app.query_one("#ccp-prompt-author-input", Input).value = app.current_prompt_author or ""
            app.query_one("#ccp-prompt-description-textarea", TextArea).text = app.current_prompt_details or ""
            app.query_one("#ccp-prompt-system-textarea", TextArea).text = app.current_prompt_system or ""
            app.query_one("#ccp-prompt-user-textarea", TextArea).text = app.current_prompt_user or ""
            app.query_one("#ccp-prompt-keywords-textarea",
                          TextArea).text = app.current_prompt_keywords_str  # Already a string

            app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False
            app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
            app.query_one("#ccp-prompt-name-input", Input).focus()
            app.notify(f"Prompt '{app.current_prompt_name}' loaded.", severity="information")
        else:
            app.notify(f"Failed to load prompt (ID/UUID: {identifier_to_fetch}).", severity="error")
            clear_ccp_prompt_fields(app)
    except Exception as e:
        logger.error(f"Error loading CCP prompt for editing: {e}", exc_info=True)
        app.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
        clear_ccp_prompt_fields(app)


async def perform_ccp_conversation_search(app: 'TldwCli') -> None:
    """Performs conversation search for the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug("Performing CCP conversation search...")
    try:
        search_input = app.query_one("#conv-char-search-input", Input)
        search_term = search_input.value.strip()

        char_select_widget = app.query_one("#conv-char-character-select", Select)
        selected_character_id = char_select_widget.value
        if selected_character_id == Select.BLANK:  # Treat BLANK as None for filtering
            selected_character_id = None

        results_list_view = app.query_one("#conv-char-search-results-list", ListView)
        await results_list_view.clear()

        if not app.notes_service:
            logger.error("Notes service not available for CCP conversation search.")
            await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
            return

        db = app.notes_service._get_db(app.notes_user_id)
        conversations: List[Dict[str, Any]] = []

        if selected_character_id:
            logger.debug(f"CCP Search: Filtering for character ID: {selected_character_id}")
            if search_term:
                logger.debug(f"CCP Search: Term '{search_term}', CharID {selected_character_id}")
                conversations = db.search_conversations_by_title(
                    title_query=search_term, character_id=selected_character_id, limit=200
                )
            else:
                logger.debug(f"CCP Search: All conversations for CharID {selected_character_id}")
                conversations = db.get_conversations_for_character(
                    character_id=selected_character_id, limit=200
                )
        else:
            logger.debug(f"CCP Search: No character selected. Global search with term: '{search_term}'")
            conversations = db.search_conversations_by_title(
                title_query=search_term, character_id=None, limit=200
            )

        if not conversations:
            msg = "Enter search term or select a character." if not search_term and not selected_character_id else "No items found matching your criteria."
            await results_list_view.append(ListItem(Label(msg)))
        else:
            for conv_data in conversations:
                base_title = conv_data.get('title') or f"Conversation ID: {conv_data['id'][:8]}..."
                display_title = base_title
                char_id_for_conv = conv_data.get('character_id')
                char_name_prefix = ""  # Initialize prefix

                if char_id_for_conv:  # If the conversation has an associated character
                    try:
                        # Always fetch character details from DB for consistent naming
                        char_details = db.get_character_card_by_id(char_id_for_conv)
                        if char_details and char_details.get('name'):
                            char_name_prefix = f"[{char_details['name']}] "
                    except Exception as e_char_name_fetch:
                        logger.warning(
                            f"Could not fetch char name for conv {conv_data['id']}, char_id {char_id_for_conv}: {e_char_name_fetch}",
                            exc_info=False)

                display_title = f"{char_name_prefix}{base_title}"
                item = ListItem(Label(display_title))
                item.details = conv_data
                await results_list_view.append(item)

        logger.info(
            f"CCP conversation search (Term: '{search_term}', CharID: {selected_character_id}) yielded {len(conversations)} results.")

    except QueryError as e_query:
        logger.error(f"UI component not found during CCP conversation search: {e_query}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: UI component missing.")))
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during CCP conversation search: {e_db}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: Database search failed.")))
    except Exception as e_unexp:
        logger.error(f"Unexpected error during CCP conversation search: {e_unexp}", exc_info=True)
        if 'results_list_view' in locals() and results_list_view.is_mounted:
            await results_list_view.clear()
            await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))


#
########################################################################################################################
#
# Event Handlers (called by app.py)
#
########################################################################################################################
async def handle_ccp_conversation_search_button_pressed(app: 'TldwCli') -> None:
    """Handles the search button press in the CCP tab's conversation section."""
    await perform_ccp_conversation_search(app)


async def handle_ccp_load_conversation_button_pressed(app: 'TldwCli') -> None:
    """Handles loading a selected conversation in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Load Conversation button pressed.")
    try:
        results_list_view = app.query_one("#conv-char-search-results-list", ListView)
        highlighted_item = results_list_view.highlighted_child

        if not (highlighted_item and hasattr(highlighted_item, 'details')):
            logger.warning("No conversation selected in CCP list or item has no details.")
            app.notify("Please select a conversation to load.", severity="warning")
            return

        conv_details_from_item = highlighted_item.details
        loaded_conversation_id = conv_details_from_item.get('id')

        if not loaded_conversation_id:
            logger.error("Selected item in CCP list is missing conversation ID.")
            app.notify("Selected item is invalid.", severity="error")
            return

        app.current_conv_char_tab_conversation_id = loaded_conversation_id
        logger.info(f"Current CCP tab conversation ID set to: {loaded_conversation_id}")

        title_input_ccp = app.query_one("#conv-char-title-input", Input)
        keywords_input_ccp = app.query_one("#conv-char-keywords-input", TextArea)
        title_input_ccp.value = conv_details_from_item.get('title', '')

        if app.notes_service:
            db = app.notes_service._get_db(app.notes_user_id)
            keywords_list_ccp = db.get_keywords_for_conversation(loaded_conversation_id)
            keywords_input_ccp.text = ", ".join(
                [kw['keyword'] for kw in keywords_list_ccp]) if keywords_list_ccp else ""
        else:
            keywords_input_ccp.text = "Service unavailable"

        app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = False
        app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True

        center_pane = app.query_one("#conv-char-center-pane", VerticalScroll)
        await center_pane.remove_children()

        if not app.notes_service:
            logger.error("Notes service not available for loading CCP messages.")
            await center_pane.mount(Static("Error: Notes service unavailable for messages."))
            return

        await app._load_branched_conversation_history(loaded_conversation_id, center_pane)

        center_pane.scroll_end(animate=False)
        logger.info(f"Loaded messages into #conv-char-center-pane for conversation {loaded_conversation_id}.")
        app.notify(f"Conversation '{title_input_ccp.value}' loaded.", severity="information")

    except QueryError as e_query:
        logger.error(f"UI component not found during CCP load conversation: {e_query}", exc_info=True)
        app.notify("Error: UI component missing for loading.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error during CCP load conversation: {e_db}", exc_info=True)
        app.notify("Error loading conversation data from database.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error during CCP load conversation: {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred while loading conversation.", severity="error")


async def handle_ccp_save_conversation_details_button_pressed(app: 'TldwCli') -> None:
    """Handles saving conversation title/keywords in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Save Conversation Details button pressed.")
    if not app.current_conv_char_tab_conversation_id:
        logger.warning("No current conversation loaded in CCP tab to save details for.")
        app.notify("No conversation loaded to save details for.", severity="warning")
        return

    if not app.notes_service:
        logger.error("Notes service is not available for CCP save details.")
        app.notify("Database service not available.", severity="error")
        return

    try:
        title_input_ccp_save = app.query_one("#conv-char-title-input", Input)
        keywords_widget_ccp_save = app.query_one("#conv-char-keywords-input", TextArea)

        new_title_ccp = title_input_ccp_save.value.strip()
        new_keywords_str_ccp = keywords_widget_ccp_save.text.strip()
        target_conversation_id_ccp = app.current_conv_char_tab_conversation_id

        db = app.notes_service._get_db(app.notes_user_id)
        current_conv_details_ccp = db.get_conversation_by_id(target_conversation_id_ccp)

        if not current_conv_details_ccp:
            logger.error(f"CCP: Conversation {target_conversation_id_ccp} not found in DB for saving details.")
            app.notify("Error: Conversation not found in database.", severity="error")
            return

        current_db_version_ccp = current_conv_details_ccp.get('version')
        if current_db_version_ccp is None:
            logger.error(f"CCP: Conversation {target_conversation_id_ccp} is missing version information.")
            app.notify("Error: Conversation version information is missing.", severity="error")
            return

        title_updated_ccp = False
        if new_title_ccp != current_conv_details_ccp.get('title', ''):
            update_payload_ccp = {'title': new_title_ccp}
            db.update_conversation(conversation_id=target_conversation_id_ccp, update_data=update_payload_ccp,
                                   expected_version=current_db_version_ccp)
            title_updated_ccp = True
            current_db_version_ccp += 1
            await perform_ccp_conversation_search(app)

        existing_db_keywords_ccp = db.get_keywords_for_conversation(target_conversation_id_ccp)
        existing_kw_texts_set_ccp = {kw['keyword'].lower() for kw in existing_db_keywords_ccp}
        ui_kw_texts_set_ccp = {kw.strip().lower() for kw in new_keywords_str_ccp.split(',') if kw.strip()}

        keywords_to_add_ccp = ui_kw_texts_set_ccp - existing_kw_texts_set_ccp
        keywords_to_remove_details_ccp = [kw for kw in existing_db_keywords_ccp if
                                          kw['keyword'].lower() not in ui_kw_texts_set_ccp]
        keywords_changed_ccp = False

        for kw_text_add in keywords_to_add_ccp:
            # db.add_keyword returns keyword_id (int) for the given text (gets or creates)
            kw_id_add = db.add_keyword(kw_text_add)  # No user_id for global keywords
            if isinstance(kw_id_add, int):  # Check if it's an int ID
                db.link_conversation_to_keyword(conversation_id=target_conversation_id_ccp, keyword_id=kw_id_add)
                keywords_changed_ccp = True
            else:
                logger.error(f"Failed to get or add keyword '{kw_text_add}', received: {kw_id_add}")

        for kw_detail_remove in keywords_to_remove_details_ccp:
            db.unlink_conversation_from_keyword(conversation_id=target_conversation_id_ccp,
                                                keyword_id=kw_detail_remove['id'])
            keywords_changed_ccp = True

        if title_updated_ccp or keywords_changed_ccp:
            logger.info(f"CCP: Details saved for conversation {target_conversation_id_ccp}.")
            app.notify("Details saved successfully!", severity="information")
            final_keywords_list_ccp = db.get_keywords_for_conversation(target_conversation_id_ccp)
            keywords_widget_ccp_save.text = ", ".join(
                [kw['keyword'] for kw in final_keywords_list_ccp]) if final_keywords_list_ccp else ""
        else:
            app.notify("No changes to save.", severity="info")

    except ConflictError as e_conflict:
        logger.error(
            f"CCP: Conflict saving conversation details for {app.current_conv_char_tab_conversation_id}: {e_conflict}",
            exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload.", severity="error")
    except QueryError as e_query:
        logger.error(f"CCP: UI component not found for saving details: {e_query}", exc_info=True)
        app.notify("Error accessing UI fields.", severity="error")
    except CharactersRAGDBError as e_db:
        logger.error(f"CCP: Database error saving details: {e_db}", exc_info=True)
        app.notify("Database error saving details.", severity="error")
    except Exception as e_unexp:
        logger.error(f"CCP: Unexpected error saving details: {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred.", severity="error")


async def handle_ccp_prompt_create_new_button_pressed(app: 'TldwCli') -> None:
    """Handles creating a new prompt in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Create New Prompt button pressed.")
    clear_ccp_prompt_fields(app)
    try:
        app.query_one("#ccp-prompt-name-input", Input).value = "New Prompt"  # Default name
        author_name = app.app_config.get("user_defaults", {}).get("author_name", "User")  # Get default from config
        app.query_one("#ccp-prompt-author-input", Input).value = author_name

        app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False
        app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
        app.query_one("#ccp-prompt-name-input", Input).focus()
        app.notify("Ready to create a new prompt.", severity="info")
    except QueryError as e_query:
        logger.error(f"CCP: UI error preparing for new prompt: {e_query}", exc_info=True)
        app.notify("UI error creating new prompt.", severity="error")


async def handle_ccp_prompt_load_selected_button_pressed(app: 'TldwCli') -> None:
    """Handles loading a selected prompt in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Load Selected Prompt button pressed.")
    try:
        list_view = app.query_one("#ccp-prompts-listview", ListView)
        selected_item = list_view.highlighted_child
        if selected_item and (hasattr(selected_item, 'prompt_id') or hasattr(selected_item, 'prompt_uuid')):
            prompt_id_to_load = getattr(selected_item, 'prompt_id', None)
            prompt_uuid_to_load = getattr(selected_item, 'prompt_uuid', None)
            await load_ccp_prompt_for_editing(app, prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
        else:
            app.notify("No prompt selected in the list.", severity="warning")
    except QueryError:
        app.notify("Prompt list not found.", severity="error")
    except Exception as e:  # Catch broader exceptions during load
        app.notify(f"Error loading prompt: {str(e)[:100]}", severity="error")  # Show part of error
        logger.error(f"Error loading prompt from list: {e}", exc_info=True)


async def handle_ccp_prompt_save_button_pressed(app: 'TldwCli') -> None:
    """Handles saving a new or existing prompt in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Save Prompt button pressed.")
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    try:
        name = app.query_one("#ccp-prompt-name-input", Input).value.strip()
        author = app.query_one("#ccp-prompt-author-input", Input).value.strip()
        details = app.query_one("#ccp-prompt-description-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-prompt-system-textarea", TextArea).text.strip()
        user_prompt = app.query_one("#ccp-prompt-user-textarea", TextArea).text.strip()
        keywords_str = app.query_one("#ccp-prompt-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

        if not name:
            app.notify("Prompt Name is required.", severity="error", timeout=4)
            app.query_one("#ccp-prompt-name-input", Input).focus()
            return

        saved_id: Optional[int] = None
        saved_uuid: Optional[str] = None
        message_from_save: str = ""  # Renamed to avoid conflict

        if app.current_prompt_id is None:
            logger.info(f"CCP: Saving new prompt: {name}")
            saved_id, saved_uuid, message_from_save = prompts_interop.add_prompt(
                name=name, author=author, details=details,
                system_prompt=system_prompt, user_prompt=user_prompt,
                keywords=keywords_list, client_id=app.prompts_client_id,
                overwrite=False
            )
        else:
            logger.info(f"CCP: Updating prompt ID: {app.current_prompt_id}, Name: {name}")
            update_payload = {
                'name': name, 'author': author, 'details': details,
                'system_prompt': system_prompt, 'user_prompt': user_prompt,
                'keywords': keywords_list
            }
            updated_uuid, message_from_save = prompts_interop.get_db_instance().update_prompt_by_id(
                prompt_id=app.current_prompt_id,
                update_data=update_payload,
                client_id=app.prompts_client_id
            )
            saved_id = app.current_prompt_id
            saved_uuid = updated_uuid

        if saved_id is not None or saved_uuid is not None:  # Check if save was successful
            app.notify(message_from_save, severity="information")
            await populate_ccp_prompts_list_view(app)
            await load_ccp_prompt_for_editing(app, prompt_id=saved_id, prompt_uuid=saved_uuid)
        else:
            app.notify(f"Failed to save prompt: {message_from_save or 'Unknown error'}", severity="error")

    except prompts_interop.InputError as e_in:
        app.notify(f"Input Error: {e_in}", severity="error", timeout=6)
    except prompts_interop.ConflictError as e_cf:
        app.notify(f"Save Conflict: {e_cf}", severity="error", timeout=6)
    except prompts_interop.DatabaseError as e_db:
        app.notify(f"Database Error: {e_db}", severity="error", timeout=6)
    except QueryError as e_query:
        logger.error(f"CCP Save Prompt: UI component error: {e_query}", exc_info=True)
        app.notify("UI Error saving prompt.", severity="error")
    except Exception as e_save:
        logger.error(f"CCP: Error saving prompt: {e_save}", exc_info=True)
        app.notify(f"Error saving prompt: {type(e_save).__name__}", severity="error")


async def handle_ccp_prompt_clone_button_pressed(app: 'TldwCli') -> None:
    """Handles cloning the currently loaded prompt in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Clone Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded to clone or service unavailable.", severity="warning")
        return
    try:
        original_name = app.current_prompt_name or "Prompt"  # Use reactive, ensure it's a string
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        cloned_name = f"Clone of {original_name} ({timestamp})"[:100]

        cloned_id, cloned_uuid, msg_clone = prompts_interop.add_prompt(  # Renamed msg to msg_clone
            name=cloned_name,
            author=app.current_prompt_author or "",
            details=f"Clone of: {app.current_prompt_details or ''}",
            system_prompt=app.current_prompt_system or "",
            user_prompt=app.current_prompt_user or "",
            keywords=[kw.strip() for kw in (app.current_prompt_keywords_str or "").split(',') if kw.strip()],
            client_id=app.prompts_client_id,
            overwrite=False
        )
        if cloned_id:
            app.notify(f"Prompt cloned as '{cloned_name}'. {msg_clone}", severity="information")
            await populate_ccp_prompts_list_view(app)
            await load_ccp_prompt_for_editing(app, prompt_id=cloned_id, prompt_uuid=cloned_uuid)
        else:
            app.notify(f"Failed to clone prompt: {msg_clone}", severity="error")
    except Exception as e_clone:
        logger.error(f"CCP: Error cloning prompt: {e_clone}", exc_info=True)
        app.notify(f"Error cloning prompt: {type(e_clone).__name__}", severity="error")


async def handle_ccp_prompt_delete_button_pressed(app: 'TldwCli') -> None:
    """Handles deleting (soft) the currently loaded prompt in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Delete Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded to delete or service unavailable.", severity="warning")
        return

    try:
        # Ensure current_prompt_id is not None before calling delete
        prompt_id_to_delete = app.current_prompt_id
        if prompt_id_to_delete is None:  # Should be caught by above check, but defensive
            app.notify("Error: No prompt ID available for deletion.", severity="error")
            return

        success = prompts_interop.soft_delete_prompt(prompt_id_to_delete, client_id=app.prompts_client_id)
        if success:
            app.notify(f"Prompt '{app.current_prompt_name or 'selected'}' deleted.",
                       severity="information")  # Use current_prompt_name
            clear_ccp_prompt_fields(app)
            await populate_ccp_prompts_list_view(app)
            app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
        else:
            app.notify(
                f"Failed to delete prompt '{app.current_prompt_name or 'selected'}'. It might have been already deleted.",
                severity="error", timeout=7)
    except prompts_interop.NotFoundError:
        app.notify(f"Prompt '{app.current_prompt_name or 'selected'}' not found for deletion.", severity="error")
    except prompts_interop.DatabaseError as e_db_del:
        logger.error(f"CCP: Database error deleting prompt: {e_db_del}", exc_info=True)
        app.notify(f"Database error deleting prompt: {type(e_db_del).__name__}", severity="error")
    except Exception as e_del:
        logger.error(f"CCP: Error deleting prompt: {e_del}", exc_info=True)
        app.notify(f"Error deleting prompt: {type(e_del).__name__}", severity="error")


async def handle_ccp_conversation_search_input_changed(app: 'TldwCli', event_value: str) -> None:
    """Handles input changes in the CCP conversation search bar with debouncing."""
    if app._conv_char_search_timer:
        app._conv_char_search_timer.stop()
    app._conv_char_search_timer = app.set_timer(0.5, lambda: perform_ccp_conversation_search(app))


async def handle_ccp_prompt_search_input_changed(app: 'TldwCli', event_value: str) -> None:
    """Handles input changes in the CCP prompt search bar with debouncing."""
    if app._prompt_search_timer:
        app._prompt_search_timer.stop()
    app._prompt_search_timer = app.set_timer(0.5, lambda: populate_ccp_prompts_list_view(app, event_value.strip()))


async def handle_ccp_character_select_changed(app: 'TldwCli', selected_value: Any) -> None:
    """Handles changes in the CCP character select dropdown."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.debug(f"CCP Character Select changed to: {selected_value}")
    await perform_ccp_conversation_search(app)


async def handle_ccp_prompts_list_view_selected(app: 'TldwCli', list_view_id: str, item: Any) -> None:
    """Handles selecting a prompt from the list in the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    if item and (hasattr(item, 'prompt_id') or hasattr(item, 'prompt_uuid')):
        prompt_id_to_load = getattr(item, 'prompt_id', None)
        prompt_uuid_to_load = getattr(item, 'prompt_uuid', None)
        logger.info(f"CCP Prompt selected from list: ID={prompt_id_to_load}, UUID={prompt_uuid_to_load}")
        await load_ccp_prompt_for_editing(app, prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
    else:
        logger.debug("CCP Prompts ListView selection was empty or item lacked ID/UUID.")

#
# End of conv_char_events.py
########################################################################################################################
