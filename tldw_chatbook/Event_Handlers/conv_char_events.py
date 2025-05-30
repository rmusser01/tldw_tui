# conv_char_events.py
# Description:
#
# Imports
import json  # For export
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, List, Dict, Union

import yaml
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
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, Filters # For File Picker
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
    # try:
    #     app.query_one("#ccp-prompt-name-input", Input).value = ""
    #     app.query_one("#ccp-prompt-author-input", Input).value = ""
    #     app.query_one("#ccp-prompt-description-textarea", TextArea).text = ""
    #     app.query_one("#ccp-prompt-system-textarea", TextArea).text = ""
    #     app.query_one("#ccp-prompt-user-textarea", TextArea).text = ""
    #     app.query_one("#ccp-prompt-keywords-textarea", TextArea).text = ""

    #     app.current_prompt_id = None
    #     app.current_prompt_uuid = None
    #     app.current_prompt_name = None  # Reactive will become None
    #     app.current_prompt_author = None
    #     app.current_prompt_details = None
    #     app.current_prompt_system = None
    #     app.current_prompt_user = None
    #     app.current_prompt_keywords_str = ""  # Reactive will become empty string
    #     app.current_prompt_version = None
    # except QueryError as e:
    #     logger.error(f"Error clearing CCP prompt fields: {e}", exc_info=True)
    app._clear_prompt_fields()

# This function is deprecated; app._load_prompt_for_editing in app.py is used instead.
# async def load_ccp_prompt_for_editing(app: 'TldwCli', prompt_id: Optional[int] = None,
#                                       prompt_uuid: Optional[str] = None) -> None:
#     """Loads prompt details into the CCP right pane for editing."""
#     logger = getattr(app, 'loguru_logger', logging)
#     if not app.prompts_service_initialized:
#         app.notify("Prompts service not available.", severity="error")
#         return
#
#     identifier_to_fetch: Union[int, str, None] = None
#     if prompt_id is not None:
#         identifier_to_fetch = prompt_id
#     elif prompt_uuid is not None:
#         identifier_to_fetch = prompt_uuid
#     else:
#         logger.warning("load_ccp_prompt_for_editing called with no ID or UUID.")
#         clear_ccp_prompt_fields(app)
#         return
#
#     logger.debug(f"CCP Load Prompt: identifier_to_fetch is: {identifier_to_fetch}")
#     try:
#         prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)
#         logger.debug(f"CCP Load Prompt: Fetched prompt_details: {prompt_details}")
#
#         if prompt_details:
#             app.current_prompt_id = prompt_details.get('id')
#             app.current_prompt_uuid = prompt_details.get('uuid')
#             app.current_prompt_name = prompt_details.get('name', '')
#             app.current_prompt_author = prompt_details.get('author', '')
#             app.current_prompt_details = prompt_details.get('details', '')
#             app.current_prompt_system = prompt_details.get('system_prompt', '')
#             app.current_prompt_user = prompt_details.get('user_prompt', '')
#             # Ensure keywords is a list before joining, default to empty list if None or not found
#             keywords_list_from_db = prompt_details.get('keywords', [])
#             app.current_prompt_keywords_str = ", ".join(keywords_list_from_db if keywords_list_from_db else [])
#             app.current_prompt_version = prompt_details.get('version')
#
#             logger.debug("CCP Load Prompt: Attempting to populate UI fields.")
#             app.query_one("#ccp-prompt-name-input", Input).value = app.current_prompt_name or ""
#             logger.debug(f"CCP Load Prompt: Set name to: {app.current_prompt_name or ''}")
#             app.query_one("#ccp-prompt-author-input", Input).value = app.current_prompt_author or ""
#             logger.debug(f"CCP Load Prompt: Set author to: {app.current_prompt_author or ''}")
#             app.query_one("#ccp-prompt-description-textarea", TextArea).text = app.current_prompt_details or ""
#             logger.debug(f"CCP Load Prompt: Set description to: {app.current_prompt_details or ''}")
#             app.query_one("#ccp-prompt-system-textarea", TextArea).text = app.current_prompt_system or ""
#             logger.debug(f"CCP Load Prompt: Set system to: {app.current_prompt_system or ''}")
#             app.query_one("#ccp-prompt-user-textarea", TextArea).text = app.current_prompt_user or ""
#             logger.debug(f"CCP Load Prompt: Set user to: {app.current_prompt_user or ''}")
#             app.query_one("#ccp-prompt-keywords-textarea",
#                           TextArea).text = app.current_prompt_keywords_str  # Already a string
#             logger.debug(f"CCP Load Prompt: Set keywords to: {app.current_prompt_keywords_str}")
#
#             logger.debug("CCP Load Prompt: Finished populating UI fields.")
#             app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False
#             app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
#             app.query_one("#ccp-prompt-name-input", Input).focus()
#             app.notify(f"Prompt '{app.current_prompt_name}' loaded.", severity="information")
#         else:
#             app.notify(f"Failed to load prompt (ID/UUID: {identifier_to_fetch}).", severity="error")
#             clear_ccp_prompt_fields(app)
#     except Exception as e:
#         logger.critical(f"CRITICAL ERROR in load_ccp_prompt_for_editing: {e}", exc_info=True)
#         app.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
#         clear_ccp_prompt_fields(app)
#         logger.debug("CCP Load Prompt: Cleared prompt fields due to exception.")

########################################################################################################################
#
# Event Handlers for Character Card Import
#
########################################################################################################################
async def _character_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    if selected_path:
        logger.info(f"Character card import selected: {selected_path}")
        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for character import.")
            return

        db = app.notes_service._get_db(app.notes_user_id)
        try:
            # import_and_save_character_from_file can take Path object directly
            char_id = ccl.import_and_save_character_from_file(db, str(selected_path))
            if char_id is not None:
                app.notify(f"Character card imported successfully (ID: {char_id}).", severity="information")
                await populate_ccp_character_select(app) # Refresh character dropdown
            else:
                # import_and_save_character_from_file logs errors, but we can notify too
                app.notify("Failed to import character card. Check logs.", severity="error")
        except ConflictError as ce:
            app.notify(f"Import conflict: {ce}", severity="warning", timeout=6)
            logger.warning(f"Conflict importing character card '{selected_path}': {ce}")
            await populate_ccp_character_select(app) # Refresh in case it was a duplicate name
        except ImportError as ie: # E.g., PyYAML missing for Markdown with frontmatter
            app.notify(f"Import error: {ie}. A required library might be missing.", severity="error", timeout=8)
            logger.error(f"Import error for character card '{selected_path}': {ie}", exc_info=True)
        except Exception as e:
            app.notify(f"Error importing character card: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing character card '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Character card import cancelled.")
        app.notify("Character import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_character_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Import Character Card button pressed.")

    defined_filters = Filters(
        ("Character Card (JSON, PNG, WebP, MD)", lambda p: p.suffix.lower() in (".json", ".png", ".webp", ".md", ".yaml", ".yml")),
        ("JSON files (*.json)", lambda p: p.suffix.lower() == ".json"),
        ("PNG images (*.png)", lambda p: p.suffix.lower() == ".png"),
        ("WebP images (*.webp)", lambda p: p.suffix.lower() == ".webp"),
        ("Markdown files (*.md)", lambda p: p.suffix.lower() == ".md"),
        ("YAML files (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(FileOpen(location=str(Path.home()), title="Select Character Card", filters=defined_filters),
                          # Use a lambda to capture `app` for the callback
                          callback=lambda path: _character_import_callback(app, path))


async def handle_ccp_left_load_character_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging) # Or however logger is typically obtained
    logger.info("CCP Load Selected Character button (left pane) pressed.")
    try:
        # Get the Select widget from the left pane
        char_select_widget = app.query_one("#conv-char-character-select", Select)
        selected_character_id = char_select_widget.value

        if selected_character_id == Select.BLANK or selected_character_id is None:
            app.notify("No character selected from the dropdown.", severity="warning")
            logger.warning("Load Character (left pane): No character selected.")
            return

        logger.info(f"Attempting to load character ID: {selected_character_id} into center editor view.")

        # Store the selected character ID (optional, but good for state tracking if needed later)
        # app.current_editing_character_id = selected_character_id

        # Switch the center pane view to the character editor
        app.ccp_active_view = "character_editor_view"
        # This will trigger the watcher in app.py to make #ccp-character-editor-view visible.
        # Population of fields will be handled in a later phase.

        # app.notify(f"Character (ID: {selected_character_id}) view activated. Editor population pending.", severity="information")

        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for loading character.")
            return

        db = app.notes_service._get_db(app.notes_user_id)

        try:
            # Load character data and image
            char_data, initial_ui_history, img = ccl.load_character_and_image(db, selected_character_id, app.notes_user_id)

            if char_data:
                app.current_ccp_character_details = char_data
                app.current_ccp_character_image = img  # Store the PIL Image object

                # Switch the center pane view to the character card display
                app.ccp_active_view = "character_card_view"

                app.notify(f"Displaying character: {char_data.get('name', 'Unknown')}", severity="information")
                logger.info(f"Character card for '{char_data.get('name', 'Unknown')}' loaded and view switched to card display.")
            else:
                app.notify("Failed to load character details.", severity="error")
                logger.warning(f"Failed to load character details for ID: {selected_character_id}")

        except CharactersRAGDBError as e_db_rag:
            logger.error(f"Database error loading character {selected_character_id}: {e_db_rag}", exc_info=True)
            app.notify(f"Database error loading character: {e_db_rag}", severity="error")
        except Exception as e_load_char: # Catch other potential errors from ccl.load_character_and_image
            logger.error(f"Error loading character data/image for ID {selected_character_id}: {e_load_char}", exc_info=True)
            app.notify(f"Error loading character: {type(e_load_char).__name__}", severity="error")

    except QueryError as e_query:
        logger.error(f"UI component not found during load character (left pane): {e_query}", exc_info=True)
        app.notify("Error: UI component missing for loading character.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error during load character (left pane): {e_unexp}", exc_info=True)
        app.notify("An unexpected error occurred while trying to load character view.", severity="error")


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
            app.notify("No changes to save.", severity="information")

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
        app.query_one("#ccp-prompt-name-input", Input).value = "New Prompt"  # Default name in right pane
        author_name = app.app_config.get("user_defaults", {}).get("author_name", "User")
        app.query_one("#ccp-prompt-author-input", Input).value = author_name # In right pane

        app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False # Expand right pane prompt details
        app.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True # Collapse right pane conv details

        app.ccp_active_view = "conversation_details_view" # Ensure center pane is NOT on its editor view.
                                                        # Or, if you want to clear the center editor too:
                                                        # app.ccp_active_view = "prompt_editor_view"
                                                        # await app._clear_prompt_fields() # This clears center editor
                                                        # Then perhaps switch back to conv view or focus right pane

        app.query_one("#ccp-prompt-name-input", Input).focus() # Focus in right pane
        app.notify("Ready to create a new prompt in the right-side editor.", severity="information")
    except QueryError as e_query:
        logger.error(f"CCP: UI error preparing for new prompt (right pane): {e_query}", exc_info=True)
        app.notify("UI error creating new prompt.", severity="error")


async def handle_ccp_prompt_load_selected_button_pressed(app: 'TldwCli') -> None:
    """Handles loading a selected prompt from the LEFT PANE list into the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Load Selected Prompt button pressed (loads into right pane editor).")
    try:
        list_view = app.query_one("#ccp-prompts-listview", ListView)
        selected_item = list_view.highlighted_child
        if selected_item and (hasattr(selected_item, 'prompt_id') or hasattr(selected_item, 'prompt_uuid')):
            prompt_id_to_load = getattr(selected_item, 'prompt_id', None)
            prompt_uuid_to_load = getattr(selected_item, 'prompt_uuid', None)
            # load_ccp_prompt_for_editing populates the RIGHT PANE editor
            await app._load_prompt_for_editing(prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
        else:
            app.notify("No prompt selected in the list.", severity="warning")
    except QueryError:
        app.notify("Prompt list not found.", severity="error")
    except Exception as e:  # Catch broader exceptions during load
        app.notify(f"Error loading prompt: {str(e)[:100]}", severity="error")  # Show part of error
        logger.error(f"Error loading prompt from list: {e}", exc_info=True)


async def handle_ccp_prompt_save_button_pressed(app: 'TldwCli') -> None:
    """Handles saving a new or existing prompt from the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Save Prompt button pressed (saves from right pane editor).")
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    try:
        # Query fields from the RIGHT PANE editor
        name = app.query_one("#ccp-prompt-name-input", Input).value.strip()
        author = app.query_one("#ccp-prompt-author-input", Input).value.strip()
        details = app.query_one("#ccp-prompt-description-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-prompt-system-textarea", TextArea).text.strip()
        user_prompt = app.query_one("#ccp-prompt-user-textarea", TextArea).text.strip()
        keywords_str = app.query_one("#ccp-prompt-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

        if not name:
            app.notify("Prompt Name is required.", severity="error", timeout=4)
            app.query_one("#ccp-prompt-name-input", Input).focus() # Focus in right pane
            return

        saved_id: Optional[int] = None
        saved_uuid: Optional[str] = None
        message_from_save: str = ""

        if app.current_prompt_id is None: # This implies a new prompt being saved from the right pane editor
            logger.info(f"CCP: Saving new prompt from right pane: {name}")
            saved_id, saved_uuid, message_from_save = prompts_interop.add_prompt(
                name=name, author=author, details=details,
                system_prompt=system_prompt, user_prompt=user_prompt,
                keywords=keywords_list, overwrite=False
            )
        else: # Updating an existing prompt loaded into the right pane editor
            logger.info(f"CCP: Updating prompt ID (from right pane): {app.current_prompt_id}, Name: {name}")
            update_payload = {
                'name': name, 'author': author, 'details': details,
                'system_prompt': system_prompt, 'user_prompt': user_prompt,
                'keywords': keywords_list
            }
            # Assuming update_prompt_by_id takes the ID from app.current_prompt_id
            # and returns (uuid, message_str)
            updated_uuid, message_from_save = prompts_interop.get_db_instance().update_prompt_by_id(
                prompt_id=app.current_prompt_id,
                update_data=update_payload,
            ) # Removed version since it's not used by Prompts_DB update.
            saved_id = app.current_prompt_id
            saved_uuid = updated_uuid

        if saved_id is not None or saved_uuid is not None:
            app.notify(message_from_save, severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # Reload the saved/updated prompt back into the RIGHT PANE editor to confirm changes and get new version
            await app._load_prompt_for_editing(prompt_id=saved_id, prompt_uuid=saved_uuid)
        else:
            app.notify(f"Failed to save prompt: {message_from_save or 'Unknown error'}", severity="error")

    except prompts_interop.InputError as e_in:
        app.notify(f"Input Error: {e_in}", severity="error", timeout=6)
    except prompts_interop.ConflictError as e_cf:
        app.notify(f"Save Conflict: {e_cf}", severity="error", timeout=6)
    except prompts_interop.DatabaseError as e_db:
        app.notify(f"Database Error: {e_db}", severity="error", timeout=6)
    except QueryError as e_query:
        logger.error(f"CCP Save Prompt (right pane): UI component error: {e_query}", exc_info=True)
        app.notify("UI Error saving prompt.", severity="error")
    except Exception as e_save:
        logger.error(f"CCP: Error saving prompt (right pane): {e_save}", exc_info=True)
        app.notify(f"Error saving prompt: {type(e_save).__name__}", severity="error")


async def handle_ccp_prompt_clone_button_pressed(app: 'TldwCli') -> None:
    """Handles cloning the currently loaded prompt in the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Clone Prompt button pressed (clones from right pane editor).")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in right pane editor to clone or service unavailable.", severity="warning")
        return
    try:
        original_name = app.current_prompt_name or "Prompt"
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        cloned_name = f"Clone of {original_name} ({timestamp})"[:100]

        cloned_id, cloned_uuid, msg_clone = prompts_interop.add_prompt(
            name=cloned_name,
            author=app.current_prompt_author or "",
            details=f"Clone of: {app.current_prompt_details or ''}",
            system_prompt=app.current_prompt_system or "",
            user_prompt=app.current_prompt_user or "",
            keywords=[kw.strip() for kw in (app.current_prompt_keywords_str or "").split(',') if kw.strip()],
            overwrite=False
        )
        if cloned_id:
            app.notify(f"Prompt cloned as '{cloned_name}'. {msg_clone}", severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # Load the newly cloned prompt into the RIGHT PANE editor
            await app._load_prompt_for_editing(prompt_id=cloned_id, prompt_uuid=cloned_uuid)
        else:
            app.notify(f"Failed to clone prompt: {msg_clone}", severity="error")
    except Exception as e_clone:
        logger.error(f"CCP: Error cloning prompt (from right pane): {e_clone}", exc_info=True)
        app.notify(f"Error cloning prompt: {type(e_clone).__name__}", severity="error")


async def handle_ccp_prompt_delete_button_pressed(app: 'TldwCli') -> None:
    """Handles deleting (soft) the currently loaded prompt from the RIGHT PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Delete Prompt button pressed (deletes based on right pane editor state).")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in right pane editor to delete or service unavailable.", severity="warning")
        return

    try:
        prompt_id_to_delete = app.current_prompt_id
        if prompt_id_to_delete is None:
            app.notify("Error: No prompt ID available for deletion.", severity="error")
            return

        # soft_delete_prompt in your DB class returns True on success, False if not found/already deleted
        # It raises ConflictError or DatabaseError on other issues.
        success = prompts_interop.soft_delete_prompt(prompt_id_to_delete) # client_id is handled internally

        if success:
            app.notify(f"Prompt '{app.current_prompt_name or 'selected'}' deleted.",
                       severity="information")
            clear_ccp_prompt_fields(app) # Clear right pane editor and reactives
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            try: # Collapse details in right pane
                app.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
            except QueryError:
                logger.warning("Could not find #ccp-prompt-details-collapsible to collapse after delete.")
        else:
            # This 'else' branch now correctly handles the "not found or already deleted" case
            app.notify(
                f"Failed to delete prompt '{app.current_prompt_name or 'selected'}'. It might have been already deleted or not found.",
                severity="error", timeout=7)
            # Refresh the list anyway to ensure UI consistency
            await populate_ccp_prompts_list_view(app)
            # If the prompt was not found, it makes sense to also clear the fields
            # as the currently displayed details might be for a non-existent prompt.
            # However, clear_ccp_prompt_fields also sets current_prompt_id to None,
            # which might be okay or you might want to just clear UI and keep ID if deletion failed for other reasons.
            # Given soft_delete_prompt returns False for "not found", clearing is appropriate.
            if app.current_prompt_id == prompt_id_to_delete: # Only clear if it was the one we tried to delete
                clear_ccp_prompt_fields(app)

    except prompts_interop.ConflictError as e_cf_del: # Specific exception from your DB class
        logger.error(f"CCP: Conflict error deleting prompt: {e_cf_del}", exc_info=True)
        app.notify(f"Conflict error deleting prompt: {e_cf_del}", severity="error")
        await populate_ccp_prompts_list_view(app) # Refresh list
    except prompts_interop.DatabaseError as e_db_del: # Specific exception from your DB class
        logger.error(f"CCP: Database error deleting prompt: {e_db_del}", exc_info=True)
        app.notify(f"Database error deleting prompt: {type(e_db_del).__name__}", severity="error")
    except Exception as e_del: # Catch any other unexpected exceptions
        logger.error(f"CCP: Unexpected error deleting prompt: {e_del}", exc_info=True)
        app.notify(f"Unexpected error deleting prompt: {type(e_del).__name__}", severity="error")


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
    """Handles selecting a prompt from the list in the CCP tab (loads to RIGHT pane)."""
    logger = getattr(app, 'loguru_logger', logging)
    if item and (hasattr(item, 'prompt_id') or hasattr(item, 'prompt_uuid')):
        prompt_id_to_load = getattr(item, 'prompt_id', None)
        prompt_uuid_to_load = getattr(item, 'prompt_uuid', None)
        logger.info(f"CCP Prompt selected from list: ID={prompt_id_to_load}, UUID={prompt_uuid_to_load}. Loading to right pane.")
        # load_ccp_prompt_for_editing loads into the RIGHT PANE editor
        await app._load_prompt_for_editing(prompt_id=prompt_id_to_load, prompt_uuid=prompt_uuid_to_load)
    else:
        logger.debug("CCP Prompts ListView selection was empty or item lacked ID/UUID.")


########################################################################################################################
#
# Event Handlers for Conversation Import
#
########################################################################################################################
async def _conversation_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    if selected_path:
        logger.info(f"Conversation import selected: {selected_path}")
        if not app.notes_service:
            app.notify("Database service not available.", severity="error")
            logger.error("Notes service not available for conversation import.")
            return

        db = app.notes_service._get_db(app.notes_user_id)
        try:
            # ccl.load_chat_history_from_file_and_save_to_db expects a string path or BytesIO
            # It returns (conversation_id, character_id)
            conv_id, _ = ccl.load_chat_history_from_file_and_save_to_db(
                db,
                str(selected_path),
                user_name_for_placeholders=app.notes_user_id # Or a more appropriate placeholder
            )
            if conv_id:
                app.notify(f"Conversation imported successfully (ID: {conv_id}).", severity="information")
                await perform_ccp_conversation_search(app) # Refresh conversation list
            else:
                app.notify("Failed to import conversation. Check logs.", severity="error")
        except Exception as e:
            app.notify(f"Error importing conversation: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing conversation from '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Conversation import cancelled.")
        app.notify("Conversation import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_conversation_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Import Conversation button pressed.")

    defined_filters = Filters(
        ("Chat Log (JSON)", lambda p: p.suffix.lower() == ".json"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(location=str(Path.home()), title="Select Conversation File", filters=defined_filters),
        callback=lambda path: _conversation_import_callback(app, path))

########################################################################################################################
#
# Event Handlers for Prompt Import
#
########################################################################################################################
def _parse_prompt_from_file_content(file_content_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses prompt data from string content.
    Tries JSON, then YAML, then a custom plain text format.
    """
    logger = loguru_logger
    if not file_content_str:
        return None
    # 1. Try JSON
    try:
        parsed_json = json.loads(file_content_str)
        if isinstance(parsed_json, dict) and parsed_json.get("name"):
            logger.debug("Successfully parsed prompt file content as JSON.")
            return parsed_json
        else:
            logger.debug("Content parsed as JSON but not a valid prompt structure (missing 'name' or not a dict).")
    except json.JSONDecodeError:
        logger.debug("Content is not valid JSON. Trying YAML.")
    except Exception as e_json_other:
        logger.warning(f"Unexpected error during JSON parsing attempt: {e_json_other}")

    # 2. Try YAML
    try:
        parsed_yaml = yaml.safe_load(file_content_str)
        if isinstance(parsed_yaml, dict) and parsed_yaml.get("name"):
            logger.debug("Successfully parsed prompt file content as YAML.")
            return parsed_yaml
        else:
            logger.debug("Content parsed as YAML but not a valid prompt structure (missing 'name' or not a dict).")
    except yaml.YAMLError:
        logger.debug("Content is not valid YAML. Trying custom plain text format.")
    except Exception as e_yaml_other:
        logger.warning(f"Unexpected error during YAML parsing attempt: {e_yaml_other}")

    # 3. Try Custom Plain Text Format
    logger.debug("Attempting to parse prompt file content as custom plain text format.")
    try:
        parsed_custom_data: Dict[str, Any] = {}
        # Regex to find sections: ### SECTION_NAME ###\nContent until next ### or EOF
        # Using re.DOTALL so . matches newlines.
        # Using re.IGNORECASE for section headers.
        # Capture group 1 is the section name, group 2 is the content.
        # This regex assumes sections are separated by ### SECTION_NAME ###
        # and the content is everything until the next ### or end of file.

        # A simpler approach based on your format:
        # Split by ###, then process pairs.
        # Example: "### TITLE ###\nSAMPLE PROMPT\n### AUTHOR ###\nrmusser01"
        # sections_raw = re.split(r'\s*###\s*([A-Z\s])\s*###\s*', file_content_str.strip(), flags=re.IGNORECASE)
        # Filter out empty strings that result from splitting at the start/end
        # sections_filtered = [s.strip() for s in sections_raw if s and s.strip()]

        # More robust parsing:
        header_map = {
            "TITLE": "name",
            "AUTHOR": "author",
            "SYSTEM": "system_prompt",
            "USER": "user_prompt",
            "KEYWORDS": "keywords_str",  # Will be split later
            "DETAILS": "details",
        }

        # Use re.finditer to find all section blocks
        # Pattern: ### SECTION_NAME ###\n(content up to next ### or EOF)
        pattern = r"^\s*###\s*("  "|".join(header_map.keys())
        r")\s*###\s*\n(.*?)(?=(?:\n\s*###|$))"
        for match in re.finditer(pattern, file_content_str, re.MULTILINE | re.DOTALL | re.IGNORECASE):
            section_name_from_file = match.group(1).upper()
            section_content = match.group(2).strip()
            dict_key = header_map.get(section_name_from_file)
            if dict_key:
                parsed_custom_data[dict_key] = section_content

        # Post-process keywords if they were read as a string
        if "keywords_str" in parsed_custom_data:
            parsed_custom_data["keywords"] = [
                kw.strip() for kw in parsed_custom_data["keywords_str"].split(',') if kw.strip()
            ]
            del parsed_custom_data["keywords_str"]  # remove the temp string version
        elif "keywords" not in parsed_custom_data:  # Ensure keywords key exists
            parsed_custom_data["keywords"] = []

        # If "name" (from TITLE) is missing after regex parsing, try to infer it from the first non-header line
        if "name" not in parsed_custom_data or not parsed_custom_data["name"]:
            first_few_lines = file_content_str.strip().split('\n', 5)
            potential_title_line = ""
            if first_few_lines:
                if first_few_lines[0].upper().strip().startswith("### TITLE ###"):
                    if len(first_few_lines) > 1 and not first_few_lines[1].upper().strip().startswith("###"):
                        potential_title_line = first_few_lines[1].strip()
                elif not first_few_lines[0].upper().strip().startswith("###"):
                    potential_title_line = first_few_lines[0].strip()
            if potential_title_line:
                parsed_custom_data["name"] = potential_title_line

        if "name" in parsed_custom_data and parsed_custom_data["name"]:
            logger.debug(
                f"Successfully parsed prompt file content as custom plain text. Data: {parsed_custom_data}")
            # Ensure all expected fields for add_prompt are present, defaulting to None or empty list
            final_data_for_add = {
                "name": parsed_custom_data.get("name"),
                "author": parsed_custom_data.get("author"),
                "details": parsed_custom_data.get("details"),
                "system_prompt": parsed_custom_data.get("system_prompt"),
                "user_prompt": parsed_custom_data.get("user_prompt"),
                "keywords": parsed_custom_data.get("keywords", []),
            }
            return final_data_for_add
        else:
            logger.debug("Custom plain text parsing did not yield a valid 'name' or was empty.")
    except Exception as e_custom:
        logger.error(f"Error parsing prompt file as custom plain text: {e_custom}", exc_info=True)
        return None
        # If all parsing attempts fail
    logger.error("All parsing attempts for prompt file failed.")
    return None

async def _prompt_import_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    logger = getattr(app, 'loguru_logger', logging)
    if selected_path:
        logger.info(f"Prompt import selected: {selected_path}")
        if not app.prompts_service_initialized:
            app.notify("Prompts service not available.", severity="error")
            return

        try:
            with open(selected_path, 'r', encoding='utf-8') as f:
                content = f.read()

            prompt_data = _parse_prompt_from_file_content(content)
            if not prompt_data:
                app.notify("Failed to parse prompt file. Invalid format or missing 'name'.", severity="error", timeout=7)
                return

            # prompts_interop.add_prompt expects individual arguments.
            # We need to extract them from the prompt_data dictionary.
            # It also returns (id, uuid, message_str)
            prompt_id, prompt_uuid, msg = prompts_interop.add_prompt(
                name=prompt_data.get("name", "Unnamed Prompt"), # name is required by _parse_prompt...
                author=prompt_data.get("author"),
                details=prompt_data.get("details"),
                system_prompt=prompt_data.get("system_prompt"),
                user_prompt=prompt_data.get("user_prompt"),
                keywords=prompt_data.get("keywords"), # Expects List[str] or None
                overwrite=False # Default to not overwriting, or make configurable
            )

            if prompt_id is not None or prompt_uuid is not None : # Check if save was successful (id or uuid exists)
                app.notify(f"Prompt imported: {msg}", severity="information")
                await populate_ccp_prompts_list_view(app) # Refresh prompt list
            else:
                app.notify(f"Failed to import prompt: {msg}", severity="error")
        except FileNotFoundError:
            app.notify(f"Prompt file not found: {selected_path}", severity="error")
            logger.error(f"Prompt file not found: {selected_path}")
        except Exception as e:
            app.notify(f"Error importing prompt: {type(e).__name__}", severity="error", timeout=6)
            logger.error(f"Error importing prompt from '{selected_path}': {e}", exc_info=True)
    else:
        logger.info("Prompt import cancelled.")
        app.notify("Prompt import cancelled.", severity="information", timeout=2)

async def handle_ccp_import_prompt_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Import Prompt button pressed.")

    defined_filters = Filters(
        ("Prompt files (TXT, MD, JSON, YAML)", lambda p: p.suffix.lower() in (".txt", ".md", ".json", ".yaml", ".yml")),
        ("JSON files (*.json)", lambda p: p.suffix.lower() == ".json"),
        ("YAML files (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
        ("Text files (*.txt)", lambda p: p.suffix.lower() == ".txt"),
        ("Markdown files (*.md)", lambda p: p.suffix.lower() == ".md"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(FileOpen(location=str(Path.home()), title="Select Prompt File", filters=defined_filters),
                          callback=lambda path: _prompt_import_callback(app, path))

########################################################################################################################
# CCP Center Pane Editor Button Handlers (these operate on the #ccp-editor-* prefixed IDs)
########################################################################################################################
async def handle_ccp_editor_prompt_save_button_pressed(app: 'TldwCli') -> None:
    """Handles saving a new or existing prompt from the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor Save Prompt button pressed (saves from center pane editor).")
    if not app.prompts_service_initialized:
        app.notify("Prompts service not available.", severity="error")
        return

    try:
        # Query fields from the CENTER PANE editor
        name = app.query_one("#ccp-editor-prompt-name-input", Input).value.strip()
        author = app.query_one("#ccp-editor-prompt-author-input", Input).value.strip()
        details = app.query_one("#ccp-editor-prompt-description-textarea", TextArea).text.strip()
        system_prompt = app.query_one("#ccp-editor-prompt-system-textarea", TextArea).text.strip()
        user_prompt = app.query_one("#ccp-editor-prompt-user-textarea", TextArea).text.strip()
        keywords_str = app.query_one("#ccp-editor-prompt-keywords-textarea", TextArea).text.strip()
        keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

        if not name:
            app.notify("Prompt Name is required for editor save.", severity="error", timeout=4)
            app.query_one("#ccp-editor-prompt-name-input", Input).focus()
            return

        saved_id: Optional[int] = None
        saved_uuid: Optional[str] = None
        message_from_save: str = ""

        # current_prompt_id, etc. are used to track the prompt loaded into *either* editor.
        # If the center pane editor is active, these reactives should reflect that.
        if app.current_prompt_id is None: # Saving a NEW prompt via center editor
            logger.info(f"CCP Editor: Saving new prompt: {name}")
            saved_id, saved_uuid, message_from_save = prompts_interop.add_prompt(
                name=name, author=author, details=details,
                system_prompt=system_prompt, user_prompt=user_prompt,
                keywords=keywords_list, overwrite=False
            )
        else: # Updating an EXISTING prompt via center editor
            logger.info(f"CCP Editor: Updating prompt ID: {app.current_prompt_id}, Name: {name}")
            update_payload = {
                'name': name, 'author': author, 'details': details,
                'system_prompt': system_prompt, 'user_prompt': user_prompt,
                'keywords': keywords_list
            }
            updated_uuid, message_from_save = prompts_interop.get_db_instance().update_prompt_by_id(
                prompt_id=app.current_prompt_id,
                update_data=update_payload,
            )
            saved_id = app.current_prompt_id
            saved_uuid = updated_uuid

        if saved_id is not None or saved_uuid is not None:
            app.notify(f"Prompt Editor: {message_from_save}", severity="information")
            await populate_ccp_prompts_list_view(app) # Refresh list in left pane
            # After saving from center editor, reload it into the center editor.
            # _load_prompt_for_editing already handles setting reactives and populating center editor.
            await app._load_prompt_for_editing(prompt_id=saved_id, prompt_uuid=saved_uuid)
        else:
            app.notify(f"Prompt Editor: Failed to save prompt: {message_from_save or 'Unknown error'}", severity="error")

    except prompts_interop.InputError as e_in:
        app.notify(f"Editor Input Error: {e_in}", severity="error", timeout=6)
    except prompts_interop.ConflictError as e_cf:
        app.notify(f"Editor Save Conflict: {e_cf}", severity="error", timeout=6)
    except prompts_interop.DatabaseError as e_db:
        app.notify(f"Editor Database Error: {e_db}", severity="error", timeout=6)
    except QueryError as e_query:
        logger.error(f"CCP Editor Save Prompt: UI component error: {e_query}", exc_info=True)
        app.notify("UI Error saving prompt from editor.", severity="error")
    except Exception as e_save:
        logger.error(f"CCP Editor: Error saving prompt: {e_save}", exc_info=True)
        app.notify(f"Editor: Error saving prompt: {type(e_save).__name__}", severity="error")


async def handle_ccp_editor_prompt_clone_button_pressed(app: 'TldwCli') -> None:
    """Handles cloning the prompt currently in the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor Clone Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in editor to clone or service unavailable.", severity="warning")
        return

    try:
        # Assume current_prompt_* reactives hold the state of the editor
        original_name = app.current_prompt_name or "Prompt"
        timestamp = datetime.now().strftime('%y%m%d%H%M%S')
        cloned_name = f"Clone of {original_name} ({timestamp})"[:100]

        cloned_id, cloned_uuid, msg_clone = prompts_interop.add_prompt(
            name=cloned_name,
            author=app.current_prompt_author or "",
            details=f"Clone of: {app.current_prompt_details or ''}",
            system_prompt=app.current_prompt_system or "",
            user_prompt=app.current_prompt_user or "",
            keywords=[kw.strip() for kw in (app.current_prompt_keywords_str or "").split(',') if kw.strip()],
            overwrite=False
        )
        if cloned_id:
            app.notify(f"Editor: Prompt cloned as '{cloned_name}'. {msg_clone}", severity="information")
            await populate_ccp_prompts_list_view(app)
            # Load the newly cloned prompt into the CENTER PANE editor
            await app._load_prompt_for_editing(prompt_id=cloned_id, prompt_uuid=cloned_uuid)
        else:
            app.notify(f"Editor: Failed to clone prompt: {msg_clone}", severity="error")
    except Exception as e_clone:
        logger.error(f"CCP Editor: Error cloning prompt: {e_clone}", exc_info=True)
        app.notify(f"Editor: Error cloning prompt: {type(e_clone).__name__}", severity="error")


async def handle_ccp_editor_prompt_delete_button_pressed(app: 'TldwCli') -> None:
    """Handles deleting the prompt currently in the CENTER PANE editor."""
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor Delete Prompt button pressed.")
    if not app.prompts_service_initialized or app.current_prompt_id is None:
        app.notify("No prompt loaded in editor to delete or service unavailable.", severity="warning")
        return

    try:
        prompt_id_to_delete = app.current_prompt_id
        if prompt_id_to_delete is None:
            app.notify("Editor: No prompt ID available for deletion.", severity="error")
            return

        success = prompts_interop.soft_delete_prompt(prompt_id_to_delete)

        if success:
            app.notify(f"Editor: Prompt '{app.current_prompt_name or 'selected'}' deleted.", severity="information")
            app._clear_prompt_fields() # Clear center editor and reactives
            await populate_ccp_prompts_list_view(app)
            app.ccp_active_view = "conversation_details_view" # Switch view back
        else:
            app.notify(f"Editor: Failed to delete prompt. It might have been already deleted.", severity="error")
            await populate_ccp_prompts_list_view(app)
            if app.current_prompt_id == prompt_id_to_delete: # If it was the one we tried to delete
                app._clear_prompt_fields()
                app.ccp_active_view = "conversation_details_view"

    except prompts_interop.ConflictError as e_cf_del:
        logger.error(f"CCP Editor: Conflict deleting prompt: {e_cf_del}", exc_info=True)
        app.notify(f"Editor: Conflict error deleting prompt: {e_cf_del}", severity="error")
    except prompts_interop.DatabaseError as e_db_del:
        logger.error(f"CCP Editor: Database error deleting prompt: {e_db_del}", exc_info=True)
        app.notify(f"Editor: Database error deleting prompt: {type(e_db_del).__name__}", severity="error")
    except Exception as e_del:
        logger.error(f"CCP Editor: Unexpected error deleting prompt: {e_del}", exc_info=True)
        app.notify(f"Editor: Unexpected error deleting prompt: {type(e_del).__name__}", severity="error")

# ##############################################################
# --- CCP Center Pane Editor Button Handlers End ---
# ##############################################################

async def handle_ccp_editor_char_save_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor: Save Character button pressed.")

    if not app.notes_service:
        app.notify("Database service not available.", severity="error")
        logger.error("Notes service not available for saving character.")
        return

    try:
        # Retrieve data from UI fields
        name = app.query_one("#ccp-editor-char-name-input", Input).value.strip()
        description = app.query_one("#ccp-editor-char-description-textarea", TextArea).text.strip()
        personality = app.query_one("#ccp-editor-char-personality-textarea", TextArea).text.strip()
        scenario = app.query_one("#ccp-editor-char-scenario-textarea", TextArea).text.strip()
        first_message = app.query_one("#ccp-editor-char-first-message-textarea", TextArea).text.strip()
        # Add other fields here if they were added to the UI, e.g., keywords

        if not name:
            app.notify("Character Name is required.", severity="error", timeout=4)
            app.query_one("#ccp-editor-char-name-input", Input).focus()
            return

        character_data = {
            "name": name,
            "description": description, # Assuming 'description' is the DB key
            "personality": personality,
            "scenario": scenario,
            "first_mes": first_message, # Assuming 'first_mes' is the DB key
            # "keywords": [], # Example if keywords were added
            # "system_prompt": "", # Example if a system prompt field for characters was added
        }

        saved_character_details: Optional[Dict[str, Any]] = None

        if app.current_editing_character_id is None: # New character
            logger.info(f"Attempting to add new character: {name}")
            # Ensure add_character_card exists and handles these fields.
            # It should return the full character data, including new ID and version.
            # The user_id is passed as the first argument as per typical service methods.
            saved_character_details = app.notes_service.add_character_card(
                user_id=app.notes_user_id,
                character_data=character_data
            )
            if saved_character_details and saved_character_details.get("id"):
                logger.info(f"New character added successfully. ID: {saved_character_details['id']}")
                app.notify(f"Character '{name}' saved successfully.", severity="information")
            else:
                logger.error(f"Failed to save new character '{name}'. Service returned: {saved_character_details}")
                app.notify(f"Failed to save new character '{name}'.", severity="error")
                return # Stop if save failed

        else: # Existing character
            logger.info(f"Attempting to update character ID: {app.current_editing_character_id}")
            current_version = app.current_editing_character_data.get("version") if app.current_editing_character_data else None

            # Ensure update_character_card exists and handles versioning.
            # It should return the full updated character data.
            saved_character_details = app.notes_service.update_character_card(
                character_id=app.current_editing_character_id,
                user_id=app.notes_user_id,
                update_data=character_data,
                expected_version=current_version
            )
            if saved_character_details:
                logger.info(f"Character {app.current_editing_character_id} updated successfully.")
                app.notify(f"Character '{name}' updated successfully.", severity="information")
            else:
                logger.error(f"Failed to update character '{name}'. Service returned: {saved_character_details}")
                app.notify(f"Failed to update character '{name}'.", severity="error")
                return # Stop if update failed

        # If save or update was successful and returned details
        if saved_character_details and saved_character_details.get("id"):
            # Reload into editor to reflect any DB-side changes (new ID, version, cleaned data)
            await app._load_character_for_editing(saved_character_details["id"])
            await populate_ccp_character_select(app) # Refresh dropdown
        else:
            # This case might be redundant if errors are caught above, but as a fallback:
            logger.warning("Save/Update operation completed but no valid character details returned.")
            # Optionally, try to reload with current_editing_character_id if it exists
            if app.current_editing_character_id:
                await app._load_character_for_editing(app.current_editing_character_id)


    except ConflictError as e_conflict:
        logger.error(f"Conflict saving character: {e_conflict}", exc_info=True)
        app.notify(f"Save conflict: {e_conflict}. Please reload and try again.", severity="error", timeout=7)
        # Optionally, reload the conflicting version:
        # if app.current_editing_character_id:
        #    await app._load_character_for_editing(app.current_editing_character_id)
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error saving character: {e_db}", exc_info=True)
        app.notify(f"Database error: {type(e_db).__name__}", severity="error")
    except QueryError as e_query:
        logger.error(f"UI component error saving character: {e_query}", exc_info=True)
        app.notify("UI Error: Could not retrieve character data from fields.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error saving character: {e_unexp}", exc_info=True)
        app.notify(f"Unexpected error: {type(e_unexp).__name__}", severity="error")


async def handle_ccp_editor_char_clone_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor: Clone Character button pressed.")

    if not app.notes_service:
        app.notify("Database service not available.", severity="error")
        logger.error("Notes service not available for cloning character.")
        return

    if app.current_editing_character_id is None or app.current_editing_character_data is None:
        app.notify("No character loaded or unsaved changes. Please load or save a character before cloning.", severity="warning", timeout=7)
        logger.warning("Clone Character: No character data loaded in editor.")
        return

    try:
        # Get current data from app state (which should reflect the editor)
        # It's generally better to use app.current_editing_character_data if it's kept in sync,
        # or query UI fields if there's a chance they are more up-to-date than reactives before a save.
        # For cloning, using the reactive `app.current_editing_character_data` is safer if it's accurate.
        # If not, query all fields like in the save handler. Let's assume `app.current_editing_character_data` is reliable here.

        original_data = app.current_editing_character_data

        original_name = original_data.get("name", "Character")
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S')
        cloned_name = f"{original_name} Clone {timestamp}"[:255] # Ensure name length fits DB schema

        cloned_character_data = {
            "name": cloned_name,
            "description": original_data.get("description", original_data.get("details", "")),
            "personality": original_data.get("personality", ""),
            "scenario": original_data.get("scenario", ""),
            "first_mes": original_data.get("first_mes", original_data.get("greeting", "")),
            # Copy other relevant fields, ensure not to copy 'id' or 'version'
            # "keywords": original_data.get("keywords", []),
            # "system_prompt": original_data.get("system_prompt", ""),
        }

        logger.info(f"Attempting to clone character '{original_name}' as '{cloned_name}'.")

        # Call add_character_card to save the cloned data as a new character
        saved_clone_details = app.notes_service.add_character_card(
            user_id=app.notes_user_id,
            character_data=cloned_character_data
        )

        if saved_clone_details and saved_clone_details.get("id"):
            new_cloned_char_id = saved_clone_details["id"]
            logger.info(f"Character cloned successfully. New ID: {new_cloned_char_id}")
            app.notify(f"Character cloned as '{cloned_name}'.", severity="information")

            # Load the newly cloned character into the editor
            await app._load_character_for_editing(new_cloned_char_id)
            await populate_ccp_character_select(app) # Refresh dropdown
        else:
            logger.error(f"Failed to save cloned character '{cloned_name}'. Service returned: {saved_clone_details}")
            app.notify(f"Failed to clone character '{original_name}'.", severity="error")

    except CharactersRAGDBError as e_db:
        logger.error(f"Database error cloning character: {e_db}", exc_info=True)
        app.notify(f"Database error cloning: {type(e_db).__name__}", severity="error")
    # QueryError might occur if we were reading directly from UI and a field was missing.
    # Since we're primarily using app.current_editing_character_data, it's less likely here.
    except QueryError as e_query:
        logger.error(f"UI component error during clone (should not happen if using reactive data): {e_query}", exc_info=True)
        app.notify("UI Error during clone operation.", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error cloning character: {e_unexp}", exc_info=True)
        app.notify(f"Unexpected error cloning: {type(e_unexp).__name__}", severity="error")


async def handle_ccp_editor_char_delete_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("CCP Editor: Delete Character button pressed.")

    if not app.notes_service:
        app.notify("Database service not available.", severity="error")
        logger.error("Notes service not available for deleting character.")
        return

    character_id_to_delete = app.current_editing_character_id
    character_name_to_delete = app.current_editing_character_data.get("name", "the loaded character") if app.current_editing_character_data else "the loaded character"


    if character_id_to_delete is None:
        app.notify("No character loaded in the editor to delete.", severity="warning")
        logger.warning("Delete Character: No character ID loaded.")
        return

    # Optional: Implement a confirmation dialog here in the future.
    # For now, deletion is direct upon button press.

    try:
        logger.info(f"Attempting to delete character ID: {character_id_to_delete} (Name: '{character_name_to_delete}').")

        # Assuming delete_character_card returns True on success, False otherwise,
        # or raises an exception for specific failures.
        success = app.notes_service.delete_character_card(
            character_id=character_id_to_delete,
            user_id=app.notes_user_id
        )

        if success:
            logger.info(f"Character {character_id_to_delete} deleted successfully.")
            app.notify(f"Character '{character_name_to_delete}' deleted.", severity="information")

            await app._clear_character_fields() # Clear the editor
            app.ccp_active_view = "conversation_details_view" # Switch to a default view
            await populate_ccp_character_select(app) # Refresh dropdown
        else:
            # This path might be taken if the DB method returns False for "not found" or "already deleted"
            # without raising an exception.
            logger.error(f"Failed to delete character '{character_name_to_delete}' (ID: {character_id_to_delete}). Service returned False.")
            app.notify(f"Failed to delete character '{character_name_to_delete}'. It might have been already deleted or an issue occurred.", severity="error", timeout=7)
            # Refresh list anyway to ensure UI consistency if it was a "not found" type of issue.
            await populate_ccp_character_select(app)


    except ConflictError as e_conflict: # If delete_character_card can raise this (e.g., if used by other data)
        logger.error(f"Conflict error deleting character {character_id_to_delete}: {e_conflict}", exc_info=True)
        app.notify(f"Conflict deleting character: {e_conflict}", severity="error")
        await populate_ccp_character_select(app) # Refresh list
    except CharactersRAGDBError as e_db:
        logger.error(f"Database error deleting character {character_id_to_delete}: {e_db}", exc_info=True)
        app.notify(f"Database error deleting: {type(e_db).__name__}", severity="error")
    except Exception as e_unexp:
        logger.error(f"Unexpected error deleting character {character_id_to_delete}: {e_unexp}", exc_info=True)
        app.notify(f"Unexpected error deleting: {type(e_unexp).__name__}", severity="error")

# ##############################################################
# --- CCP Right Pane Editor Button Handlers ---
# ##############################################################
async def handle_ccp_tab_sidebar_toggle(app: 'TldwCli', button_id: str) -> None:
    """Handles sidebar toggles specific to the CCP tab."""
    logger = getattr(app, 'loguru_logger', logging)
    if button_id == "toggle-conv-char-left-sidebar":
        app.conv_char_sidebar_left_collapsed = not app.conv_char_sidebar_left_collapsed
        logger.debug("CCP left sidebar now %s", "collapsed" if app.conv_char_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-conv-char-right-sidebar":
        app.conv_char_sidebar_right_collapsed = not app.conv_char_sidebar_right_collapsed
        logger.debug("CCP right sidebar now %s", "collapsed" if app.conv_char_sidebar_right_collapsed else "expanded")
    else:
        logger.warning(f"Unhandled sidebar toggle button ID '{button_id}' in CCP tab handler.")


#
# End of conv_char_events.py
########################################################################################################################
