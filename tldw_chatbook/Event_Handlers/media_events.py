# tldw_chatbook/Event_Handlers/media_events.py
#
#
# Imports
import logging
from typing import TYPE_CHECKING, Optional, List, Dict, Any
#
# 3rd-party Libraries
from textual.widgets import ListView, Input, TextArea, Label, ListItem, Button  # Added ListItem
from textual.css.query import QueryError
from rich.text import Text  # For formatting details
#
# Local Imports
from ..Utils.text import slugify
if TYPE_CHECKING:
    from ..app import TldwCli
    from ..DB.Client_Media_DB_v2 import MediaDatabase  # Correct import
########################################################################################################################
#
# Functions:

async def handle_media_nav_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles media navigation button presses in the Media tab."""
    logger = app.loguru_logger
    try:
        type_slug = button_id.replace("media-nav-", "")
        view_to_activate = f"media-view-{type_slug}"
        logger.debug(
            f"Media nav button '{button_id}' pressed. Activating view '{view_to_activate}', type filter: '{type_slug}'.")

        # Find the original display name for the slug
        original_media_type = ""
        try:
            nav_button = app.query_one(f"#{button_id}", Button)
            original_media_type = str(nav_button.label)
            app.current_media_type_filter_display_name = original_media_type  # Store for UI
        except QueryError:
            logger.warning(f"Could not find nav button '{button_id}' to get display name.")
            # Attempt to reverse slugify or find from a stored list if available
            # For now, we might rely on the slug if the button isn't found, or a default.
            app.current_media_type_filter_display_name = type_slug.replace("-", " ").title()

        app.media_active_view = view_to_activate
        app.current_media_type_filter_slug = type_slug

        await perform_media_search_and_display(app, type_slug, search_term="")
    except Exception as e:
        logger.error(f"Error in handle_media_nav_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error switching media view: {str(e)[:100]}", severity="error")


async def handle_media_search_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles search button press within a specific media type view."""
    logger = app.loguru_logger
    try:
        type_slug = button_id.replace("media-search-button-", "")
        search_input_id = f"media-search-input-{type_slug}"
        search_input_widget = app.query_one(f"#{search_input_id}", Input)
        search_term = search_input_widget.value.strip()
        logger.info(f"Media search triggered for type '{type_slug}' with term: '{search_term}'")
        await perform_media_search_and_display(app, type_slug, search_term)
    except QueryError as e:
        logger.error(f"UI component not found for media search button '{button_id}': {e}", exc_info=True)
        app.notify("Search UI error.", severity="error")
    except Exception as e:
        logger.error(f"Error in handle_media_search_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error performing media search: {str(e)[:100]}", severity="error")


async def handle_media_search_input_changed(app: 'TldwCli', input_id: str, value: str) -> None:
    """Handles input changes in media search bars with debouncing."""
    logger = app.loguru_logger
    type_slug = input_id.replace("media-search-input-", "")
    logger.debug(f"Media search input '{input_id}' changed to: '{value}'. Debouncing...")

    if not hasattr(app, '_media_search_timers'):
        app._media_search_timers = {}

    if type_slug in app._media_search_timers and app._media_search_timers[type_slug]:
        app._media_search_timers[type_slug].stop()

    async def debounced_search():
        logger.info(f"Debounced media search executing for type '{type_slug}', term: '{value.strip()}'")
        await perform_media_search_and_display(app, type_slug, value.strip())

    app._media_search_timers[type_slug] = app.set_timer(0.6, debounced_search)


async def handle_media_load_selected_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles loading a selected media item's details."""
    logger = app.loguru_logger
    try:
        type_slug = button_id.replace("media-load-selected-button-", "")
        details_display_id = f"media-details-display-{type_slug}"
        try:
            details_display_widget = app.query_one(f"#{details_display_id}", TextArea)
            # Ensure old text is cleared and "Loading..." is shown before new content might arrive
            details_display_widget.load_text("Loading details...")
        except QueryError as qe:
            logger.warning(f"Could not find details display widget {details_display_id} to show loading message: {qe}")
        list_view_id = f"media-list-view-{type_slug}"
        list_view_widget = app.query_one(f"#{list_view_id}", ListView)

        highlighted_item_widget = list_view_widget.highlighted_child
        if not highlighted_item_widget or not hasattr(highlighted_item_widget, 'media_data'):
            app.notify("No media item selected.", severity="warning")
            # Clear details display if no item selected
            details_display_id = f"media-details-display-{type_slug}"
            try: # Ensure this block also exists or is updated
                details_display_widget = app.query_one(f"#{details_display_id}", TextArea)
                details_display_widget.load_text("No item selected or item has no data.") # Overwrites "Loading details..." if it was set
            except QueryError:
                logger.warning(f"Details display widget #{details_display_id} not found while clearing for no selection.")
            app.current_loaded_media_item = None
            return

        media_data: Dict[str, Any] = highlighted_item_widget.media_data
        app.current_loaded_media_item = media_data

        logger.info(
            f"Media item '{media_data.get('title', 'Unknown')}' (ID: {media_data.get('id')}) loaded into reactive state.")
        # Watcher for current_loaded_media_item will update the UI.
        # app.notify is now handled by the watcher.

    except QueryError as e:
        logger.error(f"UI component not found for media load selected '{button_id}': {e}", exc_info=True)
        app.notify("Load details UI error.", severity="error")
    except Exception as e:
        logger.error(f"Error in handle_media_load_selected_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error loading media details: {str(e)[:100]}", severity="error")


async def perform_media_search_and_display(app: 'TldwCli', type_slug: str, search_term: str = "") -> None:
    """Performs search in media DB for a given type and populates the ListView."""
    logger = app.loguru_logger
    list_view_id = f"media-list-view-{type_slug}"

    try:
        list_view_widget = app.query_one(f"#{list_view_id}", ListView)
        await list_view_widget.clear() # Clear previous items
        loading_item = ListItem(Label("Loading items..."))
        loading_item.disabled = True # Make it non-selectable
        await list_view_widget.append(loading_item)

        # Also clear details display when search is performed
        details_display_id = f"media-details-display-{type_slug}"
        try:
            details_display_widget = app.query_one(f"#{details_display_id}", TextArea)
            details_display_widget.load_text("")
        except QueryError:
            logger.warning(f"Details display widget #{details_display_id} not found while clearing.")
        app.current_loaded_media_item = None

        if not app.notes_service or not hasattr(app.notes_service, '_get_db'):
            logger.error("Media DB service (via notes_service) not available for media search.")
            await list_view_widget.append(ListItem(Label("Error: Media Database service unavailable.")))
            return

        db_instance: Optional['MediaDatabase'] = app.notes_service._get_db(app.notes_user_id)
        if not db_instance or not isinstance(db_instance, app.notes_service.MediaDatabase):
            logger.error("Failed to get a valid MediaDatabase instance.")
            await list_view_widget.append(ListItem(Label("Error: Media Database instance invalid.")))
            return

        # Get the original display name corresponding to the slug
        # This assumes app.current_media_type_filter_display_name is correctly set by nav handler
        original_media_type = app.current_media_type_filter_display_name
        if not original_media_type and type_slug != "all-media":  # if 'all-media' slug, type filter is None
            logger.warning(
                f"Original media type display name not found for slug '{type_slug}'. Searching without specific type filter.")
            # If original_media_type is empty, search_media_db might search all types if media_types param is None or empty list
            media_types_filter = None
        elif type_slug == "all-media":
            media_types_filter = None  # Search all types
            original_media_type = "All Media"  # For logging
        else:
            media_types_filter = [original_media_type]

        logger.debug(
            f"Performing media search. Type Filter: '{media_types_filter}', Slug: '{type_slug}', Term: '{search_term}'")

        results_list, total_matches = db_instance.search_media_db(
            search_query=search_term if search_term else None,  # Pass None if empty for no text search
            media_types=media_types_filter,
            search_fields=['title', 'content', 'author', 'type'] if search_term else None,
            sort_by="last_modified_desc",
            page=1,
            results_per_page=200,  # Show more results
            include_trash=False,
            include_deleted=False
        )

        if not results_list:
            msg = "No media items found."
            if search_term:
                msg = f"No items matching '{search_term}'"
            if original_media_type and type_slug != "all-media":
                msg += f" for type '{original_media_type}'."
            await list_view_widget.append(ListItem(Label(msg)))
        else:
            for item_data in results_list:
                item_title = item_data.get('title', 'Untitled')
                item_id = item_data.get('id')
                if type_slug == "all-media":
                    item_data_type = item_data.get('type', 'N/A')
                    item_label_text = f"[{item_data_type}] {item_title} (ID: {item_id})"
                else:
                    item_label_text = f"{item_title} (ID: {item_id})"
                list_item = ListItem(Label(item_label_text))
                list_item.media_data = item_data  # Store full data on the ListItem
                await list_view_widget.append(list_item)
        logger.info(
            f"Media search for type '{original_media_type}' (slug: '{type_slug}', term: '{search_term}') yielded {len(results_list)} results (total DB matches: {total_matches}).")

    except QueryError as e:
        logger.error(f"UI component error during media search for type '{type_slug}': {e}", exc_info=True)
        try:
            lv_widget_err = app.query_one(f"#{list_view_id}", ListView)
            await lv_widget_err.clear()
            await lv_widget_err.append(ListItem(Label("Error: UI component missing for search.")))
        except QueryError:
            pass
    except Exception as e:
        logger.error(f"Unexpected error during media search for type '{type_slug}': {e}", exc_info=True)
        try:
            lv_widget_unexp_err = app.query_one(f"#{list_view_id}", ListView)
            await lv_widget_unexp_err.clear()
            await lv_widget_unexp_err.append(ListItem(Label(f"Error performing search: {str(e)[:100]}")))
        except QueryError:
            pass


def format_media_details(media_data: Dict[str, Any]) -> Text:
    """Formats media item details into a Rich Text object for display."""
    if not media_data:
        return Text("No media item loaded.")

    details = Text()
    details.append(f"Title: {media_data.get('title', 'N/A')}\n", style="bold")
    details.append(f"ID: {media_data.get('id', 'N/A')}\n")
    details.append(f"UUID: {media_data.get('uuid', 'N/A')}\n")
    details.append(f"Type: {media_data.get('type', 'N/A')}\n")
    details.append(f"URL: {media_data.get('url', 'N/A')}\n")
    details.append(f"Author: {media_data.get('author', 'N/A')}\n")
    details.append(f"Ingestion Date: {media_data.get('ingestion_date', 'N/A')}\n")
    details.append(f"Last Modified: {media_data.get('last_modified', 'N/A')}\n")
    details.append(f"Content Hash: {media_data.get('content_hash', 'N/A')}\n")
    details.append(f"Transcription Model: {media_data.get('transcription_model', 'N/A')}\n")
    details.append(f"Chunking Status: {media_data.get('chunking_status', 'N/A')}\n")
    details.append(f"Vector Processing: {'Yes' if media_data.get('vector_processing') else 'No'}\n")
    details.append(f"Deleted: {'Yes' if media_data.get('deleted') else 'No'}\n")
    details.append(f"In Trash: {'Yes' if media_data.get('is_trash') else 'No'}\n")
    if media_data.get('is_trash') and media_data.get('trash_date'):
        details.append(f"Trash Date: {media_data.get('trash_date')}\n")

    content_preview = media_data.get('content', '')
    if content_preview is None: content_preview = "N/A"  # Handle None content explicitly
    details.append(f"\nContent Preview:\n", style="bold underline")
    details.append(content_preview[:500] + ("..." if len(content_preview) > 500 else "") + "\n")

    # Placeholder for keywords - requires fetching them
    # details.append(f"\nKeywords: ...\n", style="bold")
    return details

#
# End of media_events.py
########################################################################################################################
