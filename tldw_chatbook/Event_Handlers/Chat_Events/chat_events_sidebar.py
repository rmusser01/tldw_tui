# chat_events_sidebar.py
#
# Imports
#
# Standard Library Imports
from typing import TYPE_CHECKING, Dict, Any, Optional
import logging

# 3rd-party Libraries
from textual.widgets import ListItem, Input, ListView, TextArea, Button, Label
from textual.css.query import QueryError
from textual.app import App
#
# Local Imports
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli  # Assuming your app class is TldwCli
#
###########################################################################################################################
#
# Globals
#
logger = logging.getLogger(__name__)  # Standard practice for logging
#
###########################################################################################################################
#
# Functions:
#

def _disable_media_copy_buttons(app: 'TldwCli'):
    """Helper to disable all media copy buttons and clear current sidebar media item."""
    app.current_sidebar_media_item = None
    try:
        app.query_one("#chat-media-copy-title-button", Button).disabled = True
        app.query_one("#chat-media-copy-content-button", Button).disabled = True
        app.query_one("#chat-media-copy-author-button", Button).disabled = True
        app.query_one("#chat-media-copy-url-button", Button).disabled = True
    except QueryError as e:
        logger.warning(f"Could not find a media copy button to disable: {e}")


async def perform_media_sidebar_search(app: 'TldwCli', search_term: str):
    """
    Performs a search for media items and populates the results in the sidebar.
    This is now an async function.
    """
    logger.debug(f"Performing media sidebar search for term: '{search_term}'")
    try:
        results_list_view = app.query_one("#chat-media-search-results-listview", ListView)
        # FIX: Query for the correct ID for the content display area.
        review_display = app.query_one("#chat-media-content-display", TextArea)
    except QueryError as e:
        logger.error(f"Error querying media search UI elements: {e}")
        app.notify(f"Error accessing media search UI: {e}", severity="error")
        return

    await results_list_view.clear()
    review_display.clear()
    _disable_media_copy_buttons(app)

    if not search_term:
        logger.debug("Search term is empty, clearing results.")
        return

    try:
        if not app.media_db:
            logger.error("app.media_db is not available.")
            app.notify("Media database service not initialized.", severity="error")
            return

        db_instance = app.media_db

        search_fields = ['title', 'content', 'author', 'keywords', 'notes']
        media_types_filter = None

        logger.debug(f"Searching media DB with term: '{search_term}', fields: {search_fields}, types: {media_types_filter}")

        media_items = db_instance.search_media_db(
            search_query=search_term,
            search_fields=search_fields,
            media_types=media_types_filter,
            date_range=None,  # No date range filtering
            must_have_keywords=None,
            must_not_have_keywords=None,
            sort_by="last_modified_desc",  # Default sort order
            media_ids_filter=None,  # No specific media IDs to filter
            page=1,  # Default to first page
            results_per_page=100,  # Limit results to 100 for performance
            include_trash=False,
            include_deleted=False,
        )
        logger.debug(f"Found {len(media_items)} media items.")

        if not media_items:
            # FIX: Await the async append method.
            await results_list_view.append(ListItem(Label("No media found.")))
        else:
            for item_dict in media_items:
                if isinstance(item_dict, dict):
                    title = item_dict.get('title', 'Untitled')
                    media_id = item_dict.get('media_id', 'Unknown ID')
                    display_label = f"{title} (ID: {media_id[:8]}...)"
                    list_item = ListItem(Label(display_label))
                    setattr(list_item, 'media_data', item_dict)
                    await results_list_view.append(list_item)
                else:
                    logger.warning(f"Skipping non-dictionary item from DB search results: {item_dict}")

    except Exception as e:
        logger.error(f"Exception during media search: {e}", exc_info=True)
        app.notify(f"Error during media search: {e}", severity="error")
        await results_list_view.clear()
        # FIX: Await the async append method.
        await results_list_view.append(ListItem(Label(f"Search error.")))


async def handle_chat_media_search_input_changed(app: 'TldwCli', input_widget: Input):
    """
    Handles changes in the media search input, debouncing the search.
    """
    search_term = input_widget.value.strip()
    logger.debug(f"Media search input changed. Current value: '{search_term}'")

    if hasattr(app, '_media_sidebar_search_timer') and app._media_sidebar_search_timer:
        app._media_sidebar_search_timer.stop()
        logger.debug("Stopped existing media search timer.")

    app._media_sidebar_search_timer = app.set_timer(
        0.5,
        lambda: app.run_worker(perform_media_sidebar_search(app, search_term), exclusive=True)
    )
    logger.debug(f"Set new media search timer for term: '{search_term}'")


async def handle_chat_media_load_selected_button_pressed(app: 'TldwCli'):
    """
    Loads the selected media item's details into the review display.
    """
    logger.debug("Load Selected Media button pressed.")
    try:
        results_list_view = app.query_one("#chat-media-search-results-listview", ListView)
        # FIX: Query for the correct ID.
        review_display = app.query_one("#chat-media-content-display", TextArea)
    except QueryError as e:
        logger.error(f"Error querying media UI elements for load: {e}")
        _disable_media_copy_buttons(app)
        return

    highlighted_item = results_list_view.highlighted_child
    if highlighted_item is None or not hasattr(highlighted_item, 'media_data'):
        app.notify("No media item selected.", severity="warning")
        review_display.clear()
        _disable_media_copy_buttons(app)
        return

    media_data: Dict[str, Any] = getattr(highlighted_item, 'media_data')
    app.current_sidebar_media_item = media_data

    # Format details for display
    details_parts = []
    if media_data.get('title'): details_parts.append(f"Title: {media_data['title']}")
    if media_data.get('author'): details_parts.append(f"Author: {media_data['author']}")
    if media_data.get('media_type'): details_parts.append(f"Type: {media_data['media_type']}")
    if media_data.get('url'): details_parts.append(f"URL: {media_data['url']}")
    details_parts.append("\n--- Content Snippet ---\n")
    content_preview = (media_data['content'][:500] + '...') if len(media_data.get('content', '')) > 500 else media_data.get('content')
    details_parts.append(content_preview or "No content available.")

    review_display.load_text("\n".join(details_parts))
    logger.info(f"Successfully loaded media ID {media_data.get('media_id')} into review display.")

    # Enable copy buttons
    app.query_one("#chat-media-copy-title-button", Button).disabled = not bool(media_data.get('title'))
    app.query_one("#chat-media-copy-content-button", Button).disabled = not bool(media_data.get('content'))
    app.query_one("#chat-media-copy-author-button", Button).disabled = not bool(media_data.get('author'))
    app.query_one("#chat-media-copy-url-button", Button).disabled = not bool(media_data.get('url'))
    logger.debug("Media copy buttons state updated.")


async def handle_chat_media_copy_title_button_pressed(app: 'TldwCli'):
    """Copies the title of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy Title button pressed.")
    if app.current_sidebar_media_item and 'title' in app.current_sidebar_media_item:
        title = str(app.current_sidebar_media_item['title'])
        app.copy_to_clipboard(title)
        app.notify("Title copied to clipboard.")
        logger.info(f"Copied title: '{title}'")
    else:
        app.notify("No media title to copy.", severity="warning")
        logger.warning("No media title available to copy.")


async def handle_chat_media_copy_content_button_pressed(app: 'TldwCli'):
    """Copies the content of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy Content button pressed.")
    if app.current_sidebar_media_item and 'content' in app.current_sidebar_media_item:
        content = str(app.current_sidebar_media_item['content'])
        app.copy_to_clipboard(content)
        app.notify("Content copied to clipboard.")
        logger.info("Copied content (length: %s)", len(content))
    else:
        app.notify("No media content to copy.", severity="warning")
        logger.warning("No media content available to copy.")


async def handle_chat_media_copy_author_button_pressed(app: 'TldwCli'):
    """Copies the author of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy Author button pressed.")
    if app.current_sidebar_media_item and 'author' in app.current_sidebar_media_item:
        author = str(app.current_sidebar_media_item['author'])
        app.copy_to_clipboard(author)
        app.notify("Author copied to clipboard.")
        logger.info(f"Copied author: '{author}'")
    else:
        app.notify("No media author to copy.", severity="warning")
        logger.warning("No media author available to copy.")


async def handle_chat_media_copy_url_button_pressed(app: 'TldwCli'):
    """Copies the URL of the currently loaded sidebar media to clipboard."""
    logger.debug("Copy URL button pressed.")
    if app.current_sidebar_media_item and 'url' in app.current_sidebar_media_item and app.current_sidebar_media_item['url']:
        url = str(app.current_sidebar_media_item['url'])
        app.copy_to_clipboard(url)
        app.notify("URL copied to clipboard.")
        logger.info(f"Copied URL: '{url}'")
    else:
        app.notify("No media URL to copy.", severity="warning")
        logger.warning("No media URL available to copy.")


# --- Button Handler Map ---
CHAT_SIDEBAR_BUTTON_HANDLERS = {
    "chat-media-load-selected-button": handle_chat_media_load_selected_button_pressed,
    "chat-media-copy-title-button": handle_chat_media_copy_title_button_pressed,
    "chat-media-copy-content-button": handle_chat_media_copy_content_button_pressed,
    "chat-media-copy-author-button": handle_chat_media_copy_author_button_pressed,
    "chat-media-copy-url-button": handle_chat_media_copy_url_button_pressed,
}


#
# End of chat_events_sidebar.py
########################################################################################################################
