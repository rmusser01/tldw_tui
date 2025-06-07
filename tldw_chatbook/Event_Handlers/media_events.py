# tldw_chatbook/Event_Handlers/media_events.py
#
#
# Imports
import logging
import math
from typing import TYPE_CHECKING, Dict, Any
#
# 3rd-party Libraries
from textual.widgets import ListView, Input, TextArea, Label, ListItem, Button  # Added ListItem
from textual.css.query import QueryError
from rich.text import Text  # For formatting details
#
# Local Imports
from ..DB.Client_Media_DB_v2 import MediaDatabase, fetch_keywords_for_media

if TYPE_CHECKING:
    from ..app import TldwCli
    from ..UI.MediaWindow import MediaWindow
########################################################################################################################
#
# Statics:
RESULTS_PER_PAGE = 20
#
# Functions:

async def handle_media_nav_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles media navigation button presses in the Media tab."""
    logger = app.loguru_logger
    button_id = event.button.id
    try:
        type_slug = button_id.replace("media-nav-", "")
        view_to_activate = f"media-view-{type_slug}"
        logger.debug(f"Media nav button '{button_id}' pressed. Activating view '{view_to_activate}', type filter: '{type_slug}'.")

        nav_button = app.query_one(f"#{button_id}", Button)
        app.current_media_type_filter_display_name = str(nav_button.label)

        # The query here works because MediaWindow is a type hint known to the checker,
        # and at runtime, Textual's query engine looks for the class instance.
        media_window = app.query_one("MediaWindow") # Query by class name as a string
        media_window.media_active_view = view_to_activate

        app.current_media_type_filter_slug = type_slug
        app.media_current_page = 1
        await perform_media_search_and_display(app, type_slug, search_term="")

    except Exception as e:
        logger.error(f"Error in handle_media_nav_button_pressed for '{button_id}': {e}", exc_info=True)
        app.notify(f"Error switching media view: {str(e)[:100]}", severity="error")


async def handle_media_search_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles search button press within a specific media type view."""
    logger = app.loguru_logger
    button_id = event.button.id
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

    if type_slug in app._media_search_timers and app._media_search_timers[type_slug]:
        app._media_search_timers[type_slug].stop()

    async def debounced_search():
        logger.info(f"Debounced media search executing for type '{type_slug}', term: '{value.strip()}'")
        app.media_current_page = 1 # Reset to page 1 for new search
        await perform_media_search_and_display(app, type_slug, value.strip())

    app._media_search_timers[type_slug] = app.set_timer(0.6, debounced_search)


async def handle_media_load_selected_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles loading a selected media item's details."""
    logger = app.loguru_logger
    button_id = event.button.id
    try:
        type_slug = button_id.replace("media-load-selected-button-", "")
        list_view = app.query_one(f"#media-list-view-{type_slug}", ListView)
        details_display = app.query_one(f"#media-details-display-{type_slug}", TextArea)

        if not list_view.highlighted_child or not hasattr(list_view.highlighted_child, 'media_data'):
            app.notify("No media item selected.", severity="warning")
            details_display.load_text("No item selected.")
            app.current_loaded_media_item = None
            return

        details_display.load_text("Loading details...")
        media_data: Dict[str, Any] = list_view.highlighted_child.media_data
        app.current_loaded_media_item = media_data
        logger.info(f"Loaded media item ID {media_data.get('id')} into reactive state.")

    except QueryError as e:
        logger.error(f"UI component not found for media load selected '{button_id}': {e}", exc_info=True)
        app.notify("Load details UI error.", severity="error")


async def handle_media_page_change_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles Next/Previous page button presses."""
    logger = app.loguru_logger
    button_id = event.button.id

    if "next" in button_id:
        app.media_current_page += 1
    elif "prev" in button_id and app.media_current_page > 1:
        app.media_current_page -= 1
    else:
        return

    type_slug = app.current_media_type_filter_slug
    search_term = ""
    try:
        search_input = app.query_one(f"#media-search-input-{type_slug}", Input)
        search_term = search_input.value
    except QueryError: pass

    logger.info(f"Changing to page {app.media_current_page} for type '{type_slug}'")
    await perform_media_search_and_display(app, type_slug, search_term)

async def perform_media_search_and_display(app: 'TldwCli', type_slug: str, search_term: str = "") -> None:
    """Performs search in media DB and populates the ListView with pagination."""
    logger = app.loguru_logger
    list_view_id = f"media-list-view-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        await list_view.clear()
        loading_item = ListItem(Label("Loading items..."))
        loading_item.disabled = True
        await list_view.append(loading_item)

        # Also clear details display
        app.query_one(f"#media-details-display-{type_slug}", TextArea).load_text("")
        app.current_loaded_media_item = None

        if not app.media_db:
            raise RuntimeError("Media DB service not available.")

        # Determine media type filter based on the reliable slug
        media_types_filter = None
        if type_slug != "all-media":
            # The slug (e.g., 'document') is the correct value for the DB query.
            # Replace hyphen with underscore for types like 'web-page' -> 'web_page'
            db_media_type = type_slug.replace('-', '_')
            media_types_filter = [db_media_type]

        # Always provide a list of fields to search_fields. The DB function should ignore
        # this list when search_query is None, but providing it satisfies the function's
        # potential requirement for the parameter to exist.
        query_arg = search_term if search_term else None
        fields_arg = ['title', 'content', 'author', 'url', 'type']

        logger.info("---EXECUTING MEDIA DB SEARCH---")
        logger.info(f"  search_query: {query_arg!r}")
        logger.info(f"  media_types: {media_types_filter!r}")
        logger.info(f"  search_fields: {fields_arg!r}")
        logger.info(f"  page: {app.media_current_page}")
        logger.info("-------------------------------")

        results, total_matches = app.media_db.search_media_db(
            search_query=query_arg,
            media_types=media_types_filter,
            search_fields=fields_arg,
            sort_by="last_modified_desc",
            page=app.media_current_page,
            results_per_page=RESULTS_PER_PAGE,
            include_trash=False,
            include_deleted=False
        )

        logger.info(f"---DB SEARCH COMPLETE---")
        logger.info(f"  Results received: {len(results)}")
        logger.info(f"  Total matches reported: {total_matches}")
        logger.info("------------------------")

        await list_view.clear() # Clear "Loading..." message

        if not results:
            msg = "No media items found."
            if search_term: msg += f" matching '{search_term}'"
            await list_view.append(ListItem(Label(msg)))
        else:
            for item in results:
                title = item.get('title', 'Untitled')
                label = f"[{item.get('type')}] {title}" if type_slug == "all-media" else title
                list_item = ListItem(Label(label))
                list_item.media_data = item
                await list_view.append(list_item)

        # Update pagination controls
        total_pages = math.ceil(total_matches / RESULTS_PER_PAGE) if total_matches > 0 else 1
        page_label = app.query_one(f"#media-page-label-{type_slug}", Label)
        prev_button = app.query_one(f"#media-prev-page-button-{type_slug}", Button)
        next_button = app.query_one(f"#media-next-page-button-{type_slug}", Button)

        page_label.update(f"Page {app.media_current_page} / {total_pages}")
        prev_button.disabled = (app.media_current_page <= 1)
        next_button.disabled = (app.media_current_page >= total_pages)

    except (QueryError, RuntimeError, Exception) as e:
        logger.error(f"Error during media search for type '{type_slug}': {e}", exc_info=True)
        try:
            lv_widget_err = app.query_one(f"#{list_view_id}", ListView)
            await lv_widget_err.clear()
            await lv_widget_err.append(ListItem(Label(f"Error: {e}")))
        except QueryError: pass


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


def format_media_details_as_markdown(app: 'TldwCli', media_data: Dict[str, Any]) -> str:
    """Formats media item details into a Markdown string for display."""
    if not media_data:
        return "No media item loaded."

    # Add a check to ensure media_db exists.
    keywords_str = "N/A"
    media_id = media_data.get('id')
    if app.media_db and media_id:
        try:
            keywords = app.media_db.get_keywords_for_media(media_id)
            if keywords:
                keywords_str = ", ".join(keywords)
        except Exception as e:
            app.loguru_logger.error(f"Failed to fetch keywords for media ID {media_id}: {e}")
            keywords_str = "Error fetching keywords"

    details = [
        f"### {media_data.get('title', 'N/A')}",
        f"**ID:** `{media_data.get('id', 'N/A')}`",
        f"**UUID:** `{media_data.get('uuid', 'N/A')}`",
        f"**Type:** {media_data.get('type', 'N/A')}",
        f"**Author:** {media_data.get('author', 'N/A')}",
        f"**URL:** {media_data.get('url', 'N/A')}",
        f"**Keywords:** {keywords_str}",
        "---",
        f"**Ingested:** {media_data.get('ingestion_date', 'N/A')}",
        f"**Modified:** {media_data.get('last_modified', 'N/A')}",
        "---",
        "#### Content",
        "```text",
        (media_data.get('content', 'N/A') or 'N/A'),
        "```"
    ]
    return "\n".join(details)

# --- Button Handler Map ---
MEDIA_BUTTON_HANDLERS = {
    # Nav buttons
    "media-nav-all-media": handle_media_nav_button_pressed,
    "media-nav-video": handle_media_nav_button_pressed,
    "media-nav-audio": handle_media_nav_button_pressed,
    "media-nav-web-page": handle_media_nav_button_pressed,
    "media-nav-pdf": handle_media_nav_button_pressed,
    "media-nav-ebook": handle_media_nav_button_pressed,
    "media-nav-document": handle_media_nav_button_pressed,
    "media-nav-xml": handle_media_nav_button_pressed,
    # Search buttons
    "media-search-button-all-media": handle_media_search_button_pressed,
    "media-search-button-video": handle_media_search_button_pressed,
    "media-search-button-audio": handle_media_search_button_pressed,
    "media-search-button-web-page": handle_media_search_button_pressed,
    "media-search-button-pdf": handle_media_search_button_pressed,
    "media-search-button-ebook": handle_media_search_button_pressed,
    "media-search-button-document": handle_media_search_button_pressed,
    "media-search-button-xml": handle_media_search_button_pressed,
    # Load selected buttons
    "media-load-selected-button-all-media": handle_media_load_selected_button_pressed,
    "media-load-selected-button-video": handle_media_load_selected_button_pressed,
    "media-load-selected-button-audio": handle_media_load_selected_button_pressed,
    "media-load-selected-button-web-page": handle_media_load_selected_button_pressed,
    "media-load-selected-button-pdf": handle_media_load_selected_button_pressed,
    "media-load-selected-button-ebook": handle_media_load_selected_button_pressed,
    "media-load-selected-button-document": handle_media_load_selected_button_pressed,
    "media-load-selected-button-xml": handle_media_load_selected_button_pressed,
}

#
# End of media_events.py
########################################################################################################################
