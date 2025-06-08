# tldw_chatbook/Event_Handlers/media_events.py
#
#
# Imports
import logging
import math
from typing import TYPE_CHECKING, Dict, Any

from textual.containers import Vertical
#
# 3rd-party Libraries
from textual.widgets import ListView, Input, TextArea, Label, ListItem, Button, Markdown, Static  # Added ListItem
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
    """
    Handles loading a selected media item's full details and displaying them
    as rendered Markdown.
    """
    logger = app.loguru_logger
    button_id = event.button.id

    try:
        type_slug = button_id.replace("media-load-selected-button-", "")
        list_view = app.query_one(f"#media-list-view-{type_slug}", ListView)
        # Query for the Markdown widget instead of TextArea
        details_display = app.query_one(f"#media-details-display-{type_slug}", Markdown)

        if not list_view.highlighted_child or not hasattr(list_view.highlighted_child, 'media_data'):
            app.notify("No media item selected.", severity="warning")
            # Use the .update() method for the Markdown widget
            await details_display.update("No item selected.")
            app.current_loaded_media_item = None
            return

        # Use .update() to show a loading message
        await details_display.update("### Loading full details...")

        # 1. Get the lightweight data from the list item and safely extract the ID
        lightweight_media_data = list_view.highlighted_child.media_data
        media_id_raw = lightweight_media_data.get('id')

        if media_id_raw is None:
            await details_display.update("### Error: Selected item has no ID.")
            return

        # 2. Ensure the ID is an integer before querying the database
        try:
            media_id = int(media_id_raw)
        except (ValueError, TypeError):
            await details_display.update(f"### Error: Invalid media ID format '{media_id_raw}'.")
            return

        if not app.media_db:
            await details_display.update("### Error: Database connection is not available.")
            return

        # 3. Fetch the FULL media item from the database using the existing DB function.
        #    We set include_trash=True so you can view items that are in the trash.
        logger.info(f"Fetching full details for media item ID: {media_id}")
        full_media_data = app.media_db.get_media_by_id(media_id, include_trash=True)

        if full_media_data is None:
            await details_display.update(
                f"### Error\n\nCould not find media item with ID `{media_id}` in the database. It may have been permanently deleted.")
            app.current_loaded_media_item = None
            return

        # 4. Update the reactive variable and format the COMPLETE data for display
        app.current_loaded_media_item = full_media_data
        logger.info(f"Loaded full media item ID {media_id} into reactive state.")

        markdown_details_string = format_media_details_as_markdown(app, full_media_data)

        # Use the .update() method to render the final Markdown string
        await details_display.update(markdown_details_string)
        details_display.scroll_home(animate=False)

    except QueryError as e:
        logger.error(f"UI component not found for media load selected '{button_id}': {e}", exc_info=True)
        app.notify("Load details UI error.", severity="error")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading media details: {e}", exc_info=True)
        # Check if details_display was defined before trying to use it
        if 'details_display' in locals():
            await details_display.update(f"### An unexpected error occurred\n\n```\n{e}\n```")
        app.notify("Error loading details.", severity="error")


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
    """Performs search in media DB and populates the ListView with rich, informative items."""
    logger = app.loguru_logger
    list_view_id = f"media-list-view-{type_slug}"

    try:
        list_view = app.query_one(f"#{list_view_id}", ListView)
        await list_view.clear()

        # Also clear the details panel for this view
        try:
            details_display = app.query_one(f"#media-details-display-{type_slug}", Markdown)
            await details_display.update("### Select an item to see details")
        except QueryError:
            pass

        if not app.media_db:
            raise RuntimeError("Media DB service not available.")

        media_types_filter = None
        if type_slug != "all-media":
            db_media_type = type_slug.replace('-', '_')
            media_types_filter = [db_media_type]

        query_arg = search_term if search_term else None
        fields_arg = ['title', 'content', 'author', 'url', 'type']

        results, total_matches = app.media_db.search_media_db(
            search_query=query_arg,
            media_types=media_types_filter,
            search_fields=fields_arg,
            sort_by="last_modified_desc",
            page=getattr(app, 'media_current_page', 1),
            results_per_page=RESULTS_PER_PAGE,
            include_trash=False,
            include_deleted=False
        )

        if not results:
            await list_view.append(ListItem(Label("No media items found.")))
        else:
            for item in results:
                title = item.get('title', 'Untitled')
                ingestion_date_str = item.get('ingestion_date', '')
                ingestion_date = ingestion_date_str.split('T')[0] if ingestion_date_str else 'N/A'

                content = item.get('content', '').strip()
                snippet = (content[:80] + '...') if content else "No content available."

                # Create a richer ListItem with a Vertical layout
                rich_list_item = ListItem(
                    Vertical(
                        Label(f"{title}", classes="media-item-title"),
                        Static(snippet, classes="media-item-snippet"),
                        Static(f"Type: {item.get('type')}  |  Ingested: {ingestion_date}", classes="media-item-meta")
                    )
                )
                rich_list_item.media_data = item  # Attach data for selection
                await list_view.append(rich_list_item)

        # Update pagination controls
        try:
            total_pages = math.ceil(total_matches / RESULTS_PER_PAGE) if total_matches > 0 else 1
            page_label = app.query_one(f"#media-page-label-{type_slug}", Label)
            prev_button = app.query_one(f"#media-prev-page-button-{type_slug}", Button)
            next_button = app.query_one(f"#media-next-page-button-{type_slug}", Button)
            current_page = getattr(app, 'media_current_page', 1)
            page_label.update(f"Page {current_page} / {total_pages}")
            prev_button.disabled = (current_page <= 1)
            next_button.disabled = (current_page >= total_pages)
        except QueryError:
            pass

    except (QueryError, RuntimeError, Exception) as e:
        logger.error(f"Error during media search for type '{type_slug}': {e}", exc_info=True)


async def handle_media_list_item_selected(app: 'TldwCli', event: ListView.Selected) -> None:
    """
    Handles a media item being selected in the ListView, populating the
    new structured details pane.
    """
    logger = app.loguru_logger

    # Get a reference to the parent MediaView to query within it
    try:
        # NOTE: The query selectors are now static IDs because only one view is visible at a time.
        title_widget = app.query_one("#details-title", Label)
        meta_widget = app.query_one("#details-meta-block", Static)
        timestamp_widget = app.query_one("#details-timestamp-block", Static)
        content_header_widget = app.query_one("#details-content-header", Label)
        content_widget = app.query_one("#details-content-area", TextArea)
    except QueryError as e:
        logger.error(f"Could not find a details widget during selection: {e}")
        return

    # Clear previous details
    title_widget.update("Loading...")
    meta_widget.update("")
    timestamp_widget.update("")
    content_header_widget.display = False
    content_widget.load_text("")

    if not hasattr(event.item, 'media_data'):
        title_widget.update("This item has no data to display.")
        return

    lightweight_media_data = event.item.media_data
    media_id = int(lightweight_media_data.get('id', 0))

    if not media_id or not app.media_db:
        title_widget.update("Error: Cannot load details.")
        return

    full_media_data = app.media_db.get_media_by_id(media_id, include_trash=True)

    if full_media_data is None:
        title_widget.update(f"Error: Could not find media item with ID {media_id}.")
        return

    # --- POPULATE THE NEW WIDGETS ---
    app.current_loaded_media_item = full_media_data

    # 1. Title
    title_widget.update(full_media_data.get('title', 'Untitled'))

    # 2. Metadata Block
    keywords_str = ", ".join(fetch_keywords_for_media(app.media_db, media_id)) or "N/A"
    meta_text = (
        f"[b]ID:[/] {full_media_data.get('id', 'N/A')}  [b]UUID:[/] {full_media_data.get('uuid', 'N/A')}\n"
        f"[b]Type:[/] {full_media_data.get('type', 'N/A')}  [b]Author:[/] {full_media_data.get('author', 'N/A')}\n"
        f"[b]URL:[/] {full_media_data.get('url', 'N/A')}\n"
        f"[b]Keywords:[/] {keywords_str}"
    )
    meta_widget.update(Text.from_markup(meta_text))

    # 3. Timestamp Block
    timestamp_text = (
        f"[dim]Ingested: {full_media_data.get('ingestion_date', 'N/A')} | "
        f"Modified: {full_media_data.get('last_modified', 'N/A')}[/dim]"
    )
    timestamp_widget.update(Text.from_markup(timestamp_text))

    # 4. Content
    content_header_widget.display = True
    content_widget.load_text(full_media_data.get('content', 'No content available.'))

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
    """Formats media item details into a clean, scannable Markdown string."""
    if not media_data:
        return "### No Media Item Loaded"

    keywords_str = "N/A"
    media_id = media_data.get('id')
    if app.media_db and media_id:
        try:
            keywords = fetch_keywords_for_media(app.media_db, media_id)
            keywords_str = ", ".join(keywords) if keywords else "N/A"
        except Exception as e:
            keywords_str = f"*Error fetching keywords: {e}*"

    title = media_data.get('title', 'Untitled')

    # Create a compact, multi-line metadata block
    meta_header = (
        f"**ID:** `{media_data.get('id', 'N/A')}`  **UUID:** `{media_data.get('uuid', 'N/A')}`\n"
        f"**Type:** `{media_data.get('type', 'N/A')}`  **Author:** `{media_data.get('author', 'N/A')}`\n"
        f"**URL:** `{media_data.get('url', 'N/A')}`\n"
        f"**Keywords:** {keywords_str}"
    )

    # Format timestamps
    ingested = f"**Ingested:** `{media_data.get('ingestion_date', 'N/A')}`"
    modified = f"**Modified:** `{media_data.get('last_modified', 'N/A')}`"

    content = media_data.get('content', 'N/A') or 'N/A'

    # Assemble the final markdown string
    final_markdown = (
        f"## {title}\n\n"
        f"{meta_header}\n\n"
        "---\n\n"
        f"{ingested}\n{modified}\n\n"
        "### Content\n\n"
        "```text\n"
        f"{content}\n"
        "```"
    )

    return final_markdown

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
    # Pagination buttons
    "media-prev-page-button-all-media": handle_media_page_change_button_pressed,
    "media-next-page-button-all-media": handle_media_page_change_button_pressed,
}

#
# End of media_events.py
########################################################################################################################
