# tldw_chatbook/UI/MediaWindow.py
#
#
# Imports
from typing import TYPE_CHECKING, List, Optional
#
# Third-party Libraries
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, Input, ListView, TextArea, Markdown
#
# Local Imports
from ..Utils.text import slugify
from ..Event_Handlers import media_events
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

MEDIA_SUB_TABS = [
    ("Video/Audio", "video-audio"),
    ("Documents", "documents"),
    ("PDFs", "pdfs"),
    ("Ebooks", "ebooks"),
    ("Websites", "websites"),
    ("MediaWiki", "mediawiki"),
    ("Placeholder", "placeholder")
]

class MediaWindow(Container):
    """
    Container for the Media Tab's UI, featuring a left navigation pane
    and content areas for different media types.
    """
    media_active_view: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.media_types_from_db: List[str] = getattr(self.app_instance, '_media_types_for_ui', [])
        self.log.debug(f"MediaWindow __init__: Received media types: {self.media_types_from_db}")

    def activate_initial_view(self) -> None:
        """Sets the initial active view for the media tab. Called by the app."""
        # This method is called from the app after this window is mounted.
        if not self.media_active_view and self.media_types_from_db:
            initial_slug = slugify("All Media")
            self.log.info(f"MediaWindow: Activating initial view for slug '{initial_slug}'")
            self.media_active_view = f"media-view-{initial_slug}"

            # Set the filter slug and display name on the app so the search function has context
            self.app_instance.current_media_type_filter_slug = initial_slug
            self.app_instance.current_media_type_filter_display_name = "All Media"

            # Set the active view to trigger the watcher that makes the pane visible
            self.media_active_view = f"media-view-{initial_slug}"

            # Now that the view is set, schedule the function that loads its content.
            self.app_instance.call_later(
                media_events.perform_media_search_and_display,
                app=self.app_instance,
                type_slug=initial_slug,
                search_term=""
            )

    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides media sub-views when the active view changes."""
        self.log.debug(f"MediaWindow active view changing from '{old_view}' to: '{new_view}'")

        # Since this watcher is inside MediaWindow, it can safely query its own children.
        for child in self.query(".media-view-area"):
            child.styles.display = "none"

        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
                self.log.info(f"Switched Media view to: {new_view}")
            except QueryError as e:
                self.log.error(f"UI component '{new_view}' not found within MediaWindow: {e}", exc_info=True)

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.log.info(f"MediaWindow on_mount: UI composed with types: {self.media_types_from_db}")
        # The on_mount is now much simpler and no longer needs to fetch data.
        pass

    def compose(self) -> ComposeResult:
        self.log.debug(f"MediaWindow composing. Initial types from __init__: {self.media_types_from_db}")

        # Left Navigation Pane
        with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
            yield Static("Media Types", classes="sidebar-title")
            if not self.media_types_from_db or (
                    len(self.media_types_from_db) == 1 and self.media_types_from_db[0] in ["Error Loading Types",
                                                                                           "DB Error", "Service Error",
                                                                                           "DB Error or No Media in DB",
                                                                                           "No media types loaded."]):
                error_message = "No media types loaded."
                if self.media_types_from_db and isinstance(self.media_types_from_db[0], str):
                    error_message = self.media_types_from_db[0]
                yield Label(error_message)
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    yield Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button")

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            # Create a view for "All Media"
            with Horizontal(id="media-view-all-media", classes="media-view-area"):
                # --- LEFT PANE (for list and controls) ---
                with VerticalScroll(classes="media-left-pane"):
                    yield Label("All Media Management", classes="pane-title")
                    yield Input(placeholder="Search in All Media...", id="media-search-input-all-media",
                                classes="sidebar-input media-search-input")
                    yield ListView(id="media-list-view-all-media", classes="sidebar-listview media-items-list")
                    with Horizontal(classes="media-pagination-bar"):
                        yield Button("Previous", id="media-prev-page-button-all-media", disabled=True)
                        yield Label("Page 1 / 1", id="media-page-label-all-media", classes="media-page-label")
                        yield Button("Next", id="media-next-page-button-all-media", disabled=True)

                # --- RIGHT PANE (for details) ---
                with VerticalScroll(classes="media-right-pane"):
                    yield Markdown("Select an item from the list to see its details.",
                                   id="media-details-display-all-media")

            # Create views for each specific media type
            for media_type_display_name in self.media_types_from_db:
                if media_type_display_name == "All Media": continue  # Already created above
                type_slug = slugify(media_type_display_name)
                with Horizontal(id=f"media-view-{type_slug}", classes="media-view-area"):
                    # --- LEFT PANE ---
                    with VerticalScroll(classes="media-left-pane"):
                        yield Label(f"{media_type_display_name} Management", classes="pane-title")
                        yield Input(placeholder=f"Search in {media_type_display_name}...",
                                    id=f"media-search-input-{type_slug}", classes="sidebar-input media-search-input")
                        yield ListView(id=f"media-list-view-{type_slug}", classes="sidebar-listview media-items-list")
                        with Horizontal(classes="media-pagination-bar"):
                            yield Button("Previous", id=f"media-prev-page-button-{type_slug}", disabled=True)
                            yield Label("Page 1 / 1", id=f"media-page-label-{type_slug}", classes="media-page-label")
                            yield Button("Next", id=f"media-next-page-button-{type_slug}", disabled=True)

                    # --- RIGHT PANE ---
                    with VerticalScroll(classes="media-right-pane"):
                        yield Markdown("Select an item from the list to see its details.",
                                       id=f"media-details-display-{type_slug}")

            # Hide all views by default; app.py watcher will manage visibility
            for view_area in self.query(".media-view-area"):
                view_area.styles.display = "none"

#
# End of MediaWindow.py
#######################################################################################################################
