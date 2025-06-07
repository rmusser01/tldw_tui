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
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, Input, ListView, TextArea
#
# Local Imports
from ..Utils.text import slugify
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

        with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
            yield Static("Media Types", classes="sidebar-title")
            if not self.media_types_from_db or any(err_msg in self.media_types_from_db for err_msg in ["Error Loading Types", "DB Error", "Service Error", "DB Error or No Media in DB", "No media types loaded."]):
                error_message = "No media types loaded."
                if self.media_types_from_db and isinstance(self.media_types_from_db[0], str):
                    error_message = self.media_types_from_db[0]
                yield Label(error_message)
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    yield Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button")

        with Container(classes="media-content-pane", id="media-content-pane"):
            if not self.media_types_from_db or any(err_msg in self.media_types_from_db for err_msg in ["Error Loading Types", "DB Error", "Service Error", "DB Error or No Media in DB", "No media types loaded."]):
                yield Static("No media content areas to display due to issues loading media types.", classes="placeholder-window")
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    with VerticalScroll(id=f"media-view-{type_slug}", classes="media-view-area"):
                        yield Label(f"{media_type_display_name.title()} Management", classes="pane-title")
                        with Horizontal(classes="media-controls-bar"):
                            yield Input(placeholder=f"Search in {media_type_display_name.title()}...",
                                        id=f"media-search-input-{type_slug}",
                                        classes="media-search-input")
                        with Vertical(classes="media-list-container"):
                            yield ListView(id=f"media-list-view-{type_slug}", classes="media-items-list")
                            with Horizontal(classes="media-pagination-bar"):
                                yield Button("Previous", id=f"media-prev-page-button-{type_slug}", disabled=True)
                                yield Label("Page 1 / 1", id=f"media-page-label-{type_slug}")
                                yield Button("Next", id=f"media-next-page-button-{type_slug}", disabled=True)
                        yield Button("Display Item Details", id=f"media-load-selected-button-{type_slug}", variant="primary")
                        with VerticalScroll(id=f"media-details-scroll-{type_slug}", classes="media-details-scroll"):
                            yield TextArea("", id=f"media-details-display-{type_slug}", classes="media-details-display", read_only=True, language="markdown")

#
# End of MediaWindow.py
#######################################################################################################################
