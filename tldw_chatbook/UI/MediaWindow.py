# tldw_chatbook/UI/MediaWindow.py
#
#
# Imports
from typing import TYPE_CHECKING, List

from loguru import logger
#
# Third-party Libraries
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Label, Input, ListView, TextArea
#
# Local Imports
from ..DB.Client_Media_DB_v2 import MediaDatabase
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

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        # media_types will be fetched and passed by app.py during compose_content_area
        self.media_types_from_db: List[str] = []

    async def on_mount(self) -> None:
        """Fetch media types and build UI elements after mount."""
        self.log.debug("MediaWindow on_mount: Fetching media types...")
        logger.debug("MediaWindow on_mount: Fetching media types...")
        if self.app_instance.notes_service and hasattr(self.app_instance.notes_service, '_get_db'):
            db = self.app_instance.notes_service._get_db(self.app_instance.notes_user_id)
            if db and isinstance(db, self.app_instance.MediaDatabase):  # Use MediaDatabase for type check
                try:
                    self.media_types_from_db = ["All Media"] + sorted(
                        list(set(db.get_distinct_media_types(include_deleted=False, include_trash=False))))
                    self.log.info(
                        f"MediaWindow: Fetched {len(self.media_types_from_db)} distinct media types: {self.media_types_from_db}")
                    logger.info(f"MediaWindow: Fetched {len(self.media_types_from_db)} distinct media types: {self.media_types_from_db}")

                    # Rebuild nav pane with fetched types
                    nav_pane = self.query_one("#media-nav-pane", VerticalScroll)
                    await nav_pane.remove_children()
                    await nav_pane.mount(Static("Media Types", classes="sidebar-title"))
                    if not self.media_types_from_db or self.media_types_from_db == ["Error Loading Types"] or self.media_types_from_db == ["DB Error or No Media in DB"] or self.media_types_from_db == ["Service Error"]:
                        await nav_pane.mount(Label("No media types loaded." if not self.media_types_from_db else self.media_types_from_db[0]))
                    else:
                        for media_type_display_name in self.media_types_from_db:
                            type_slug = slugify(media_type_display_name) # slugify is now imported
                            await nav_pane.mount(Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button"))

                except Exception as e:
                    self.log.error(f"MediaWindow: Error fetching media types: {e}", exc_info=True)
                    logger.error(f"MediaWindow: Error fetching media types: {e}", exc_info=True)
                    self.media_types_from_db = ["Error Loading Types"]
                    # Attempt to update nav_pane even on error to show the error message
                    try:
                        nav_pane = self.query_one("#media-nav-pane", VerticalScroll)
                        await nav_pane.remove_children()
                        await nav_pane.mount(Static("Media Types", classes="sidebar-title"))
                        await nav_pane.mount(Label("Error Loading Types"))
                    except Exception as e_nav_pane_update:
                        self.log.error(f"MediaWindow: Error updating nav_pane after fetch error: {e_nav_pane_update}")
                        logger.error(f"MediaWindow: Error updating nav_pane after fetch error: {e_nav_pane_update}")
            else:
                self.log.error("MediaWindow: MediaDatabase instance not available or invalid.")
                self.media_types_from_db = ["DB Error"]
                logger.error("MediaWindow: MediaDatabase instance not available or invalid.")
                try:
                    nav_pane = self.query_one("#media-nav-pane", VerticalScroll)
                    await nav_pane.remove_children()
                    await nav_pane.mount(Static("Media Types", classes="sidebar-title"))
                    await nav_pane.mount(Label("DB Error"))
                except Exception as e_nav_pane_update:
                    self.log.error(f"MediaWindow: Error updating nav_pane after DB error: {e_nav_pane_update}")
                    logger.error(f"MediaWindow: Error updating nav_pane after DB error: {e_nav_pane_update}")
        else:
            self.log.error("MediaWindow: Notes service or _get_db method not available.")
            logger.error("MediaWindow: Notes service or _get_db method not available.")
            self.media_types_from_db = ["Service Error"]
            try:
                nav_pane = self.query_one("#media-nav-pane", VerticalScroll)
                await nav_pane.remove_children()
                await nav_pane.mount(Static("Media Types", classes="sidebar-title"))
                await nav_pane.mount(Label("Service Error"))
            except Exception as e_nav_pane_update:
                self.log.error(f"MediaWindow: Error updating nav_pane after service error: {e_nav_pane_update}")
                logger.error(f"MediaWindow: Error updating nav_pane after service error: {e_nav_pane_update}")

    def compose(self) -> ComposeResult:
        # self.media_types_from_db should be populated by app.py passing it to constructor
        # or by a dedicated on_mount fetch if compose_content_area isn't re-run.
        # For now, assume it's passed by app.py to constructor or set before compose is called.
        # If it's empty at compose time, it will show "No media types".

        self.log.debug(f"MediaWindow composing. Types available: {self.media_types_from_db}")
        logger.debug(f"MediaWindow composing. Types available: {self.media_types_from_db}")

        # Left Navigation Pane
        with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
            yield Static("Media Types", classes="sidebar-title")
            if not self.media_types_from_db:  # If list is empty after on_mount attempt
                yield Label("No media types loaded.")
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    yield Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button")

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            if not self.media_types_from_db:
                yield Static("No media content areas to display.", classes="placeholder-window")
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    with Container(id=f"media-view-{type_slug}", classes="media-view-area"):
                        yield Label(f"{media_type_display_name} Management",
                                    classes="pane-title")  # More descriptive title
                        with Horizontal(classes="media-controls-bar"):  # Group search and load
                            yield Input(placeholder=f"Search in {media_type_display_name}...",
                                        id=f"media-search-input-{type_slug}",
                                        classes="sidebar-input media-search-input")
                            yield Button("Search", id=f"media-search-button-{type_slug}",
                                         classes="sidebar-button media-search-button")
                        yield ListView(id=f"media-list-view-{type_slug}", classes="sidebar-listview media-items-list")
                        yield Button("View/Load Selected", id=f"media-load-selected-button-{type_slug}",
                                     classes="sidebar-button media-load-button")

                        with VerticalScroll(id=f"media-details-scroll-{type_slug}",
                                            classes="media-details-scroll"):  # Scrollable details
                            yield TextArea("", id=f"media-details-display-{type_slug}",
                                           classes="sidebar-textarea media-details-display", read_only=True)

                    # Hide all view areas by default; app.py watcher will manage visibility
                    self.query_one(f"#media-view-{type_slug}", Container).styles.display = "none"

#
# End of MediaWindow.py
#######################################################################################################################
