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
        # Initialize from app_instance._media_types_for_ui, which should be pre-fetched
        self.media_types_from_db: List[str] = getattr(self.app_instance, '_media_types_for_ui', [])
        self.log.debug(f"MediaWindow __init__: Received media types: {self.media_types_from_db}")

    async def _rebuild_nav_pane(self, types_to_display: List[str]):
        """Helper function to rebuild the navigation pane."""
        nav_pane = self.query_one("#media-nav-pane", VerticalScroll)
        await nav_pane.remove_children()
        await nav_pane.mount(Static("Media Types", classes="sidebar-title"))

        if not types_to_display or types_to_display == ["Error Loading Types"] or types_to_display == ["DB Error"] or types_to_display == ["Service Error"] or types_to_display == ["DB Error or No Media in DB"]:
            error_message = "No media types loaded."
            if types_to_display and isinstance(types_to_display[0], str): # Check if it's an error message
                error_message = types_to_display[0]
            await nav_pane.mount(Label(error_message))
            self.log.info(f"MediaWindow: Nav pane rebuilt with message: {error_message}")
        else:
            for media_type_display_name in types_to_display:
                type_slug = slugify(media_type_display_name)
                await nav_pane.mount(Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button"))
            self.log.info(f"MediaWindow: Nav pane rebuilt with {len(types_to_display)} media types.")

    async def on_mount(self) -> None:
        """Validate pre-fetched media types and conditionally refresh if needed."""
        self.log.info(f"MediaWindow on_mount: Initial media types from constructor: {self.media_types_from_db}")

        # Check if pre-fetched types are valid and sufficient
        # Valid means not empty and does not solely contain error indicators.
        initial_types_are_valid = bool(self.media_types_from_db) and not \
            (len(self.media_types_from_db) == 1 and self.media_types_from_db[0] in ["Error Loading Types", "DB Error", "Service Error", "DB Error or No Media in DB"])

        if initial_types_are_valid:
            self.log.info("MediaWindow on_mount: Pre-fetched media types are considered valid. Using them.")
            # If compose already built the nav pane correctly with these, we might not need to rebuild.
            # However, to ensure consistency if compose had an issue or if types were somehow empty during compose:
            await self._rebuild_nav_pane(self.media_types_from_db)
            return # Skip re-fetching

        self.log.info("MediaWindow on_mount: Pre-fetched types are invalid or empty. Attempting to fetch.")
        # Conditional Re-fetch Logic (if pre-fetched were not valid)
        if self.app_instance.media_db: # Directly use app_instance.media_db
            db = self.app_instance.media_db
            self.log.debug(f"MediaWindow on_mount: Got media_db instance: {db}")
            is_media_db_instance = isinstance(db, MediaDatabase) # Corrected class name
            self.log.debug(f"MediaWindow on_mount: Is instance of MediaDatabase? {is_media_db_instance}")

            if is_media_db_instance:
                try:
                    fetched_types = ["All Media"] + sorted(
                        list(set(db.get_distinct_media_types(include_deleted=False, include_trash=False))))
                    self.log.info(
                        f"MediaWindow on_mount: Fetched {len(fetched_types)} distinct media types: {fetched_types}")

                    # Compare with initial to see if an update is truly needed
                    if self.media_types_from_db != fetched_types:
                        self.media_types_from_db = fetched_types
                        await self._rebuild_nav_pane(self.media_types_from_db)
                    else:
                        self.log.info("MediaWindow on_mount: Fetched types are same as initial valid types. No UI rebuild needed for nav pane.")

                except Exception as e:
                    self.log.error(f"MediaWindow on_mount: Error fetching media types: {e}", exc_info=True)
                    self.media_types_from_db = ["Error Loading Types"]
                    await self._rebuild_nav_pane(self.media_types_from_db)
            else:
                self.log.error("MediaWindow on_mount: self.app_instance.media_db is not a valid MediaDatabase instance.")
                self.media_types_from_db = ["DB Error"] # More specific error
                await self._rebuild_nav_pane(self.media_types_from_db)
        else:
            self.log.error("MediaWindow on_mount: self.app_instance.media_db is None. Cannot fetch types.")
            self.media_types_from_db = ["Service Error"] # Indicates media_db service itself is missing
            await self._rebuild_nav_pane(self.media_types_from_db)

    def compose(self) -> ComposeResult:
        self.log.debug(f"MediaWindow composing. Initial types from __init__: {self.media_types_from_db}")

        # Left Navigation Pane
        with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
            yield Static("Media Types", classes="sidebar-title")
            # Logic to display types or error message based on self.media_types_from_db
            # This will be populated by on_mount if types were invalid/empty initially
            if not self.media_types_from_db or \
               (len(self.media_types_from_db) == 1 and self.media_types_from_db[0] in ["Error Loading Types", "DB Error", "Service Error", "DB Error or No Media in DB", "No media types loaded."]):
                error_message = "No media types loaded."
                if self.media_types_from_db and isinstance(self.media_types_from_db[0], str):
                    error_message = self.media_types_from_db[0]
                yield Label(error_message)
                self.log.info(f"MediaWindow compose: Nav pane shows message: {error_message}")
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    yield Button(media_type_display_name, id=f"media-nav-{type_slug}", classes="media-nav-button")
                self.log.info(f"MediaWindow compose: Nav pane composed with {len(self.media_types_from_db)} media types.")

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            if not self.media_types_from_db or \
               (len(self.media_types_from_db) == 1 and self.media_types_from_db[0] in ["Error Loading Types", "DB Error", "Service Error", "DB Error or No Media in DB", "No media types loaded."]):
                yield Static("No media content areas to display due to issues loading media types.", classes="placeholder-window")
                self.log.info("MediaWindow compose: Content pane shows placeholder due to no valid media types.")
            else:
                for media_type_display_name in self.media_types_from_db:
                    type_slug = slugify(media_type_display_name)
                    with Container(id=f"media-view-{type_slug}", classes="media-view-area"):
                        yield Label(f"{media_type_display_name} Management",
                                    classes="pane-title")
                        with Horizontal(classes="media-controls-bar"):
                            yield Input(placeholder=f"Search in {media_type_display_name}...",
                                        id=f"media-search-input-{type_slug}",
                                        classes="sidebar-input media-search-input")
                            yield Button("Search", id=f"media-search-button-{type_slug}",
                                         classes="sidebar-button media-search-button")
                        yield ListView(id=f"media-list-view-{type_slug}", classes="sidebar-listview media-items-list")
                        yield Button("View/Load Selected", id=f"media-load-selected-button-{type_slug}",
                                     classes="sidebar-button media-load-button")

                        with VerticalScroll(id=f"media-details-scroll-{type_slug}",
                                            classes="media-details-scroll"):
                            yield TextArea("", id=f"media-details-display-{type_slug}",
                                           classes="sidebar-textarea media-details-display", read_only=True)

                    # Hide all view areas by default; app.py watcher will manage visibility
                    self.query_one(f"#media-view-{type_slug}", Container).styles.display = "none"
                self.log.info(f"MediaWindow compose: Content pane composed for {len(self.media_types_from_db)} media types.")

#
# End of MediaWindow.py
#######################################################################################################################
