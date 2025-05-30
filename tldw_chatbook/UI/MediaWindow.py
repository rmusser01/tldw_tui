# tldw_chatbook/UI/MediaWindow.py
#
#
# Imports
from typing import TYPE_CHECKING, List
#
# Third-party Libraries
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Button, Label, Input, ListView, TextArea

from ..DB.Client_Media_DB_v2 import MediaDatabase

#
# Local Imports
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

def slugify(text: str) -> str:
    """Simple slugify function, robust for empty or non-string."""
    if not isinstance(text, str) or not text:
        return "unknown_type" # Default slug for unexpected types
    return text.lower().replace(" ", "-").replace("/", "-").replace("&", "and").replace("(", "").replace(")", "").replace(":", "").replace(",", "")

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

    def on_mount(self) -> None:
        """Fetch media types and build UI elements after mount."""
        self.log.debug("MediaWindow on_mount: Fetching media types...")
        if self.app_instance.notes_service and hasattr(self.app_instance.notes_service, '_get_db'):
            db = self.app_instance.notes_service._get_db(self.app_instance.notes_user_id)
            if db and isinstance(db, self.app_instance.MediaDatabase):  # Use MediaDatabase for type check
                try:
                    self.media_types_from_db = ["All Media"] + sorted(
                        list(set(db.get_distinct_media_types(include_deleted=False, include_trash=False))))
                    self.log.info(
                        f"MediaWindow: Fetched {len(self.media_types_from_db)} distinct media types: {self.media_types_from_db}")
                except Exception as e:
                    self.log.error(f"MediaWindow: Error fetching media types: {e}", exc_info=True)
                    self.media_types_from_db = ["Error Loading Types"]
            else:
                self.log.error("MediaWindow: MediaDatabase instance not available or invalid.")
                self.media_types_from_db = ["DB Error"]
        else:
            self.log.error("MediaWindow: Notes service or _get_db method not available.")
            self.media_types_from_db = ["Service Error"]

        # Now that media_types_from_db is populated, we can mount the children.
        # Textual generally prefers defining children in compose, but for dynamic content
        # based on async/DB calls at mount, we might need to mount them here.
        # For simplicity, we'll ensure compose can handle an empty list and re-compose if types change significantly.
        # Or, better, make compose use self.media_types_from_db and call self.recompose() if it changes.
        # The provided `app.py` calls `compose_content_area` once. If media_types_from_db needs to be fetched
        # *before* compose, it should happen in app.py and be passed to MediaWindow constructor.
        # Let's assume for now that `app.py` will handle fetching and passing.
        # If `compose` is called *after* `on_mount` completes (which is not standard for initial compose),
        # then the current logic in `compose` using `self.media_types_from_db` would work.
        #
        # Given the structure of app.py's compose_content_area, it's better if app.py fetches the types
        # and passes them to MediaWindow's constructor. Let's adjust that.
        # If not, MediaWindow would need to be more complex, possibly using app.call_later to mount children.

    def compose(self) -> ComposeResult:
        # self.media_types_from_db should be populated by app.py passing it to constructor
        # or by a dedicated on_mount fetch if compose_content_area isn't re-run.
        # For now, assume it's passed by app.py to constructor or set before compose is called.
        # If it's empty at compose time, it will show "No media types".

        self.log.debug(f"MediaWindow composing. Types available: {self.media_types_from_db}")

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
