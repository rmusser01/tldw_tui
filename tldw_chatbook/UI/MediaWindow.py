# tldw_chatbook/UI/MediaWindow.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# Third-party Libraries
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button
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


class MediaWindow(Container):
    """
   Container for the Media Tab's UI, featuring a left navigation pane
   and content areas for different media types.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with VerticalScroll(classes="media-nav-pane", id="media-nav-pane"):
            yield Static("Media Types", classes="sidebar-title")  # Optional title for the nav
            for label, id_suffix in MEDIA_SUB_TABS:
                yield Button(label, id=f"media-nav-{id_suffix}", classes="media-nav-button")

        # Main Content Pane
        with Container(classes="media-content-pane", id="media-content-pane"):
            for label, id_suffix in MEDIA_SUB_TABS:
                # Create a container for each view area
                # Initially, all will be composed, but only one will be visible (controlled by watcher in app.py)
                view_container = Container(
                    Static(f"{label} Content Area - Coming Soon!", classes="placeholder-window"),
                    id=f"media-view-{id_suffix}",
                    classes="media-view-area"
                )
                view_container.styles.display = "none"  # Hide all by default
                yield view_container

#
# End of MediaWindow.py
#######################################################################################################################
