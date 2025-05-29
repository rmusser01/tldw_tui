# tldw_chatbook/UI/Ingest_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button
#
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class IngestWindow(Container):
    """
    Container for the Ingest Content Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ingest-nav-pane", classes="ingest-nav-pane"):
            yield Static("Ingestion Methods", classes="sidebar-title")
            yield Button("Ingest Prompts", id="ingest-nav-prompts", classes="ingest-nav-button")
            yield Button("Ingest Characters", id="ingest-nav-characters", classes="ingest-nav-button")
            yield Button("Ingest Media", id="ingest-nav-media", classes="ingest-nav-button")
            yield Button("Ingest Notes", id="ingest-nav-notes", classes="ingest-nav-button")
            yield Button("Ingest Media via tldw", id="ingest-nav-tldw", classes="ingest-nav-button")

        with Container(id="ingest-content-pane", classes="ingest-content-pane"):
            yield Container(
                Static("Prompt Ingestion Area - Content Coming Soon!"),
                id="ingest-view-prompts",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Character Ingestion Area - Content Coming Soon!"),
                id="ingest-view-characters",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Media Ingestion Area - Content Coming Soon!"),
                id="ingest-view-media",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Note Ingestion Area - Content Coming Soon!"),
                id="ingest-view-notes",
                classes="ingest-view-area",
            )
            # Placeholder for tldw view if it's different
            yield Container(
                Static("TLDW Media Ingestion Area - Content Coming Soon!"),
                id="ingest-view-tldw", # Assuming this ID corresponds to the button
                classes="ingest-view-area",
            )

#
# End of Logs_Window.py
#######################################################################################################################
