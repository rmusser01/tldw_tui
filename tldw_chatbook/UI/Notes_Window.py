# Notes_Window.py
# Description: This file contains the UI components for the Notes Window 
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, TextArea, Static
#
# Local Imports
from ..Widgets.notes_sidebar_left import NotesSidebarLeft
from ..Widgets.notes_sidebar_right import NotesSidebarRight
# from ..Constants import TAB_NOTES # Not strictly needed if IDs are hardcoded here
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class NotesWindow(Container):
    """
    Container for the Notes Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance # Not strictly used in compose below, but good practice if needed later

    def compose(self) -> ComposeResult:
        yield NotesSidebarLeft(id="notes-sidebar-left")

        with Container(id="notes-main-content"):
            yield TextArea(id="notes-editor-area", classes="notes-editor")
            with Horizontal(id="notes-controls-area"):
                yield Button("☰ L", id="toggle-notes-sidebar-left", classes="sidebar-toggle")
                yield Static()  # Spacer
                yield Button("Save Note", id="notes-save-button", variant="primary")
                yield Static()  # Spacer
                yield Button("R ☰", id="toggle-notes-sidebar-right", classes="sidebar-toggle")

        yield NotesSidebarRight(id="notes-sidebar-right")

#
# End of Notes_Window.py
#######################################################################################################################
