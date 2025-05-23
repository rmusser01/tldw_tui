# character_sidebar.py
# Description: character sidebar widget
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Collapsible, Placeholder
#
# Local Imports
# (Add any necessary local imports here if needed for actual content later)
#
#######################################################################################################################
#
# Functions:

def create_character_sidebar(id_prefix: str) -> ComposeResult:
    """Yield the widgets for the character settings sidebar.

    The sidebar is intended for managing character information and conversation data.
    For now, it contains placeholder content.
    """

    # id_prefix will likely be "chat" when called from the main app for the chat window's right sidebar.
    # The main container ID should be unique, e.g., f"{id_prefix}-character-sidebar" or just "character-sidebar".
    # For consistency with the app.py modification, we'll use "character-sidebar" as the main ID.
    with VerticalScroll(id="character-sidebar", classes="sidebar"): # Ensure this ID matches the watcher in app.py
        yield Static("Character Info", classes="sidebar-title")

        with Collapsible(title="Character Details", collapsed=False):
            yield Placeholder("Character Name")
            yield Placeholder("Character Background")
            yield Placeholder("Character Personality")

        with Collapsible(title="Conversation Data", collapsed=True):
            yield Placeholder("Current Conversation Stats")
            yield Placeholder("Mood Analysis (Placeholder)")

        with Collapsible(title="Other Character Tools", collapsed=True):
            yield Placeholder("Tool 1")
            yield Placeholder("Tool 2")

        logging.debug(f"Character sidebar created with id_prefix: {id_prefix}")

#
# End of character_sidebar.py
#######################################################################################################################
