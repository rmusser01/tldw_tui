# tldw_chatbook/UI/Tools_Settings_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class ToolsSettingsWindow(Container):
    """
    Container for the Tools & Settings Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="tools-settings-nav-pane", classes="tools-nav-pane"):
            yield Static("Navigation", classes="sidebar-title")
            yield Button("General Settings", id="ts-nav-general-settings", classes="ts-nav-button")
            yield Button("API Keys", id="ts-nav-api-keys", classes="ts-nav-button")
            yield Button("Database Tools", id="ts-nav-db-tools", classes="ts-nav-button")
            yield Button("Appearance", id="ts-nav-appearance", classes="ts-nav-button")

        with Container(id="tools-settings-content-pane", classes="tools-content-pane"):
            yield Container(
                Static("General Settings Area - Content Coming Soon!"),
                id="ts-view-general-settings",
                classes="ts-view-area",
            )
            yield Container(
                Static("API Keys Management Area - Content Coming Soon!"),
                id="ts-view-api-keys",
                classes="ts-view-area",
            )
            yield Container(
                Static("Database Tools Area - Content Coming Soon!"),
                id="ts-view-db-tools",
                classes="ts-view-area",
            )
            yield Container(
                Static("Appearance Settings Area - Content Coming Soon!"),
                id="ts-view-appearance",
                classes="ts-view-area",
            )

#
# End of Tools_Settings_Window.py
#######################################################################################################################
