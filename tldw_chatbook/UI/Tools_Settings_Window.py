# tldw_chatbook/UI/Tools_Settings_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
import toml
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button, TextArea
# Local Imports
from tldw_chatbook.config import load_cli_config_and_ensure_existence, DEFAULT_CONFIG_PATH
from tldw_chatbook.config import chachanotes_db, media_db, prompts_db # DB instances
from datetime import datetime
from pathlib import Path
#
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
            yield Button("Configuration File Settings", id="ts-nav-config-file-settings", classes="ts-nav-button")
            yield Button("Database Tools", id="ts-nav-db-tools", classes="ts-nav-button")
            yield Button("Appearance", id="ts-nav-appearance", classes="ts-nav-button")

        with Container(id="tools-settings-content-pane", classes="tools-content-pane"):
            yield Container(
                Static("General Settings Area - Content Coming Soon!"),
                id="ts-view-general-settings",
                classes="ts-view-area",
            )
            yield Container(
                TextArea(
                    text=toml.dumps(load_cli_config_and_ensure_existence()),
                    language="toml",
                    read_only=False, # Made editable
                    id="config-text-area"
                ),
                Container(
                    Button("Save", id="save-config-button", variant="primary"),
                    Button("Reload", id="reload-config-button"),
                    classes="config-button-container"
                ),
                id="ts-view-config-file-settings",
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        config_text_area = self.query_one("#config-text-area", TextArea)

        if button_id == "save-config-button":
            try:
                config_data = toml.loads(config_text_area.text)
                with open(DEFAULT_CONFIG_PATH, "w") as f:
                    toml.dump(config_data, f)
                self.app_instance.notify("Configuration saved successfully.")
            except toml.TOMLDecodeError:
                self.app_instance.notify("Error: Invalid TOML format.", severity="error")
            except IOError:
                self.app_instance.notify("Error: Could not write to configuration file.", severity="error")

        elif button_id == "reload-config-button":
            try:
                config_data = load_cli_config_and_ensure_existence(force_reload=True)
                config_text_area.text = toml.dumps(config_data)
                self.app_instance.notify("Configuration reloaded.")
            except Exception as e:
                self.app_instance.notify(f"Error reloading configuration: {e}", severity="error")

#
# End of Tools_Settings_Window.py
#######################################################################################################################
