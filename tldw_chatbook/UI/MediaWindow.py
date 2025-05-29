# tldw_chatbook/UI/MediaWindow.py
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, Button
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import TldwCli

class MediaWindow(Container):
    """
    Container for the Media Tab's UI (Placeholder).
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield Static("Media Window - Content Coming Soon!", classes="placeholder-window") # Add class for centering
        yield Button("Media Placeholder Action", disabled=True)