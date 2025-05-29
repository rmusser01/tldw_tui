# tldw_chatbook/UI/LLM_Management_Window.py
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

class LLMManagementWindow(Container):
    """
    Container for the LLM Management Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        # Use a unique ID for this window if it's also set in app.py, e.g., "llm_management-window"
        # The id passed from app.py during instantiation will take precedence if set there.
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="llm-nav-pane", classes="llm-nav-pane"):
            yield Static("LLM Options", classes="sidebar-title")
            yield Button("Llama.cpp", id="llm-nav-llama-cpp", classes="llm-nav-button")
            yield Button("Llamafile", id="llm-nav-llamafile", classes="llm-nav-button")
            yield Button("vLLM", id="llm-nav-vllm", classes="llm-nav-button")
            yield Button("Transformers", id="llm-nav-transformers", classes="llm-nav-button")
            yield Button("Local Models", id="llm-nav-local-models", classes="llm-nav-button")
            yield Button("Download Models", id="llm-nav-download-models", classes="llm-nav-button")

        with Container(id="llm-content-pane", classes="llm-content-pane"):
            yield Container(
                Static("Llama.cpp Management Area - Content Coming Soon!"),
                id="llm-view-llama-cpp",
                classes="llm-view-area",
            )
            yield Container(
                Static("Llamafile Management Area - Content Coming Soon!"),
                id="llm-view-llamafile",
                classes="llm-view-area",
            )
            yield Container(
                Static("vLLM Management Area - Content Coming Soon!"),
                id="llm-view-vllm",
                classes="llm-view-area",
            )
            yield Container(
                Static("Transformers Library Management Area - Content Coming Soon!"),
                id="llm-view-transformers",
                classes="llm-view-area",
            )
            yield Container(
                Static("Local Model Management Area - Content Coming Soon!"),
                id="llm-view-local-models",
                classes="llm-view-area",
            )
            yield Container(
                Static("Model Download Area - Content Coming Soon!"),
                id="llm-view-download-models",
                classes="llm-view-area",
            )

#
# End of LLM_Management_Window.py
#######################################################################################################################
