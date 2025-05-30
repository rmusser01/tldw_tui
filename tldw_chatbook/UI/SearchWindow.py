# tldw_chatbook/UI/SearchWindow.py
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:


# Define constants for sub-tab view IDs and button IDs for clarity
# These could also be in Constants.py but are kept here for encapsulation with SearchWindow's structure
# View IDs
SEARCH_VIEW_RAG_QA = "search-view-rag-qa"
SEARCH_VIEW_RAG_CHAT = "search-view-rag-chat"
SEARCH_VIEW_EMBEDDINGS_CREATION = "search-view-embeddings-creation"
SEARCH_VIEW_RAG_MANAGEMENT = "search-view-rag-management"
SEARCH_VIEW_EMBEDDINGS_MANAGEMENT = "search-view-embeddings-management"

# Button IDs
SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_EMBEDDINGS_CREATION = "search-nav-embeddings-creation"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_EMBEDDINGS_MANAGEMENT = "search-nav-embeddings-management"


class SearchWindow(Container):
    """
    Container for the Search Tab's UI, featuring a vertical tab bar and content areas.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        # Ensure the main window has the ID used in app.py for tab switching
        # The id="search-window" is set when this class is instantiated in app.py's compose_content_area

    def compose(self) -> ComposeResult:
        # Main horizontal layout for the Search tab.
        # The SearchWindow itself (id="search-window") will have layout: horizontal from .window class.

        # Left Vertical Tab Bar
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("Embeddings Creation", id=SEARCH_NAV_EMBEDDINGS_CREATION, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            yield Button("Embeddings Management", id=SEARCH_NAV_EMBEDDINGS_MANAGEMENT, classes="search-nav-button")

        # Right Content Pane
        with Container(id="search-content-pane", classes="search-content-pane"):
            # Individual view areas, only one visible at a time. Watcher handles display.
            yield Container(Static("RAG QA Content - Coming Soon!"), id=SEARCH_VIEW_RAG_QA, classes="search-view-area")
            yield Container(Static("RAG Chat Content - Coming Soon!"), id=SEARCH_VIEW_RAG_CHAT,
                            classes="search-view-area")
            yield Container(Static("Embeddings Creation Content - Coming Soon!"), id=SEARCH_VIEW_EMBEDDINGS_CREATION,
                            classes="search-view-area")
            yield Container(Static("RAG Management Content - Coming Soon!"), id=SEARCH_VIEW_RAG_MANAGEMENT,
                            classes="search-view-area")
            yield Container(Static("Embeddings Management Content - Coming Soon!"),
                            id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area")

#
# End of SearchWindow.py
########################################################################################################################
