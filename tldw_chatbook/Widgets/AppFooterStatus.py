# tldw_chatbook/Widgets/AppFooterStatus.py
#
# Imports
#
# 3rd-party Libraries
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static
#
# Local Imports
#
########################################################################################################################
#
# AppFooterStatus

class AppFooterStatus(Widget):
    DEFAULT_CSS = """
    AppFooterStatus {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-1;
        width: 100%;
        layout: horizontal; 
        padding: 0 1; 
        /* Removed align: right middle; from parent, will control children individually */
    }

    #footer-key-palette, #footer-key-quit {
        width: auto;
        padding: 0 1; /* Padding around each key binding */
        color: $text-muted;
        dock: left; /* Dock key bindings to the left */
    }
    
    #footer-spacer {
        width: 1fr; /* Takes up remaining space in the middle */
    }

    #internal-db-size-indicator { /* This is for the DB sizes */
        width: auto;
        /* content-align: right; Textual doesn't have content-align for Static directly */
        /* dock: right; Docking within Horizontal might be tricky, align on parent is better */
        color: $text-muted;
        dock: right; /* Dock DB sizes to the right */
        padding: 0 1; /* Add padding to the right of DB sizes as well */
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._key_palette = Static("Ctrl+P (palette)", id="footer-key-palette")
        self._key_quit = Static("Ctrl+Q (quit)", id="footer-key-quit")
        self._db_status_display: Static = Static("", id="internal-db-size-indicator")

    def compose(self) -> ComposeResult:
        yield self._key_palette
        yield self._key_quit
        yield Static(id="footer-spacer") # This will push db_status_display to the right
        yield self._db_status_display # This is the existing DB size display

    def update_db_sizes_display(self, status_string: str) -> None:
        try:
            self._db_status_display.update(status_string)
        except Exception as e:
            # If the app is shutting down, the widget might be gone
            # In a real scenario, you'd use self.log from the widget
            print(f"Error updating AppFooterStatus display: {e}")

#
# End of AppFooterStatus.py
########################################################################################################################
