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
        background: $primary-background-darken-1; /* Or any color from your theme */
        width: 100%;
        layout: horizontal; /* To arrange items inside if needed */
        align: right middle; /* Aligns children to the right */
        padding: 0 1; /* Padding for the footer itself */
    }

    #internal-db-size-indicator {
        width: auto;
        /* content-align: right; Textual doesn't have content-align for Static directly */
        /* dock: right; Docking within Horizontal might be tricky, align on parent is better */
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_status_display: Static = Static("", id="internal-db-size-indicator")

    def compose(self) -> ComposeResult:
        yield self._db_status_display

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
