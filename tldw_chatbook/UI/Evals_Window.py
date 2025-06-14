# tldw_chatbook/UI/Evals_Window.py
#
# Imports
#
# 3rd-Party Libraries
from typing import TYPE_CHECKING, Optional
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Label
#
# Local Imports
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE
if TYPE_CHECKING:
    from ..app import TldwCli
#
# Configure logger with context
logger = logger.bind(module="EvalsWindow")
#
# #######################################################################################################################
#
# Functions:

# Constants for clarity
EVALS_VIEW_SETUP = "evals-view-setup"
EVALS_VIEW_RESULTS = "evals-view-results"
EVALS_VIEW_MODELS = "evals-view-models"
EVALS_VIEW_DATASETS = "evals-view-datasets"

EVALS_NAV_SETUP = "evals-nav-setup"
EVALS_NAV_RESULTS = "evals-nav-results"
EVALS_NAV_MODELS = "evals-nav-models"
EVALS_NAV_DATASETS = "evals-nav-datasets"

class EvalsWindow(Container):
    """
    A fully self-contained component for the Evals Tab, featuring a collapsible
    sidebar and content areas for evaluation-related functionality.
    """
    # --- STATE LIVES HERE NOW ---
    evals_sidebar_collapsed: reactive[bool] = reactive(False)
    evals_active_view: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    # --- WATCHERS LIVE HERE NOW ---
    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """Dynamically adjusts the evals browser panes when the sidebar is collapsed or expanded."""
        try:
            nav_pane = self.query_one("#evals-nav-pane")
            toggle_button = self.query_one("#evals-sidebar-toggle-button")
            nav_pane.set_class(collapsed, "collapsed")
            toggle_button.set_class(collapsed, "collapsed")
        except QueryError as e:
            logger.warning(f"UI component not found during evals sidebar collapse: {e}")

    def watch_evals_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Shows/hides the relevant content view when the active view slug changes."""
        if old_view:
            try:
                self.query_one(f"#{old_view}").styles.display = "none"
            except QueryError: pass
        if new_view:
            try:
                view_to_show = self.query_one(f"#{new_view}")
                view_to_show.styles.display = "block"
            except QueryError:
                logger.error(f"Could not find new evals view to display: #{new_view}")

    # --- EVENT HANDLERS LIVE HERE NOW ---
    @on(Button.Pressed, "#evals-sidebar-toggle-button")
    def handle_sidebar_toggle(self) -> None:
        """Toggles the sidebar's collapsed state."""
        self.evals_sidebar_collapsed = not self.evals_sidebar_collapsed

    @on(Button.Pressed, ".evals-nav-button")
    def handle_nav_button_press(self, event: Button.Pressed) -> None:
        """Handles a click on an evals navigation button."""
        if event.button.id:
            type_slug = event.button.id.replace("evals-nav-", "")
            self.evals_active_view = f"evals-view-{type_slug}"

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.info(f"EvalsWindow on_mount: UI composed")
        # Set initial active view
        if not self.evals_active_view:
            self.evals_active_view = EVALS_VIEW_SETUP

    def compose(self) -> ComposeResult:
        # Left Navigation Pane
        with VerticalScroll(classes="evals-nav-pane", id="evals-nav-pane"):
            yield Static("Evaluation Tools", classes="sidebar-title")
            yield Button("Evaluation Setup", id=EVALS_NAV_SETUP, classes="evals-nav-button")
            yield Button("Results Dashboard", id=EVALS_NAV_RESULTS, classes="evals-nav-button")
            yield Button("Model Management", id=EVALS_NAV_MODELS, classes="evals-nav-button")
            yield Button("Dataset Management", id=EVALS_NAV_DATASETS, classes="evals-nav-button")

        # Main Content Pane
        with Container(classes="evals-content-pane", id="evals-content-pane"):
            yield Button(
                get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                id="evals-sidebar-toggle-button",
                classes="sidebar-toggle"
            )

            # Create a view for Evaluation Setup
            with Container(id=EVALS_VIEW_SETUP, classes="evals-view-area"):
                yield Static("Evaluation Setup", classes="pane-title")
                yield Label("Configure evaluation runs, datasets, models, etc.")

            # Create a view for Results Dashboard
            with Container(id=EVALS_VIEW_RESULTS, classes="evals-view-area"):
                yield Static("Results Dashboard", classes="pane-title")
                yield Label("View evaluation metrics, comparisons, and reports.")

            # Create a view for Model Management
            with Container(id=EVALS_VIEW_MODELS, classes="evals-view-area"):
                yield Static("Model Management", classes="pane-title")
                yield Label("Manage models under evaluation.")

            # Create a view for Dataset Management
            with Container(id=EVALS_VIEW_DATASETS, classes="evals-view-area"):
                yield Static("Dataset Management", classes="pane-title")
                yield Label("Manage datasets used for evaluations.")

            # Hide all views by default; on_mount will manage visibility
            for view_area in self.query(".evals-view-area"):
                view_area.styles.display = "none"

#
# End of Evals_Window.py
# #######################################################################################################################
