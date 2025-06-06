# llm_nav_events.py
# Description: Contains event handlers for managing navigation between different LLM views.
#
# Imports
from __future__ import annotations
#
import logging
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.css.query import QueryError
from textual.widgets import Button
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli  # pragma: no cover – runtime import only
# Import the specific handler
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events_ollama import handle_ollama_nav_button_pressed
#
#######################################################################################################################
#
# Functions:

__all__ = ["handle_llm_nav_button_pressed"]

async def handle_llm_nav_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """
    Handles navigation button presses in the LLM Management tab.
    Dispatches to specific handlers if available, otherwise uses a generic approach.
    """
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    button_id = event.button.id
    logger.info(f"LLM nav button pressed: {button_id}")

    view_to_activate = button_id.replace("llm-nav-", "llm-view-")

    try:
        nav_pane = app.query_one("#llm-nav-pane")
        for nav_button in nav_pane.query(".llm-nav-button"):
            nav_button.remove_class("-active")
    except QueryError as e:
        logger.error(f"Could not query #llm-nav-pane or .llm-nav-button: {e}", exc_info=True)
        # Proceeding because view switching might still work

    # Add active class to the specifically clicked button
    try:
        clicked_button = app.query_one(f"#{button_id}", Button)
        clicked_button.add_class("-active")
    except QueryError as e:
        logger.error(f"Could not query clicked button #{button_id}: {e}", exc_info=True)
        # Proceeding because view switching might still work

    # Specific handlers
    if button_id == "llm-nav-ollama":
        await handle_ollama_nav_button_pressed(app)
    else:
        # Generic fallback for other LLM nav buttons
        logger.debug(f"Using generic activation for LLM view: {view_to_activate}")
        try:
            # Update app's reactive property to show the selected view
            # This relies on the watcher `watch_llm_active_view` in app.py
            app.llm_active_view = view_to_activate
            logger.info(f"Successfully set app.llm_active_view to: {view_to_activate} for generic handling.")
        except Exception as e: # Catch errors related to setting reactive or if watcher fails
            logger.error(f"Error in generic LLM view activation for '{view_to_activate}': {e}", exc_info=True)

    # Note: The original code had a single try-except.
    # Splitting it can help isolate where QueryErrors happen (nav pane vs content pane switching).
    # The specific handlers (like handle_ollama_nav_button_pressed) have their own try-except.

# --- Button Handler Map ---
LLM_NAV_BUTTON_HANDLERS = {
    "llm-nav-llamafile": handle_llm_nav_button_pressed,
    "llm-nav-llamacpp": handle_llm_nav_button_pressed,
    "llm-nav-ollama": handle_llm_nav_button_pressed,
    "llm-nav-vllm": handle_llm_nav_button_pressed,
    "llm-nav-transformers": handle_llm_nav_button_pressed,
    "llm-nav-mlx-lm": handle_llm_nav_button_pressed,
    "llm-nav-onnx": handle_llm_nav_button_pressed,
}

#
# End of llm_nav_events.py
#######################################################################################################################
