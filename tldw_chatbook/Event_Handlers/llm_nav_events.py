"""llm_nav_events.py

Contains event handlers for managing navigation between different LLM views.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.css.query import QueryError
from textual.widgets import Button

if TYPE_CHECKING:
    from ..app import TldwCli  # pragma: no cover – runtime import only

# Import the specific handler
from tldw_chatbook.Event_Handlers.llm_management_events.llm_management_events_ollama import handle_ollama_nav_button_pressed
from tldw_chatbook.Event_Handlers.llm_management_events.llm_management_events import handle_mlx_lm_nav_button_pressed

__all__ = [
    "handle_llm_nav_button_pressed",
]

async def handle_llm_nav_button_pressed(app: "TldwCli", button_id: str) -> None:
    """
    Handles the navigation button presses in the LLM Management tab.
    Dispatches to specific handlers if available, otherwise uses a generic approach.

    Args:
        app: The TldwCli app instance
        button_id: The ID of the button that was pressed
    """
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info(f"LLM nav button pressed: {button_id}")

    # Map button IDs to view IDs
    view_to_activate = button_id.replace("llm-nav-", "llm-view-")
    logger.debug(f"Activating LLM view: {view_to_activate}")

    try:
        # Update app's reactive property to show the selected view
        app.llm_active_view = view_to_activate

        # Remove active class from all nav buttons
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
        # The reactive variable app.llm_active_view should also be set if the
        # specific handler doesn't do it, to keep external state consistent.
        # However, handle_ollama_nav_button_pressed directly manipulates view display.
        # For consistency, we might want specific handlers to also update app.llm_active_view.
        # For now, let's assume specific handlers manage the view entirely.
        # If watch_llm_active_view is still active, this might cause double view changes or conflicts.
        # The handle_ollama_nav_button_pressed already sets display properties.
        # To prevent conflict with the watcher, we might avoid setting app.llm_active_view here for this specific case,
        # OR ensure the watcher is idempotent / specific handlers also set app.llm_active_view.
        # For now, let the specific handler do its job. The watcher will run if app.llm_active_view changes.
        # If handle_ollama_nav_button_pressed doesn't change app.llm_active_view, the watcher won't re-hide other views.
        # This seems acceptable.
    elif button_id == "llm-nav-mlx-lm":
        await handle_mlx_lm_nav_button_pressed(app)
    # Add other specific handlers here:
    # elif button_id == "llm-nav-another":
    #     await handle_another_llm_nav_button_pressed(app)
    else:
        # Generic fallback for other LLM navigation buttons
        view_to_activate = button_id.replace("llm-nav-", "llm-view-")
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
