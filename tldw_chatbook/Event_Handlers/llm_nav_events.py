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

__all__ = [
    "handle_llm_nav_button_pressed",
]

async def handle_llm_nav_button_pressed(app: "TldwCli", button_id: str) -> None:
    """
    Handles the navigation button presses in the LLM Management tab.
    
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
        
        # Add active class to the clicked button
        clicked_button = app.query_one(f"#{button_id}", Button)
        clicked_button.add_class("-active")
        
        logger.info(f"Successfully switched to LLM view: {view_to_activate}")
    except QueryError as e:
        logger.error(f"UI component not found during LLM view switch: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in handle_llm_nav_button_pressed: {e}", exc_info=True)
