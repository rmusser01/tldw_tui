"""llm_management_events_ollama.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **Ollama** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates Ollama-specific logic from the main llm_management_events.py.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.containers import Container
from textual.css.query import QueryError

if TYPE_CHECKING:
    from ..app import TldwCli

__all__ = [
    "handle_ollama_nav_button_pressed",
]

###############################################################################
# ─── Ollama UI helpers ──────────────────────────────────────────────────────
###############################################################################


async def handle_ollama_nav_button_pressed(app: "TldwCli") -> None:
    """Handle the Ollama navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        view_areas = content_pane.query(".llm-view-area")

        for view in view_areas:
            if view.id:  # Only hide if it has an ID
                logger.debug(f"Hiding view #{view.id}")
                view.styles.display = "none"
            else: # pragma: no cover
                logger.warning("Found a .llm-view-area without an ID, not hiding it.")

        ollama_view = app.query_one("#llm-view-ollama", Container)
        logger.debug(f"Showing view #{ollama_view.id}")
        ollama_view.styles.display = "block"
        #app.notify("Switched to Ollama view.")

    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to Ollama view: Could not find required UI elements.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while switching to Ollama view.", severity="error")

#
# End of llm_management_events_ollama.py
########################################################################################################################
