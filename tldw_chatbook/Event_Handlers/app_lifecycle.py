# app_lifecycle.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING

from textual.css.query import QueryError
#
# 3rd-Party Imports
from textual.widgets import RichLog, Button

#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_copy_logs_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Copy All Logs to Clipboard' button press."""
    logger = getattr(app, 'loguru_logger', logging)  # Use app's logger
    logger.info("Copy logs button pressed.")
    try:
        # Use the actual RichLog type, not a string
        log_widget = app.query_one("#app-log-display", RichLog)  # <--- FIX HERE

        if log_widget.lines:
            # Your existing logic for extracting text from log_widget.lines
            # Make sure this part is robust (e.g. check type of line_item)
            all_log_text_parts = []
            for line_item in log_widget.lines:
                if hasattr(line_item, 'text'):  # Safer check for Strip or similar
                    all_log_text_parts.append(line_item.text)
                else:
                    all_log_text_parts.append(str(line_item))  # Fallback

            all_log_text = "\n".join(all_log_text_parts)

            app.copy_to_clipboard(all_log_text)  # Assuming app has this method
            app.notify(
                "Logs copied to clipboard!",
                title="Clipboard",
                severity="information",
                timeout=4
            )
            logger.debug(
                f"Copied {len(log_widget.lines)} lines ({len(all_log_text)} chars) from RichLog to clipboard.")
        else:
            app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)

    # except app.query_one("QueryError"): # <--- THIS IS ALSO WRONG
    # `QueryError` is an exception type, not something to query.
    # You need to import it and use it directly in the except block.
    except QueryError:  # <--- CORRECT WAY TO CATCH QueryError
        app.notify("Log widget not found. Cannot copy.", title="Error", severity="error", timeout=4)
        logger.error("Could not find #app-log-display to copy logs.")
    except AttributeError as ae:
        app.notify(f"Error processing log line: {str(ae)}", title="Error", severity="error", timeout=6)
        logger.error(f"AttributeError while processing RichLog lines: {ae}", exc_info=True)
    except Exception as e:  # General catch-all
        app.notify(f"Error copying logs: {str(e)}", title="Error", severity="error", timeout=6)
        logger.error(f"Failed to copy logs: {e}", exc_info=True)

# --- Button Handler Map ---
APP_LIFECYCLE_BUTTON_HANDLERS = {
    "copy-logs-button": handle_copy_logs_button_pressed,
}

#
# End of app_lifecycle.py
########################################################################################################################
