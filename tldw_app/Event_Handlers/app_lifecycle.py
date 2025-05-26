# app_lifecycle.py
# Description:
#
# Imports
import logging
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:

async def handle_copy_logs_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Copy All Logs to Clipboard' button press."""
    logging.info("Copy logs button pressed.")
    try:
        log_widget = app.query_one("#app-log-display", "RichLog") # RichLog is a string here for Textual type hint
        if log_widget.lines:
            all_log_text_parts = [line_item.text for line_item in log_widget.lines if hasattr(line_item, 'text')]
            all_log_text = "\n".join(all_log_text_parts)

            app.copy_to_clipboard(all_log_text)
            app.notify(
                "Logs copied to clipboard!",
                title="Clipboard",
                severity="information",
                timeout=4
            )
            logging.debug(
                f"Copied {len(log_widget.lines)} lines ({len(all_log_text)} chars) from RichLog to clipboard.")
        else:
            app.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
    except app.query_one("QueryError"): # Adjusted for app.query_one
        app.notify("Log widget not found. Cannot copy.", title="Error", severity="error", timeout=4)
        logging.error("Could not find #app-log-display to copy logs.")
    except AttributeError as ae:
        app.notify(f"Error processing log line: {str(ae)}", title="Error", severity="error", timeout=6)
        logging.error(f"AttributeError while processing RichLog lines: {ae}", exc_info=True)
    except Exception as e:
        app.notify(f"Error copying logs: {str(e)}", title="Error", severity="error", timeout=6)
        logging.error(f"Failed to copy logs: {e}", exc_info=True)



#
# End of app_lifecycle.py
########################################################################################################################
