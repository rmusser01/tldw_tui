# tab_events.py
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

async def handle_tab_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles tab switching button presses."""
    new_tab_id = button_id.replace("tab-", "")
    logging.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")
    if new_tab_id != app.current_tab:
        app.current_tab = new_tab_id  # This will trigger the watch_current_tab method in TldwCli
    else:
        logging.debug(f"Already on tab '{new_tab_id}'. Ignoring.")

#
# End of tab_events.py
########################################################################################################################
