# side_bar_events.py
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

async def handle_sidebar_toggle_button_pressed(app: 'TldwCli', button_id: str) -> None:
    """Handles all sidebar toggle button presses."""
    if button_id == "toggle-chat-sidebar":
        app.chat_sidebar_collapsed = not app.chat_sidebar_collapsed
        logging.debug("Chat sidebar now %s", "collapsed" if app.chat_sidebar_collapsed else "expanded")
    elif button_id == "toggle-character-sidebar":
        app.character_sidebar_collapsed = not app.character_sidebar_collapsed
        logging.debug("Character sidebar now %s", "collapsed" if app.character_sidebar_collapsed else "expanded")
    elif button_id == "toggle-notes-sidebar-left":
        app.notes_sidebar_left_collapsed = not app.notes_sidebar_left_collapsed
        logging.debug("Notes left sidebar now %s", "collapsed" if app.notes_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-notes-sidebar-right":
        app.notes_sidebar_right_collapsed = not app.notes_sidebar_right_collapsed
        logging.debug("Notes right sidebar now %s", "collapsed" if app.notes_sidebar_right_collapsed else "expanded")
    elif button_id == "toggle-conv-char-left-sidebar":
        app.conv_char_sidebar_left_collapsed = not app.conv_char_sidebar_left_collapsed
        logging.debug("CCP left sidebar now %s", "collapsed" if app.conv_char_sidebar_left_collapsed else "expanded")
    elif button_id == "toggle-conv-char-right-sidebar":
        app.conv_char_sidebar_right_collapsed = not app.conv_char_sidebar_right_collapsed
        logging.debug("CCP right sidebar now %s", "collapsed" if app.conv_char_sidebar_right_collapsed else "expanded")
    else:
        logging.warning(f"Unhandled sidebar toggle button ID: {button_id}")

#
# End of side_bar_events.py
########################################################################################################################
