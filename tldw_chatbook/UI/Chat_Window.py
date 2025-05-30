# Chat_Window.py
# Description: This file contains the UI components for the chat window
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea
#
# Local Imports
from ..Widgets.settings_sidebar import create_settings_sidebar
from ..Widgets.chat_right_sidebar import create_chat_right_sidebar
from ..Constants import TAB_CHAT
from ..Utils.Emoji_Handling import get_char, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, EMOJI_SEND, FALLBACK_SEND, \
    EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, EMOJI_STOP, FALLBACK_STOP

#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class ChatWindow(Container):
    """
    Container for the Chat Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        # Settings Sidebar (Left)
        yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)

        # Main Chat Content Area
        with Container(id="chat-main-content"):
            yield VerticalScroll(id="chat-log")
            with Horizontal(id="chat-input-area"):
                yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), id="toggle-chat-left-sidebar",
                             classes="sidebar-toggle")
                yield TextArea(id="chat-input", classes="chat-input")
                yield Button(get_char(EMOJI_SEND, FALLBACK_SEND), id="send-chat", classes="send-button")
                yield Button(get_char(EMOJI_STOP, FALLBACK_STOP), id="stop-chat-generation", classes="stop-button",
                             disabled=True)
                yield Button(get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), id="toggle-chat-right-sidebar",
                             classes="sidebar-toggle")

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )

#
# End of Chat_Window.py
#######################################################################################################################
