# Chat_Window.py
# Description: This file contains the UI components for the chat window
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, TextArea, Label
from textual.screen import Screen
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
# Constants:
MAX_PASTE_LENGTH = 5000  # 5KB threshold for paste confirmation
#
# Functions:

class ConfirmPasteScreen(Screen):
    """A screen to confirm pasting large text."""

    def __init__(self, text_area_widget: TextArea, full_text: str, previous_text: str) -> None:
        super().__init__()
        self.text_area_widget = text_area_widget
        self.full_text = full_text
        self.previous_text = previous_text

    def compose(self) -> ComposeResult:
        yield VerticalScroll(
            Label(f"You are pasting a large amount of text ({len(self.full_text) // 1024} KB). Are you sure?", id="confirm-message"),
            Horizontal(
                Button("Yes", variant="primary", id="confirm-paste-yes"),
                Button("No", variant="error", id="confirm-paste-no"),
                id="confirm-buttons"
            ),
            id="confirm-dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        chat_window = self.app.query_one(ChatWindow)

        if event.button.id == "confirm-paste-yes":
            # Load the text first. This might trigger ChatWindow.on_text_area_changed.
            self.text_area_widget.load_text(self.full_text)
            # Then update previous_chat_input_text to the new, confirmed text.
            chat_window.previous_chat_input_text = self.full_text
        elif event.button.id == "confirm-paste-no":
            # Revert to the original text. This might trigger ChatWindow.on_text_area_changed.
            self.text_area_widget.load_text(self.previous_text)
            # previous_chat_input_text in chat_window correctly remains as it was (which is self.previous_text).

        # Crucially, set confirming_paste to False *after* text is loaded and previous_text updated.
        # This ensures that any TextArea.Changed event triggered by load_text() within this method
        # sees confirming_paste as true, and then it's set to false for future, unrelated changes.
        chat_window.confirming_paste = False
        self.app.pop_screen()

class ChatWindow(Container):
    """
    Container for the Chat Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.previous_chat_input_text: str = ""
        self.confirming_paste: bool = False

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
                yield Button("💡", id="respond-for-me-button", classes="action-button suggest-button") # Suggest button
                self.app_instance.loguru_logger.debug("ChatWindow: 'respond-for-me-button' composed.")
                yield Button(get_char(EMOJI_STOP, FALLBACK_STOP), id="stop-chat-generation", classes="stop-button",
                             disabled=True)
                yield Button(get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), id="toggle-chat-right-sidebar",
                             classes="sidebar-toggle")

        # Character Details Sidebar (Right)
        yield from create_chat_right_sidebar(
            "chat",
            initial_ephemeral_state=self.app_instance.current_chat_is_ephemeral
        )

    # Decorator removed, relies on Textual's event dispatch by method name conventions
    # or general event bubbling if ChatWindow is an ancestor.
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes, including paste events."""
        # Ensure we are handling the event from the correct TextArea
        if not event.text_area.id or event.text_area.id != "chat-input":
            return

        text_area = event.text_area # Direct access via event attribute
        current_text = text_area.text

        if self.confirming_paste:
            # If a confirmation process is active (dialog is up or just handled an action),
            # subsequent TextArea.Changed events (e.g., from load_text in ConfirmPasteScreen)
            # should not trigger new checks or updates to previous_chat_input_text here.
            # The ConfirmPasteScreen method is responsible for these updates and for resetting confirming_paste.
            return

        # Heuristic: A paste is indicated by a significant increase in text length,
        # not just single character typing.
        text_changed_significantly = len(current_text) > len(self.previous_chat_input_text) + 1

        if text_changed_significantly and len(current_text) > MAX_PASTE_LENGTH:
            self.confirming_paste = True # Signal that we are starting a confirmation

            # Revert to the previous text in the TextArea. This is crucial because:
            # 1. It prevents the large pasted text from briefly appearing if the user clicks "No".
            # 2. The call to load_text() will itself fire a TextArea.Changed event.
            #    By setting self.confirming_paste = True *before* this call,
            #    the subsequent event will be correctly ignored by the `if self.confirming_paste:` check above.
            original_previous_text = self.previous_chat_input_text # Save before it's changed by load_text
            text_area.load_text(original_previous_text)
            # Note: After load_text, previous_chat_input_text might be updated if the above 'if self.confirming_paste'
            # didn't exist. However, with that check, previous_chat_input_text remains unchanged by this specific load_text.

            self.app.push_screen(
                ConfirmPasteScreen(
                    text_area_widget=text_area,
                    full_text=current_text,        # The large text that was pasted
                    previous_text=original_previous_text # The text before this paste attempt
                )
            )
        else:
            # This branch handles:
            # - Regular typing.
            # - Small pastes (not exceeding MAX_PASTE_LENGTH).
            # - TextArea.Changed events fired by load_text() from ConfirmPasteScreen,
            #   but *only after* self.confirming_paste has been set to False.
            self.previous_chat_input_text = current_text


#
# End of Chat_Window.py
#######################################################################################################################
