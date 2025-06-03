# tldw_app/widgets/chat_message.py
# Description: This file contains the ChatMessage widget for tldw_cli
#
# Imports
#
# 3rd-party Libraries
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static, Button, Label # Added Label
from textual.reactive import reactive
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

class ChatMessage(Widget):
    """A widget to display a single chat message with action buttons."""

    DEFAULT_CSS = """
    ChatMessage {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    ChatMessage > Vertical {
        border: round $surface;
        background: $panel;
        padding: 0 1;
        width: 100%;
        height: auto;
    }
    ChatMessage.-user > Vertical {
        background: $boost; /* Different background for user */
        border: round $accent;
    }
    .message-header {
        width: 100%;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
    }
    .message-text {
        padding: 1;
        width: 100%;
        height: auto;
    }
    .message-actions {
        height: auto;
        width: 100%;
        padding: 0 1;
        /* margin-top: 1; */ /* Removed top margin */      
        border-top: solid $surface-lighten-1;
        align: right middle; /* Align buttons to the right */
        display: block; /* Default display state */
    }
    .message-actions Button {
        min-width: 8;
        height: 1;
        margin: 0 0 0 1; /* Space between buttons */
        border: none;
        background: $surface-lighten-2;
        color: $text-muted;
    }
    .message-actions Button:hover {
        background: $surface;
        color: $text;
    }
    /* Specific hover style for delete */
    .message-actions .delete-button:hover {
        background: $error; /* Use the theme's error color (usually red) */
        color: white; /* Adjust text color for contrast if needed */
    }
    /* Initially hide AI actions until generation is complete */
    ChatMessage.-ai .message-actions.-generating {
        display: none;
    }
    """

    # Store the raw text content
    message_text = reactive("", repaint=True)
    role = reactive("User", repaint=True) # "User" or "AI"
    generation_complete = reactive(True) # Used for AI messages to show actions

    # -- Internal state for message metadata ---
    message_id_internal: reactive[Optional[str]] = reactive(None)
    message_version_internal: reactive[Optional[int]] = reactive(None)
    # Store timestamp if provided, e.g. when loading from DB
    timestamp: reactive[Optional[str]] = reactive(None) # Store as ISO string
    # Store image data if message has an image
    image_data: reactive[Optional[bytes]] = reactive(None)
    image_mime_type: reactive[Optional[str]] = reactive(None)

    def __init__(self,
                 message: str,
                 role: str,
                 generation_complete: bool = True,
                 message_id: Optional[str] = None,
                 message_version: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 image_data: Optional[bytes] = None,
                 image_mime_type: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.message_text = message
        self.role = role
        self.generation_complete = generation_complete

        self.message_id_internal = message_id
        self.message_version_internal = message_version
        self.timestamp = timestamp
        self.image_data = image_data
        self.image_mime_type = image_mime_type

        #self.add_class(f"-{role.lower()}") # Add role-specific class
        # For CSS styling, we use a generic class based on whether it's the user or not.
        # The actual self.role (e.g., "Default Assistant") is used for display in the header.
        if role.lower() == "user":
            self.add_class("-user")
        else: # Any role other than "user" (e.g., "AI", "Default Assistant", "Character Name") gets the -ai style
            self.add_class("-ai")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"{self.role}", classes="message-header")
            yield Static(self.message_text, classes="message-text")
            actions_class = "message-actions"
            # Logic for the '-generating' class on the actions container
            # This should only apply if it's an AI message AND generation is not complete
            if self.has_class("-ai") and not self.generation_complete:
                actions_class += " -generating"
            with Horizontal(classes=actions_class):
                # Common buttons
                yield Button("Edit", classes="action-button edit-button")
                yield Button("📋", classes="action-button copy-button", id="copy") # Emoji for copy
                yield Button("🔊", classes="action-button speak-button", id="speak") # Emoji for speak

                # AI-specific buttons
                if self.has_class("-ai"):
                    yield Button("👍", classes="action-button thumb-up-button", id="thumb-up")
                    yield Button("👎", classes="action-button thumb-down-button", id="thumb-down")
                    yield Button("🔄", classes="action-button regenerate-button", id="regenerate") # Emoji for regenerate
                    if self.generation_complete: # Only show continue if generation is complete
                        yield Button("↪️", id="continue-response-button", classes="action-button continue-button")

                # Add delete button for all messages at very end
                yield Button("🗑️", classes="action-button delete-button")  # Emoji for delete ; Label: Delete, Class: delete-button

    def update_message_chunk(self, chunk: str):
        if self.has_class("-ai"):
            self.query_one(".message-text", Static).update(self.message_text + chunk)
            self.message_text += chunk

    def mark_generation_complete(self):
        if self.has_class("-ai"):
            self.generation_complete = True
            actions_container = self.query_one(".message-actions")
            actions_container.remove_class("-generating")
            # Ensure it's displayed if CSS might still hide it via other means,
            # though removing '-generating' should be enough if the CSS is specific.
            actions_container.styles.display = "block" # or "flex" if it's a flex container

#
#
#######################################################################################################################
