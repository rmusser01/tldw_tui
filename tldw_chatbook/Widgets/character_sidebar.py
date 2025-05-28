# character_sidebar.py
# Description: character sidebar widget
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical
from textual.widgets import Static, Collapsible, Placeholder, Select, Input, Label, TextArea, Button, Checkbox, ListView


#
# Local Imports
# (Add any necessary local imports here if needed for actual content later)
#
#######################################################################################################################
#
# Functions:

def create_character_sidebar(id_prefix: str, initial_ephemeral_state: bool = True) -> ComposeResult:
    """
    Yield the widgets for the character and chat session settings sidebar.
    id_prefix is typically "chat".
    initial_ephemeral_state determines the initial state of controls related to saving.
    """
    with VerticalScroll(id="character-sidebar", classes="sidebar"): # Main ID for the whole sidebar
        yield Static("Session & Character", classes="sidebar-title")

        # Section for current chat session details (title, keywords, etc.)
        with Collapsible(title="Current Chat Details", collapsed=False, id=f"{id_prefix}-chat-details-collapsible"):
            yield Button(
                "New Chat",
                id=f"{id_prefix}-new-conversation-button", # Matches app.py query
                classes="sidebar-button"
            )
            yield Label("Conversation ID:", classes="sidebar-label", id=f"{id_prefix}-uuid-label-displayonly")
            yield Input(
                id=f"{id_prefix}-conversation-uuid-display", # Matches app.py query
                value="Ephemeral Chat" if initial_ephemeral_state else "N/A",
                disabled=True, # Always disabled display
                classes="sidebar-input"
            )

            yield Label("Chat Title:", classes="sidebar-label", id=f"{id_prefix}-title-label-displayonly") # Keep consistent ID for query if needed elsewhere
            yield Input(
                id=f"{id_prefix}-conversation-title-input", # Matches app.py query
                placeholder="Chat title...",
                disabled=initial_ephemeral_state,
                classes="sidebar-input"
            )
            yield Label("Keywords (comma-sep):", classes="sidebar-label", id=f"{id_prefix}-keywords-label-displayonly")
            yield TextArea(
                "",
                id=f"{id_prefix}-conversation-keywords-input", # Matches app.py query
                classes="sidebar-textarea chat-keywords-textarea",
                disabled=initial_ephemeral_state
            )
            # Button to make an EPHEMERAL chat PERSISTENT (Save Chat to DB)
            yield Button(
                "Save Invis Chat",
                id=f"{id_prefix}-save-current-chat-button", # Matches app.py's expected ID
                classes="sidebar-button save-chat-button",
                variant="success",
                disabled=not initial_ephemeral_state # Enabled if ephemeral, disabled if already saved
            )
            # Button to save METADATA (title/keywords) of a PERSISTENT/ALREADY EXISTING chat
            yield Button(
                "Save Details",
                id=f"{id_prefix}-save-conversation-details-button", # ID for app.py handler
                classes="sidebar-button save-details-button", # Specific class
                variant="primary", # Or "default"
                disabled=initial_ephemeral_state # Disabled if ephemeral, enabled if persistent
            )
        # ===================================================================
        # Prompts (only for chat tab)
        # ===================================================================
<<<<<<< Updated upstream
        if id_prefix == "chat": # This is the outer condition for adding the Prompts section
            with Collapsible(title="Prompts", collapsed=True, id=f"{id_prefix}-sidebar-prompts-collapsible"): # START OF COLLAPSIBLE
                # ALL THE FOLLOWING YIELDS MUST BE INDENTED UNDER THIS 'with'
                yield Label("Search Prompts:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-sidebar-prompt-search-input", # e.g., chat-sidebar-prompt-search-input
                    placeholder="Search name/details...",
                    classes="sidebar-input"
                )
                yield Label("Filter by Keywords (comma-sep):", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-sidebar-prompt-keyword-filter-input", # e.g., chat-sidebar-prompt-keyword-filter-input
                    placeholder="e.g., writing, technical",
                    classes="sidebar-input"
                )
                yield ListView(id=f"{id_prefix}-sidebar-prompts-listview") # e.g., chat-sidebar-prompts-listview

                yield Label("Selected Prompt - System:", classes="sidebar-label")
                yield TextArea(
                    id=f"{id_prefix}-sidebar-prompt-system-display", # e.g., chat-sidebar-prompt-system-display
                    read_only=True, classes="sidebar-textarea ccp-prompt-textarea",
                )
                yield Button("Copy System", id=f"{id_prefix}-sidebar-copy-system-prompt-button", classes="sidebar-button") # e.g., chat-sidebar-copy-system-prompt-button

                yield Label("Selected Prompt - User:", classes="sidebar-label")
                yield TextArea(id=f"{id_prefix}-sidebar-prompt-user-display", read_only=True, classes="sidebar-textarea ccp-prompt-textarea") # e.g., chat-sidebar-prompt-user-display
                yield Button("Copy User", id=f"{id_prefix}-sidebar-copy-user-prompt-button", classes="sidebar-button") # e.g., chat-sidebar-copy-user-prompt-button
=======
        if id_prefix == "chat":
            with Collapsible(title="Prompts", collapsed=True, id=f"{id_prefix}-prompts-collapsible"):
                yield Label("Search Prompts:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-prompt-search-input",
                    placeholder="Title or keywords...",
                    classes="sidebar-input"
                )
                yield ListView(
                    id=f"{id_prefix}-prompt-list-view",
                    classes="sidebar-listview"
                )
                yield Button(
                    "Load Selected Prompt",
                    id=f"{id_prefix}-prompt-load-selected-button",
                    variant="default",
                    classes="sidebar-button"
                )

                # Display area for loaded prompt
                # Use a Vertical container instead of VerticalScroll if scrolling isn't desired for this small section
                with Vertical(id=f"{id_prefix}-loaded-prompt-display-area", classes="loaded-prompt-area hidden"):
                    yield Label("Title:", classes="sidebar-label", id=f"{id_prefix}-loaded-prompt-title-label")
                    yield Static("", id=f"{id_prefix}-loaded-prompt-title-static", classes="sidebar-static-display")

                    yield Label("System Prompt:", classes="sidebar-label", id=f"{id_prefix}-loaded-prompt-system-label")
                    yield TextArea(
                        "",
                        id=f"{id_prefix}-loaded-prompt-system-text",
                        read_only=True,
                        classes="sidebar-textarea loaded-prompt-textarea"
                    )
                    yield Button(
                        "Copy System Prompt",
                        id=f"{id_prefix}-copy-system-prompt-button",
                        variant="default",
                        classes="sidebar-button copy-prompt-button",
                        disabled=True
                    )

                    yield Label("User Prompt:", classes="sidebar-label", id=f"{id_prefix}-loaded-prompt-user-label")
                    yield TextArea(
                        "",
                        id=f"{id_prefix}-loaded-prompt-user-text",
                        read_only=True,
                        classes="sidebar-textarea loaded-prompt-textarea"
                    )
                    yield Button(
                        "Copy User Prompt",
                        id=f"{id_prefix}-copy-user-prompt-button",
                        variant="default",
                        classes="sidebar-button copy-prompt-button",
                        disabled=True
                    )

                    yield Label("Keywords:", classes="sidebar-label", id=f"{id_prefix}-loaded-prompt-keywords-label")
                    yield Static("", id=f"{id_prefix}-loaded-prompt-keywords-static", classes="sidebar-static-display")
>>>>>>> Stashed changes

        # ===================================================================
        # Saved Conversations (only for chat tab)
        # ===================================================================
        with Collapsible(title="Search & Load Conversations", collapsed=True):
            yield Input(
                id=f"{id_prefix}-conversation-search-bar",
                placeholder="Search all chats...",
                classes="sidebar-input"
            )
            yield Checkbox(
                "Include Character Chats",
                id=f"{id_prefix}-conversation-search-include-character-checkbox"
                # value=False by default for Checkbox
            )
            yield Select(
                [],  # Empty options initially
                id=f"{id_prefix}-conversation-search-character-filter-select",
                allow_blank=True,  # User can select nothing to clear filter
                prompt="Filter by Character...",
                classes="sidebar-select"  # Assuming a general class for selects or use default
            )
            yield Checkbox(
                "All Characters",
                id=f"{id_prefix}-conversation-search-all-characters-checkbox",
                value=True  # Default to True
            )
            yield ListView(
                id=f"{id_prefix}-conversation-search-results-list",
                classes="sidebar-listview"  # Add specific styling if needed
            )
            # Set initial height for ListView via styles property if not handled by class
            # Example: self.query_one(f"#{id_prefix}-conversation-search-results-list", ListView).styles.height = 10
            yield Button(
                "Load Selected Chat",
                id=f"{id_prefix}-conversation-load-selected-button",
                variant="default",  # Or "primary"
                classes="sidebar-button"  # Use existing class or new one
            )


        # Placeholder for actual character details (if a specific character is active beyond default)
        # This part would be more relevant if the chat tab directly supported switching active characters
        # for the ongoing conversation, rather than just for filtering searches.
        # For now, keeping it minimal.
        with Collapsible(title="Active Character Info", collapsed=True):
            yield Placeholder("Display Active Character Name")
            # Could add a select here to change the character for the *current* chat,
            # which would then influence the AI's persona for subsequent messages.
            # This is a more advanced feature than just "regular" vs. "specific character" chats.

        with Collapsible(title="Other Character Tools", collapsed=True):
            yield Placeholder("Tool 1")
            yield Placeholder("Tool 2")

        logging.debug(f"Character sidebar (id='character-sidebar', prefix='{id_prefix}') created with ephemeral state: {initial_ephemeral_state}")

#
# End of character_sidebar.py
#######################################################################################################################
