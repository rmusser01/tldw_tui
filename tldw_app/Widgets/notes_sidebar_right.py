from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, TextArea, Button

class NotesSidebarRight(VerticalScroll):
    """A sidebar for displaying and editing note details."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the notes details sidebar."""
        yield Static("Note Details", classes="sidebar-title", id="notes-details-sidebar-title")
        
        yield Static("Title:", classes="sidebar-label")
        yield Input(placeholder="Note title...", id="notes-title-input")
        
        yield Static("Keywords:", classes="sidebar-label")
        # Initialize TextArea with a few lines of text to suggest its size.
        # Actual height will be controlled by CSS using the 'notes-keywords-textarea' class.
        yield TextArea("", id="notes-keywords-area", classes="notes-keywords-textarea")
        
        yield Button("Save Keywords", id="notes-save-keywords-button", variant="primary")
        
        yield Static("Export Options", classes="sidebar-label")
        yield Button("Export as Markdown", id="notes-export-markdown-button")
        yield Button("Export as Text", id="notes-export-text-button")
