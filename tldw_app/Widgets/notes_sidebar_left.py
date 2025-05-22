from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, ListView, Button, ListItem, Label

class NotesSidebarLeft(VerticalScroll):
    """A sidebar for managing notes."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the notes sidebar."""
        yield Static("My Notes", classes="sidebar-title", id="notes-sidebar-title")
        yield Input(placeholder="Search notes...", id="notes-search-input")
        yield ListView(id="notes-list-view") # Ensure ListView is created
        yield Button("New Note", id="notes-new-button", variant="success")
        yield Button("Delete Selected Note", id="notes-delete-button", variant="error")

    async def populate_notes_list(self, notes_data: list[dict]) -> None:
        """Clears and populates the notes list."""
        list_view = self.query_one("#notes-list-view", ListView)
        await list_view.clear() # Clear existing items

        for note in notes_data:
            list_item = ListItem(Label(note['title']))
            list_item.note_id = note['id']
            list_item.note_version = note['version']
            await list_view.append(list_item)
