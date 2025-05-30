from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, ListView, Button, ListItem, Label, Collapsible

class NotesSidebarLeft(VerticalScroll):
    """A sidebar for managing notes with collapsible action sections."""

    DEFAULT_CSS = """
    NotesSidebarLeft {
        dock: left;
        width: 25%;
        min-width: 20;
        max-width: 80;
        background: $boost;
        padding: 1;
        border-right: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    NotesSidebarLeft > .sidebar-title {
        text-style: bold underline;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }
    NotesSidebarLeft > Static.sidebar-label { /* More specific selector for labels */
        margin-top: 1; /* Add space above labels */
    }
    NotesSidebarLeft > Input {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarLeft > Button, NotesSidebarLeft > Collapsible > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarLeft > ListView {
        width: 100%;
        height: 1fr;
        border: round $surface;
        margin-bottom: 1;
    }
    NotesSidebarLeft > Collapsible { /* Style for the collapsible itself */
        width: 100%;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the notes sidebar."""
        yield Static("My Notes", classes="sidebar-title", id="notes-sidebar-title-main")
        yield Button("Create New Note", id="notes-create-new-button", variant="success")
        yield Button("Import Note", id="notes-import-button", variant="default")

        yield Static("Search & Filter:", classes="sidebar-label")
        yield Input(placeholder="Search notes content...", id="notes-search-input")
        yield Input(placeholder="Keywords (e.g., projectA, urgent)", id="notes-keyword-filter-input")
        yield Button("Search / Filter", id="notes-search-button", variant="default") # Combined button

        yield ListView(id="notes-list-view") # Ensure ListView is created

        with Collapsible(title="Selected Note Actions", collapsed=False): # Start expanded
            yield Button("Load Selected Note", id="notes-load-selected-button", variant="default")
            yield Button("Edit Selected Note", id="notes-edit-selected-button", variant="primary")
            # "Save Current Note" (acting on the editor) is better in main controls or right sidebar.
            # If it's meant to be a quick save from here, ensure its action is clear.

        with Collapsible(title="Advanced Actions", collapsed=True):
            yield Button("Placeholder button")

    async def populate_notes_list(self, notes_data: list[dict]) -> None:
        """Clears and populates the notes list."""
        list_view = self.query_one("#notes-list-view", ListView)
        await list_view.clear()

        if not notes_data:
            await list_view.append(ListItem(Label("No notes found.")))
            return

        for note in notes_data:
            title = note.get('title', "Untitled Note") # Default if title is missing or None
            if not title.strip(): # If title is just whitespace
                title = "Untitled Note"
            list_item = ListItem(Label(title))
            setattr(list_item, 'note_id', note['id'])
            setattr(list_item, 'note_version', note['version'])
            await list_view.append(list_item)
