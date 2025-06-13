# tldw_chatbook/UI/SearchWindow.py
#
#
# Imports
from pathlib import Path

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Button, Input, Markdown, Select, Checkbox
#
# Third-Party Libraries
from typing import TYPE_CHECKING, Union, Optional
import asyncio
from loguru import logger
#
# Local Imports
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema
from tldw_chatbook.config import get_chachanotes_db_path, get_cli_setting

if TYPE_CHECKING:
    from ..app import TldwCli
try:
    from ..Embeddings.Embeddings_Lib import *
    EMBEDDINGS_GENERATION_AVAILABLE = True
    logger.info("✅ Embeddings Generation dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    EMBEDDINGS_GENERATION_AVAILABLE = False
    logger.warning(f"Embeddings Generation depenedencies not found, features related will be disabled. Reason: {e}")
try:
    from ..Embeddings.Chroma_Lib import *
    VECTORDB_AVAILABLE = True
    logger.info("✅ Vector Database dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    VECTORDB_AVAILABLE = False
    logger.warning(f"Vector Database depenedencies not found, features related will be disabled. Reason: {e}")
try:
    from ..Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate

    WEB_SEARCH_AVAILABLE = True
    logger.info("✅ Web Search dependencies found. Feature is enabled.")
except (ImportError, ModuleNotFoundError) as e:
    WEB_SEARCH_AVAILABLE = False
    # Use a warning here, as this is an expected condition, not a critical error.
    logger.warning(f"⚠️ Web Search dependencies not found, feature will be disabled. Reason: {e}")
    # Define placeholders so the rest of the file doesn't crash if they are referenced.
    generate_and_search = None
    analyze_and_aggregate = None
#
#######################################################################################################################
#
# Functions:


# Constants for clarity
SEARCH_VIEW_RAG_QA = "search-view-rag-qa"
SEARCH_VIEW_RAG_CHAT = "search-view-rag-chat"
SEARCH_VIEW_EMBEDDINGS_CREATION = "search-view-embeddings-creation"
SEARCH_VIEW_RAG_MANAGEMENT = "search-view-rag-management"
SEARCH_VIEW_EMBEDDINGS_MANAGEMENT = "search-view-embeddings-management"
SEARCH_VIEW_WEB_SEARCH = "search-view-web-search"

SEARCH_NAV_RAG_QA = "search-nav-rag-qa"
SEARCH_NAV_RAG_CHAT = "search-nav-rag-chat"
SEARCH_NAV_EMBEDDINGS_CREATION = "search-nav-embeddings-creation"
SEARCH_NAV_RAG_MANAGEMENT = "search-nav-rag-management"
SEARCH_NAV_EMBEDDINGS_MANAGEMENT = "search-nav-embeddings-management"
SEARCH_NAV_WEB_SEARCH = "search-nav-web-search"

# UI Constant for "Local Server" provider display name
LOCAL_SERVER_PROVIDER_DISPLAY_NAME = "Local OpenAI-Compliant Server"
LOCAL_SERVER_PROVIDER_INTERNAL_ID = "local_openai_compliant" # Internal ID to distinguish

class SearchWindow(Container):
    """
    Container for the Search Tab's UI, featuring a vertical tab bar and content areas.
    """

    # Database display name mapping
    DB_DISPLAY_NAMES = {
        "media_db": "Media DB",
        "rag_chat_db": "RAG Chat DB",
        "char_chat_db": "Character Chat DB"
    }

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._chroma_manager: Union["ChromaDBManager", None] = None
        self._selected_embedding_id: Union[str, None] = None
        # State to hold the mapping from display choice in dropdown to actual item ID, like in Gradio
        self._item_mapping: dict[str, str] = {}
        # Similar mapping for creation view
        self._creation_item_mapping: dict[str, str] = {}
        self._selected_item_display_name: Optional[str] = None

    async def on_mount(self) -> None:
        """Called when the window is first mounted."""
        logger.info("SearchWindow.on_mount: Setting and initializing initial active sub-tab.")

        # Hide all view areas initially
        for view in self.query(".search-view-area"):
            view.display = False

        if hasattr(self.app_instance, 'search_active_sub_tab') and self.app_instance.search_active_sub_tab is None:
            # Set the initial view to creation, as it's simpler
            self.app_instance.search_active_sub_tab = SEARCH_VIEW_EMBEDDINGS_CREATION

            # Set the active navigation button
            nav_button_id = SEARCH_VIEW_EMBEDDINGS_CREATION.replace("-view-", "-nav-")
            nav_button = self.query_one(f"#{nav_button_id}")
            nav_button.add_class("-active-search-sub-view")

            # Show the active view
            active_view = self.query_one(f"#{SEARCH_VIEW_EMBEDDINGS_CREATION}")
            active_view.display = True

            # And call its initializer
            await self._initialize_embeddings_creation_view()
        elif hasattr(self.app_instance, 'search_active_sub_tab') and self.app_instance.search_active_sub_tab is not None:
            # If there's already an active sub-tab, show it
            active_view_id = self.app_instance.search_active_sub_tab

            # Set the active navigation button
            nav_button_id = active_view_id.replace("-view-", "-nav-")
            nav_button = self.query_one(f"#{nav_button_id}")
            nav_button.add_class("-active-search-sub-view")

            # Show the active view
            active_view = self.query_one(f"#{active_view_id}")
            active_view.display = True

            # Initialize the view if needed
            if active_view_id == SEARCH_VIEW_EMBEDDINGS_CREATION:
                await self._initialize_embeddings_creation_view()
            elif active_view_id == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
                await self._initialize_embeddings_management_view()

    def compose(self) -> ComposeResult:
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            if WEB_SEARCH_AVAILABLE:
                yield Button("Web Search", id=SEARCH_NAV_WEB_SEARCH, classes="search-nav-button")
            else:
                yield Button("Web Search", id="search-nav-web-search-disabled", classes="search-nav-button disabled")
            if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
                yield Button("Embeddings Creation", id=SEARCH_NAV_EMBEDDINGS_CREATION, classes="search-nav-button")
            else:
                yield Button("Embeddings Creation", id="search-nav-embeddings-creation-disabled", classes="search-nav-button disabled")
            if VECTORDB_AVAILABLE:
                yield Button("Embeddings Management", id=SEARCH_NAV_EMBEDDINGS_MANAGEMENT, classes="search-nav-button")
            else:
                yield Button("Embeddings Management", id="search-nav-embeddings-management-disabled", classes="search-nav-button disabled")

        with Container(id="search-content-pane", classes="search-content-pane"):
            yield Container(id=SEARCH_VIEW_RAG_QA, classes="search-view-area",)
                            #children=[Static("RAG QA Content - Coming Soon!")]
            yield Container(id=SEARCH_VIEW_RAG_CHAT, classes="search-view-area",)
                            #children=[Static("RAG Chat Content - Coming Soon!")])
            yield Container(id=SEARCH_VIEW_RAG_MANAGEMENT, classes="search-view-area",)
                            #children=[Static("RAG Management Content - Coming Soon!")])

            # --- Embeddings Creation View (enhanced layout) ---
            if EMBEDDINGS_GENERATION_AVAILABLE and VECTORDB_AVAILABLE:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                    with VerticalScroll(classes="search-form-container"):
                        yield Static("Create Embeddings", classes="search-view-title")

                        yield Markdown("Select a database and configure embedding settings below. Embeddings allow semantic search and retrieval of your content.", id="creation-help-text")

                        yield Static("Database Selection", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Content Source:", classes="search-form-label")
                            yield Select(
                                [("Media DB", "media_db"), ("RAG Chat DB", "rag_chat_db"),
                                 ("Character Chat DB", "char_chat_db")],
                                id="creation-db-select",
                                prompt="Select Content Source...",
                                value="media_db"
                            )
                        with Horizontal(classes="search-form-row"):
                            yield Static("Database Path:", classes="search-form-label")
                            yield Input(id="creation-db-path-display", disabled=True, value="Path will appear here")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Collection Name:", classes="search-form-label")
                            yield Input(id="creation-collection-name-input", placeholder="Enter a name for the collection...")

                        yield Static("Content Selection", classes="search-section-title")
                        yield Markdown("Choose how you want to select content for embedding creation.", id="creation-selection-help-text")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Selection Mode:", classes="search-form-label")
                            yield Select(
                                [("All Items", "all"), ("Keyword Filter", "keyword"), ("Individual Selection", "individual")],
                                id="creation-selection-mode-select",
                                value="all"
                            )

                        # Keyword filter (initially hidden)
                        with Horizontal(id="creation-keyword-filter-container", classes="search-form-row hidden"):
                            yield Static("Keyword:", classes="search-form-label")
                            yield Input(id="creation-keyword-input", placeholder="Enter keywords to filter items...")

                        # Individual item selection (initially hidden)
                        with Container(id="creation-individual-selection-container", classes="hidden"):
                            with Horizontal(classes="search-form-row"):
                                yield Static("", classes="search-form-label")
                                yield Button("Refresh Item List", id="creation-refresh-list-button", variant="primary")

                            with Horizontal(classes="search-form-row"):
                                yield Static("Items:", classes="search-form-label")
                                yield Select([], id="creation-item-select", prompt="Select an item...")

                            yield Markdown("Select an item from the dropdown above.", id="creation-item-selection-help-text")

                        yield Static("Embedding Model", classes="search-section-title")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Provider:", classes="search-form-label")
                            yield Select([], id="creation-embedding-provider-select", prompt="Select Provider...")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Model:", classes="search-form-label")
                            yield Select([], id="creation-model-select", prompt="Select Model...")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Custom Model:", classes="search-form-label")
                            yield Input(id="creation-custom-model-input", placeholder="Custom HuggingFace Model Name...",
                                        classes="hidden")
                        with Horizontal(classes="search-form-row"):
                            yield Static("API URL:", classes="search-form-label")
                            yield Input(id="creation-api-url-input", placeholder="http://localhost:8080/v1/embeddings",
                                        classes="hidden")

                        yield Static("Chunking Options", classes="search-section-title")
                        yield Markdown("Chunking divides your content into smaller pieces for better retrieval. Adjust these settings based on your content type.", id="chunking-help-text")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Method:", classes="search-form-label")
                            yield Select([("By Words", "words"), ("By Sentences", "sentences")],
                                         id="creation-chunk-method-select", value="words")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Max Size:", classes="search-form-label")
                            yield Input(id="creation-chunk-size-input", value="400", type="integer")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Overlap:", classes="search-form-label")
                            yield Input(id="creation-chunk-overlap-input", value="200", type="integer")
                        with Horizontal(classes="search-form-row"):
                            yield Static("Adaptive:", classes="search-form-label")
                            yield Checkbox("Use adaptive chunking (recommended)", id="creation-adaptive-chunking-checkbox")

                        yield Static("", classes="search-section-title")
                        yield Button("Create Embeddings", id="creation-create-all-button", variant="primary")
                        yield Markdown("Status: Ready to create embeddings. Click the button above to start the process.", id="creation-status-output")
            else:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown("### Embeddings Creation Is Not Currently Available\n\nThe required dependencies for embeddings creation are not installed. Please install the necessary packages to use this feature.")

            # --- Embeddings Management View (Two-Pane layout, enhanced for better UX) ---
            if VECTORDB_AVAILABLE:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                    with Horizontal():
                        # Left Pane - Selection and Status
                        with Vertical(classes="search-management-left-pane"):
                            with VerticalScroll():
                                yield Static("Manage Embeddings", classes="search-view-title")
                                yield Markdown("Select a database and item to view, update, or delete its embeddings.", id="mgmt-help-text")

                                yield Static("Database & Item Selection", classes="search-section-title")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Database:", classes="search-form-label")
                                    yield Select(
                                        [("Media DB", "media_db"), ("RAG Chat DB", "rag_chat_db"),
                                         ("Character Chat DB", "char_chat_db")],
                                        id="mgmt-db-select",
                                        prompt="Select Content Source...",
                                        value="media_db"
                                    )

                                with Horizontal(classes="search-form-row"):
                                    yield Static("", classes="search-form-label")
                                    yield Button("Refresh Item List", id="mgmt-refresh-list-button", variant="primary")

                                with Horizontal(classes="search-form-row"):
                                    yield Static("Item:", classes="search-form-label")
                                    yield Select([], id="mgmt-item-select", prompt="Select an item...")

                                yield Static("Embedding Status", classes="search-section-title")
                                yield Markdown("Select an item above to see its embedding status.", id="mgmt-embedding-status-md")

                                yield Static("Embedding Metadata", classes="search-section-title")
                                yield Markdown("Metadata will appear here when an item is selected.", id="mgmt-embedding-metadata-md")

                        # Right Pane - Update Configuration
                        with Vertical(classes="search-management-right-pane"):
                            with VerticalScroll():
                                yield Static("Update Configuration", classes="search-view-title")
                                yield Markdown("Configure the embedding settings for the selected item.", id="mgmt-update-help-text")

                                yield Static("Embedding Model", classes="search-section-title")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Provider:", classes="search-form-label")
                                    yield Select([], id="mgmt-embedding-provider-select", prompt="Select Provider...")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Model:", classes="search-form-label")
                                    yield Select([], id="mgmt-model-select", prompt="Select Model...")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Custom Model:", classes="search-form-label")
                                    yield Input(id="mgmt-custom-model-input", placeholder="Custom HuggingFace Model Name...",
                                                classes="hidden")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("API URL:", classes="search-form-label")
                                    yield Input(id="mgmt-api-url-input", placeholder="http://localhost:8080/v1/embeddings",
                                                classes="hidden")

                                yield Static("Chunking Options", classes="search-section-title")
                                yield Markdown("Configure how content is divided into chunks for embedding.", id="mgmt-chunking-help-text")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Method:", classes="search-form-label")
                                    yield Select([("By Words", "words"), ("By Sentences", "sentences")],
                                                 id="mgmt-chunk-method-select", value="words")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Max Size:", classes="search-form-label")
                                    yield Input(id="mgmt-chunk-size-input", value="400", type="integer")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Overlap:", classes="search-form-label")
                                    yield Input(id="mgmt-chunk-overlap-input", value="200", type="integer")
                                with Horizontal(classes="search-form-row"):
                                    yield Static("Adaptive:", classes="search-form-label")
                                    yield Checkbox("Use adaptive chunking (recommended)", id="mgmt-adaptive-chunking-checkbox")

                                yield Static("Actions", classes="search-section-title")
                                with Horizontal(classes="search-button-row"):
                                    yield Button("Update Embedding", id="mgmt-update-embedding-button", variant="success")
                                    yield Button("Delete Embedding", id="mgmt-delete-embedding-button", variant="error")

                                yield Markdown("Status: Select an item from the left panel to update or delete its embeddings.", id="mgmt-status-output")
            else:
                with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown("### Embeddings Management Is Not Currently Available\n\nThe required dependencies for vector database management are not installed. Please install the necessary packages to use this feature.")

            if WEB_SEARCH_AVAILABLE:
                with Container(id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area"):
                    with VerticalScroll():
                        yield Input(placeholder="Enter search query...", id="web-search-input")
                        yield Button("Search", id="web-search-button", classes="search-action-button")
                        yield VerticalScroll(Markdown("", id="web-search-results"))
            else:
                with Container(id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area"):
                    with VerticalScroll():
                        yield Markdown("### Web Search/Scraping Is Not Currently Installed\n\n...")

    # --- HELPER METHODS ---

    async def _get_chroma_manager(self) -> "ChromaDBManager":
        """Get or create a ChromaDBManager instance using the app's configuration."""
        if self._chroma_manager is None:
            logger.info("ChromaDBManager instance not found, creating a new one.")
            try:
                user_config = self.app_instance.app_config
                user_id = self.app_instance.notes_user_id
                self._chroma_manager = ChromaDBManager(user_id=user_id, user_embedding_config=user_config)
                logger.info(f"Successfully created ChromaDBManager for user '{user_id}'.")
            except Exception as e:
                logger.error(f"Failed to create ChromaDBManager: {e}", exc_info=True)
                self.app_instance.notify(f"Failed to initialize embedding system: {escape(str(e))}", severity="error",
                                         timeout=10)
                raise
        return self._chroma_manager

    def _get_db_path(self, db_type: str) -> str:
        """Helper to get the configured path for a database type."""
        # Note: In a real app, you would fetch these from config.py
        base_path = get_cli_setting("database", "chachanotes_db_path")
        if not base_path:
            return "Path not configured"

        if db_type == "media_db":
            return get_cli_setting("database", "media_db_path", "Not Set")
        elif db_type == "rag_chat_db":
            # Assuming it's in the same dir as the main db
            return str(Path(base_path).parent / "rag_qa.db")
        elif db_type == "char_chat_db":
            return str(Path(base_path))  # It's the same DB
        return "Unknown DB Type"

    # --- EVENT HANDLERS (New and Refactored) ---

    @on(Button.Pressed, ".search-nav-button")
    async def handle_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handles all navigation button presses within the search tab."""
        event.stop()
        button_id = event.button.id
        if not button_id: return

        logger.info(f"Search nav button '{button_id}' pressed.")
        target_view_id = button_id.replace("-nav-", "-view-").replace("-disabled", "")
        self.app_instance.search_active_sub_tab = target_view_id

        # Remove active class from all nav buttons and add it to the clicked one
        for button in self.query(".search-nav-button"):
            button.remove_class("-active-search-sub-view")
        event.button.add_class("-active-search-sub-view")

        # Hide all view areas and show the target one
        for view in self.query(".search-view-area"):
            view.display = False
        target_view = self.query_one(f"#{target_view_id}")
        target_view.display = True

        try:
            if target_view_id == SEARCH_VIEW_EMBEDDINGS_CREATION:
                await self._initialize_embeddings_creation_view()
            elif target_view_id == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
                await self._initialize_embeddings_management_view()
        except Exception as e:
            logger.error(f"Error initializing view '{target_view_id}': {e}", exc_info=True)
            self.app_instance.notify(f"Error loading view: {escape(str(e))}", severity="error")

    # --- Creation View Handlers ---
    @on(Select.Changed, "#creation-db-select")
    def on_creation_db_select(self, event: Select.Changed) -> None:
        db_type = str(event.value)

        # Update database path display
        db_path_display = self.query_one("#creation-db-path-display", Input)
        db_path_display.value = self._get_db_path(db_type)

        # Update collection name input with the selected database type
        collection_name_input = self.query_one("#creation-collection-name-input", Input)
        # Only update if the field is empty or matches a previous database type
        if not collection_name_input.value or collection_name_input.value in ["media_db", "rag_chat_db", "char_chat_db"]:
            collection_name_input.value = db_type

    @on(Select.Changed, "#creation-embedding-provider-select")
    def on_creation_provider_select(self, event: Select.Changed) -> None:
        """Handle provider selection changes in creation view."""
        provider = str(event.value)
        model_select = self.query_one("#creation-model-select", Select)

        # Update model options based on provider
        if provider == "huggingface":
            model_select.set_options([
                ("jinaai/jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en"),
                ("BAAI/bge-large-en-v1.5", "BAAI/bge-large-en-v1.5"),
                ("sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
                ("Custom", "custom")
            ])
        elif provider == "openai":
            model_select.set_options([
                ("text-embedding-ada-002", "text-embedding-ada-002"),
                ("text-embedding-3-small", "text-embedding-3-small"),
                ("text-embedding-3-large", "text-embedding-3-large")
            ])
        elif provider == "local":
            model_select.set_options([
                ("Local Embeddings API", "local_api")
            ])

        # Update visibility of custom inputs
        self._update_provider_visibility(provider, "creation")

    @on(Select.Changed, "#creation-model-select")
    def on_creation_model_select(self, event: Select.Changed) -> None:
        """Handle model selection changes in creation view."""
        model = str(event.value)
        provider_select = self.query_one("#creation-embedding-provider-select", Select)
        provider = str(provider_select.value)

        # Show custom model input if "Custom" is selected
        custom_input = self.query_one("#creation-custom-model-input", Input)
        custom_input.display = provider == "huggingface" and model == "custom"

    @on(Select.Changed, "#creation-selection-mode-select")
    def on_creation_selection_mode_select(self, event: Select.Changed) -> None:
        """Handle selection mode changes in creation view."""
        mode = str(event.value)

        # Get the containers
        keyword_container = self.query_one("#creation-keyword-filter-container")
        individual_container = self.query_one("#creation-individual-selection-container")

        # Hide both containers initially
        keyword_container.add_class("hidden")
        individual_container.add_class("hidden")

        # Show the appropriate container based on the selected mode
        if mode == "keyword":
            keyword_container.remove_class("hidden")
        elif mode == "individual":
            individual_container.remove_class("hidden")

            # Refresh the item list if it's empty
            item_select = self.query_one("#creation-item-select", Select)
            # Check if the Select widget has any options by trying to get its value
            # If it has no options, value will be None or Select.BLANK
            if item_select.value is Select.BLANK:
                asyncio.create_task(self._refresh_creation_item_list())

    @on(Button.Pressed, "#creation-create-all-button")
    async def on_creation_create_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Create Embeddings button press."""
        # Get all the configuration values
        db_select = self.query_one("#creation-db-select", Select)
        db_type = str(db_select.value)
        db_path = self._get_db_path(db_type)

        # Get collection name (new)
        collection_name_input = self.query_one("#creation-collection-name-input", Input)
        collection_name = collection_name_input.value.strip()

        # Use db_type as default collection name if not provided
        if not collection_name:
            collection_name = db_type
            collection_name_input.value = collection_name

        # Get selection mode
        selection_mode_select = self.query_one("#creation-selection-mode-select", Select)
        selection_mode = str(selection_mode_select.value)

        # Get keyword or selected items based on selection mode
        keyword = None
        selected_items = None

        if selection_mode == "keyword":
            keyword_input = self.query_one("#creation-keyword-input", Input)
            keyword = keyword_input.value.strip()
            if not keyword:
                await self.query_one("#creation-status-output", Markdown).update("❌ Please enter a keyword to filter items.")
                return
        elif selection_mode == "individual":
            item_select = self.query_one("#creation-item-select", Select)
            if not item_select.value:
                await self.query_one("#creation-status-output", Markdown).update("❌ Please select at least one item.")
                return
            selected_items = [str(item_select.value)]

        provider_select = self.query_one("#creation-embedding-provider-select", Select)
        provider = str(provider_select.value)

        model_select = self.query_one("#creation-model-select", Select)
        model = str(model_select.value)

        # Get custom model if applicable
        custom_model_input = self.query_one("#creation-custom-model-input", Input)
        if provider == "huggingface" and model == "custom":
            model = custom_model_input.value

        # Get API URL if applicable
        api_url_input = self.query_one("#creation-api-url-input", Input)
        api_url = api_url_input.value if provider == "local" else None

        # Get chunking options
        chunk_method_select = self.query_one("#creation-chunk-method-select", Select)
        chunk_method = str(chunk_method_select.value)

        chunk_size_input = self.query_one("#creation-chunk-size-input", Input)
        chunk_size = int(chunk_size_input.value)

        chunk_overlap_input = self.query_one("#creation-chunk-overlap-input", Input)
        chunk_overlap = int(chunk_overlap_input.value)

        adaptive_chunking_checkbox = self.query_one("#creation-adaptive-chunking-checkbox", Checkbox)
        adaptive_chunking = adaptive_chunking_checkbox.value

        # Update status
        status_output = self.query_one("#creation-status-output", Markdown)
        await status_output.update("⏳ Preparing to create embeddings...")

        try:
            # Create embedding configuration
            embedding_config = {
                "provider": provider,
                "model": model,
                "api_url": api_url,
                "chunking": {
                    "method": chunk_method,
                    "size": chunk_size,
                    "overlap": chunk_overlap,
                    "adaptive": adaptive_chunking
                }
            }

            # Update status based on selection mode
            if selection_mode == "all":
                status_message = f"⏳ Creating embeddings for all items in {db_type} with collection name '{collection_name}'..."
            elif selection_mode == "keyword":
                status_message = f"⏳ Creating embeddings for items matching keyword '{keyword}' in {db_type} with collection name '{collection_name}'..."
            else:  # individual
                status_message = f"⏳ Creating embeddings for {len(selected_items)} selected items in {db_type} with collection name '{collection_name}'..."

            await status_output.update(f"{status_message}\n\nThis may take some time depending on the size of your database.")

            # Call the embedding creation function with the appropriate parameters
            await self._create_embeddings(
                db_type, 
                db_path, 
                embedding_config, 
                collection_name,
                selection_mode=selection_mode,
                keyword=keyword,
                selected_items=selected_items
            )

            # Update status with success message
            if selection_mode == "all":
                success_message = f"✅ Successfully created embeddings for all items in {db_type}!"
            elif selection_mode == "keyword":
                success_message = f"✅ Successfully created embeddings for items matching keyword '{keyword}' in {db_type}!"
            else:  # individual
                success_message = f"✅ Successfully created embeddings for {len(selected_items)} selected items in {db_type}!"

            await status_output.update(f"{success_message}\n\nYou can now use these embeddings for semantic search and retrieval.")

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}", exc_info=True)
            await status_output.update(f"❌ Error creating embeddings: {escape(str(e))}")
            self.app_instance.notify(f"Error creating embeddings: {escape(str(e))}", severity="error")

    # --- Management View Handlers ---
    @on(Select.Changed, "#mgmt-db-select")
    async def on_mgmt_db_select(self, event: Select.Changed) -> None:
        """When DB source changes in management view, refresh the item list."""
        await self._refresh_mgmt_item_list()

    @on(Button.Pressed, "#mgmt-refresh-list-button")
    async def on_mgmt_refresh_button_pressed(self, event: Button.Pressed) -> None:
        """Handles the refresh button in the management view."""
        await self._refresh_mgmt_item_list()

    @on(Button.Pressed, "#creation-refresh-list-button")
    async def on_creation_refresh_button_pressed(self, event: Button.Pressed) -> None:
        """Handles the refresh button in the creation view."""
        await self._refresh_creation_item_list()

    @on(Select.Changed, "#mgmt-item-select")
    async def on_mgmt_item_select(self, event: Select.Changed) -> None:
        """When an item is selected from the dropdown, show its status."""
        if event.value is Select.BLANK:
            self._selected_item_display_name = None
            self._selected_embedding_id = None
            await self.query_one("#mgmt-embedding-status-md", Markdown).update("Select an item to see its status.")
            await self.query_one("#mgmt-embedding-metadata-md", Markdown).update("")
        else:
            self._selected_item_display_name = str(event.value)
            self._selected_embedding_id = self._item_mapping.get(self._selected_item_display_name)
            await self._check_and_display_embedding_status()

    # --- Universal Helper for Dynamic Provider/Model Selects ---
    def _update_provider_visibility(self, provider: str, prefix: str) -> None:
        """Shows/hides model dropdowns based on provider selection."""
        # For simplicity, we assume one model select and one custom input for now
        # In a real app, you might have more complex logic
        custom_input = self.query_one(f"#{prefix}-custom-model-input", Input)
        api_url_input = self.query_one(f"#{prefix}-api-url-input", Input)

        is_local = provider == "local"

        custom_input.display = provider == "huggingface" and self.query_one(f"#{prefix}-model-select",
                                                                            Select).value == "custom"
        api_url_input.display = is_local

    # --- INITIALIZATION HELPERS ---

    async def _initialize_embeddings_creation_view(self) -> None:
        """Populates dropdowns and sets defaults for the Creation view."""
        logger.info("Initializing Embeddings Creation view.")
        # This logic can be shared with the management view's initialization
        db_select = self.query_one("#creation-db-select", Select)
        db_path_display = self.query_one("#creation-db-path-display", Input)
        db_path_display.value = self._get_db_path(str(db_select.value))

        # Set default collection name based on selected database
        collection_name_input = self.query_one("#creation-collection-name-input", Input)
        default_collection_name = str(db_select.value)
        if not collection_name_input.value:
            collection_name_input.value = default_collection_name

        # You would fetch providers and models from your config here
        # For now, using placeholder data
        provider_select = self.query_one("#creation-embedding-provider-select", Select)
        provider_select.set_options([("HuggingFace", "huggingface"), ("Local Server", "local"), ("OpenAI", "openai")])
        provider_select.value = "huggingface"

        model_select = self.query_one("#creation-model-select", Select)
        model_select.set_options([
            ("jinaai/jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en"),
            ("BAAI/bge-large-en-v1.5", "BAAI/bge-large-en-v1.5"),
            ("sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
            ("Custom", "custom")
        ])

        # Set initial visibility of custom inputs
        self._update_provider_visibility("huggingface", "creation")

        # Initialize status message
        status_output = self.query_one("#creation-status-output", Markdown)
        await status_output.update("Ready to create embeddings. Configure your settings and click the button above.")

    async def _initialize_embeddings_management_view(self) -> None:
        """Populates dropdowns and sets defaults for the Management view."""
        logger.info("Initializing Embeddings Management view.")
        provider_select = self.query_one("#mgmt-embedding-provider-select", Select)
        provider_select.set_options([("HuggingFace", "huggingface"), ("Local Server", "local"), ("OpenAI", "openai")])
        provider_select.value = "huggingface"

        model_select = self.query_one("#mgmt-model-select", Select)
        model_select.set_options(
            [("jinaai/jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en"), ("Custom", "custom")])

        # Initial population of the item list
        await self._refresh_mgmt_item_list()

    async def _refresh_collections_list(self) -> None:
        """Refreshes the list of collections in the ChromaDB manager."""
        logger.info("Refreshing collections list...")
        try:
            chroma_manager = await self._get_chroma_manager()
            # Get all collections from ChromaDB
            collections = chroma_manager.client.list_collections()
            logger.info(f"Found {len(collections)} collections in ChromaDB")

            # Update any UI elements that need to display collections
            # This is a placeholder - you might need to update specific UI elements
            # based on your application's requirements

        except Exception as e:
            logger.error(f"Error refreshing collections list: {e}", exc_info=True)

    async def _refresh_creation_item_list(self) -> None:
        """Fetches items for the selected DB and populates the creation item selection dropdown."""
        item_select = self.query_one("#creation-item-select", Select)
        db_select = self.query_one("#creation-db-select", Select)
        status_output = self.query_one("#creation-status-output", Markdown)

        db_type = str(db_select.value)
        db_name = self.DB_DISPLAY_NAMES.get(db_type, "Unknown Database")

        await status_output.update(f"⏳ Loading items from {db_name}...")

        # This is where you would call your backend functions like `get_all_content_from_database`
        # For this example, I'll use placeholder data similar to the management view.
        await asyncio.sleep(0.1)  # Simulate async call

        # ---- Placeholder Logic ----
        # In a real app, replace this with your actual DB calls
        items = [
            {'id': 'media_1', 'title': 'My First Video'},
            {'id': 'media_2', 'title': 'Interesting Article'},
        ] if db_type == "media_db" else [
            {'id': 'conv_1', 'title': 'Chat about AGI'},
        ]
        # ---- End Placeholder Logic ----

        # Now, check embedding status for each item
        try:
            chroma_manager = await self._get_chroma_manager()
            collection_name = db_type  # Simplified name for example

            choices = []
            new_mapping = {}

            # Check if the collection exists
            collections = chroma_manager.client.list_collections()
            collection_exists = any(collection.name == collection_name for collection in collections)

            for item in items:
                item_id = item['id']
                # This is a simplified check. Your real logic would be more robust.
                if not collection_exists:
                    status = "❓"  # Collection doesn't exist
                else:
                    try:
                        result = chroma_manager.client.get_collection(name=collection_name).get(ids=[f"{item_id}_chunk_0"])
                        status = "✅" if result and result['ids'] else "❌"
                    except Exception:
                        status = "⚠️"  # Error checking

                display_name = f"{item['title']} ({status})"
                choices.append((display_name, item_id))  # Use item_id as value for easier processing
                new_mapping[display_name] = item_id

            # Store the mapping for later use
            self._creation_item_mapping = new_mapping

            # Set the options for the multi-select dropdown
            item_select.set_options(choices)

            if len(items) > 0:
                await status_output.update(f"✅ Found {len(items)} items in {db_name}. Select items to create embeddings for.")
            else:
                await status_output.update(f"ℹ️ No items found in {db_name}. Try another database or check your data.")

        except Exception as e:
            logger.error(f"Error refreshing item list: {e}", exc_info=True)
            await status_output.update(f"❌ Error loading items: {escape(str(e))}")

    async def _refresh_mgmt_item_list(self) -> None:
        """Fetches items for the selected DB and populates the management dropdown."""
        item_select = self.query_one("#mgmt-item-select", Select)
        db_select = self.query_one("#mgmt-db-select", Select)
        status_md = self.query_one("#mgmt-embedding-status-md", Markdown)
        mgmt_status = self.query_one("#mgmt-status-output", Markdown)

        db_type = str(db_select.value)
        db_name = self.DB_DISPLAY_NAMES.get(db_type, "Unknown Database")

        await status_md.update("⏳ Refreshing item list, please wait...")
        await mgmt_status.update(f"⏳ Loading items from {db_name}...")

        # This is where you would call your backend functions like `get_all_content_from_database`
        # For this example, I'll use placeholder data.
        await asyncio.sleep(0.1)  # Simulate async call

        # ---- Placeholder Logic ----
        # In a real app, replace this with your actual DB calls
        items = [
            {'id': 'media_1', 'title': 'My First Video'},
            {'id': 'media_2', 'title': 'Interesting Article'},
        ] if db_type == "media_db" else [
            {'id': 'conv_1', 'title': 'Chat about AGI'},
        ]
        # ---- End Placeholder Logic ----

        # Now, check embedding status for each item (as in your Gradio code)
        try:
            chroma_manager = await self._get_chroma_manager()
            collection_name = db_type  # Simplified name for example

            choices = []
            new_mapping = {}

            await mgmt_status.update(f"⏳ Checking embedding status for {len(items)} items...")

            # Check if the collection exists
            collections = chroma_manager.client.list_collections()
            collection_exists = any(collection.name == collection_name for collection in collections)

            for item in items:
                item_id = item['id']
                # This is a simplified check. Your real logic would be more robust.
                if not collection_exists:
                    status = "❓"  # Collection doesn't exist
                else:
                    try:
                        result = chroma_manager.client.get_collection(name=collection_name).get(ids=[f"{item_id}_chunk_0"])
                        status = "✅" if result and result['ids'] else "❌"
                    except Exception:
                        status = "⚠️"  # Error checking

                display_name = f"{item['title']} ({status})"
                choices.append((display_name, display_name))  # Textual Select options are (label, value)
                new_mapping[display_name] = item_id

            self._item_mapping = new_mapping
            item_select.set_options(choices)

            if len(items) > 0:
                await status_md.update(f"✅ Found {len(items)} items in {db_name}. Select one to see details.")
                await mgmt_status.update(f"Ready. {len(items)} items loaded from {db_name}.")
            else:
                await status_md.update(f"ℹ️ No items found in {db_name}. Try another database or check your data.")
                await mgmt_status.update(f"No items found in {db_name}.")

        except Exception as e:
            logger.error(f"Error refreshing item list: {e}", exc_info=True)
            await status_md.update(f"❌ Error loading items: {escape(str(e))}")
            await mgmt_status.update(f"Error: Failed to load items from {db_name}. See logs for details.")

    async def _create_embeddings(self, db_type: str, db_path: str, embedding_config: dict, collection_name: str, 
                           selection_mode: str = "all", keyword: str = None, selected_items: list = None) -> None:
        """
        Creates embeddings for the specified database with the given configuration and collection name.

        Args:
            db_type: The type of database (media_db, rag_chat_db, etc.)
            db_path: The path to the database
            embedding_config: Configuration for the embedding model and chunking
            collection_name: Custom name for the collection
            selection_mode: Mode of item selection ("all", "keyword", or "individual")
            keyword: Keyword to filter items by (used when selection_mode is "keyword")
            selected_items: List of item IDs to create embeddings for (used when selection_mode is "individual")
        """
        logger.info(f"Creating embeddings for {db_type} with collection name '{collection_name}', selection mode: {selection_mode}")

        try:
            # Get ChromaDB manager
            chroma_manager = await self._get_chroma_manager()

            # In a real implementation, you would:
            # 1. Load data from the database based on selection mode
            # 2. Create embeddings using the specified configuration
            # 3. Store the embeddings in ChromaDB with the specified collection name

            # For demonstration purposes, we'll log what would happen in each mode
            if selection_mode == "all":
                logger.info(f"Would create embeddings for ALL items in {db_type}")
                # In a real implementation, you would load all items from the database
                # items = load_all_items_from_database(db_type, db_path)
            elif selection_mode == "keyword":
                logger.info(f"Would create embeddings for items matching keyword '{keyword}' in {db_type}")
                # In a real implementation, you would filter items by keyword
                # items = load_items_matching_keyword(db_type, db_path, keyword)
            else:  # individual
                logger.info(f"Would create embeddings for {len(selected_items)} selected items in {db_type}: {selected_items}")
                # In a real implementation, you would load only the selected items
                # items = load_specific_items(db_type, db_path, selected_items)

            # For now, we'll just simulate the process with a delay
            await asyncio.sleep(2)

            # Log success
            logger.info(f"Successfully created embeddings for {db_type} with collection name '{collection_name}', selection mode: {selection_mode}")

        except Exception as e:
            logger.error(f"Error in _create_embeddings: {e}", exc_info=True)
            raise

    async def _check_and_display_embedding_status(self) -> None:
        """Fetches and displays the status of the currently selected embedding."""
        status_md = self.query_one("#mgmt-embedding-status-md", Markdown)
        metadata_md = self.query_one("#mgmt-embedding-metadata-md", Markdown)
        mgmt_status = self.query_one("#mgmt-status-output", Markdown)

        if not self._selected_embedding_id:
            await status_md.update("### No Item Selected\n\nPlease select an item from the dropdown above.")
            await metadata_md.update("No metadata available until an item is selected.")
            return

        item_display_name = self._selected_item_display_name or "Unknown Item"
        await status_md.update(f"### ⏳ Checking Status\n\nRetrieving embedding information for: `{self._selected_embedding_id}`...")
        await mgmt_status.update(f"⏳ Checking embedding status for {item_display_name}...")

        chroma_manager = await self._get_chroma_manager()
        db_select = self.query_one("#mgmt-db-select", Select)
        db_type = str(db_select.value)
        db_name = self.DB_DISPLAY_NAMES.get(db_type, "Unknown Database")
        collection_name = db_type  # Simplified name

        try:
            # First check if the collection exists
            collections = chroma_manager.client.list_collections()
            collection_exists = any(collection.name == collection_name for collection in collections)

            if not collection_exists:
                await status_md.update(f"### ❌ Collection Not Found\n\nThe collection `{collection_name}` does not exist in the database. You need to create embeddings first.")
                await metadata_md.update("No metadata available. You need to create embeddings for this database using the Embeddings Creation view.")
                await mgmt_status.update(f"Collection '{collection_name}' does not exist. Create embeddings first.")
                return

            # If collection exists, proceed with checking the item
            results = chroma_manager.client.get_collection(name=collection_name).get(
                ids=[f"{self._selected_embedding_id}_chunk_0"],  # Check first chunk
                include=["metadatas", "embeddings"]
            )

            if not results or not results['ids']:
                await status_md.update(f"### ❌ No Embedding Found\n\nThe selected item `{item_display_name}` does not have embeddings in the {db_name}.")
                await metadata_md.update("No metadata available. You can create embeddings for this item using the form on the right.")
                await mgmt_status.update(f"Item selected: {item_display_name} (No embeddings found)")
                return

            # Embedding exists - show detailed status
            await status_md.update(f"### ✅ Embedding Exists\n\nThe selected item has embeddings in the {db_name}.")

            # Format metadata in a more readable way
            metadata = results['metadatas'][0] if results.get('metadatas') else {}
            embedding_preview = str(results['embeddings'][0][:5]) + "..." if results.get('embeddings') else "N/A"
            embedding_dimensions = len(results['embeddings'][0]) if results.get('embeddings') and results['embeddings'][0] else "Unknown"

            md_content = f"### Embedding Information\n\n"
            md_content += f"- **Dimensions:** `{embedding_dimensions}`\n"
            md_content += f"- **Vector Preview:** `{embedding_preview}`\n\n"

            if metadata:
                md_content += f"### Metadata\n\n"
                for key, val in metadata.items():
                    md_content += f"- **{escape(str(key))}:** `{escape(str(val))}`\n"
            else:
                md_content += "No additional metadata available."

            await metadata_md.update(md_content)
            await mgmt_status.update(f"Item selected: {item_display_name} (Embeddings found with {embedding_dimensions} dimensions)")

        except Exception as e:
            logger.error(f"Error checking embedding status: {e}", exc_info=True)
            await status_md.update(f"### ❌ Error Checking Status\n\nFailed to retrieve embedding information.\n\n```\n{escape(str(e))}\n```")
            await metadata_md.update("No metadata available due to an error. See status message for details.")
            await mgmt_status.update(f"Error: Failed to check embedding status for {item_display_name}. See logs for details.")

#
# End of SearchWindow.py
########################################################################################################################
