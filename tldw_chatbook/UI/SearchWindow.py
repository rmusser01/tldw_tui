# tldw_chatbook/UI/SearchWindow.py
#
#
# Imports
from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Button, Input, Markdown, Select
#
# Third-Party Libraries
from typing import TYPE_CHECKING, Union
import asyncio
from loguru import logger
#
# Local Imports
from tldw_chatbook.Embeddings.Chroma_Lib import ChromaDBManager
from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory, EmbeddingConfigSchema
from tldw_chatbook.config import get_chachanotes_db_path
if TYPE_CHECKING:
    from ..app import TldwCli
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


class SearchWindow(Container):
    """
    Container for the Search Tab's UI, featuring a vertical tab bar and content areas.
    """

    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._chroma_manager: Union["ChromaDBManager", None] = None
        self._selected_embedding_id: Union[str, None] = None

    async def on_mount(self) -> None:
        """Called when the window is first mounted."""
        logger.info("SearchWindow.on_mount: Setting and initializing initial active sub-tab.")
        if hasattr(self.app_instance, 'search_active_sub_tab') and self.app_instance.search_active_sub_tab is None:
            # Set the initial view
            self.app_instance.search_active_sub_tab = SEARCH_VIEW_EMBEDDINGS_CREATION
            # And call its initializer
            await self._initialize_embeddings_creation_view()

    def compose(self) -> ComposeResult:
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("Embeddings Creation", id=SEARCH_NAV_EMBEDDINGS_CREATION, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            yield Button("Embeddings Management", id=SEARCH_NAV_EMBEDDINGS_MANAGEMENT, classes="search-nav-button")
            if WEB_SEARCH_AVAILABLE:
                yield Button("Web Search", id=SEARCH_NAV_WEB_SEARCH, classes="search-nav-button")
            else:
                yield Button("Web Search", id="search-nav-web-search-disabled", classes="search-nav-button disabled")

        with Container(id="search-content-pane", classes="search-content-pane"):
            # --- Each view is now a Container with a VerticalScroll inside ---
            # --- This isolates scrolling and fixes the layout problem.    ---

            with Container(id=SEARCH_VIEW_RAG_QA, classes="search-view-area"):
                with VerticalScroll():
                    yield Static("RAG QA Content - Coming Soon!")

            with Container(id=SEARCH_VIEW_RAG_CHAT, classes="search-view-area"):
                with VerticalScroll():
                    yield Static("RAG Chat Content - Coming Soon!")

            with Container(id=SEARCH_VIEW_EMBEDDINGS_CREATION, classes="search-view-area"):
                with VerticalScroll():
                    yield Static("Create New Embeddings", classes="search-view-title")
                    yield Horizontal(Static("Select Embedding Model:", classes="search-label"), Select([], id="embeddings-model-select", prompt="Select a model..."), classes="search-input-row")
                    yield Horizontal(Static("Content to Embed:", classes="search-label"), Input(placeholder="Enter text or file path...", id="embeddings-content-input"), classes="search-input-row")
                    yield Horizontal(Static("Collection Name:", classes="search-label"), Input(placeholder="Enter collection name (optional)...", id="embeddings-collection-input"), classes="search-input-row")
                    yield Horizontal(Button("Create Embeddings", id="embeddings-create-button", classes="search-action-button"), Button("Clear", id="embeddings-clear-button", classes="search-action-button"), classes="search-button-row")
                    yield Static("Status: Ready", id="embeddings-status-display")

            with Container(id=SEARCH_VIEW_RAG_MANAGEMENT, classes="search-view-area"):
                with VerticalScroll():
                    yield Static("RAG Management Content - Coming Soon!")

            with Container(id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area"):
                with VerticalScroll():
                    yield Static("Manage Existing Embeddings", classes="search-view-title")
                    yield Horizontal(Static("Collection:", classes="search-label"), Select([], id="embeddings-management-collection-select", prompt="Select a collection..."), Button("Refresh Collections", id="embeddings-management-refresh-button", classes="search-action-button"), classes="search-input-row")
                    yield Horizontal(Static("Search:", classes="search-label"), Input(placeholder="Search within collection...", id="embeddings-management-search-input"), Button("Search", id="embeddings-management-search-button", classes="search-action-button"), classes="search-input-row")
                    yield Static("Embeddings:", classes="search-section-title")
                    yield VerticalScroll(id="embeddings-management-results-container", classes="search-results-container")
                    yield Horizontal(Static("Selected Embedding:", classes="search-label"), Input(id="embeddings-management-selected-id", disabled=True), classes="search-input-row")
                    yield Horizontal(Button("View Details", id="embeddings-management-view-button", classes="search-action-button"), Button("Delete", id="embeddings-management-delete-button", classes="search-action-button"), classes="search-button-row")
                    yield Static("Content:", classes="search-section-title")
                    yield VerticalScroll(Markdown("", id="embeddings-management-view-content"), classes="search-results-container")
                    yield Static("Status: Ready", id="embeddings-management-status-display")

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

    # --- DECORATED EVENT HANDLERS ---

    @on(Button.Pressed, ".search-nav-button")
    async def handle_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handles all navigation button presses within the search tab."""
        event.stop()
        button_id = event.button.id
        if not button_id: return

        logger.info(f"Search nav button '{button_id}' pressed.")
        target_view_id = button_id.replace("-nav-", "-view-").replace("-disabled", "")

        # 1. Tell the app to switch the view. The watcher in app.py will handle the visual change.
        self.app_instance.search_active_sub_tab = target_view_id

        # 2. Immediately call the initialization/data-loading function for that view.
        try:
            if target_view_id == SEARCH_VIEW_EMBEDDINGS_CREATION:
                await self._initialize_embeddings_creation_view()
            elif target_view_id == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
                await self._refresh_collections_list()
        except Exception as e:
            logger.error(f"Error initializing view '{target_view_id}': {e}", exc_info=True)
            self.app_instance.notify(f"Error loading view: {escape(str(e))}", severity="error")

    # --- Embeddings Creation Event Handlers ---

    @on(Button.Pressed, "#embeddings-create-button")
    async def handle_embeddings_create_button_pressed(self, event: Button.Pressed) -> None:
        logger.info("Create Embeddings button pressed.")
        status_display = self.query_one("#embeddings-status-display", Static)
        try:
            model_select = self.query_one("#embeddings-model-select", Select)
            content_input = self.query_one("#embeddings-content-input", Input)
            collection_input = self.query_one("#embeddings-collection-input", Input)

            if not model_select.value or model_select.value is Select.BLANK or not content_input.value:
                self.app_instance.notify("Model and Content are required.", severity="warning")
                status_display.update("Status: Model and Content are required.")
                return

            status_display.update("Status: Creating embeddings...")
            chroma_manager = await self._get_chroma_manager()
            collection_name = collection_input.value or None
            content = content_input.value
            import time, hashlib
            media_id = f"manual_{hashlib.md5(content.encode()).hexdigest()[:8]}_{int(time.time())}"

            def create_embeddings_sync():
                chroma_manager.process_and_store_content(
                    content=content, media_id=media_id, file_name=f"manual_entry_{media_id}",
                    collection_name=collection_name, embedding_model_id_override=str(model_select.value)
                )
                return f"Successfully created embedding for media_id '{media_id}'"

            result_message = await asyncio.to_thread(create_embeddings_sync)
            status_display.update(f"Status: {result_message}")
            self.app_instance.notify(result_message, severity="information")
            logger.info(f"Embeddings creation result: {result_message}")

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to create embeddings: {escape(str(e))}", severity="error")
            status_display.update(f"Status: Error creating embeddings: {escape(str(e))}")

    @on(Button.Pressed, "#embeddings-clear-button")
    async def handle_embeddings_clear_button_pressed(self, event: Button.Pressed) -> None:
        self.query_one("#embeddings-content-input", Input).value = ""
        self.query_one("#embeddings-collection-input", Input).value = ""
        self.query_one("#embeddings-status-display", Static).update("Status: Ready")

    # --- Embeddings Management Event Handlers ---

    @on(Button.Pressed, "#embeddings-management-refresh-button")
    async def handle_embeddings_management_refresh_button_pressed(self, event: Button.Pressed) -> None:
        await self._refresh_collections_list()

    @on(Select.Changed, "#embeddings-management-collection-select")
    async def handle_collection_select_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            await self._update_embeddings_list(str(event.value))

    @on(Button.Pressed, "#embeddings-management-search-button")
    async def handle_embeddings_search_button_pressed(self, event: Button.Pressed) -> None:
        collection_select = self.query_one("#embeddings-management-collection-select", Select)
        search_input = self.query_one("#embeddings-management-search-input", Input)
        if collection_select.value is Select.BLANK:
            self.app_instance.notify("Please select a collection first.", severity="warning")
            return
        await self._update_embeddings_list(str(collection_select.value), search_input.value)

    @on(Button.Pressed, ".embedding-item-button")
    async def handle_embedding_item_button_pressed(self, event: Button.Pressed) -> None:
        self._selected_embedding_id = event.button.id.replace("embedding-item-", "")
        self.query_one("#embeddings-management-selected-id", Input).value = self._selected_embedding_id
        await self.query_one("#embeddings-management-view-content", Markdown).update("")

    @on(Button.Pressed, "#embeddings-management-delete-button")
    async def handle_delete_embedding_button_pressed(self, event: Button.Pressed) -> None:
        status_display = self.query_one("#embeddings-management-status-display", Static)
        if not self._selected_embedding_id:
            self.app_instance.notify("Please select an embedding to delete.", severity="warning")
            return
        collection_select = self.query_one("#embeddings-management-collection-select", Select)
        if collection_select.value is Select.BLANK:
            self.app_instance.notify("Please select a collection first.", severity="warning")
            return

        try:
            collection_name = str(collection_select.value)
            chroma_manager = await self._get_chroma_manager()
            chroma_manager.delete_from_collection(ids=[self._selected_embedding_id], collection_name=collection_name)

            self.app_instance.notify(f"Deleted embedding {self._selected_embedding_id}", severity="information")
            status_display.update(f"Status: Deleted embedding {self._selected_embedding_id}")
            self._selected_embedding_id = None
            self.query_one("#embeddings-management-selected-id", Input).value = ""
            await self._update_embeddings_list(collection_name)
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to delete embedding: {escape(str(e))}", severity="error")
            status_display.update(f"Status: Error deleting embedding: {escape(str(e))}")

    @on(Button.Pressed, "#embeddings-management-view-button")
    async def handle_view_embedding_button_pressed(self, event: Button.Pressed) -> None:
        if not self._selected_embedding_id:
            self.app_instance.notify("Please select an embedding to view.", severity="warning")
            return
        collection_select = self.query_one("#embeddings-management-collection-select", Select)
        if collection_select.value is Select.BLANK:
            self.app_instance.notify("Please select a collection first.", severity="warning")
            return

        try:
            collection_name = str(collection_select.value)
            chroma_manager = await self._get_chroma_manager()
            results = chroma_manager.client.get_collection(name=collection_name).get(ids=[self._selected_embedding_id],
                                                                                     include=["metadatas", "documents"])

            if not results['ids']:
                self.app_instance.notify(f"Could not find details for embedding {self._selected_embedding_id}",
                                         severity="warning")
                return

            metadata = results['metadatas'][0] if results['metadatas'] else {}
            content = results['documents'][0] if results['documents'] else "No content available"

            markdown_content = f"### ID: `{self._selected_embedding_id}`\n\n"
            for key, value in metadata.items():
                markdown_content += f"- **{key}:** `{value}`\n"
            markdown_content += f"\n---\n\n### Content\n\n```\n{content}\n```"

            await self.query_one("#embeddings-management-view-content", Markdown).update(markdown_content)
            self.app_instance.notify("Embedding details loaded", severity="information", timeout=3)
        except Exception as e:
            logger.error(f"Failed to view embedding details: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to view embedding details: {escape(str(e))}", severity="error")

    @on(Button.Pressed, "#web-search-button")
    async def handle_web_search_button_pressed(self, event: Button.Pressed) -> None:
        # Placeholder for your web search logic
        pass

    # --- Initialization and Helper Functions (to be called by the handlers) ---

    async def _initialize_embeddings_creation_view(self) -> None:
        logger.info("Initializing Embeddings Creation view.")
        try:
            chroma_manager = await self._get_chroma_manager()
            if not chroma_manager: return
            model_select = self.query_one("#embeddings-model-select", Select)
            embedding_models = chroma_manager.embedding_config_schema.models.keys()
            default_model = chroma_manager.embedding_config_schema.default_model_id
            model_select.set_options([(model, model) for model in embedding_models])
            if default_model and default_model in embedding_models:
                model_select.value = default_model
            elif embedding_models:
                model_select.value = next(iter(embedding_models))
            self.query_one("#embeddings-content-input", Input).value = ""
            self.query_one("#embeddings-collection-input", Input).value = ""
            self.query_one("#embeddings-status-display", Static).update("Status: Ready")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings creation view: {e}", exc_info=True)
            self.query_one("#embeddings-status-display", Static).update(f"Status: Error - {escape(str(e))}")

    async def _refresh_collections_list(self) -> None:
        collection_select = self.query_one("#embeddings-management-collection-select", Select)
        status_display = self.query_one("#embeddings-management-status-display", Static)
        try:
            chroma_manager = await self._get_chroma_manager()
            if not chroma_manager:
                status_display.update("Status: Embedding system not available.")
                collection_select.set_options([])
                return
            collections = chroma_manager.list_collections()
            collection_options = [(c.name, c.name) for c in collections]
            current_selection = collection_select.value
            collection_select.set_options(collection_options)
            if current_selection is not Select.BLANK and any(opt[1] == current_selection for opt in collection_options):
                collection_select.value = current_selection
            else:
                collection_select.value = Select.BLANK
            status_display.update(f"Status: Found {len(collections)} collections.")
        except Exception as e:
            logger.error(f"Failed to refresh collections list: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to refresh collections: {escape(str(e))}", severity="error")
            status_display.update(f"Status: Error refreshing collections: {escape(str(e))}")

    async def _update_embeddings_list(self, collection_name: str, search_query: str = "") -> None:
        results_container = self.query_one("#embeddings-management-results-container", VerticalScroll)
        status_display = self.query_one("#embeddings-management-status-display", Static)

        await results_container.remove_children()
        self._selected_embedding_id = None
        self.query_one("#embeddings-management-selected-id", Input).value = ""
        await self.query_one("#embeddings-management-view-content", Markdown).update("")

        if not collection_name:
            status_display.update("Status: Please select a collection.")
            return

        try:
            chroma_manager = await self._get_chroma_manager()
            status_display.update(f"Status: Searching in '{collection_name}'...")

            def do_search():
                if search_query:
                    return chroma_manager.vector_search(query=search_query, collection_name=collection_name, k=20,
                                                        include_fields=["metadatas"])
                else:
                    count = chroma_manager.count_items_in_collection(collection_name)
                    if count == 0: return []
                    raw_results = chroma_manager.client.get_collection(name=collection_name).get(limit=min(count, 200),
                                                                                                 include=["metadatas"])
                    return [{"id": id, "metadata": meta} for id, meta in
                            zip(raw_results.get('ids', []), raw_results.get('metadatas', []))]

            results = await asyncio.to_thread(do_search)

            for item in results:
                item_id = item.get("id", "Unknown ID")
                item_button = Button(f"ID: {item_id}", id=f"embedding-item-{item_id}", classes="embedding-item-button")
                await results_container.mount(item_button)

            status_display.update(f"Status: Found {len(results)} embeddings in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update embeddings list: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to update embeddings: {escape(str(e))}", severity="error")
            status_display.update(f"Status: Error updating embeddings: {escape(str(e))}")

#
# End of SearchWindow.py
########################################################################################################################
