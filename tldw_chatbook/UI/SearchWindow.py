# tldw_chatbook/UI/SearchWindow.py
#
#
# Imports
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


# Define constants for sub-tab view IDs and button IDs for clarity
# These could also be in Constants.py but are kept here for encapsulation with SearchWindow's structure
# View IDs
SEARCH_VIEW_RAG_QA = "search-view-rag-qa"
SEARCH_VIEW_RAG_CHAT = "search-view-rag-chat"
SEARCH_VIEW_EMBEDDINGS_CREATION = "search-view-embeddings-creation"
SEARCH_VIEW_RAG_MANAGEMENT = "search-view-rag-management"
SEARCH_VIEW_EMBEDDINGS_MANAGEMENT = "search-view-embeddings-management"
SEARCH_VIEW_WEB_SEARCH = "search-view-web-search"

# Button IDs
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
        """Called when the window is first mounted. Sets the initial view via the app's reactive state."""
        self.app_instance.log.info("SearchWindow.on_mount: Setting initial active sub-tab.")
        # FIX: Set the app's reactive variable. The watcher in app.py will handle showing the view.
        # This is the single source of truth for which view is active.
        if hasattr(self.app_instance, 'search_active_sub_tab'):
            self.app_instance.search_active_sub_tab = SEARCH_VIEW_EMBEDDINGS_CREATION

        # Initialize the view that is now visible.
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
            yield Container(Static("RAG QA Content - Coming Soon!"), id=SEARCH_VIEW_RAG_QA, classes="search-view-area")
            yield Container(Static("RAG Chat Content - Coming Soon!"), id=SEARCH_VIEW_RAG_CHAT,
                            classes="search-view-area")

            # Embeddings Creation View
            yield Container(
                Static("Create New Embeddings", classes="search-view-title"),
                Horizontal(
                    Static("Select Embedding Model:", classes="search-label"),
                    Select([], id="embeddings-model-select", prompt="Select a model..."),
                    classes="search-input-row"
                ),
                Horizontal(
                    Static("Content to Embed:", classes="search-label"),
                    Input(placeholder="Enter text or file path...", id="embeddings-content-input"),
                    classes="search-input-row"
                ),
                Horizontal(
                    Static("Collection Name:", classes="search-label"),
                    Input(placeholder="Enter collection name (optional)...", id="embeddings-collection-input"),
                    classes="search-input-row"
                ),
                Horizontal(
                    Button("Create Embeddings", id="embeddings-create-button", classes="search-action-button"),
                    Button("Clear", id="embeddings-clear-button", classes="search-action-button"),
                    classes="search-button-row"
                ),
                Static("Status: Ready", id="embeddings-status-display"),
                id=SEARCH_VIEW_EMBEDDINGS_CREATION,
                classes="search-view-area"
            )

            yield Container(Static("RAG Management Content - Coming Soon!"), id=SEARCH_VIEW_RAG_MANAGEMENT,
                            classes="search-view-area")

            # Embeddings Management View
            yield Container(
                Static("Manage Existing Embeddings", classes="search-view-title"),
                Horizontal(
                    Static("Collection:", classes="search-label"),
                    Select([], id="embeddings-management-collection-select", prompt="Select a collection..."),
                    Button("Refresh Collections", id="embeddings-management-refresh-button",
                           classes="search-action-button"),
                    classes="search-input-row"
                ),
                Horizontal(
                    Static("Search:", classes="search-label"),
                    Input(placeholder="Search within collection...", id="embeddings-management-search-input"),
                    Button("Search", id="embeddings-management-search-button", classes="search-action-button"),
                    classes="search-input-row"
                ),
                Static("Embeddings:", classes="search-section-title"),
                VerticalScroll(id="embeddings-management-results-container", classes="search-results-container"),
                Horizontal(
                    Static("Selected Embedding:", classes="search-label"),
                    Input(id="embeddings-management-selected-id", disabled=True),
                    classes="search-input-row"
                ),
                Horizontal(
                    Button("View Details", id="embeddings-management-view-button", classes="search-action-button"),
                    Button("Delete", id="embeddings-management-delete-button", classes="search-action-button"),
                    classes="search-button-row"
                ),
                Static("Status: Ready", id="embeddings-management-status-display"),
                id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT,
                classes="search-view-area"
            )

            if WEB_SEARCH_AVAILABLE:
                yield Container(
                    Input(placeholder="Enter search query...", id="web-search-input"),
                    Button("Search", id="web-search-button", classes="search-action-button"),
                    VerticalScroll(Markdown("", id="web-search-results")),
                    id=SEARCH_VIEW_WEB_SEARCH,
                    classes="search-view-area"
                )
            else:
                yield Container(
                    Markdown("### Web Search/Scraping Is Not Currently Installed\n\n..."),
                    id=SEARCH_VIEW_WEB_SEARCH,
                    classes="search-view-area"
                )

    @on(Button.Pressed, ".search-nav-button")
    async def handle_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handles all navigation button presses within the search tab."""
        event.stop()
        button_id = event.button.id
        if not button_id: return

        self.app_instance.log.info(f"Search nav button '{button_id}' pressed.")

        target_view_id = button_id.replace("-nav-", "-view-").replace("-disabled", "")
        self.app_instance.search_active_sub_tab = target_view_id

        # Perform view-specific initialization after switching
        if target_view_id == SEARCH_VIEW_EMBEDDINGS_CREATION:
            await self._initialize_embeddings_creation_view()
        elif target_view_id == SEARCH_VIEW_EMBEDDINGS_MANAGEMENT:
            await self._refresh_collections_list()

    ### Navigation handlers for all search tab views
    @on(Button.Pressed, f"#{SEARCH_NAV_WEB_SEARCH}, #search-nav-web-search-disabled")
    async def handle_web_search_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the web search navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to Web Search view.")
        await self._switch_to_view(SEARCH_VIEW_WEB_SEARCH)

    @on(Button.Pressed, f"#{SEARCH_NAV_EMBEDDINGS_CREATION}")
    async def handle_embeddings_creation_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the embeddings creation navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to Embeddings Creation view.")
        await self._switch_to_view(SEARCH_VIEW_EMBEDDINGS_CREATION)
        # Initialize the Embeddings Creation view
        await self._initialize_embeddings_creation_view()

    @on(Button.Pressed, f"#{SEARCH_NAV_EMBEDDINGS_MANAGEMENT}")
    async def handle_embeddings_management_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the embeddings management navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to Embeddings Management view.")
        await self._switch_to_view(SEARCH_VIEW_EMBEDDINGS_MANAGEMENT)
        # When switching to this view, refresh the collections list
        await self._refresh_collections_list()

    @on(Button.Pressed, f"#{SEARCH_NAV_RAG_QA}")
    async def handle_rag_qa_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the RAG QA navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to RAG QA view.")
        await self._switch_to_view(SEARCH_VIEW_RAG_QA)

    @on(Button.Pressed, f"#{SEARCH_NAV_RAG_CHAT}")
    async def handle_rag_chat_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the RAG Chat navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to RAG Chat view.")
        await self._switch_to_view(SEARCH_VIEW_RAG_CHAT)

    @on(Button.Pressed, f"#{SEARCH_NAV_RAG_MANAGEMENT}")
    async def handle_rag_management_nav_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the RAG Management navigation button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to RAG Management view.")
        await self._switch_to_view(SEARCH_VIEW_RAG_MANAGEMENT)

    async def _switch_to_view(self, target_view_id: str) -> None:
        """Helper method to switch between views in the search tab."""
        all_view_ids = [
            SEARCH_VIEW_RAG_QA, SEARCH_VIEW_RAG_CHAT, SEARCH_VIEW_EMBEDDINGS_CREATION,
            SEARCH_VIEW_RAG_MANAGEMENT, SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, SEARCH_VIEW_WEB_SEARCH
        ]
        for view_id in all_view_ids:
            # Using query() which returns a list, safer if an ID is missing
            for view in self.query(f"#{view_id}"):
                is_target_view = (view_id == target_view_id)
                view.styles.display = "block" if is_target_view else "none"

    async def _get_chroma_manager(self) -> "ChromaDBManager":
        """Get or create a ChromaDBManager instance using the app's configuration."""
        if self._chroma_manager is None:
            self.app_instance.log.info("ChromaDBManager instance not found, creating a new one.")
            logger.info("ChromaDBManager instance not found, creating a new one.")
            try:
                # Use the app's central configuration
                user_config = self.app_instance.app_config
                user_id = self.app_instance.notes_user_id

                # Your ChromaDBManager expects the whole config dictionary
                from ..Embeddings.Chroma_Lib import ChromaDBManager
                self._chroma_manager = ChromaDBManager(user_id=user_id, user_embedding_config=user_config)

                self.app_instance.log.info(f"Successfully created ChromaDBManager for user '{user_id}'.")
                logger.info(f"Successfully created ChromaDBManager for user '{user_id}'.")
            except Exception as e:
                self.app_instance.log.error(f"Failed to create ChromaDBManager: {e}", exc_info=True)
                logger.error(f"Failed to create ChromaDBManager: {e}")
                self.app_instance.notify(f"Failed to initialize embedding system: {e}", severity="error", timeout=10)
                logger.error(f"Failed to notify user about embedding system initialization error: {e}")
                # Re-raise to prevent dependent functions from running with a None manager
                raise
        return self._chroma_manager

    async def _initialize_embeddings_creation_view(self) -> None:
        """Initialize the Embeddings Creation view."""
        self.app_instance.log.info("Initializing Embeddings Creation view.")
        try:
            chroma_manager = await self._get_chroma_manager()
            if not chroma_manager: return

            # Populate the model select from the config
            model_select = self.query_one("#embeddings-model-select", Select)
            embedding_models = chroma_manager.embedding_config_schema.models.keys()
            default_model = chroma_manager.embedding_config_schema.default_model_id

            model_select.set_options([(model, model) for model in embedding_models])
            if default_model and default_model in embedding_models:
                model_select.value = default_model
            elif embedding_models:
                model_select.value = next(iter(embedding_models))  # Select the first one

            self.query_one("#embeddings-content-input", Input).value = ""
            self.query_one("#embeddings-collection-input", Input).value = ""
            self.query_one("#embeddings-status-display", Static).update("Status: Ready")
        except Exception as e:
            self.app_instance.log.error(f"Failed to initialize embeddings creation view: {e}", exc_info=True)
            try:
                self.query_one("#embeddings-status-display", Static).update(f"Status: Error - {e}")
            except Exception as e_status:
                self.app_instance.log.error(f"Failed to update status display: {e_status}")

    async def _refresh_collections_list(self) -> None:
        """Refresh the list of collections in the embeddings management view."""
        collection_select = self.query_one("#embeddings-management-collection-select", Select)
        status_display = self.query_one("#embeddings-management-status-display", Static)

        try:
            chroma_manager = await self._get_chroma_manager()
            if not chroma_manager:
                status_display.update("Status: Embedding system not available.")
                collection_select.set_options([])  # Clear any old options
                return

            # 1. Get the list of Collection objects
            collections = chroma_manager.list_collections()

            # 2. Create a list of (label, value) tuples from the collection names
            #    This is the core of the fix.
            collection_options = [(collection.name, collection.name) for collection in collections]

            # 3. Preserve the currently selected value if it exists in the new list
            current_selection = collection_select.value

            # 4. Set the new options on the widget all at once.
            #    This replaces collection_select.clear() and the loop.
            collection_select.set_options(collection_options)

            # 5. Re-apply the selection if it's still a valid option
            if current_selection and any(opt[1] == current_selection for opt in collection_options):
                collection_select.value = current_selection
            else:
                # If the old selection is gone, clear it.
                collection_select.value = None  # This will show the prompt

            status_display.update(f"Status: Found {len(collections)} collections.")

        except Exception as e:
            self.app_instance.log.error(f"Failed to refresh collections list: {e}", exc_info=True)
            logger.error(f"Failed to refresh collections list: {e}")
            self.app_instance.notify(f"Failed to refresh collections: {e}", severity="error")
            status_display.update(f"Status: Error refreshing collections: {e}")

    async def _update_embeddings_list(self, collection_name: str, search_query: str = "") -> None:
        """Update the list of embeddings in the embeddings management view."""
        try:
            # Get the results container
            results_container = self.query_one("#embeddings-management-results-container", VerticalScroll)

            # Clear the container
            await results_container.remove_children()

            # Get the ChromaDBManager
            chroma_manager = await self._get_chroma_manager()

            # If search query is provided, use it to filter results
            if search_query:
                # Search for embeddings matching the query
                results = chroma_manager.vector_search(
                    query=search_query,
                    collection_name=collection_name,
                    k=20  # Limit to 20 results
                )
            else:
                # Get all embeddings in the collection (up to a limit)
                # This is a simplified approach - in a real app, you might want pagination
                count = chroma_manager.count_items_in_collection(collection_name)
                if count > 0:
                    # Use vector_search with an empty query to get all items
                    # This is not ideal but works for demonstration purposes
                    results = chroma_manager.vector_search(
                        query="",
                        collection_name=collection_name,
                        k=min(count, 100)  # Limit to 100 results
                    )
                else:
                    results = []

            # Add the results to the container
            for item in results:
                item_id = item.get("id", "Unknown ID")
                metadata = item.get("metadata", {})
                source = metadata.get("source", "Unknown source")

                # Create a button for each item that will select it when clicked
                item_button = Button(f"{item_id} - {source}", id=f"embedding-item-{item_id}", classes="embedding-item-button")
                item_button.data = item  # Store the full item data in the button

                # Add the button to the container
                await results_container.mount(item_button)

            # Update the status display
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Found {len(results)} embeddings in collection '{collection_name}'")
            logger.info(f"Found {len(results)} embeddings in collection '{collection_name}'")

        except Exception as e:
            self.app_instance.log.error(f"Failed to update embeddings list: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to update embeddings: {e}", severity="error")
            logger.error(f"Failed to update embeddings list: {e}")
            # Update the status display
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Error updating embeddings: {e}")

    # --- Embeddings Creation Event Handlers ---

    @on(Button.Pressed, "#embeddings-create-button")
    async def handle_embeddings_create_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Create Embeddings button press."""
        self.app_instance.log.info("Create Embeddings button pressed.")
        logger.info("Create Embeddings button pressed in SearchWindow.")
        status_display = self.query_one("#embeddings-status-display", Static)
        try:
            model_select = self.query_one("#embeddings-model-select", Select)
            content_input = self.query_one("#embeddings-content-input", Input)
            collection_input = self.query_one("#embeddings-collection-input", Input)

            if not model_select.value or not content_input.value:
                self.app_instance.notify("Model and Content are required.", severity="warning")
                status_display.update("Status: Model and Content are required.")
                logger.warning("Model and Content are required for embeddings creation.")
                return

            status_display.update("Status: Creating embeddings...")
            chroma_manager = await self._get_chroma_manager()

            collection_name = collection_input.value or None
            content = content_input.value

            import time, hashlib
            media_id = f"manual_{hashlib.md5(content.encode()).hexdigest()[:8]}_{int(time.time())}"

            # Using a worker to avoid blocking the UI
            def create_embeddings_sync():
                chroma_manager.process_and_store_content(
                    content=content,
                    media_id=media_id,
                    file_name=f"manual_entry_{media_id}",
                    collection_name=collection_name,
                    embedding_model_id_override=str(model_select.value)
                )
                return f"Successfully created embedding for media_id '{media_id}'"

            loop = asyncio.get_running_loop()
            result_message = await loop.run_in_executor(None, create_embeddings_sync, )

            status_display.update(f"Status: {result_message}")
            self.app_instance.notify(result_message, severity="information")
            logger.info(f"Successfully created embedding: {result_message}")

        except Exception as e:
            self.app_instance.log.error(f"Failed to create embeddings: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to create embeddings: {e}", severity="error")
            logger.error(f"Failed to create embeddings: {e}")
            status_display.update(f"Status: Error creating embeddings: {e}")

    @on(Button.Pressed, "#embeddings-clear-button")
    async def handle_embeddings_clear_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Clear button press in the Embeddings Creation view."""
        self.app_instance.log.info("Clear button pressed in Embeddings Creation view.")

        try:
            # Clear the input fields
            content_input = self.query_one("#embeddings-content-input", Input)
            collection_input = self.query_one("#embeddings-collection-input", Input)

            content_input.value = ""
            collection_input.value = ""

            # Reset the status
            status_display = self.query_one("#embeddings-status-display", Static)
            status_display.update("Status: Ready")

        except Exception as e:
            self.app_instance.log.error(f"Failed to clear embeddings creation form: {e}", exc_info=True)

    # --- Embeddings Management Event Handlers ---

    @on(Button.Pressed, "#embeddings-management-refresh-button")
    async def handle_embeddings_management_refresh_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Refresh Collections button press."""
        self.app_instance.log.info("Refresh Collections button pressed.")
        await self._refresh_collections_list()

    @on(Select.Changed, "#embeddings-management-collection-select")
    async def handle_embeddings_management_collection_select_changed(self, event: Select.Changed) -> None:
        """Handle the Collection select change."""
        self.app_instance.log.info(f"Collection select changed to {event.value}.")

        if event.value:
            # Update the embeddings list for the selected collection
            await self._update_embeddings_list(event.value)

    @on(Button.Pressed, "#embeddings-management-search-button")
    async def handle_embeddings_management_search_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Search button press in the Embeddings Management view."""
        self.app_instance.log.info("Search button pressed in Embeddings Management view.")

        try:
            # Get the collection and search query
            collection_select = self.query_one("#embeddings-management-collection-select", Select)
            search_input = self.query_one("#embeddings-management-search-input", Input)

            if not collection_select.value:
                self.app_instance.notify("Please select a collection first.", severity="warning")
                return

            # Update the embeddings list with the search query
            await self._update_embeddings_list(collection_select.value, search_input.value)

        except Exception as e:
            self.app_instance.log.error(f"Failed to search embeddings: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to search embeddings: {e}", severity="error")

    @on(Button.Pressed, ".embedding-item-button")
    async def handle_embedding_item_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the embedding item button press."""
        self.app_instance.log.info(f"Embedding item button {event.button.id} pressed.")

        try:
            # Store the selected embedding ID
            self._selected_embedding_id = event.button.id.replace("embedding-item-", "")

            # Update the selected ID input
            selected_id_input = self.query_one("#embeddings-management-selected-id", Input)
            selected_id_input.value = self._selected_embedding_id

        except Exception as e:
            self.app_instance.log.error(f"Failed to handle embedding item selection: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to select embedding: {e}", severity="error")

    @on(Button.Pressed, "#embeddings-management-delete-button")
    async def handle_embeddings_management_delete_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Delete button press in the Embeddings Management view."""
        self.app_instance.log.info("Delete button pressed in Embeddings Management view.")

        try:
            # Check if an embedding is selected
            if not self._selected_embedding_id:
                self.app_instance.notify("Please select an embedding to delete.", severity="warning")
                return

            # Get the collection
            collection_select = self.query_one("#embeddings-management-collection-select", Select)
            if not collection_select.value:
                self.app_instance.notify("Please select a collection first.", severity="warning")
                return

            # Get the ChromaDBManager
            chroma_manager = await self._get_chroma_manager()

            # Delete the embedding
            chroma_manager.delete_from_collection(
                ids=[self._selected_embedding_id],
                collection_name=collection_select.value
            )

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Deleted embedding {self._selected_embedding_id}")
            self.app_instance.notify(f"Deleted embedding {self._selected_embedding_id}", severity="information")

            # Clear the selected ID
            self._selected_embedding_id = None
            selected_id_input = self.query_one("#embeddings-management-selected-id", Input)
            selected_id_input.value = ""

            # Refresh the embeddings list
            await self._update_embeddings_list(collection_select.value)

        except Exception as e:
            self.app_instance.log.error(f"Failed to delete embedding: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to delete embedding: {e}", severity="error")

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Error deleting embedding: {e}")

    @on(Button.Pressed, "#embeddings-management-view-button")
    async def handle_embeddings_management_view_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the View Details button press in the Embeddings Management view."""
        self.app_instance.log.info("View Details button pressed in Embeddings Management view.")

        try:
            # Check if an embedding is selected
            if not self._selected_embedding_id:
                self.app_instance.notify("Please select an embedding to view.", severity="warning")
                return

            # Get the collection
            collection_select = self.query_one("#embeddings-management-collection-select", Select)
            if not collection_select.value:
                self.app_instance.notify("Please select a collection first.", severity="warning")
                return

            # Get the ChromaDBManager
            chroma_manager = await self._get_chroma_manager()

            # Search for the embedding to get its details
            results = chroma_manager.vector_search(
                query="",  # Empty query to get all items
                collection_name=collection_select.value,
                k=100  # Limit to 100 results
            )

            # Find the selected embedding
            selected_item = None
            for item in results:
                if item.get("id") == self._selected_embedding_id:
                    selected_item = item
                    break

            if not selected_item:
                self.app_instance.notify(f"Could not find details for embedding {self._selected_embedding_id}", severity="warning")
                return

            # Display the details
            metadata = selected_item.get("metadata", {})
            content = selected_item.get("document", "No content available")

            # Format the details as a message
            details = f"ID: {self._selected_embedding_id}\n"
            details += f"Source: {metadata.get('source', 'Unknown')}\n"
            details += f"Model: {metadata.get('embedding_model', 'Unknown')}\n"
            details += f"Created: {metadata.get('created_at', 'Unknown')}\n"
            details += f"Content: {content[:500]}{'...' if len(content) > 500 else ''}"

            # Show the details in a notification
            self.app_instance.notify(details, severity="information", timeout=20)

        except Exception as e:
            self.app_instance.log.error(f"Failed to view embedding details: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to view embedding details: {e}", severity="error")

    @on(Button.Pressed, "#embeddings-management-recreate-selected-button")
    async def handle_embeddings_management_recreate_selected_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Recreate Selected button press in the Embeddings Management view."""
        self.app_instance.log.info("Recreate Selected button pressed in Embeddings Management view.")

        try:
            # Check if an embedding is selected
            if not self._selected_embedding_id:
                self.app_instance.notify("Please select an embedding to recreate.", severity="warning")
                return

            # Get the collection and model
            collection_select = self.query_one("#embeddings-management-collection-select", Select)
            model_select = self.query_one("#embeddings-management-model-select", Select)

            if not collection_select.value:
                self.app_instance.notify("Please select a collection first.", severity="warning")
                return

            if not model_select.value:
                self.app_instance.notify("Please select a model for recreation.", severity="warning")
                return

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Recreating embedding {self._selected_embedding_id} with model {model_select.value}...")

            # Get the ChromaDBManager
            chroma_manager = await self._get_chroma_manager()

            # Search for the embedding to get its content
            results = chroma_manager.vector_search(
                query="",  # Empty query to get all items
                collection_name=collection_select.value,
                k=100  # Limit to 100 results
            )

            # Find the selected embedding
            selected_item = None
            for item in results:
                if item.get("id") == self._selected_embedding_id:
                    selected_item = item
                    break

            if not selected_item:
                self.app_instance.notify(f"Could not find content for embedding {self._selected_embedding_id}", severity="warning")
                status_display.update(f"Status: Error - Could not find content for embedding {self._selected_embedding_id}")
                return

            # Get the content
            content = selected_item.get("document", "")
            if not content:
                self.app_instance.notify("No content found in the selected embedding.", severity="warning")
                status_display.update("Status: Error - No content found in the selected embedding.")
                return

            # Delete the old embedding
            chroma_manager.delete_from_collection(
                ids=[self._selected_embedding_id],
                collection_name=collection_select.value
            )

            # Create a new embedding with the same ID but different model
            chroma_manager.process_and_store_content(
                content=content,
                media_id=self._selected_embedding_id,
                file_name=f"recreated_{self._selected_embedding_id}",
                collection_name=collection_select.value,
                embedding_model_id_override=model_select.value
            )

            # Update the status
            status_display.update(f"Status: Successfully recreated embedding {self._selected_embedding_id} with model {model_select.value}")
            self.app_instance.notify(f"Successfully recreated embedding with model {model_select.value}", severity="information")

            # Refresh the embeddings list
            await self._update_embeddings_list(collection_select.value)

        except Exception as e:
            self.app_instance.log.error(f"Failed to recreate embedding: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to recreate embedding: {e}", severity="error")

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Error recreating embedding: {e}")

    @on(Button.Pressed, "#embeddings-management-recreate-all-button")
    async def handle_embeddings_management_recreate_all_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Recreate All button press in the Embeddings Management view."""
        self.app_instance.log.info("Recreate All button pressed in Embeddings Management view.")

        try:
            # Get the collection and model
            collection_select = self.query_one("#embeddings-management-collection-select", Select)
            model_select = self.query_one("#embeddings-management-model-select", Select)

            if not collection_select.value:
                self.app_instance.notify("Please select a collection first.", severity="warning")
                return

            if not model_select.value:
                self.app_instance.notify("Please select a model for recreation.", severity="warning")
                return

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Recreating all embeddings in collection {collection_select.value} with model {model_select.value}...")

            # Get the ChromaDBManager
            chroma_manager = await self._get_chroma_manager()

            # Get all embeddings in the collection
            results = chroma_manager.vector_search(
                query="",  # Empty query to get all items
                collection_name=collection_select.value,
                k=100  # Limit to 100 results
            )

            if not results:
                self.app_instance.notify("No embeddings found in the selected collection.", severity="warning")
                status_display.update("Status: No embeddings found in the selected collection.")
                return

            # Reset the collection with the new model
            # This is more efficient than recreating each embedding individually
            chroma_manager.reset_chroma_collection(
                collection_name=collection_select.value,
                new_metadata={"default_embedding_model": model_select.value}
            )

            # Re-add all the embeddings with the new model
            for item in results:
                item_id = item.get("id")
                content = item.get("document", "")
                if content:
                    chroma_manager.process_and_store_content(
                        content=content,
                        media_id=item_id,
                        file_name=f"recreated_{item_id}",
                        collection_name=collection_select.value,
                        embedding_model_id_override=model_select.value
                    )

            # Update the status
            status_display.update(f"Status: Successfully recreated {len(results)} embeddings with model {model_select.value}")
            self.app_instance.notify(f"Successfully recreated {len(results)} embeddings with model {model_select.value}", severity="information")

            # Refresh the embeddings list
            await self._update_embeddings_list(collection_select.value)

        except Exception as e:
            self.app_instance.log.error(f"Failed to recreate all embeddings: {e}", exc_info=True)
            self.app_instance.notify(f"Failed to recreate all embeddings: {e}", severity="error")

            # Update the status
            status_display = self.query_one("#embeddings-management-status-display", Static)
            status_display.update(f"Status: Error recreating all embeddings: {e}")

    # --- Web Search Event Handlers ---

    @on(Button.Pressed, "#web-search-button")
    async def handle_perform_web_search_button_pressed(self, event: Button.Pressed) -> None:
        self.app_instance.log.info(f"Button {event.button.id} pressed. Performing web search.")
        query_input = self.query_one("#web-search-input", Input)
        results_markdown = self.query_one("#web-search-results", Markdown)

        query = query_input.value
        if not query:
            await results_markdown.update("Please enter a search query.")
            return

        await results_markdown.update("Searching...")  # Placeholder message

        try:
            # Default search_params; these might need to be configurable later
            search_params = {
                "engine": "google",  # Or another default engine
                "content_country": "US",
                "search_lang": "en",
                "output_lang": "en",
                "result_count": 5, # Keep it low for initial testing
                "subquery_generation": False, # Keep it simple for now
                # Add other necessary params from WebSearch_APIs.py if they are mandatory
                # e.g. "relevance_analysis_llm": "openai", "final_answer_llm": "openai"
                # These might require loading API keys or further configuration.
                # For now, let's assume they can be omitted or have defaults in the backend.
            }

            # Running synchronous function in an executor to avoid blocking UI thread
            loop = asyncio.get_event_loop()

            await results_markdown.update(f"Performing search for: {query}...")

            # Step 1: Generate and Search (Synchronous)
            phase1_results = await loop.run_in_executor(None, generate_and_search, query, search_params)

            web_search_results = phase1_results.get("web_search_results_dict", {})
            # sub_query_details = phase1_results.get("sub_query_dict", {}) # Not used for now

            if web_search_results.get("error") or web_search_results.get("processing_error"):
                error_message = web_search_results.get("error") or web_search_results.get("processing_error")
                await results_markdown.update(f"Error during search: {error_message}")
                return

            formatted_results = "## Raw Search Results:\n\n"
            if web_search_results.get("results"):
                for i, res in enumerate(web_search_results["results"][:5]): # Display top 5
                    formatted_results += f"### {res.get('title', 'No Title')}\n"
                    formatted_results += f"*URL:* {res.get('url', '#')}\n"
                    # Ensure snippet is a string and escape markdown characters like '_'
                    snippet = str(res.get('content', 'No snippet available')).replace('_', '\\_')
                    formatted_results += f"*Snippet:* {snippet}\n\n"
            else:
                formatted_results += "No results found."

            await results_markdown.update(formatted_results)

            # TODO: Implement Phase 2 (analyze_and_aggregate) later if phase 1 works
            # This would involve:
            # phase2_results = await analyze_and_aggregate(web_search_results, sub_query_details, search_params)
            # final_answer = phase2_results.get("final_answer", {}).get("Report", "Could not generate final answer.")
            # results_markdown.update(f"## Final Answer:\n\n{final_answer}")

        except Exception as e:
            self.app_instance.log.error(f"Error during web search: {e}", exc_info=True)
            await results_markdown.update(f"An unexpected error occurred: {e}")

#
# End of SearchWindow.py
########################################################################################################################
