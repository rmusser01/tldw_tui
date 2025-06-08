# tldw_chatbook/UI/SearchWindow.py
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Button, Input, Markdown

from typing import TYPE_CHECKING
import asyncio
from ..Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate


if TYPE_CHECKING:
    from ..app import TldwCli
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
        # Ensure the main window has the ID used in app.py for tab switching
        # The id="search-window" is set when this class is instantiated in app.py's compose_content_area

    def compose(self) -> ComposeResult:
        # Main horizontal layout for the Search tab.
        # The SearchWindow itself (id="search-window") will have layout: horizontal from .window class.

        # Left Vertical Tab Bar
        with Vertical(id="search-left-nav-pane", classes="search-nav-pane"):
            yield Button("RAG QA", id=SEARCH_NAV_RAG_QA, classes="search-nav-button")
            yield Button("RAG Chat", id=SEARCH_NAV_RAG_CHAT, classes="search-nav-button")
            yield Button("Embeddings Creation", id=SEARCH_NAV_EMBEDDINGS_CREATION, classes="search-nav-button")
            yield Button("RAG Management", id=SEARCH_NAV_RAG_MANAGEMENT, classes="search-nav-button")
            yield Button("Embeddings Management", id=SEARCH_NAV_EMBEDDINGS_MANAGEMENT, classes="search-nav-button")
            yield Button("Web Search", id=SEARCH_NAV_WEB_SEARCH, classes="search-nav-button")

        # Right Content Pane
        with Container(id="search-content-pane", classes="search-content-pane"):
            # Individual view areas, only one visible at a time. Watcher handles display.
            yield Container(Static("RAG QA Content - Coming Soon!"), id=SEARCH_VIEW_RAG_QA, classes="search-view-area")
            yield Container(Static("RAG Chat Content - Coming Soon!"), id=SEARCH_VIEW_RAG_CHAT,
                            classes="search-view-area")
            yield Container(Static("Embeddings Creation Content - Coming Soon!"), id=SEARCH_VIEW_EMBEDDINGS_CREATION,
                            classes="search-view-area")
            yield Container(Static("RAG Management Content - Coming Soon!"), id=SEARCH_VIEW_RAG_MANAGEMENT,
                            classes="search-view-area")
            yield Container(Static("Embeddings Management Content - Coming Soon!"),
                            id=SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, classes="search-view-area")
            # yield Container(Static("Web Search Content - Coming Soon!"), id=SEARCH_VIEW_WEB_SEARCH, classes="search-view-area")
            yield Container(
                Input(placeholder="Enter search query...", id="web-search-input"),
                Button("Search", id="web-search-button", classes="search-action-button"),
                VerticalScroll(Markdown("", id="web-search-results")), # To display results
                id=SEARCH_VIEW_WEB_SEARCH,
                classes="search-view-area"
            )

    @on(Button.Pressed, f"#{SEARCH_NAV_WEB_SEARCH}")
    async def handle_web_search_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the web search button press to switch views."""
        self.app_instance.log.info(f"Button {event.button.id} pressed. Switching to Web Search view.")
        self.query_one("#search-content-pane").set_styles("display: none;")
        for view_id in [SEARCH_VIEW_RAG_QA, SEARCH_VIEW_RAG_CHAT, SEARCH_VIEW_EMBEDDINGS_CREATION, SEARCH_VIEW_RAG_MANAGEMENT, SEARCH_VIEW_EMBEDDINGS_MANAGEMENT, SEARCH_VIEW_WEB_SEARCH]:
            view = self.query_one(f"#{view_id}")
            if view_id == SEARCH_VIEW_WEB_SEARCH:
                view.set_styles("display: block;")
            else:
                view.set_styles("display: none;")
        self.query_one("#search-content-pane").set_styles("display: block;")

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
