# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Imports
import logging
import logging.handlers
import sys
from pathlib import Path
import traceback
from typing import Union, Optional, List, Dict, Any
#
# 3rd-Party Libraries
# --- Textual Imports ---
from loguru import logger as loguru_logger # Keep if app.py uses it directly, or pass app.loguru_logger
from rich.text import Text
# --- Textual Imports ---
from textual.app import App, ComposeResult
from textual.logging import TextualHandler
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select, ListView, Checkbox, Label, Collapsible
)
from textual.containers import Horizontal, Container, VerticalScroll
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual.dom import DOMNode  # For type hinting if needed
from textual.timer import Timer
from textual.css.query import QueryError  # For specific error handling
#
# --- Local API library Imports ---
from tldw_app.Constants import ALL_TABS, TAB_CCP, TAB_CHAT, TAB_LOGS, TAB_NOTES, TAB_STATS
from tldw_app.Logging_Config import RichLogHandler
from tldw_app.Prompt_Management import Prompts_Interop as prompts_interop
from tldw_app.Utils.Emoji_Handling import get_char, EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN, EMOJI_TITLE_NOTE, \
    FALLBACK_TITLE_NOTE, EMOJI_TITLE_SEARCH, FALLBACK_TITLE_SEARCH, EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE, \
    EMOJI_SEND, FALLBACK_SEND, EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON, supports_emoji
from tldw_app.Utils.Utils import safe_float, safe_int
from .config import (
    CONFIG_TOML_CONTENT,
    DEFAULT_CONFIG_PATH,
    load_settings,
    get_cli_setting,
    get_cli_log_file_path,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS
)
from .Screens.Stats_screen import StatsScreen
from .Event_Handlers import (
    app_lifecycle as app_lifecycle_handlers,
    tab_events as tab_handlers,
    sidebar_events as sidebar_handlers,
    chat_events as chat_handlers,
    conv_char_events as ccp_handlers,
    notes_events as notes_handlers,
    worker_events as worker_handlers, worker_events,
)
from .Notes.Notes_Library import NotesInteropService
from .DB.ChaChaNotes_DB import CharactersRAGDBError, ConflictError
from .Widgets.chat_message import ChatMessage
from .Widgets.settings_sidebar import create_settings_sidebar
from .Widgets.character_sidebar import create_character_sidebar  # Import for character sidebar
from .Widgets.notes_sidebar_left import NotesSidebarLeft
from .Widgets.notes_sidebar_right import NotesSidebarRight
from .Widgets.titlebar import TitleBar
from .LLM_Calls.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google,
)
from .LLM_Calls.LLM_API_Calls_Local import (
    # Add local API functions if they are in the same file
    chat_with_llama, chat_with_kobold, chat_with_oobabooga,
    chat_with_vllm, chat_with_tabbyapi, chat_with_aphrodite,
    chat_with_ollama, chat_with_custom_openai, chat_with_custom_openai_2, chat_with_local_llm
)
API_IMPORTS_SUCCESSFUL = True
#
#######################################################################################################################
#
# Statics

if API_IMPORTS_SUCCESSFUL:
    API_FUNCTION_MAP = {
        "OpenAI": chat_with_openai,
        "Anthropic": chat_with_anthropic,
        "Cohere": chat_with_cohere,
        "HuggingFace": chat_with_huggingface,
        "DeepSeek": chat_with_deepseek,
        "Google": chat_with_google, # Key from config
        "Groq": chat_with_groq,
        "koboldcpp": chat_with_kobold,  # Key from config
        "llama_cpp": chat_with_llama,  # Key from config
        "MistralAI": chat_with_mistral,  # Key from config
        "Oobabooga": chat_with_oobabooga,  # Key from config
        "OpenRouter": chat_with_openrouter,
        "vllm": chat_with_vllm,  # Key from config
        "TabbyAPI": chat_with_tabbyapi,  # Key from config
        "Aphrodite": chat_with_aphrodite,  # Key from config
        "Ollama": chat_with_ollama,  # Key from config
        "Custom": chat_with_custom_openai,  # Key from config
        "Custom_2": chat_with_custom_openai_2,  # Key from config
        "local-llm": chat_with_local_llm
    }
    logging.info(f"API_FUNCTION_MAP populated with {len(API_FUNCTION_MAP)} entries.")
else:
    API_FUNCTION_MAP = {}
    logging.error("API_FUNCTION_MAP is empty due to import failures.")

ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS} # If needed for sidebar defaults
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys()) # If needed
#
#
#####################################################################################################################
#
# Functions:

# --- Global variable for config ---
APP_CONFIG = load_settings()

# Configure root logger based on config BEFORE app starts fully
_initial_log_level_str = APP_CONFIG.get("general", {}).get("log_level", "INFO").upper()
_initial_log_level = getattr(logging, _initial_log_level_str, logging.INFO)
# Define a basic initial format
_initial_log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
# Remove existing handlers before basicConfig to avoid duplicates if script is re-run
logging.basicConfig(level=_initial_log_level, format=_initial_log_format,
                    force=True)  # force=True might help override defaults
logging.info("Initial basic logging configured.")


# --- Main App ---
class TldwCli(App[None]):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""
    #TITLE = "🧠📝🔍  tldw CLI"
    TITLE = f"{get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}{get_char(EMOJI_TITLE_NOTE, FALLBACK_TITLE_NOTE)}{get_char(EMOJI_TITLE_SEARCH, FALLBACK_TITLE_SEARCH)}  tldw CLI"
    # Use forward slashes for paths, works cross-platform
    CSS_PATH = "css/tldw_cli.tcss"
    BINDINGS = [Binding("ctrl+q", "quit", "Quit App", show=True)]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive(TAB_CHAT, layout=True)
    ccp_active_view: reactive[str] = reactive("conversation_details_view", layout=True)

    # Add state to hold the currently streaming AI message widget
    current_ai_message_widget: Optional[ChatMessage] = None

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "Ollama")
    _default_ccp_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic") # Changed from character_defaults

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    # Renamed character_api_provider_value to ccp_api_provider_value for clarity with TAB_CCP
    ccp_api_provider_value: reactive[Optional[str]] = reactive(_default_ccp_provider)

    # Reactives for sidebar
    chat_sidebar_collapsed: reactive[bool] = reactive(False, layout=True)
    character_sidebar_collapsed: reactive[bool] = reactive(False, layout=True)  # For character sidebar
    notes_sidebar_left_collapsed: reactive[bool] = reactive(False, layout=True)
    notes_sidebar_right_collapsed: reactive[bool] = reactive(False, layout=True)
    conv_char_sidebar_left_collapsed: reactive[bool] = reactive(False, layout=True)
    conv_char_sidebar_right_collapsed: reactive[bool] = reactive(False, layout=True)

    # Reactive variables for selected note details
    current_selected_note_id: reactive[Optional[str]] = reactive(None)
    current_selected_note_version: reactive[Optional[int]] = reactive(None)
    current_selected_note_title: reactive[Optional[str]] = reactive(None)
    current_selected_note_content: reactive[Optional[str]] = reactive("")

    # Chats
    current_chat_is_ephemeral: reactive[bool] = reactive(True)  # Start new chats as ephemeral
    # Reactive variable for current chat conversation ID
    current_chat_conversation_id: reactive[Optional[str]] = reactive(None)
    # Reactive variable for current conversation loaded in the Conversations, Characters & Prompts tab
    current_conv_char_tab_conversation_id: reactive[Optional[str]] = reactive(None)

    # Prompts
    current_prompt_id: reactive[Optional[int]] = reactive(None)
    current_prompt_uuid: reactive[Optional[str]] = reactive(None)
    current_prompt_name: reactive[Optional[str]] = reactive(None)
    current_prompt_author: reactive[Optional[str]] = reactive(None)
    current_prompt_details: reactive[Optional[str]] = reactive(None)
    current_prompt_system: reactive[Optional[str]] = reactive(None)
    current_prompt_user: reactive[Optional[str]] = reactive(None)
    current_prompt_keywords_str: reactive[Optional[str]] = reactive("") # Store as comma-sep string for UI
    current_prompt_version: reactive[Optional[int]] = reactive(None) # If DB provides it and you need it
    # is_new_prompt can be inferred from current_prompt_id being None

    _prompt_search_timer: Optional[Timer] = None

    # De-Bouncers
    _conv_char_search_timer: Optional[Timer] = None
    _conversation_search_timer: Optional[Timer] = None
    _notes_search_timer: Optional[Timer] = None

    # Make API_IMPORTS_SUCCESSFUL accessible if needed by old methods or directly
    API_IMPORTS_SUCCESSFUL = API_IMPORTS_SUCCESSFUL

    def __init__(self):
        super().__init__()
        self.app_config = load_settings() # Already loading this
        self.loguru_logger = loguru_logger # Make loguru_logger an instance variable for handlers
        self.prompts_client_id = "tldw_tui_client_v1" # Store client ID for prompts service

        # --- Initialize NotesInteropService ---
        # FIXME - add option to config to set username for app/DBs
        self.notes_user_id = "default_tui_user"  # Or any default user ID string
        notes_db_base_dir = Path.home() / ".config/tldw_cli/user_notes"
        try:
            self.notes_service = NotesInteropService(
                base_db_directory=notes_db_base_dir,
                api_client_id="tldw_tui_client" # Client ID for operations by TUI
            )
            logging.info(f"NotesInteropService initialized for user '{self.notes_user_id}' at {notes_db_base_dir}")
        except CharactersRAGDBError as e:
            logging.error(f"Failed to initialize NotesInteropService: {e}", exc_info=True)
            self.notes_service = None
        except Exception as e_notes_init:
            logging.error(f"Unexpected error during NotesInteropService initialization: {e_notes_init}", exc_info=True)
            self.notes_service = None

        # --- Providers & Models ---
        logging.debug("__INIT__: Attempting to get providers and models...")
        try:
            # Call the function from the config module
            self.providers_models = get_cli_providers_and_models()
            logging.info(
                f"__INIT__: Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e_providers:
            logging.error(f"__INIT__: Failed to get providers and models: {e_providers}", exc_info=True)
            self.providers_models = {}

        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = initial_tab_from_config if initial_tab_from_config in ALL_TABS else TAB_CHAT
        if self._initial_tab_value != initial_tab_from_config: # Log if fallback occurred
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to '{self._initial_tab_value}'.")
        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        # current_tab reactive will be set in on_mount after UI is composed

        self._rich_log_handler: Optional[RichLogHandler] = None # For the RichLog widget in Logs tab

        # --- PromptsInteropService Initialization ---
        self.prompts_service_initialized = False
        try:
            prompts_db_path_str = get_cli_setting("database", "prompts_db_path", str(Path.home() / ".local/share/tldw_cli/tldw_cli_prompts_v2.db"))
            prompts_db_path = Path(prompts_db_path_str).expanduser().resolve()
            prompts_db_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_interop.initialize_interop(db_path=prompts_db_path, client_id=self.prompts_client_id)
            self.prompts_service_initialized = True
            logging.info(f"Prompts Interop Service initialized with DB: {prompts_db_path}")
        except Exception as e_prompts:
            self.prompts_service_initialized = False
            logging.error(f"Failed to initialize Prompts Interop Service: {e_prompts}", exc_info=True)

        self._prompt_search_timer = None  # Initialize here

        # --- Setup Default view for CCP tab ---
        # Initialize self.ccp_active_view based on initial tab or default state if needed
        if self._initial_tab_value == TAB_CCP:
            self.ccp_active_view = "conversation_details_view"  # Default view for CCP tab
        # else: it will default to "conversation_details_view" anyway

    def _setup_logging(self):
        """Sets up all logging handlers, including Loguru integration."""
        # This first logging.info will go to the stderr handler from the initial basicConfig
        logging.info("--- _setup_logging START ---")
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # --- BEGIN LOGURU MANAGEMENT (Your existing code is mostly fine here) ---
        try:
            loguru_logger.remove()  # Good: removes Loguru's default stderr sink
            logging.info("Loguru: All pre-existing sinks removed.")

            def sink_to_standard_logging(message):
                # ... (your existing sink_to_standard_logging function)
                record = message.record
                level_mapping = {
                    "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
                    "SUCCESS": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL,
                }
                std_level = level_mapping.get(record["level"].name, logging.INFO)
                std_logger = logging.getLogger(record["name"])
                if record["exception"]:
                    std_logger.log(std_level, record["message"], exc_info=record["exception"])
                else:
                    std_logger.log(std_level, record["message"])

            loguru_logger.add(
                sink_to_standard_logging,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="TRACE"
            )
            # This log message will also currently go to the initial basicConfig stderr handler
            logging.info("Loguru: Configured to forward its messages to standard Python logging system.")
        except Exception as e:
            # This log message will also currently go to the initial basicConfig stderr handler
            logging.error(f"Loguru: Error during Loguru reconfiguration: {e}", exc_info=True)
        # --- END LOGURU MANAGEMENT ---

        # --- CONFIGURE STANDARD PYTHON LOGGING ROOT LOGGER ---
        root_logger = logging.getLogger()

        # !!! IMPORTANT FIX: Remove all existing handlers from the root logger !!!
        # This will get rid of the StreamHandler (to stderr) added by the initial
        # global logging.basicConfig() call.
        initial_handlers_removed_count = 0
        for handler in root_logger.handlers[:]:  # Iterate over a copy
            root_logger.removeHandler(handler)
            if hasattr(handler, 'close') and callable(handler.close):
                try:
                    handler.close()
                except Exception:
                    pass  # Ignore errors during close of old handlers
            initial_handlers_removed_count += 1

        # Log this removal using Loguru, as standard logging has no handlers yet.
        # This message will go to Loguru's sink (which forwards to std logging,
        # but std logging has no handlers yet, so it might hit Python's "last resort" stderr).
        # Or, better, print to stderr just for this one-off setup message if needed, then rely on proper handlers.
        if initial_handlers_removed_count > 0:
            # Using print here because logging state is actively being changed.
            # This should be one of the last messages to hit raw stderr if setup is correct.
            print(
                f"INFO: _setup_logging: Removed {initial_handlers_removed_count} pre-existing handler(s) from root logger.",
                file=sys.stderr)

        # Now that root_logger is clean, set its overall level.
        # This level acts as a filter before messages reach any of its handlers.
        initial_log_level_str = self.app_config.get("general", {}).get("log_level", "INFO").upper()
        initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
        root_logger.setLevel(initial_log_level)
        # (A temporary print to confirm, as logging to root_logger now might go to "last resort" until a handler is added)
        print(f"INFO: _setup_logging: Root logger level set to {logging.getLevelName(root_logger.level)}",
              file=sys.stderr)

        # --- Add TextualHandler (to standard logging) ---
        # (Your existing TextualHandler setup code is fine)
        # Ensure it's added AFTER clearing old handlers and setting root level.
        # ...
        has_textual_handler = any(isinstance(h, TextualHandler) for h in root_logger.handlers)
        if not has_textual_handler:
            textual_console_handler = TextualHandler()
            textual_console_handler.setLevel(initial_log_level)  # Respects app_config
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            textual_console_handler.setFormatter(console_formatter)
            root_logger.addHandler(textual_console_handler)
            # Now, logging.info should go to Textual's dev console (and other handlers added below)
            logging.info(
                f"Standard Logging: Added TextualHandler (Level: {logging.getLevelName(textual_console_handler.level)}).")
        else:
            logging.info("Standard Logging: TextualHandler already exists.")

        # Test Loguru message again. It should now go to TextualHandler (and others).
        loguru_logger.info(
            "Loguru Test: This message from Loguru should now appear in Textual dev console (and other configured handlers).")

        # --- Setup RichLog Handler (to standard logging) ---
        # (Your existing RichLogHandler setup code is fine, ensure it's added AFTER clearing)
        # ...
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            # Check if it's already added by a previous call (should not happen if _setup_logging is called once)
            if not any(isinstance(h, RichLogHandler) and h.rich_log_widget is log_display_widget for h in
                       root_logger.handlers):
                if not self._rich_log_handler:  # Create if it doesn't exist
                    self._rich_log_handler = RichLogHandler(log_display_widget)
                # Configure and add
                rich_log_handler_level_str = self.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
                rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
                self._rich_log_handler.setLevel(rich_log_handler_level)
                root_logger.addHandler(self._rich_log_handler)
                logging.info(
                    f"Standard Logging: Added RichLogHandler (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            else:
                logging.info("Standard Logging: RichLogHandler already exists and is added.")
        except QueryError:
            logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            self._rich_log_handler = None
        except Exception as e:
            logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
            self._rich_log_handler = None

        # --- Setup File Logging (to standard logging) ---
        # (Your existing FileHandler setup code is fine, ensure it's added AFTER clearing)
        # ... (your existing code to add file_handler to root_logger) ...
        try:
            log_file_path = get_cli_log_file_path()
            log_dir = log_file_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)

            has_file_handler = any(
                isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_file_path) for h in
                root_logger.handlers)

            if not has_file_handler:
                max_bytes_default = 10485760
                backup_count_default = 5
                file_log_level_default_str = "INFO"
                max_bytes = int(get_cli_setting("logging", "log_max_bytes", max_bytes_default))
                backup_count = int(get_cli_setting("logging", "log_backup_count", backup_count_default))
                file_log_level_str = get_cli_setting("logging", "file_log_level", file_log_level_default_str).upper()
                file_log_level = getattr(logging, file_log_level_str, logging.INFO)

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setLevel(file_log_level)
                file_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                logging.info(
                    f"Standard Logging: Added RotatingFileHandler (File: '{log_file_path}', Level: {logging.getLevelName(file_log_level)}).")
            else:
                logging.info("Standard Logging: RotatingFileHandler already exists for this file path.")
        except Exception as e:
            logging.warning(f"!!! ERROR setting up file logging: {e}", exc_info=True)

        # Re-evaluate lowest level for standard logging root logger
        # (Your existing logic for this is fine)
        all_std_handlers = root_logger.handlers
        if all_std_handlers:
            handler_levels = [h.level for h in all_std_handlers if h.level > 0]
            if handler_levels:
                lowest_effective_level = min(handler_levels)
                current_root_level = root_logger.level
                # Only adjust root logger level if it's currently *less* verbose (higher numeric value)
                # than the most verbose handler.
                if current_root_level > lowest_effective_level:
                    logging.info(
                        f"Standard Logging: Adjusting root logger level from {logging.getLevelName(current_root_level)} to {logging.getLevelName(lowest_effective_level)} to match most verbose handler.")
                    root_logger.setLevel(lowest_effective_level)
            logging.info(f"Standard Logging: Final Root logger level is: {logging.getLevelName(root_logger.level)}")
        else:
            logging.warning("Standard Logging: No handlers found on root logger after setup!")

        logging.info("Logging setup complete.")
        logging.info("--- _setup_logging END ---")

    def compose(self) -> ComposeResult:
        logging.debug("App composing UI...")
        yield Header()
        # Set up the main title bar with a static title
        yield TitleBar()
        yield from self.compose_tabs()
        yield from self.compose_content_area()
        yield Footer()
        logging.debug("App compose finished.")

    def compose_tabs(self) -> ComposeResult:
        with Horizontal(id="tabs"):
            for tab_id_loop in ALL_TABS:
                # Use "CCP" as label for TAB_CCP for brevity
                label_text = "CCP" if tab_id_loop == TAB_CCP else tab_id_loop.replace('_', ' ').capitalize()
                yield Button(
                    label_text,
                    id=f"tab-{tab_id_loop}",
                    classes="-active" if tab_id_loop == self._initial_tab_value else ""
                )

    ############################################################
    #
    # Code that builds the content area of the app aka the main UI.
    #
    ###########################################################
    def compose_content_area(self) -> ComposeResult:
        logging.debug(f"Compose: Composing content area...")

        with Container(id="content"):
            # --- Chat Window ---
            # Assign specific reactive variables to the Select widgets
            chat_window = Container(id=f"{TAB_CHAT}-window", classes="window")
            if self._initial_tab_value != TAB_CHAT:
                chat_window.styles.display = False  # Hide if not the initial tab
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
                # Pass self.current_chat_is_ephemeral to create_character_sidebar if it needs to adjust UI
                yield from create_settings_sidebar(TAB_CHAT, self.app_config) # This is fine

                with Container(id="chat-main-content"):
                    yield VerticalScroll(id="chat-log")
                    with Horizontal(id="chat-input-area"):
                        #yield Button("☰", id="toggle-chat-sidebar", classes="sidebar-toggle")
                        yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE), id="toggle-chat-sidebar",
                                     classes="sidebar-toggle")
                        yield TextArea(id="chat-input", classes="chat-input") # Ensure prompt is used if needed
                        #yield Button("Send ▶", id="send-chat", classes="send-button")
                        yield Button(get_char(EMOJI_SEND, FALLBACK_SEND), id="send-chat", classes="send-button")
                        #yield Button("👤", id="toggle-character-sidebar", classes="sidebar-toggle")
                        yield Button(get_char(EMOJI_CHARACTER_ICON, FALLBACK_CHARACTER_ICON), id="toggle-character-sidebar",
                                     classes="sidebar-toggle")

                # Right sidebar (new character specific settings) for chat window
                # The create_character_sidebar function will define a widget with id="character-sidebar"
                # Pass a string prefix, for example, "chat" or "character_chat"
                yield from create_character_sidebar("chat", initial_ephemeral_state=self.current_chat_is_ephemeral)

            # --- Conversations, Characters & Prompts Window ---
            with Container(id=f"{TAB_CCP}-window", classes="window"):
                # Left Pane (remains the same)
                with VerticalScroll(id="conv-char-left-pane", classes="cc-left-pane"):
                    yield Static("CCP Menu", classes="sidebar-title cc-section-title-text")
                    with Collapsible(title="Characters", id="conv-char-characters-collapsible"):
                        yield Select([], prompt="Select Character...", allow_blank=True,id="conv-char-character-select")
                    with Collapsible(title="Conversations", id="conv-char-conversations-collapsible"):
                        yield Input(id="conv-char-search-input", placeholder="Search conversations...", classes="sidebar-input")
                        yield Button("Search", id="conv-char-conversation-search-button", classes="sidebar-button")
                        yield ListView(id="conv-char-search-results-list")
                        yield Button("Load Selected", id="conv-char-load-button", classes="sidebar-button")
                    with Collapsible(title="Prompts", id="ccp-prompts-collapsible"):
                        yield Button("Create New Prompt", id="ccp-prompt-create-new-button", classes="sidebar-button")
                        yield Input(id="ccp-prompt-search-input", placeholder="Search prompts...", classes="sidebar-input")
                        yield ListView(id="ccp-prompts-listview", classes="sidebar-listview")
                        yield Button("Load Selected Prompt", id="ccp-prompt-load-selected-button", classes="sidebar-button")
                yield Button("☰", id="toggle-conv-char-left-sidebar", classes="cc-sidebar-toggle-button")

                # Center Pane
                with VerticalScroll(id="conv-char-center-pane", classes="cc-center-pane"):
                    # Container for conversation messages - visible by default CSS
                    with Container(id="ccp-conversation-messages-view", classes="ccp-view-area"):  # <<< ADD THIS BACK
                        yield Static("Conversation History", classes="pane-title",
                                     id="ccp-center-pane-title-conv")  # Keep your Static or original
                        # Messages will be mounted dynamically here

                    # Container for prompt editing UI (initially hidden)
                    prompt_editor_container = Container(id="ccp-prompt-editor-view",
                                                        classes="ccp-view-area")  # <<< ADD THIS BACK
                    prompt_editor_container.styles.display = "none"  # Hide it initially

                    with prompt_editor_container:  # Now compose its children (using original IDs)
                        yield Static("Prompt Editor", classes="pane-title",
                                     id="ccp-center-pane-title-prompt")  # Or your preferred title
                        yield Label("Prompt Name:", classes="sidebar-label")
                        yield Input(id="ccp-editor-prompt-name-input", placeholder="Unique prompt name...",
                                    classes="sidebar-input")  # <<< USE "ccp-editor-" PREFIX
                        yield Label("Author:", classes="sidebar-label")
                        yield Input(id="ccp-editor-prompt-author-input", placeholder="Author name...",
                                    classes="sidebar-input")  # <<< USE "ccp-editor-" PREFIX
                        yield Label("Details/Description:", classes="sidebar-label")
                        yield TextArea("", id="ccp-editor-prompt-description-textarea",
                                       classes="sidebar-textarea ccp-prompt-textarea")  # <<< USE "ccp-editor-" PREFIX
                        yield Label("System Prompt:", classes="sidebar-label")
                        yield TextArea("", id="ccp-editor-prompt-system-textarea",
                                       classes="sidebar-textarea ccp-prompt-textarea")  # <<< USE "ccp-editor-" PREFIX
                        yield Label("User Prompt (Template):", classes="sidebar-label")
                        yield TextArea("", id="ccp-editor-prompt-user-textarea",
                                       classes="sidebar-textarea ccp-prompt-textarea")  # <<< USE "ccp-editor-" PREFIX
                        yield Label("Keywords (comma-separated):", classes="sidebar-label")
                        yield TextArea("", id="ccp-editor-prompt-keywords-textarea",
                                       classes="sidebar-textarea ccp-prompt-textarea")  # <<< USE "ccp-editor-" PREFIX
                        with Horizontal(classes="ccp-prompt-action-buttons"):
                            yield Button("Save Prompt", id="ccp-editor-prompt-save-button", variant="success",
                                         classes="sidebar-button")  # <<< USE "ccp-editor-" PREFIX
                            yield Button("Clone Prompt", id="ccp-editor-prompt-clone-button",
                                         classes="sidebar-button")  # <<< USE "ccp-editor-" PREFIX
                            yield Button("Delete Prompt", id="ccp-editor-prompt-delete-button", variant="error",
                                         classes="sidebar-button")  # <<< USE "ccp-editor-" PREFIX

                # Button to toggle the right sidebar for CCP tab
                yield Button(get_char(EMOJI_SIDEBAR_TOGGLE, FALLBACK_SIDEBAR_TOGGLE),
                             id="toggle-conv-char-right-sidebar", classes="cc-sidebar-toggle-button")
                # --- Right Pane (Details & Settings) ---
                with VerticalScroll(id="conv-char-right-pane", classes="cc-right-pane"): # Corrected: Only one
                    #yield from create_settings_sidebar(TAB_CCP, self.app_config) # CCP settings sidebar
                    with Collapsible(title="Conversation Details", id="ccp-conversation-details-collapsible", collapsed=True):
                        yield Static("Title:", classes="sidebar-label")
                        yield Input(id="conv-char-title-input", placeholder="Conversation title...", classes="sidebar-input")
                        yield Static("Keywords:", classes="sidebar-label")
                        yield TextArea("", id="conv-char-keywords-input", classes="conv-char-keywords-textarea")
                        yield Button("Save Conversation Details", id="conv-char-save-details-button", classes="sidebar-button")
                        yield Static("Export Options", classes="sidebar-label export-label")
                        yield Button("Export as Text", id="conv-char-export-text-button", classes="sidebar-button")
                        yield Button("Export as JSON", id="conv-char-export-json-button", classes="sidebar-button")
                    with Collapsible(title="Prompt Details", id="ccp-prompt-details-collapsible", collapsed=True):
                        yield Label("Prompt Name:", classes="sidebar-label")
                        yield Input(id="ccp-prompt-name-input", placeholder="Unique prompt name...", classes="sidebar-input")
                        yield Label("Author:", classes="sidebar-label")
                        yield Input(id="ccp-prompt-author-input", placeholder="Author name...", classes="sidebar-input")
                        yield Label("Details/Description:", classes="sidebar-label")
                        yield TextArea("", id="ccp-prompt-description-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                        yield Label("System Prompt:", classes="sidebar-label")
                        yield TextArea("", id="ccp-prompt-system-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                        yield Label("User Prompt (Template):", classes="sidebar-label")
                        yield TextArea("", id="ccp-prompt-user-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                        yield Label("Keywords (comma-separated):", classes="sidebar-label")
                        yield TextArea("", id="ccp-prompt-keywords-textarea", classes="sidebar-textarea ccp-prompt-textarea")
                        with Horizontal(classes="ccp-prompt-action-buttons"):
                            yield Button("Save Prompt", id="ccp-prompt-save-button", variant="success", classes="sidebar-button")
                            yield Button("Clone Prompt", id="ccp-prompt-clone-button", classes="sidebar-button")
                            yield Button("Delete Prompt", id="ccp-prompt-delete-button", variant="error", classes="sidebar-button")

                # FIXME
                # # Right Pane (dynamic content visibility)
                # with VerticalScroll(id="conv-char-right-pane",
                #                     classes="cc-right-pane"):  # Ensure this ID is unique if used elsewhere for styling
                #     # Container for LLM settings (visibility toggled)
                #     with Container(id="ccp-right-pane-llm-settings-container"):  # Initially visible by default CSS
                #         yield from create_settings_sidebar(TAB_CCP, self.app_config)
                #
                #     # Conversation Details (always present, expanded/collapsed state might change by watcher)
                #     with Collapsible(title="Conversation Details", id="ccp-conversation-details-collapsible",
                #                      collapsed=False):  # Start expanded
                #         yield Static("Title:", classes="sidebar-label")
                #         yield Input(id="conv-char-title-input", placeholder="Conversation title...",
                #                     classes="sidebar-input")
                #         yield Static("Keywords:", classes="sidebar-label")
                #         yield TextArea("", id="conv-char-keywords-input", classes="conv-char-keywords-textarea")
                #         yield Button("Save Conversation Details", id="conv-char-save-details-button",
                #                      classes="sidebar-button")
                #         yield Static("Export Options", classes="sidebar-label export-label")
                #         yield Button("Export as Text", id="conv-char-export-text-button", classes="sidebar-button")
                #         yield Button("Export as JSON", id="conv-char-export-json-button", classes="sidebar-button")

            # --- Notes Tab Window ---
            chat_window = Container(id=f"{TAB_CHAT}-window", classes="window")
            if self._initial_tab_value != TAB_CHAT:
                chat_window.styles.display = False  # Hide if not the initial tab
            with Container(id=f"{TAB_NOTES}-window", classes="window"):
                # Instantiate the left sidebar (ensure it has a unique ID for the watcher)
                yield NotesSidebarLeft(id="notes-sidebar-left", classes="sidebar")

                # Main content area for notes (editor and toggles)
                with Container(id="notes-main-content"):  # Similar to chat-main-content
                    yield TextArea(id="notes-editor-area", classes="notes-editor")  # Make it take up 1fr height
                    # Container for toggle buttons, similar to chat-input-area
                    with Horizontal(id="notes-controls-area"):
                        yield Button("☰ L", id="toggle-notes-sidebar-left", classes="sidebar-toggle")
                        yield Static()  # Spacer
                        yield Button("Save Note", id="notes-save-button", variant="primary")
                        yield Static()  # Spacer
                        yield Button("R ☰", id="toggle-notes-sidebar-right", classes="sidebar-toggle")

                # Instantiate the right sidebar (ensure it has a unique ID for the watcher)
                yield NotesSidebarRight(id="notes-sidebar-right", classes="sidebar")

            # --- Logs Tab ---

            # --- Media Tab Goes Here

            # ---- Search Tab ---

            # ---- Ingest Tab ---

            # --- Logs Window ---
            with Container(id=f"{TAB_LOGS}-window", classes="window"):
                yield RichLog(id="app-log-display", wrap=True, highlight=True, markup=True, auto_scroll=True)
                yield Button("Copy All Logs to Clipboard", id="copy-logs-button", classes="logs-action-button")

            # --- Stats Window (Placeholder) ---
            with Container(id=f"{TAB_STATS}-window", classes="window"):
                yield StatsScreen(
                    id="stats_screen_content")  # You can give the StatsScreen instance an ID if needed, or omit it

            # --- Other Placeholder Windows ---
            for tab_id_placeholder in ALL_TABS:
                if tab_id_placeholder not in [TAB_CHAT, TAB_CCP, TAB_NOTES, TAB_LOGS, TAB_STATS]:  # Updated to TAB_CCP
                    with Container(id=f"{tab_id_placeholder}-window", classes="window placeholder-window"):
                        yield Static(f"{tab_id_placeholder.replace('_', ' ').capitalize()} Window Placeholder")
                        yield Button("Coming Soon", id=f"{tab_id_placeholder}-placeholder-button", disabled=True)

    # --- Watcher for CCP Active View ---
    def watch_ccp_active_view(self, old_view: Optional[str], new_view: str) -> None:
        loguru_logger.debug(f"CCP active view changing from '{old_view}' to: '{new_view}'")
        try:
            conversation_messages_view = self.query_one("#ccp-conversation-messages-view")
            prompt_editor_view = self.query_one("#ccp-prompt-editor-view")

            # Right pane elements
            llm_settings_container_right = self.query_one("#ccp-right-pane-llm-settings-container")
            conv_details_collapsible_right = self.query_one("#ccp-conversation-details-collapsible", Collapsible)

            if new_view == "prompt_editor_view":
                # Center Pane: Show Prompt Editor, Hide Conversation Messages
                conversation_messages_view.display = False
                prompt_editor_view.display = True

                # Right Pane: Hide LLM Settings, Keep Conversation Details (can be collapsed)
                llm_settings_container_right.display = False
                conv_details_collapsible_right.display = True  # Ensure it's displayed
                # conv_details_collapsible_right.collapsed = True # Optionally collapse it

                # Focus an element in prompt editor
                try:
                    self.query_one("#ccp-editor-prompt-name-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus prompt name input in editor view.")

            elif new_view == "conversation_details_view":
                # Center Pane: Show Conversation Messages, Hide Prompt Editor
                conversation_messages_view.display = True
                prompt_editor_view.display = False

                # Right Pane: Show LLM Settings, Show and Expand Conversation Details
                llm_settings_container_right.display = True
                conv_details_collapsible_right.display = True
                conv_details_collapsible_right.collapsed = False  # Expand when viewing conversation

                # Potentially focus conversation search or title in the left/right pane
                try:
                    # If a conversation is loaded, maybe focus its title in right pane
                    if self.current_conv_char_tab_conversation_id:
                        self.query_one("#conv-char-title-input", Input).focus()
                    else:  # Otherwise, maybe focus the search in left pane
                        self.query_one("#conv-char-search-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus default element in conversation details view.")
            else:  # Default or unknown view (treat as conversation_details_view)
                conversation_messages_view.display = True
                prompt_editor_view.display = False
                llm_settings_container_right.display = True
                conv_details_collapsible_right.display = True
                conv_details_collapsible_right.collapsed = False
                loguru_logger.warning(
                    f"Unknown ccp_active_view: {new_view}, defaulting to conversation_details_view.")

        except QueryError as e:
            loguru_logger.exception(f"UI component not found during CCP view switch: {e}")
        except Exception as e_watch:
            loguru_logger.exception(f"Unexpected error in watch_ccp_active_view: {e_watch}")

    # --- Watcher for Right Sidebar in CCP Tab ---
    # The existing `watch_conv_char_sidebar_right_collapsed` should work if you target
    # `#conv-char-right-pane` and use `add_class("collapsed")` or `remove_class("collapsed")`
    # in conjunction with the CSS you'll add. Or, stick to `display = not collapsed`.
    # Let's use the class-based approach for consistency with your provided CSS for other sidebars.
    def watch_conv_char_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts right sidebar pane."""
        try:
            sidebar_pane = self.query_one("#conv-char-right-pane")  # Target the VerticalScroll itself
            if collapsed:
                sidebar_pane.add_class("collapsed")
            else:
                sidebar_pane.remove_class("collapsed")
            # The .cc-right-pane.collapsed CSS rule will handle display:none and width:0
            loguru_logger.debug(f"CCP right pane collapsed state: {collapsed}")
        except QueryError:
            loguru_logger.error("CCP right pane (#conv-char-right-pane) not found for collapse toggle.")
        except Exception as e:
            loguru_logger.error(f"Error toggling CCP right pane: {e}", exc_info=True)

    # --- Modify _clear_prompt_fields and _load_prompt_for_editing ---
    def _clear_prompt_fields(self) -> None:
        """Clears prompt input fields in the CENTER PANE editor."""
        try:
            self.query_one("#ccp-editor-prompt-name-input", Input).value = ""
            self.query_one("#ccp-editor-prompt-author-input", Input).value = ""
            self.query_one("#ccp-editor-prompt-description-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-system-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-user-textarea", TextArea).text = ""
            self.query_one("#ccp-editor-prompt-keywords-textarea", TextArea).text = ""
            loguru_logger.debug("Cleared prompt editor fields in center pane.")
        except QueryError as e:
            loguru_logger.error(f"Error clearing prompt editor fields in center pane: {e}")

    async def _load_prompt_for_editing(self, prompt_id: Optional[int], prompt_uuid: Optional[str] = None) -> None:
        if not self.prompts_service_initialized:
            self.notify("Prompts service not available.", severity="error")
            return

        # Switch to prompt editor view
        self.ccp_active_view = "prompt_editor_view"  # This will trigger the watcher

        identifier_to_fetch = prompt_id if prompt_id is not None else prompt_uuid
        if identifier_to_fetch is None:
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset all reactive prompt states
            self.current_prompt_uuid = None
            self.current_prompt_name = None
            # ... etc. for other prompt reactives
            loguru_logger.warning("_load_prompt_for_editing called with no ID/UUID after view switch.")
            return

        try:
            prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)

            if prompt_details:
                self.current_prompt_id = prompt_details.get('id')
                self.current_prompt_uuid = prompt_details.get('uuid')
                self.current_prompt_name = prompt_details.get('name', '')
                self.current_prompt_author = prompt_details.get('author', '')
                self.current_prompt_details = prompt_details.get('details', '')
                self.current_prompt_system = prompt_details.get('system_prompt', '')
                self.current_prompt_user = prompt_details.get('user_prompt', '')
                self.current_prompt_keywords_str = ", ".join(prompt_details.get('keywords', []))
                self.current_prompt_version = prompt_details.get('version')

                # Populate UI in the CENTER PANE editor
                self.query_one("#ccp-editor-prompt-name-input", Input).value = self.current_prompt_name
                self.query_one("#ccp-editor-prompt-author-input", Input).value = self.current_prompt_author
                self.query_one("#ccp-editor-prompt-description-textarea",
                               TextArea).text = self.current_prompt_details
                self.query_one("#ccp-editor-prompt-system-textarea", TextArea).text = self.current_prompt_system
                self.query_one("#ccp-editor-prompt-user-textarea", TextArea).text = self.current_prompt_user
                self.query_one("#ccp-editor-prompt-keywords-textarea",
                               TextArea).text = self.current_prompt_keywords_str

                self.query_one("#ccp-editor-prompt-name-input", Input).focus()  # Focus after loading
                self.notify(f"Prompt '{self.current_prompt_name}' loaded for editing.", severity="info")
            else:
                self.notify(f"Failed to load prompt (ID/UUID: {identifier_to_fetch}).", severity="error")
                self._clear_prompt_fields()  # Clear editor if load fails
                self.current_prompt_id = None  # Reset reactives
        except Exception as e:
            loguru_logger.error(f"Error loading prompt for editing: {e}", exc_info=True)
            self.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset reactives

    def watch_current_chat_is_ephemeral(self, is_ephemeral: bool) -> None:
        self.loguru_logger.debug(f"Chat ephemeral state changed to: {is_ephemeral}")
        try:
            # --- Controls for EPHEMERAL chat actions ---
            save_current_chat_button = self.query_one("#chat-save-current-chat-button", Button)
            save_current_chat_button.disabled = not is_ephemeral  # Enable if ephemeral

            # --- Controls for PERSISTENT chat metadata ---
            title_input = self.query_one("#chat-conversation-title-input", Input)
            keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
            save_details_button = self.query_one("#chat-save-conversation-details-button", Button)
            uuid_display = self.query_one("#chat-conversation-uuid-display", Input)

            title_input.disabled = is_ephemeral  # Disable if ephemeral
            keywords_input.disabled = is_ephemeral  # Disable if ephemeral
            save_details_button.disabled = is_ephemeral  # Disable if ephemeral (cannot save details for non-existent chat)

            if is_ephemeral:
                # Clear details and set UUID display when switching TO ephemeral
                title_input.value = ""
                keywords_input.text = ""
                # Ensure UUID display is also handled
                try:
                    uuid_display = self.query_one("#chat-conversation-uuid-display", Input)
                    uuid_display.value = "Ephemeral Chat"
                except QueryError:
                    loguru_logger.warning(
                        "Could not find #chat-conversation-uuid-display to update for ephemeral state.")
            # ELSE: If switching TO persistent (is_ephemeral is False),
            # the calling function (e.g., load chat, save ephemeral chat button handler)
            # is responsible for POPULATING the title/keywords fields.
            # This watcher correctly enables them here.

        except QueryError as e:
            self.loguru_logger.warning(f"UI component not found while watching ephemeral state: {e}. Tab might not be fully composed or active.")
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_current_chat_is_ephemeral: {e_watch}", exc_info=True)

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        self.chat_api_provider_value = new_value # Watcher will call _update_model_select

    def update_ccp_provider_reactive(self, new_value: Optional[str]) -> None: # Renamed
        self.ccp_api_provider_value = new_value # Watcher will call _update_model_select

    def on_mount(self) -> None:
        """Configure logging, set initial tab, bind selects, and start processors."""
        self._setup_logging()
        if self._rich_log_handler:
            loguru_logger.debug("Starting RichLogHandler processor task...")
            self._rich_log_handler.start_processor(self)

        # --- Bind Select Widgets ---
        logging.info("App on_mount: Binding Select widgets to reactive updaters...")
        # (Keep your existing binding logic with self.watch here)
        try:
            chat_select = self.query_one(f"#{TAB_CHAT}-api-provider", Select)
            # Watch the Select's internal value, and when it changes, update our app-level reactive var
            self.watch(chat_select, "value", self.update_chat_provider_reactive, init=False)
            logging.debug(f"Bound chat provider Select ({chat_select.id}) value to update_chat_provider_reactive")
            loguru_logger.warning(f">>> DEBUG: Bound chat provider Select to reactive update method.")
        except QueryError:
            logging.error(f"on_mount: Failed to find chat provider select: #{TAB_CHAT}-api-provider")
            loguru_logger.debug(f">>> DEBUG: ERROR - Failed to bind chat provider select.")
        except Exception as e:
            logging.error(f"on_mount: Error binding chat provider select: {e}", exc_info=True)
            print(f">>> DEBUG: ERROR - Exception during chat provider select binding: {e}")

        try:
            ccp_select = self.query_one(f"#{TAB_CCP}-api-provider", Select)
            self.watch(ccp_select, "value", self.update_ccp_provider_reactive, init=False) # Use renamed reactive updater
        except QueryError:
            logging.error(f"on_mount: Failed to find character provider select: #{TAB_CCP}-api-provider")
            loguru_logger.debug(f">>> DEBUG: ERROR - Failed to bind character provider select.")
        except Exception as e:
            logging.error(f"on_mount: Error binding character provider select: {e}", exc_info=True)
            loguru_logger.debug(f">>> DEBUG: ERROR - Exception during character provider select binding: {e}")
        # --- END BINDING LOGIC ---

        # Set initial tab *after* UI is composed and bindings potentially set up
        self.current_tab = self._initial_tab_value # This will trigger watch_current_tab

        logging.info("App mount process completed.")

        # Populate dynamic selects that depend on DB/config after mount
        self.call_later(self._populate_chat_conversation_character_filter_select) # For Chat tab
        # Delegate CCP tab population to its handler module
        self.call_later(ccp_handlers.populate_ccp_character_select, self)
        self.call_later(ccp_handlers.populate_ccp_prompts_list_view, self)
        # Initial search/list for CCP might also be triggered if it's the default tab
        if self.current_tab == TAB_CCP:
            self.call_later(ccp_handlers.perform_ccp_conversation_search, self)

    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        if self._rich_log_handler: # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")
        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()
                    root_logger.removeHandler(handler)
                    logging.info("RotatingFileHandler removed and closed.")
                except Exception as e_fh_close:
                    logging.error(f"Error removing/closing file handler: {e_fh_close}")
        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    # WATCHER - Handles UI changes when current_tab's VALUE changes
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        # (Your existing watcher code is likely fine, just ensure the QueryErrors aren't hiding a problem)
        loguru_logger.debug(f"\n>>> DEBUG: watch_current_tab triggered! Old: '{old_tab}', New: '{new_tab}'")
        if not isinstance(new_tab, str) or not new_tab:
            print(f">>> DEBUG: watch_current_tab: Invalid new_tab '{new_tab!r}', aborting.")
            logging.error(f"Watcher received invalid new_tab value: {new_tab!r}. Aborting tab switch.")
            return
        if old_tab and not isinstance(old_tab, str):
            print(f">>> DEBUG: watch_current_tab: Invalid old_tab '{old_tab!r}', setting to None.")
            logging.warning(f"Watcher received invalid old_tab value: {old_tab!r}.")
            old_tab = None

        logging.debug(f"Watcher: Switching tab from '{old_tab}' to '{new_tab}'")

        # --- Hide Old Tab ---
        if old_tab and old_tab != new_tab:
            try: self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
            except QueryError: logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            try: self.query_one(f"#{old_tab}-window").display = False
            except QueryError: logging.warning(f"Watcher: Could not find old window #{old_tab}-window")

        # Show New Tab UI
        try:
            self.query_one(f"#tab-{new_tab}", Button).add_class("-active")
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True

            # Focus input logic (as in original, adjust if needed)
            if new_tab not in [TAB_LOGS, TAB_STATS]: # Don't focus input on these tabs
                input_to_focus: Optional[Union[TextArea, Input]] = None
                try: input_to_focus = new_window.query_one(TextArea)
                except QueryError:
                    try: input_to_focus = new_window.query_one(Input) # Check for Input if TextArea not found
                    except QueryError: pass # No primary input found

                if input_to_focus:
                    self.set_timer(0.1, input_to_focus.focus) # Slight delay for focus
                    logging.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    logging.debug(f"Watcher: No primary input (TextArea or Input) found to focus in '{new_tab}'")
        except QueryError:
            logging.error(f"Watcher: Could not find new button or window for #tab-{new_tab} / #{new_tab}-window")
        except Exception as e_show_new:
            logging.error(f"Watcher: Error showing new tab '{new_tab}': {e_show_new}", exc_info=True)

        loguru_logger.debug(">>> DEBUG: watch_current_tab finished.")

        # Tab-specific actions on switch
        if new_tab == TAB_NOTES:
            self.call_later(notes_handlers.load_and_display_notes_handler, self)
        elif new_tab == TAB_CCP:
            # Initial population for CCP tab when switched to
            self.call_later(ccp_handlers.populate_ccp_character_select, self)
            self.call_later(ccp_handlers.populate_ccp_prompts_list_view, self)
            self.call_later(ccp_handlers.perform_ccp_conversation_search, self) # Initial search/list for conversations
        elif new_tab == TAB_CHAT:
            # If chat tab becomes active, maybe re-focus chat input
            try: self.query_one("#chat-input", TextArea).focus()
            except QueryError: pass


    # Watchers for sidebar collapsed states (keep as is)
    def watch_chat_sidebar_collapsed(self, collapsed: bool) -> None:
        try:
            self.query_one("#chat-sidebar").display = not collapsed
        except QueryError:
            logging.error("Chat sidebar (#chat-sidebar) not found by watcher.")

    def watch_character_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the character settings sidebar."""
        try:
            sidebar = self.query_one("#character-sidebar")  # ID from create_character_sidebar
            sidebar.display = not collapsed
        except QueryError:
            logging.error("Character sidebar widget (#character-sidebar) not found.")

    def watch_notes_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes left sidebar."""
        try:
            sidebar = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes left sidebar widget (#notes-sidebar-left) not found.")

    def watch_notes_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes right sidebar."""
        try:
            sidebar = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes right sidebar widget (#notes-sidebar-right) not found.")

    def watch_conv_char_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts left sidebar pane."""
        try:
            sidebar_pane = self.query_one("#conv-char-left-pane") # The ID of the VerticalScroll
            sidebar_pane.display = not collapsed # True means visible, False means hidden
            logging.debug(f"Conversations, Characters & Prompts left pane display set to {not collapsed}")
        except QueryError:
            logging.error("Conversations, Characters & Prompts left sidebar pane (#conv-char-left-pane) not found.")
        except Exception as e:
            logging.error(f"Error toggling Conversations, Characters & Prompts left sidebar pane: {e}", exc_info=True)

    async def save_current_note(self) -> bool:
        """Saves the currently selected note's title and content to the database."""
        if not self.notes_service or not self.current_selected_note_id or self.current_selected_note_version is None:
            logging.warning("No note selected or service unavailable. Cannot save.")
            # Optionally: self.notify("No note selected to save.", severity="warning")
            return False

        try:
            editor = self.query_one("#notes-editor-area", TextArea)
            title_input = self.query_one("#notes-title-input", Input)
            current_content = editor.text
            current_title = title_input.value

            # Check if title or content actually changed to avoid unnecessary saves.
            # This requires storing the original loaded title/content if not already done by reactives.
            # For now, we save unconditionally if a note is selected.
            # A more advanced check could compare with self.current_selected_note_title and self.current_selected_note_content

            logging.info(
                f"Attempting to save note ID: {self.current_selected_note_id}, Version: {self.current_selected_note_version}")
            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.current_selected_note_id,
                update_data={'title': current_title, 'content': current_content},
                expected_version=self.current_selected_note_version
            )
            if success:
                logging.info(f"Note {self.current_selected_note_id} saved successfully.")
                # Update version and potentially title/content reactive vars if update_note returns new state
                # For now, we re-fetch to get the new version.
                updated_note_details = self.notes_service.get_note_by_id(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id
                )
                if updated_note_details:
                    self.current_selected_note_version = updated_note_details.get('version')
                    self.current_selected_note_title = updated_note_details.get('title')  # Update reactive
                    # self.current_selected_note_content = updated_note_details.get('content') # Update reactive

                    # Refresh the list in the left sidebar to reflect title changes and update item version
                    await notes_handlers.load_and_display_notes_handler(self)
                    # self.notify("Note saved!", severity="information") # If notifications are setup
                else:
                    # Note might have been deleted by another client after our successful update, though unlikely.
                    logging.warning(f"Note {self.current_selected_note_id} not found after presumably successful save.")
                    # self.notify("Note saved, but failed to refresh details.", severity="warning")

                return True
            else:
                # This path should ideally not be reached if update_note raises exceptions on failure.
                logging.warning(
                    f"notes_service.update_note for {self.current_selected_note_id} returned False without raising error.")
                # self.notify("Failed to save note (unknown reason).", severity="error")
                return False

        except ConflictError as e:
            logging.error(f"Conflict saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify(f"Save conflict: {e}. Please reload the note.", severity="error")
            # Optionally, offer to reload the note or overwrite. For now, just log.
            # await self.handle_save_conflict() # A new method to manage this
            return False
        except CharactersRAGDBError as e:
            logging.error(f"Database error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Error saving note to database.", severity="error")
            return False
        except QueryError as e:
            logging.error(f"UI component not found while saving note: {e}", exc_info=True)
            # self.notify("UI error while saving note.", severity="error")
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Unexpected error saving note.", severity="error")
            return False

    # --- MODIFIED EVENT DISPATCHERS ---
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs, sending messages, and message actions."""
        button = event.button
        button_id = button.id
        self.loguru_logger.debug(f"Button pressed: {button_id} on tab {self.current_tab}")

        # Tab Switching
        if button_id and button_id.startswith("tab-"):
            await tab_handlers.handle_tab_button_pressed(self, button_id)
            return # Handled

        # Sidebar Toggles
        if button_id and button_id.startswith("toggle-"):
            await sidebar_handlers.handle_sidebar_toggle_button_pressed(self, button_id)
            return # Handled

        # --- Tab-Specific Button Actions ---
        current_active_tab = self.current_tab # Use the reactive value

        if current_active_tab == TAB_CHAT:
            if button_id == "send-chat": await chat_handlers.handle_chat_send_button_pressed(self, TAB_CHAT)
            elif button_id == "chat-new-conversation-button": await chat_handlers.handle_chat_new_conversation_button_pressed(self)
            elif button_id == "chat-save-current-chat-button": await chat_handlers.handle_chat_save_current_chat_button_pressed(self)
            elif button_id == "chat-save-conversation-details-button": await chat_handlers.handle_chat_save_details_button_pressed(self)
            elif button_id == "chat-conversation-load-selected-button": await chat_handlers.handle_chat_load_selected_button_pressed(self)
            elif button.parent and isinstance(button.parent.parent, ChatMessage):
                action_widget = self._get_chat_message_widget_from_button(button)
                if action_widget: await chat_handlers.handle_chat_action_button_pressed(self, button, action_widget)
            else: logging.warning(f"Unhandled button on CHAT tab: {button_id}")

        elif current_active_tab == TAB_CCP:
            if button_id == "conv-char-conversation-search-button": await ccp_handlers.handle_ccp_conversation_search_button_pressed(self)
            elif button_id == "conv-char-load-button": await ccp_handlers.handle_ccp_load_conversation_button_pressed(self)
            elif button_id == "conv-char-save-details-button": await ccp_handlers.handle_ccp_save_conversation_details_button_pressed(self)
            elif button_id == "ccp-prompt-create-new-button": await ccp_handlers.handle_ccp_prompt_create_new_button_pressed(self)
            elif button_id == "ccp-prompt-load-selected-button": await ccp_handlers.handle_ccp_prompt_load_selected_button_pressed(self)
            elif button_id == "ccp-prompt-save-button": await ccp_handlers.handle_ccp_prompt_save_button_pressed(self)
            elif button_id == "ccp-prompt-clone-button": await ccp_handlers.handle_ccp_prompt_clone_button_pressed(self)
            elif button_id == "ccp-prompt-delete-button": await ccp_handlers.handle_ccp_prompt_delete_button_pressed(self)
            # Note: CCP tab might also have a chat interface. If its send button is different, handle it here.
            # Example: if button_id == "send-ccp-chat": await ccp_handlers.handle_ccp_chat_send_button_pressed(self, TAB_CCP)
            # And its ChatMessage actions could be distinct or shared.
            elif button.parent and isinstance(button.parent.parent, ChatMessage): # If CCP has ChatMessages
                 action_widget_ccp = self._get_chat_message_widget_from_button(button)
                 # if action_widget_ccp: await ccp_handlers.handle_ccp_chat_action_button_pressed(self, button, action_widget_ccp)
                 # For now, assume ChatMessage actions are handled by chat_handlers if they are generic enough
                 if action_widget_ccp: await chat_handlers.handle_chat_action_button_pressed(self, button, action_widget_ccp)

            else: logging.warning(f"Unhandled button on CCP tab: {button_id}")

        elif current_active_tab == TAB_NOTES:
            if button_id == "notes-create-new-button": await notes_handlers.handle_notes_create_new_button_pressed(self)
            elif button_id == "notes-edit-selected-button": await notes_handlers.handle_notes_edit_selected_button_pressed(self)
            elif button_id == "notes-search-button": await notes_handlers.handle_notes_search_button_pressed(self)
            elif button_id == "notes-load-selected-button": await notes_handlers.handle_notes_load_selected_button_pressed(self)
            elif button_id == "notes-save-current-button": await notes_handlers.handle_notes_save_current_button_pressed(self)
            elif button_id == "notes-save-button": await notes_handlers.handle_notes_main_save_button_pressed(self)
            elif button_id == "notes-delete-button": await notes_handlers.handle_notes_delete_button_pressed(self)
            elif button_id == "notes-save-keywords-button": await notes_handlers.handle_notes_save_keywords_button_pressed(self)
            else: logging.warning(f"Unhandled button on NOTES tab: {button_id}")

        elif current_active_tab == TAB_LOGS:
            if button_id == "copy-logs-button": await app_lifecycle_handlers.handle_copy_logs_button_pressed(self)
            else: logging.warning(f"Unhandled button on LOGS tab: {button_id}")

        else: # Fallback for unhandled tabs or buttons not caught by above
            logging.warning(f"Button '{button_id}' pressed on unhandled/inactive tab '{current_active_tab}' or unhandled button ID.")

    def _get_chat_message_widget_from_button(self, button: Button) -> Optional[ChatMessage]:
        """Helper to find the parent ChatMessage widget from an action button within it."""
        node: Optional[DOMNode] = button.parent # Start search from button's direct parent
        while node is not None:
            if isinstance(node, ChatMessage):
                return node
            node = node.parent
        logging.warning(f"Could not find parent ChatMessage for button: {button.id}")
        return None

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        current_active_tab = self.current_tab

        if input_id == "notes-search-input" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_search_input_changed(self, event.value)
        elif input_id == "chat-conversation-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "conv-char-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_search_input_changed(self, event.value)
        elif input_id == "ccp-prompt-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompt_search_input_changed(self, event.value)
        # Add more specific input handlers if needed, e.g., for title inputs if they need live validation/reaction

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view_id = event.list_view.id
        current_active_tab = self.current_tab

        if list_view_id == "notes-list-view" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_list_view_selected(self, list_view_id, event.item)
        elif list_view_id == "ccp-prompts-listview" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)
        # Note: chat-conversation-search-results-list and conv-char-search-results-list selections
        # are typically handled by their respective "Load Selected" buttons rather than direct on_list_view_selected.

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id
        current_active_tab = self.current_tab

        if checkbox_id.startswith("chat-conversation-search-") and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_search_checkbox_changed(self, checkbox_id, event.value)
        # Add handlers for checkboxes in other tabs if any

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handles changes in Select widgets if specific actions are needed beyond watchers."""
        select_id = event.select.id
        current_active_tab = self.current_tab

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)
        # Note: API provider/model Selects are handled by watchers on their reactive variables
        # (e.g., watch_chat_api_provider_value) which call _update_model_select.

    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # Delegate all worker state changes to the central handler
        await worker_handlers.handle_api_call_worker_state_changed(self, event)

    # --- Helper methods that remain in app.py (mostly for UI orchestration or complex state) ---
    def _safe_float(self, value: str, default: float, name: str) -> float:
        return safe_float(value, default, name) # Delegate to imported helper

    def _safe_int(self, value: str, default: Optional[int], name: str) -> Optional[int]:
        return safe_int(value, default, name) # Delegate to imported helper

    def _get_api_name(self, provider: str, endpoints: dict) -> Optional[str]:
        # ... (Keep original implementation as it's specific to config structure) ...
        provider_key_map = { "llama_cpp": "llama_cpp", "Ollama": "Ollama", "Oobabooga": "Oobabooga", "koboldcpp": "koboldcpp", "vllm": "vllm", "Custom": "Custom", "Custom-2": "Custom_2", }
        endpoint_key = provider_key_map.get(provider)
        if endpoint_key:
            url = endpoints.get(endpoint_key)
            if url: return url
            else: logging.warning(f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
        return None


    def watch_chat_api_provider_value(self, new_value: Optional[str]) -> None:
        self.loguru_logger.debug(f"Watcher: chat_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CHAT, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CHAT, models)

    def watch_ccp_api_provider_value(self, new_value: Optional[str]) -> None: # Renamed from watch_character_...
        self.loguru_logger.debug(f"Watcher: ccp_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CCP, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CCP, models)


    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        # ... (Keep original implementation as is) ...
        model_select_id = f"#{id_prefix}-api-model"
        try:
            model_select = self.query_one(model_select_id, Select)
            new_model_options = [(model, model) for model in models]
            # Store current value if it's valid for new options, otherwise try default
            current_value = model_select.value
            model_select.set_options(new_model_options) # This might clear the value

            if current_value in models:
                model_select.value = current_value
            elif models:
                model_select.value = models[0] # Default to first if old value invalid
            else:
                model_select.value = Select.BLANK # No models, ensure blank

            model_select.prompt = "Select Model..." if models else "No models available"
        except QueryError:
            logging.error(f"Helper ERROR: Cannot find model select '{model_select_id}'")
        except Exception as e_set_options:
             logging.error(f"Helper ERROR setting options/value for {model_select_id}: {e_set_options}")


    async def _populate_chat_conversation_character_filter_select(self) -> None:
        """Populates the character filter select in the Chat tab's conversation search."""
        # ... (Keep original implementation as is) ...
        logging.info("Attempting to populate #chat-conversation-search-character-filter-select.")
        if not self.notes_service:
            logging.error("Notes service not available for char filter select (Chat Tab).")
            # Optionally update the select to show an error state
            try:
                char_filter_select_err = self.query_one("#chat-conversation-search-character-filter-select", Select)
                char_filter_select_err.set_options([("Service Offline", Select.BLANK)])
            except QueryError: pass
            return
        try:
            db = self.notes_service._get_db(self.notes_user_id)
            character_cards = db.list_character_cards(limit=1000)
            options = [(char['name'], char['id']) for char in character_cards if char.get('name') and char.get('id')]

            char_filter_select = self.query_one("#chat-conversation-search-character-filter-select", Select)
            char_filter_select.set_options(options if options else [("No characters", Select.BLANK)])
            # Default to BLANK, user must explicitly choose or use "All Characters" checkbox
            char_filter_select.value = Select.BLANK
            logging.info(f"Populated #chat-conversation-search-character-filter-select with {len(options)} chars.")
        except QueryError as e_q:
            logging.error(f"Failed to find #chat-conversation-search-character-filter-select: {e_q}", exc_info=True)
        except CharactersRAGDBError as e_db: # Catch specific DB error
            logging.error(f"DB error populating char filter select (Chat Tab): {e_db}", exc_info=True)
        except Exception as e_unexp:
            logging.error(f"Unexpected error populating char filter select (Chat Tab): {e_unexp}", exc_info=True)

    def chat_wrapper(self, **kwargs: Any) -> Any:
        """
        Delegates to the actual worker target function in worker_events.py.
        This method is called by app.run_worker.
        """
        # All necessary parameters (message, history, api_endpoint, model, etc.)
        # are passed via kwargs from the calling event handler (e.g., handle_chat_send_button_pressed).
        return worker_events.chat_wrapper_function(self, **kwargs) # Pass self as 'app_instance'

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding='utf-8') as f:
                f.write(CONFIG_TOML_CONTENT)
    except Exception as e_cfg_main:
        logging.error(f"Could not ensure creation of default config file: {e_cfg_main}", exc_info=True)

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        # Ensure the CSS file exists, using its path defined in TldwCli.CSS_PATH
        # The actual css_content should be in Constants.py
        from .Constants import css_content # Assuming css_content is still in Constants.py
        css_file_path = Path(TldwCli.CSS_PATH)
        if not css_file_path.is_file():
            css_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(css_file_path, "w", encoding='utf-8') as f:
                f.write(css_content)
            logging.info(f"Created default CSS file: {css_file_path}")
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file '{TldwCli.CSS_PATH}': {e_css_main}", exc_info=True)

    app_instance = TldwCli() # Create instance
    try:
        app_instance.run()
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

#
# End of app.py
#######################################################################################################################
