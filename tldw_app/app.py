# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Imports
import asyncio
import json
import logging
import logging.handlers
import platform
import sys
from datetime import datetime
from pathlib import Path
import traceback
import os
from typing import Union, Generator, Optional, List, Dict, Any
#
# 3rd-Party Libraries
from loguru import logger as loguru_logger
from rich.text import Text
from rich.markup import escape
# --- Textual Imports ---
from textual.app import App, ComposeResult
from rich.markup import escape as escape_markup
from textual.logging import TextualHandler
from textual.strip import Strip
from textual.widgets import (
    Static, Button, Input, Header, Footer, RichLog, TextArea, Select, ListView, Checkbox, ListItem, Label, Collapsible
)
from textual.containers import Horizontal, Container, VerticalScroll
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.binding import Binding
from textual.dom import DOMNode  # For type hinting if needed
from textual.timer import Timer
from textual.css.query import QueryError  # For specific error handling
#
# --- Local API library Imports ---
from tldw_app.Chat.Chat_Functions import chat
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

#
#
#####################################################################################################################
#
# Functions:

# --- Constants ---
TAB_CHAT = "chat"
TAB_CONV_CHAR = "conversations_characters"
TAB_NOTES = "notes"
TAB_MEDIA = "media"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_STATS = "stats"
TAB_LOGS = "logs"
ALL_TABS = [TAB_CHAT, TAB_CONV_CHAR, TAB_NOTES, TAB_MEDIA, TAB_SEARCH, TAB_INGEST, TAB_LOGS, TAB_STATS]  # Updated list


# FIXME - this is only referenced in the sidebar, should consolidate with the config loading so there's only one set of these
ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS}
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys())

# --- ASCII Portrait ---
ASCII_PORTRAIT = r"""
  .--./)
 /.''.')
 | \ '/
 W `-'
 \\    '.
  '.    /
    `~~`
"""

#####################################
# --- Emoji Checker/Support ---
#####################################
# Cache the result so we don't re-calculate every time
_emoji_support_cached = None

def supports_emoji() -> bool:
    """
    Detects if the current terminal likely supports emojis.
    This is heuristic-based and not 100% foolproof.
    Caches the result for efficiency.
    """
    global _emoji_support_cached
    if _emoji_support_cached is not None:
        return _emoji_support_cached

    # 1. Must be a TTY
    if not sys.stdout.isatty():
        _emoji_support_cached = False
        return False

    # 2. Encoding should ideally be UTF-8
    # (getattr is used for safety, e.g., if sys.stdout is mocked)
    encoding = getattr(sys.stdout, 'encoding', '').lower()
    if 'utf-8' not in encoding and 'utf8' not in encoding:
        # Some terminals might still render emojis with other encodings,
        # but UTF-8 is the most reliable indicator.
        # For cmd.exe, even with chcp 65001 (UTF-8), font support is the main issue.
        # Don't immediately fail, let OS checks decide more
        #_emoji_support_cached = False
        #return False
        pass

    #####################################
    # 3. OS-specific checks
    #####################################
    os_name = platform.system()

    if os_name == 'Windows':
        # Windows Terminal has good emoji support.
        if 'WT_SESSION' in os.environ or 'TERMINUS_SUBLIME' in os.environ: # WT_SESSION for Windows Terminal, TERMINUS_SUBLIME for Terminus
            _emoji_support_cached = True
            return True
        # For older cmd.exe or PowerShell without Windows Terminal,
        # emoji support is unreliable or poor even with UTF-8 codepage.
        # Check if running in ConEmu, which has better support
        if 'CONEMUBUILD' in os.environ or 'CMDER_ROOT' in os.environ :
             _emoji_support_cached = True
             return True
        # Check if it's Fluent Terminal
        if os.environ.get('FLUENT_TERMINAL_PROFILE_NAME'):
            _emoji_support_cached = True
            return True

        # For standard cmd.exe or older PowerShell, be pessimistic.
        _emoji_support_cached = False
        return False

    # For macOS and Linux:
    # If it's a UTF-8 TTY, support is generally good on modern systems.
    # We can check for TERM=dumb as a negative indicator.
    if os.environ.get('TERM') == 'dumb':
        _emoji_support_cached = False
        return False

    # If encoding wasn't explicitly UTF-8 earlier, but it's Linux/macOS not TERM=dumb,
    # it's still likely okay on modern systems.
    # However, to be safer, if not UTF-8, tend towards no.
    if 'utf-8' not in encoding and 'utf8' not in encoding:
        _emoji_support_cached = False
        return False

    # Default to True for non-Windows UTF-8 (or generally capable) TTYs not being 'dumb'
    _emoji_support_cached = True
    return True

# Define your emoji and fallback pairs
# You can centralize these or define them where needed.
# Example:
# --- Emoji and Fallback Definitions ---
EMOJI_TITLE_BRAIN = "🧠"
FALLBACK_TITLE_BRAIN = "[B]"
EMOJI_TITLE_NOTE = "📝"
FALLBACK_TITLE_NOTE = "[N]"
EMOJI_TITLE_SEARCH = "🔍"
FALLBACK_TITLE_SEARCH = "[S]"

EMOJI_SEND = "▶" # Or "➡️"
FALLBACK_SEND = "Send"

EMOJI_SIDEBAR_TOGGLE = "☰"
FALLBACK_SIDEBAR_TOGGLE = "Menu"

EMOJI_CHARACTER_ICON = "👤"
FALLBACK_CHARACTER_ICON = "Char"

EMOJI_THINKING = "🤔" # Or "⏳" "💭"
FALLBACK_THINKING = "..."

EMOJI_COPY = "📋"
FALLBACK_COPY = "Copy"
EMOJI_COPIED = "✅" # For feedback
FALLBACK_COPIED = "[OK]"

EMOJI_EDIT = "✏️"
FALLBACK_EDIT = "Edit"
EMOJI_SAVE_EDIT = "💾" # Or use check/OK
FALLBACK_SAVE_EDIT = "Save"

ROCKET_EMOJI = "🚀"
ROCKET_FALLBACK = "[Go!]"

CHECK_EMOJI = "✅"
CHECK_FALLBACK = "[OK]"

CROSS_EMOJI = "❌"
CROSS_FALLBACK = "[FAIL]"

SPARKLES_EMOJI = "✨"
SPARKLES_FALLBACK = "*"

def get_char(emoji_char: str, fallback_char: str) -> str:
    """Returns the emoji if supported, otherwise the fallback."""
    return emoji_char if supports_emoji() else fallback_char


# --- Custom Logging Handler ---
class RichLogHandler(logging.Handler):
    def __init__(self, rich_log_widget: RichLog):
        super().__init__()
        self.rich_log_widget = rich_log_widget
        self.log_queue = asyncio.Queue()
        self.formatter = logging.Formatter(
            "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
            style="{", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.setFormatter(self.formatter)
        self._queue_processor_task = None

    def start_processor(self, app: App):  # Keep 'app' param for context if needed elsewhere, but don't use for run_task
        """Starts the log queue processing task using the widget's run_task."""
        if not self._queue_processor_task or self._queue_processor_task.done():
            try:
                # Get the currently running event loop
                loop = asyncio.get_running_loop()
                # Create the task using the standard asyncio function
                self._queue_processor_task = loop.create_task(
                    self._process_log_queue(),
                    name="RichLogProcessor"
                )
                logging.debug("RichLog queue processor task started via asyncio.create_task.")
            except RuntimeError as e:
                # Handle cases where the loop might not be running (shouldn't happen if called from on_mount)
                logging.error(f"Failed to get running loop to start log processor: {e}")
            except Exception as e:
                logging.error(f"Failed to start log processor task: {e}", exc_info=True)

    async def stop_processor(self):
        """Signals the queue processor task to stop and waits for it."""
        # This cancellation logic works for tasks created with asyncio.create_task
        if self._queue_processor_task and not self._queue_processor_task.done():
            logging.debug("Attempting to stop RichLog queue processor task...")
            self._queue_processor_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                await self._queue_processor_task
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task cancelled successfully.")
            except Exception as e:
                # Log errors during cancellation itself
                logging.error(f"Error occurred while awaiting cancelled log processor task: {e}", exc_info=True)
            finally:
                self._queue_processor_task = None  # Ensure it's cleared

    async def _process_log_queue(self):
        """Coroutine to process logs from the queue and write to the widget."""
        while True:
            try:
                message = await self.log_queue.get()
                if self.rich_log_widget.is_mounted and self.rich_log_widget.app:
                    self.rich_log_widget.write(message)
                self.log_queue.task_done()
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task received cancellation.")
                # Process any remaining items? Might be risky if app is shutting down.
                # while not self.log_queue.empty():
                #    try: message = self.log_queue.get_nowait(); # process...
                #    except asyncio.QueueEmpty: break
                break  # Exit the loop on cancellation
            except Exception as e:
                loguru_logger.critical(f"!!! CRITICAL ERROR in RichLog processor: {e}")  # Use print as fallback
                traceback.print_exc()
                # Avoid continuous loop on error, maybe sleep?
                await asyncio.sleep(1)

    def emit(self, record: logging.LogRecord):
        """Format the record and put it onto the async queue."""
        try:
            message = self.format(record)
            # Use call_soon_threadsafe if emit might be called from non-asyncio threads (workers)
            # For workers started with thread=True, this is necessary.
            if hasattr(self.rich_log_widget, 'app') and self.rich_log_widget.app:
                self.rich_log_widget.app._loop.call_soon_threadsafe(self.log_queue.put_nowait, message)
            else:  # Fallback during startup/shutdown
                if record.levelno >= logging.WARNING: logging.warning(f"LOG_FALLBACK: {message}")
        except Exception:
            loguru_logger.warning(f"!!!!!!!! ERROR within RichLogHandler.emit !!!!!!!!!!")  # Use print as fallback
            traceback.print_exc()


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
    current_tab: reactive[str] = reactive("chat", layout=True)

    # Add state to hold the currently streaming AI message widget
    current_ai_message_widget: Optional[ChatMessage] = None

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "Ollama")
    _default_character_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic")

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    character_api_provider_value: reactive[Optional[str]] = reactive(_default_character_provider)

    # Reactives for sidebar
    chat_sidebar_collapsed: reactive[bool] = reactive(False, layout=True)
    character_sidebar_collapsed: reactive[bool] = reactive(False, layout=True)  # For character sidebar
    notes_sidebar_left_collapsed: reactive[bool] = reactive(False, layout=True)
    notes_sidebar_right_collapsed: reactive[bool] = reactive(False, layout=True)
    conv_char_sidebar_collapsed: reactive[bool] = reactive(False, layout=True)

    # Reactive variables for selected note details
    current_selected_note_id: reactive[Optional[str]] = reactive(None)
    current_selected_note_version: reactive[Optional[int]] = reactive(None)
    current_selected_note_title: reactive[Optional[str]] = reactive(None)
    current_selected_note_content: reactive[Optional[str]] = reactive("")

    # Reactive variable for current chat conversation ID
    current_chat_conversation_id: reactive[Optional[str]] = reactive(None)
    # Reactive variable for current conversation loaded in the Conversations & Characters tab
    current_conv_char_tab_conversation_id: reactive[Optional[str]] = reactive(None)

    # De-Bouncers
    _conv_char_search_timer: Optional[Timer] = None # For conv-char-search-input
    _conversation_search_timer: Optional[Timer] = None # For chat-conversation-search-bar

    def __init__(self):
        super().__init__()
        # Load config ONCE
        self.app_config = load_settings()  # Ensure this is called

        # --- Initialize NotesInteropService ---
        self.notes_user_id = "default_tui_user"  # Or any default user ID string
        notes_db_base_dir = Path.home() / ".config/tldw_cli/user_notes"
        try:
            self.notes_service = NotesInteropService(
                base_db_directory=notes_db_base_dir,
                api_client_id="tldw_tui_client"  # Client ID for operations done by the TUI
            )
            logging.info(f"NotesInteropService initialized for user '{self.notes_user_id}' at {notes_db_base_dir}")
        except CharactersRAGDBError as e:
            logging.error(f"Failed to initialize NotesInteropService: {e}", exc_info=True)
            self.notes_service = None
        except Exception as e:  # Catch any other unexpected error during init
            logging.error(f"An unexpected error occurred during NotesInteropService initialization: {e}", exc_info=True)
            self.notes_service = None

        logging.debug("__INIT__: Attempting to get providers and models...")
        try:
            # Call the function from the config module
            self.providers_models = get_cli_providers_and_models()
            # *** ADD THIS LOGGING ***
            logging.info(
                f"__INIT__: Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e:
            logging.error(f"__INIT__: Failed to get providers and models: {e}", exc_info=True)
            self.providers_models = {}  # Set empty on error
        # Determine the *value* for the initial tab but don't set the reactive var yet
        initial_tab_from_config = get_cli_setting("general", "default_tab", "chat")
        if initial_tab_from_config not in ALL_TABS:
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to 'chat'.")
            self._initial_tab_value = "chat"
        else:
            self._initial_tab_value = initial_tab_from_config

        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        self._rich_log_handler: Optional[RichLogHandler] = None  # Initialize handler attribute
        self._conv_char_search_timer = None
        self._conversation_search_timer = None

    def _setup_logging(self):
        """Sets up all logging handlers, including Loguru integration."""
        # Use standard logging for messages during this setup phase itself,
        # as Loguru is being reconfigured.
        logging.info("--- _setup_logging START ---")
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # --- BEGIN LOGURU MANAGEMENT (Improved) ---
        try:
            # Remove ALL existing Loguru handlers. This ensures we clear out the
            # default stderr sink and any other sinks that might have been added
            # by Loguru or other parts of the code before this point.
            loguru_logger.remove()
            logging.info("Loguru: All pre-existing sinks removed.")

            # Now, add a handler that bridges Loguru to Python's standard logging.
            # This PropagateHandler takes Loguru's records and passes them to standard logging.
            class PropagateHandler(logging.Handler):
                def emit(self, record: logging.LogRecord):  # Standard logging LogRecord
                    # This handler is added to standard logging, but it's meant to be
                    # used as a target for Loguru. The logic for Loguru to *send* to
                    # standard logging is simpler.

                    # Let's redefine: We want a Loguru *sink* that pushes to standard logging.
                    # Loguru's documentation shows how to do this:
                    # https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
                    pass  # This handler isn't used in the Loguru -> standard logging direction

            # The official way to make Loguru messages available to standard logging:
            # Create a standard logging handler that Loguru will write to.
            class InterceptHandler(logging.Handler):
                def emit(self, record):
                    # Get corresponding Loguru level if it exists
                    try:
                        level = loguru_logger.level(record.levelname).name
                    except ValueError:
                        level = record.levelno

                    # Find caller from where Loguru was logged
                    frame, depth = logging.currentframe(), 0
                    # IMPORTANT: Adjust depth if emit is called from another helper.
                    # We want the frame where loguru_loguru_logger.info() etc. was called.
                    # This simple depth calculation might need tweaking if emit is nested.
                    # For direct calls from loguru_logger.add(InterceptHandler()), depth might be low.
                    # If InterceptHandler is added to root logger and loguru calls logging.getLogger().log(),
                    # then depth should find the original loguru call site.

                    # A simpler approach for depth might be needed if the above is complex.
                    # For now, let's assume a reasonable depth or let Loguru infer it.
                    # The crucial part is that Loguru has an `opt(depth=...)`

                    # Re-evaluate how to get the correct call site information
                    # For this use case, we are configuring Loguru to send its messages
                    # TO standard logging.
                    # So when you call loguru_loguru_logger.info(), it should go to a sink that
                    # then calls logging.getLogger().handle().

                    # Let's use Loguru's recommended way for its messages to be caught by standard logging handlers:
                    # Standard logging.LogRecord attributes:
                    # record.name, record.levelno, record.pathname, record.lineno,
                    # record.msg, record.args, record.exc_info, record.funcName

                    # Create a new standard logging record from Loguru's record
                    # This is tricky because Loguru's record object is internal.
                    # The easiest way is for Loguru to have a sink that *calls* standard logging.
                    pass  # This will be replaced

            # --- Loguru Configuration to send its logs to standard logging handlers ---
            # This configuration ensures that when `loguru_loguru_logger.info("...")` is called,
            # it will be passed to your standard logging handlers (TextualHandler, RichLogHandler).

            # 1. Remove any default/existing Loguru sinks.
            loguru_logger.remove()  # Already did this above.

            # 2. Add a new Loguru sink that calls the standard `logging` module.
            #    This effectively forwards Loguru messages.
            def sink_to_standard_logging(message):
                """
                A Loguru sink function that takes a Loguru Message object
                and logs it using the standard Python logging module.
                """
                record = message.record  # Loguru's internal record

                # Map Loguru level to standard logging level
                level_mapping = {
                    "TRACE": logging.DEBUG,  # Or a custom level if you define TRACE
                    "DEBUG": logging.DEBUG,
                    "INFO": logging.INFO,
                    "SUCCESS": logging.INFO,  # Standard logging doesn't have SUCCESS, map to INFO
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL,
                }
                std_level = level_mapping.get(record["level"].name, logging.INFO)

                # Get a standard logger instance (e.g., based on Loguru's record name)
                std_logger = logging.getLogger(record["name"])

                # Log using standard logger
                # We need to pass message, and potentially exc_info etc.
                # record["message"] is the formatted string from Loguru
                if record["exception"]:
                    std_logger.log(
                        std_level,
                        record["message"],  # Loguru has already formatted the message
                        exc_info=record["exception"]
                    )
                else:
                    std_logger.log(
                        std_level,
                        record["message"]  # Loguru has already formatted the message
                    )

            # Add this custom sink to Loguru.
            # You can choose a format for how Loguru prepares the message
            # before it hits your standard logging formatters.
            # Using a simple format here, as standard logging will reformat.
            loguru_logger.add(
                sink_to_standard_logging,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                # Loguru's own formatting
                level="TRACE"  # Capture all levels from Loguru to pass to standard logging
            )
            logging.info("Loguru: Configured to forward its messages to standard Python logging system.")

        except Exception as e:
            logging.error(f"Loguru: Error during Loguru reconfiguration: {e}", exc_info=True)
        # --- END LOGURU MANAGEMENT ---

        # Configure the root logger FIRST (standard logging)
        root_logger = logging.getLogger()
        # Determine initial log level for standard logging from app_config
        # If loguru messages are now going to standard logging, the root_logger's level
        # will also affect if they are processed by standard logging handlers.
        initial_log_level_str = self.app_config.get("general", {}).get("log_level", "INFO").upper()
        initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
        root_logger.setLevel(initial_log_level)  # Set overall level for standard logging

        # Important: Loguru's own level (in `loguru_logger.add`) determines what it SENDS.
        # Standard logging's root_logger.setLevel and handler levels determine what they PROCESS.
        # For example, if Loguru sends a DEBUG, but root_logger is INFO, the DEBUG won't be processed
        # by standard handlers unless a specific standard handler has its level set to DEBUG.

        # Log this message using standard logging, now that Loguru should be forwarding.
        logging.info(f"Standard Logging: Root logger level initially set to: {logging.getLevelName(root_logger.level)}")
        # You can test with loguru here too, it should appear in your Textual logs if bridge is working
        loguru_logger.info("Loguru Test: This message from Loguru should appear in Textual logs.")

        # --- Add TextualHandler (to standard logging) ---
        # (Your existing TextualHandler setup code is fine)
        has_textual_handler = any(isinstance(h, TextualHandler) for h in root_logger.handlers)
        if not has_textual_handler:
            textual_console_handler = TextualHandler()
            textual_console_handler.setLevel(initial_log_level)  # Respects app_config
            # Formatter for Textual's dev console
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",  # Standard format
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            textual_console_handler.setFormatter(console_formatter)
            root_logger.addHandler(textual_console_handler)
            logging.info(
                f"Standard Logging: Added TextualHandler (Level: {logging.getLevelName(textual_console_handler.level)}).")
        else:
            logging.info("Standard Logging: TextualHandler already exists.")

        # --- Setup RichLog Handler (to standard logging) ---
        # (Your existing RichLogHandler setup code is fine)
        try:
            log_display_widget = self.query_one("#app-log-display", RichLog)
            if self._rich_log_handler and self._rich_log_handler in root_logger.handlers:
                logging.info("Standard Logging: RichLogHandler already exists and is added.")
            elif not self._rich_log_handler:
                self._rich_log_handler = RichLogHandler(log_display_widget)
                # RichLogHandler can have its own level, e.g., DEBUG to see more details in-app
                rich_log_handler_level_str = self.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
                rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
                self._rich_log_handler.setLevel(rich_log_handler_level)
                # RichLogHandler will use its own formatter defined in its class
                root_logger.addHandler(self._rich_log_handler)
                logging.info(
                    f"Standard Logging: Added RichLogHandler (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            # ... (rest of your RichLogHandler error handling)
        except QueryError:
            logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            self._rich_log_handler = None
        except Exception as e:
            logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
            self._rich_log_handler = None

        # --- Setup File Logging (to standard logging) ---
        # (Your existing FileHandler setup code is fine)
        try:
            log_file_path = get_cli_log_file_path()
            log_dir = log_file_path.parent
            log_dir.mkdir(parents=True, exist_ok=True)

            has_file_handler = any(
                isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_file_path) for h in
                root_logger.handlers)

            if not has_file_handler:
                # Max bytes and backup count from config, with direct defaults if not found
                max_bytes_default = 10485760  # 10MB
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
                # Formatter for the file
                file_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",  # Standard format
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
        # This ensures the root logger is at least as verbose as its most verbose handler.
        all_std_handlers = root_logger.handlers
        if all_std_handlers:
            # Filter out level 0 handlers if any (shouldn't be typical)
            handler_levels = [h.level for h in all_std_handlers if h.level > 0]
            if handler_levels:
                lowest_effective_level = min(handler_levels)
                # Also consider the initial_log_level explicitly set on root_logger
                # The root logger's level should be min(its_explicit_level, lowest_handler_level)
                # but practically, setting it to the lowest handler level is common.
                # However, if root_logger.setLevel(INFO) was called, and a handler is DEBUG,
                # root needs to be at least DEBUG.
                if root_logger.level > lowest_effective_level:
                    logging.info(
                        f"Standard Logging: Adjusting root logger level from {logging.getLevelName(root_logger.level)} to {logging.getLevelName(lowest_effective_level)} to match most verbose handler.")
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
            for tab_id in ALL_TABS:
                label = "Conversations & Characters" if tab_id == TAB_CONV_CHAR else tab_id.replace('_',
                                                                                                    ' ').capitalize()
                yield Button(
                    label,
                    id=f"tab-{tab_id}",
                    # Initial active state based on the value determined in __init__
                    classes="-active" if tab_id == self._initial_tab_value else ""
                )

    def compose_content_area(self) -> ComposeResult:
        logging.debug(f"Compose: Composing content area...")

        with Container(id="content"):
            # --- Chat Window ---
            # Assign specific reactive variables to the Select widgets
            with Container(id=f"{TAB_CHAT}-window", classes="window"):
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
                yield from create_character_sidebar("chat")

            # --- Conversations & Characters Window (Redesigned) ---
            with Container(id=f"{TAB_CONV_CHAR}-window", classes="window"):
                # Toggle button for the left pane
                yield Button("☰", id="toggle-conv-char-sidebar")

                # Left Pane
                with VerticalScroll(id="conv-char-left-pane", classes="cc-left-pane"):
                    yield Static("My Characters & Conversations",
                                 classes="sidebar-title")  # Reusing sidebar-title class

                    with Collapsible(title="Characters", id="conv-char-characters-collapsible"):
                        yield Select(
                            options=[("", "<placeholder>")],  # Placeholder option
                            prompt="Select Character...",
                            allow_blank=True,
                            id="conv-char-character-select"
                        )
                        # Add other character-related buttons here later if needed

                    with Collapsible(title="Conversations", id="conv-char-conversations-collapsible"):
                        yield Input(id="conv-char-search-input", placeholder="Search conversations...",
                                    classes="sidebar-input")  # Moved
                        yield Button("Search", id="conv-char-conversation-search-button",
                                     classes="sidebar-button")  # New
                        yield ListView(
                            id="conv-char-search-results-list")  # Moved
                        yield Button("Load Selected", id="conv-char-load-button",
                                     classes="sidebar-button")  # Moved

                # Center Pane
                with VerticalScroll(id="conv-char-center-pane", classes="cc-center-pane"):
                    yield Static("Conversation History", classes="pane-title")
                    # Message widgets will be mounted here dynamically

                # Right Pane (also includes settings previously in a separate sidebar for this tab)
                with VerticalScroll(id="conv-char-right-pane", classes="cc-right-pane"):
                    yield Static("Character/Conversation Details", classes="sidebar-title")  # Reusing sidebar-title

                    # General Settings (from create_settings_sidebar for TAB_CONV_CHAR)
                    # This assumes create_settings_sidebar for TAB_CONV_CHAR would now be integrated here
                    # or this section is for new character/conv details separate from global settings.
                    # For now, adding new specific fields as per instruction:
                    yield Static("Title:", classes="sidebar-label")
                    yield Input(id="conv-char-title-input", placeholder="Conversation title...",
                                classes="sidebar-input")

                    yield Static("Keywords:", classes="sidebar-label")
                    yield TextArea("", id="conv-char-keywords-input",
                                   classes="conv-char-keywords-textarea")  # Placeholder text removed for specific class

                    yield Button("Save Details", id="conv-char-save-details-button", classes="sidebar-button",
                                 variant="primary")

                    yield Static("Export Options", classes="sidebar-label export-label")
                    yield Button("Export as Text", id="conv-char-export-text-button", classes="sidebar-button")
                    yield Button("Export as JSON", id="conv-char-export-json-button", classes="sidebar-button")

                    # Placeholder for where old settings from create_settings_sidebar(TAB_CONV_CHAR,...) would go
                    # For this step, we are just adding the new UI elements.
                    # The existing call to create_settings_sidebar for TAB_CONV_CHAR is removed as per the instruction
                    # to "Remove all existing child widgets currently composed within this container".
                    # If global settings (like API provider, model) are still needed for this tab,
                    # they would need to be re-integrated here or the design re-evaluated.
                    # The instruction was to remove existing children of the TAB_CONV_CHAR-window,
                    # which included the create_settings_sidebar(TAB_CONV_CHAR, self.app_config) call.
                    yield from create_settings_sidebar(TAB_CONV_CHAR, self.app_config)

            # --- Logs Window ---
            with Container(id=f"{TAB_LOGS}-window", classes="window"):
                yield RichLog(id="app-log-display", wrap=True, highlight=True, markup=True, auto_scroll=True)
                yield Button("Copy All Logs to Clipboard", id="copy-logs-button", classes="logs-action-button")

            # --- Other Placeholder Windows ---
            for tab_id in ALL_TABS:
                if tab_id not in [TAB_CHAT, TAB_CONV_CHAR, TAB_LOGS, TAB_NOTES]:  # Updated to TAB_CONV_CHAR
                    with Container(id=f"{tab_id}-window", classes="window placeholder-window"):
                        yield Static(f"{tab_id.replace('_', ' ').capitalize()} Window Placeholder")
                        yield Button("Coming Soon", id=f"{tab_id}-placeholder-button", disabled=True)

            # --- Notes Tab Window ---
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

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        """Called when the chat provider Select value changes internally."""
        loguru_logger.debug(f">>> DEBUG: update_chat_provider_reactive called with: {new_value!r}")
        self.chat_api_provider_value = new_value

    def update_character_provider_reactive(self, new_value: Optional[str]) -> None:
        """Called when the character provider Select value changes internally."""
        loguru_logger.debug(f">>> DEBUG: update_character_provider_reactive called with: {new_value!r}")
        self.character_api_provider_value = new_value

    # --- END Add explicit methods ---

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
            char_select = self.query_one(f"#{TAB_CONV_CHAR}-api-provider", Select)
            self.watch(char_select, "value", self.update_character_provider_reactive, init=False)
            logging.debug(
                f"Bound character provider Select ({char_select.id}) value to update_character_provider_reactive")
            loguru_logger.debug(f">>> DEBUG: Bound character provider Select to reactive update method.")
        except QueryError:
            logging.error(f"on_mount: Failed to find character provider select: #{TAB_CONV_CHAR}-api-provider")
            loguru_logger.debug(f">>> DEBUG: ERROR - Failed to bind character provider select.")
        except Exception as e:
            logging.error(f"on_mount: Error binding character provider select: {e}", exc_info=True)
            loguru_logger.debug(f">>> DEBUG: ERROR - Exception during character provider select binding: {e}")
        # --- END BINDING LOGIC ---

        # --- Set initial reactive tab value ---
        # This MUST be done AFTER the UI exists and AFTER bindings are set up (if they depend on it)
        # Crucially, this will trigger watch_current_tab to set the initial visibility.
        print(f">>> DEBUG: on_mount: Setting self.current_tab = {self._initial_tab_value} (will trigger watcher)")
        logging.info(
            f"App on_mount: Setting current_tab reactive value to {self._initial_tab_value} to trigger initial view.")
        self.current_tab = self._initial_tab_value

        logging.info("App mount process completed.")

        # Populate character filter for conversation search
        try:
            if self.notes_service:
                db = self.notes_service._get_db(self.notes_user_id)
                character_cards = db.list_character_cards(limit=1000)  # Fetch a reasonable number of characters

                options = [(char['name'], char['id']) for char in character_cards if
                           char.get('name') and char.get('id') is not None]

                char_filter_select = self.query_one("#chat-conversation-search-character-filter-select", Select)
                char_filter_select.set_options(options)
                logging.info(
                    f"Populated #chat-conversation-search-character-filter-select with {len(options)} characters.")
            else:
                logging.warning(
                    "Notes service not available, cannot populate character filter for conversation search.")
        except QueryError as e:
            logging.error(f"on_mount: Could not find #chat-conversation-search-character-filter-select: {e}",
                          exc_info=True)
        except CharactersRAGDBError as e:
            logging.error(f"on_mount: Database error populating character filter: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"on_mount: Unexpected error populating character filter: {e}", exc_info=True)

        # Populate the character select dropdown in the Conversations & Characters tab
        self.call_later(self._populate_conv_char_character_select)

    async def on_shutdown_request(self, event) -> None:
        """Stop background tasks before shutdown."""
        logging.info("--- App Shutdown Requested ---")
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        logging.info("--- App Unmounting ---")
        # Processor should already be stopped by on_shutdown_request if graceful
        # Ensure handlers are removed here regardless
        if self._rich_log_handler:
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")
        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()  # Ensure file is closed
                    root_logger.removeHandler(handler)
                    logging.info("File handler removed.")
                except Exception as e:
                    logging.error(f"Error removing file handler: {e}")
        logging.shutdown()  # Ensure logs are flushed
        loguru_logger.info("--- App Unmounted ---")  # Use print as logging might be shut down

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
            try:
                old_button = self.query_one(f"#tab-{old_tab}")
                old_button.remove_class("-active")
                print(f">>> DEBUG: Deactivated button #tab-{old_tab}")
            except QueryError:
                print(f">>> DEBUG: Could not find old button #tab-{old_tab}")
                logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            except Exception as e:
                logging.error(f"Watcher: Error deactivating old button: {e}", exc_info=True)

            try:
                old_window = self.query_one(f"#{old_tab}-window")
                old_window.display = False  # Set style directly
                print(f">>> DEBUG: Set display=False for window #{old_tab}-window")
            except QueryError:
                print(f">>> DEBUG: Could not find old window #{old_tab}-window")
                logging.warning(f"Watcher: Could not find old window #{old_tab}-window")
            except Exception as e:
                logging.error(f"Watcher: Error hiding old window: {e}", exc_info=True)

        # --- Show New Tab ---
        try:
            new_button = self.query_one(f"#tab-{new_tab}")
            new_button.add_class("-active")
            print(f">>> DEBUG: Activated button #tab-{new_tab}")
        except QueryError:
            print(f">>> DEBUG: Could not find new button #tab-{new_tab}")
            logging.error(f"Watcher: Could not find new button #tab-{new_tab}")
        except Exception as e:
            logging.error(f"Watcher: Error activating new button: {e}", exc_info=True)

        try:
            new_window = self.query_one(f"#{new_tab}-window")
            new_window.display = True  # Set style directly
            print(f">>> DEBUG: Set display=True for window #{new_tab}-window")

            # Focus input (your existing logic here is fine)
            if new_tab not in [TAB_LOGS]:
                input_widget: Optional[Union[TextArea, Input]] = None
                try:
                    input_widget = new_window.query_one(TextArea)
                except QueryError:
                    try:
                        input_widget = new_window.query_one(Input)
                    except QueryError:
                        pass

                if input_widget:
                    def _focus_input():
                        try:
                            input_widget.focus()
                        except Exception as focus_err:
                            logging.warning(f"Focus failed: {focus_err}")

                    # Slightly longer delay might sometimes help if layout is complex
                    self.set_timer(0.1, _focus_input)
                    logging.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    logging.debug(f"Watcher: No input found to focus in '{new_tab}'")

        except QueryError:
            print(f">>> DEBUG: Could not find new window #{new_tab}-window")
            logging.error(f"Watcher: Could not find new window #{new_tab}-window")
        except Exception as e:
            print(f">>> DEBUG: Error showing new window #{new_tab}-window: {e}")
            logging.error(f"Watcher: Error showing new window: {e}", exc_info=True)

        loguru_logger.debug(">>> DEBUG: watch_current_tab finished.")

        # If the new tab is TAB_NOTES, load the notes
        if new_tab == TAB_NOTES:
            self.call_later(self.load_and_display_notes)

    def watch_chat_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the chat sidebar."""
        try:
            sidebar = self.query_one("#chat-sidebar")  # id from create_settings_sidebar
            sidebar.display = not collapsed  # True → visible
        except QueryError:
            logging.error("Chat sidebar widget not found.")

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

    def watch_conv_char_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations & Characters left sidebar pane."""
        try:
            sidebar_pane = self.query_one("#conv-char-left-pane") # The ID of the VerticalScroll
            sidebar_pane.display = not collapsed # True means visible, False means hidden
            # Optional: you might want to set a class to control width/border as well,
            # similar to how other sidebars might be handled, e.g., adding/removing "collapsed" class.
            # For now, direct display toggle is simplest.
            # If using a class:
            # sidebar_pane.set_class(collapsed, "collapsed") # Adds "collapsed" class if true, removes if false

            # Also, ensure the toggle button itself is not part of the pane being hidden.
            # Based on Step 1, the button "toggle-conv-char-sidebar" is outside "conv-char-left-pane".
            logging.debug(f"Conversations & Characters left pane display set to {not collapsed}")
        except QueryError:
            logging.error("Conversations & Characters left sidebar pane (#conv-char-left-pane) not found.")
        except Exception as e:
            logging.error(f"Error toggling Conversations & Characters left sidebar pane: {e}", exc_info=True)

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
                    await self.load_and_display_notes()
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

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for tabs, sending messages, and message actions."""
        button = event.button
        button_id = button.id
        print(f"\n>>> DEBUG: on_button_pressed called! Button ID: {event.button.id}\n")
        logging.debug(f"Button pressed: {button_id}, Classes: {button.classes}")

        # --- Search button in Conversations & Characters Tab ---
        if button_id == "conv-char-conversation-search-button":
            logging.debug("conv-char-conversation-search-button pressed. Performing search.")
            await self._perform_conv_char_search()
            return

        elif button_id == "conv-char-load-button":
            logging.info("conv-char-load-button pressed.")
            try:
                results_list_view = self.query_one("#conv-char-search-results-list", ListView)
                highlighted_item = results_list_view.highlighted_child

                if not (highlighted_item and hasattr(highlighted_item, 'details')):
                    logging.warning("No conversation selected in conv-char list or item has no details.")
                    # self.notify("Please select a conversation to load.", severity="warning") # If notifier is set up
                    return

                # FIXME
                conv_details = highlighted_item.details
                loaded_conversation_id = conv_details.get('id')

                if not loaded_conversation_id:
                    logging.error("Selected item in conv-char list is missing conversation ID.")
                    # self.notify("Selected item is invalid.", severity="error")
                    return

                self.current_conv_char_tab_conversation_id = loaded_conversation_id
                logging.info(f"Current conv-char tab conversation ID set to: {loaded_conversation_id}")

                # Populate Right Pane
                title_input = self.query_one("#conv-char-title-input", Input)
                keywords_input = self.query_one("#conv-char-keywords-input", TextArea) # This is a TextArea

                title_input.value = conv_details.get('title', '')

                # Fetch and populate keywords
                if self.notes_service:
                    db_for_keywords = self.notes_service._get_db(self.notes_user_id)
                    keywords_list = db_for_keywords.get_keywords_for_conversation(loaded_conversation_id)
                    keywords_input.text = ", ".join([kw['keyword'] for kw in keywords_list]) if keywords_list else ""
                    logging.info(f"Populated keywords for conversation {loaded_conversation_id} into #conv-char-keywords-input.")
                else:
                    keywords_input.text = "" # Clear if no service
                    logging.warning("Notes service not available, cannot load keywords for right pane.")


                # Populate Center Pane (Messages)
                center_pane = self.query_one("#conv-char-center-pane", VerticalScroll)
                await center_pane.remove_children() # Clear previous messages

                if not self.notes_service:
                    logging.error("Notes service not available, cannot load messages for center pane.")
                    await center_pane.mount(Static("Error: Notes service unavailable. Cannot load messages."))
                    return

                db = self.notes_service._get_db(self.notes_user_id)
                messages = db.get_messages_for_conversation(loaded_conversation_id, order_by_timestamp="ASC", limit=1000)

                if not messages:
                    await center_pane.mount(Static("No messages in this conversation."))
                else:
                    for msg_data in messages:
                        # Ensure image_data is handled correctly (it might be None or bytes)
                        image_data_for_widget = msg_data.get('image_data')
                        chat_message_widget = ChatMessage(
                            message=msg_data['content'],
                            role=msg_data['sender'],
                            timestamp=msg_data.get('timestamp'),
                            image_data=image_data_for_widget,
                            image_mime_type=msg_data.get('image_mime_type'),
                            message_id=msg_data['id']
                        )
                        await center_pane.mount(chat_message_widget)

                center_pane.scroll_end(animate=False)
                logging.info(f"Loaded {len(messages)} messages into #conv-char-center-pane for conversation {loaded_conversation_id}.")

            except QueryError as e:
                logging.error(f"UI component not found during conv-char load: {e}", exc_info=True)
                # Optionally, notify the user if a specific component was expected but not found
                try:
                    center_pane_err_fallback = self.query_one("#conv-char-center-pane", VerticalScroll)
                    await center_pane_err_fallback.remove_children()
                    await center_pane_err_fallback.mount(Static("Error: UI component missing for loading."))
                except QueryError: pass # Center pane itself might be the issue
            except CharactersRAGDBError as e:
                logging.error(f"Database error during conv-char load: {e}", exc_info=True)
                # self.notify("Error loading conversation data from database.", severity="error")
                try:
                    center_pane_db_err = self.query_one("#conv-char-center-pane", VerticalScroll)
                    await center_pane_db_err.remove_children()
                    await center_pane_db_err.mount(Static("Error: Database issue loading messages."))
                except QueryError: pass
            except Exception as e:
                logging.error(f"Unexpected error during conv-char load: {e}", exc_info=True)
                # self.notify("An unexpected error occurred while loading the conversation.", severity="error")
                try:
                    center_pane_unexp_err = self.query_one("#conv-char-center-pane", VerticalScroll)
                    await center_pane_unexp_err.remove_children()
                    await center_pane_unexp_err.mount(Static("Error: Unexpected issue loading conversation."))
                except QueryError: pass
            return

        elif button_id == "conv-char-save-details-button":
            logging.info("conv-char-save-details-button pressed.")
            if not self.current_conv_char_tab_conversation_id:
                logging.warning("No current conversation loaded in ConvChar tab to save details for.")
                # self.notify("No conversation loaded.", severity="warning")
                return

            if not self.notes_service:
                logging.error("Notes service is not available.")
                # self.notify("Database service not available.", severity="error")
                return

            try:
                title_input = self.query_one("#conv-char-title-input", Input)
                keywords_widget = self.query_one("#conv-char-keywords-input", TextArea)

                new_title = title_input.value.strip()
                new_keywords_str = keywords_widget.text.strip()
                target_conversation_id = self.current_conv_char_tab_conversation_id

                db = self.notes_service._get_db(self.notes_user_id)
                current_conv_details = db.get_conversation_by_id(target_conversation_id)

                if not current_conv_details:
                    logging.error(f"Conversation {target_conversation_id} not found in DB for saving details.")
                    # self.notify("Error: Conversation not found.", severity="error")
                    return

                current_db_version = current_conv_details.get('version')
                if current_db_version is None:
                    logging.error(f"Conversation {target_conversation_id} is missing version information.")
                    # self.notify("Error: Conversation version missing.", severity="error")
                    return

                # 1. Update Title if changed
                title_updated = False
                if new_title != current_conv_details.get('title'):
                    update_payload = {'title': new_title}
                    logging.debug(f"Attempting to update title for conv {target_conversation_id} to '{new_title}' from version {current_db_version}")
                    # update_conversation returns True on success, or raises ConflictError/DBError
                    db.update_conversation(conversation_id=target_conversation_id, update_data=update_payload, expected_version=current_db_version)
                    logging.info(f"Conversation {target_conversation_id} title updated successfully. New version will be {current_db_version + 1}.")
                    title_updated = True
                    current_db_version += 1 # Version for the conversation row is now incremented.
                    # Refresh the list view if title changed
                    await self._perform_conv_char_search() # Re-run search to update list

                # 2. Update Keywords
                existing_db_keywords = db.get_keywords_for_conversation(target_conversation_id)
                existing_keyword_texts_set = {kw['keyword'].lower() for kw in existing_db_keywords}

                ui_keyword_texts_set = {kw.strip().lower() for kw in new_keywords_str.split(',') if kw.strip()}

                keywords_to_add = ui_keyword_texts_set - existing_keyword_texts_set
                keywords_to_remove_details = [kw for kw in existing_db_keywords if kw['keyword'].lower() not in ui_keyword_texts_set]

                keywords_changed = False
                for kw_text_to_add in keywords_to_add:
                    # db.add_keyword returns keyword_id (str) for the given text (gets or creates)
                    kw_id = db.add_keyword(user_id=self.notes_user_id, keyword_text=kw_text_to_add)
                    if kw_id:
                        db.link_conversation_to_keyword(conversation_id=target_conversation_id, keyword_id=kw_id)
                        logging.debug(f"Linked keyword '{kw_text_to_add}' (ID: {kw_id}) to conv {target_conversation_id}")
                        keywords_changed = True

                for kw_detail_to_remove in keywords_to_remove_details:
                    db.unlink_conversation_from_keyword(conversation_id=target_conversation_id, keyword_id=kw_detail_to_remove['id'])
                    logging.debug(f"Unlinked keyword '{kw_detail_to_remove['keyword']}' (ID: {kw_detail_to_remove['id']}) from conv {target_conversation_id}")
                    keywords_changed = True

                if title_updated or keywords_changed:
                    logging.info(f"Details saved for conversation {target_conversation_id}. Title updated: {title_updated}, Keywords changed: {keywords_changed}")
                    # self.notify("Details saved successfully!", severity="information")
                    # Refresh keywords in UI to reflect any casing changes from add_keyword or if some failed.
                    final_keywords_list = db.get_keywords_for_conversation(target_conversation_id)
                    keywords_widget.text = ", ".join([kw['keyword'] for kw in final_keywords_list]) if final_keywords_list else ""
                else:
                    logging.info(f"No changes detected to save for conversation {target_conversation_id}.")
                    # self.notify("No changes to save.", severity="info")

            except ConflictError as e:
                logging.error(f"Conflict saving conversation details for {self.current_conv_char_tab_conversation_id}: {e}", exc_info=True)
                # self.notify(f"Save conflict: {e}. Please reload.", severity="error")
            except QueryError as e:
                logging.error(f"UI component not found for saving conv-char details: {e}", exc_info=True)
                # self.notify("Error accessing UI fields.", severity="error")
            except CharactersRAGDBError as e:
                logging.error(f"Database error saving conv-char details: {e}", exc_info=True)
                # self.notify("Database error saving details.", severity="error")
            except Exception as e:
                logging.error(f"Unexpected error saving conv-char details: {e}", exc_info=True)
                # self.notify("An unexpected error occurred.", severity="error")
            return

        # --- Tab Switching --- ────────────────────────────
        if button_id and button_id.startswith("tab-"):
            print(f">>> DEBUG: Tab button detected: {button_id}")
            new_tab_id = button_id.replace("tab-", "")
            logging.info(f"Tab button {button_id} pressed. Requesting switch to '{new_tab_id}'")
            if new_tab_id != self.current_tab:
                print(f">>> DEBUG: Changing current_tab from '{self.current_tab}' to '{new_tab_id}'")
                self.current_tab = new_tab_id
            else:
                print(f">>> DEBUG: Already on tab '{new_tab_id}'. Ignoring.")
                logging.debug(f"Already on tab '{new_tab_id}'. Ignoring.")
            return

        # ── sidebar toggle ────────────────────────────────────────────
        if button_id == "toggle-chat-sidebar":
            self.chat_sidebar_collapsed = not self.chat_sidebar_collapsed
            logging.debug("Chat sidebar now %s", "collapsed" if self.chat_sidebar_collapsed else "expanded")
            return

        if button_id == "toggle-character-sidebar":
            self.character_sidebar_collapsed = not self.character_sidebar_collapsed
            logging.debug("Character sidebar now %s", "collapsed" if self.character_sidebar_collapsed else "expanded")
            return

        if button_id == "toggle-notes-sidebar-left":
            self.notes_sidebar_left_collapsed = not self.notes_sidebar_left_collapsed
            logging.debug("Notes left sidebar now %s", "collapsed" if self.notes_sidebar_left_collapsed else "expanded")
            return

        if button_id == "toggle-notes-sidebar-right":
            self.notes_sidebar_right_collapsed = not self.notes_sidebar_right_collapsed
            logging.debug("Notes right sidebar now %s",
                          "collapsed" if self.notes_sidebar_right_collapsed else "expanded")
            return

        if button_id == "toggle-conv-char-sidebar":
            self.conv_char_sidebar_collapsed = not self.conv_char_sidebar_collapsed
            logging.debug("Conversations & Characters sidebar now %s", "collapsed" if self.conv_char_sidebar_collapsed else "expanded")
            return

        if button_id == "notes-new-button":
            if not self.notes_service:
                logging.error("Notes service not available, cannot create new note.")
                # TODO: Show user error
                return
            try:
                new_note_title = "New Note"
                # Ensure a unique title if many "New Note"s exist, or let user rename immediately
                # For now, simple title.
                new_note_id = self.notes_service.add_note(
                    user_id=self.notes_user_id,
                    title=new_note_title,
                    content=""  # Start with empty content
                )
                if new_note_id:
                    logging.info(f"New note created with ID: {new_note_id}")
                    await self.load_and_display_notes()  # Refresh list
                    # Optionally, select the new note automatically:
                    # self.current_selected_note_id = new_note_id
                    # new_note_details = self.notes_service.get_note_by_id(self.notes_user_id, new_note_id)
                    # if new_note_details:
                    #    self.current_selected_note_version = new_note_details.get('version')
                    #    self.current_selected_note_title = new_note_details.get('title')
                    #    self.current_selected_note_content = new_note_details.get('content', "")
                    #    self.query_one("#notes-editor-area", TextArea).text = self.current_selected_note_content
                    #    self.query_one("#notes-title-input", Input).value = self.current_selected_note_title
                else:
                    logging.error("Failed to create new note (ID was None).")
            except CharactersRAGDBError as e:
                logging.error(f"Database error creating new note: {e}", exc_info=True)
                # TODO: Show user error
            except Exception as e:
                logging.error(f"Unexpected error creating new note: {e}", exc_info=True)
            return

        # --- Button Functionalities for Collapsible Menu ---
        if button_id == "notes-create-new-button":  # Matches existing "notes-new-button"
            if not self.notes_service:
                logging.error("Notes service not available, cannot create new note.")
                # TODO: Show user error (e.g., self.notify)
                return
            try:
                new_note_title = "New Note"
                new_note_id = self.notes_service.add_note(
                    user_id=self.notes_user_id,
                    title=new_note_title,
                    content=""  # Start with empty content
                )
                if new_note_id:
                    logging.info(f"New note created with ID: {new_note_id} via 'notes-create-new-button'")
                    await self.load_and_display_notes()  # Refresh list
                    # Optionally, select the new note:
                    # This would involve finding the ListItem, setting current_selected_note_id,
                    # and then calling the logic from on_list_view_selected or a helper.
                    # For now, just refresh the list.
                else:
                    logging.error("Failed to create new note (ID was None) via 'notes-create-new-button'.")
            except CharactersRAGDBError as e:
                logging.error(f"Database error creating new note via 'notes-create-new-button': {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error creating new note via 'notes-create-new-button': {e}", exc_info=True)
            return

        if button_id == "notes-edit-selected-button":
            try:
                if self.current_selected_note_id:  # Only focus if a note is selected/loaded
                    self.query_one("#notes-editor-area", TextArea).focus()
                    logging.info("Focused notes editor for editing selected note.")
                else:
                    logging.info("Edit selected note button pressed, but no note is currently selected.")
                    # Optionally: self.notify("No note selected to edit.", severity="warning")
            except QueryError as e:
                logging.error(f"UI component not found for 'notes-edit-selected-button': {e}", exc_info=True)
            return

        if button_id == "notes-search-button":
            try:
                self.query_one("#notes-search-input", Input).focus()
                logging.info("Focused notes search input.")
            except QueryError as e:
                logging.error(f"UI component not found for 'notes-search-button': {e}", exc_info=True)
            return

        if button_id == "notes-load-selected-button":
            logging.info("Attempting to load selected note via button.")
            if not self.notes_service:
                logging.error("Notes service not available, cannot load selected note.")
                # self.notify("Notes service unavailable.", severity="error")
                return
            try:
                notes_list_view = self.query_one("#notes-list-view", ListView)
                selected_item = notes_list_view.highlighted_child

                if selected_item and hasattr(selected_item, 'note_id') and hasattr(selected_item, 'note_version'):
                    note_id = selected_item.note_id
                    # note_version = selected_item.note_version # We get the latest version from DB
                    logging.info(f"Re-loading note: ID={note_id} from highlighted item.")

                    note_details = self.notes_service.get_note_by_id(
                        user_id=self.notes_user_id,
                        note_id=note_id
                    )
                    if note_details:
                        self.current_selected_note_id = note_id
                        self.current_selected_note_version = note_details.get('version')
                        self.current_selected_note_title = note_details.get('title')
                        self.current_selected_note_content = note_details.get('content', "")

                        self.query_one("#notes-editor-area", TextArea).text = self.current_selected_note_content
                        self.query_one("#notes-title-input", Input).value = self.current_selected_note_title or ""

                        keywords_area = self.query_one("#notes-keywords-area", TextArea)
                        keywords_for_note = self.notes_service.get_keywords_for_note(
                            user_id=self.notes_user_id, note_id=note_id
                        )
                        keywords_area.text = ", ".join(
                            [kw['keyword'] for kw in keywords_for_note]) if keywords_for_note else ""

                        logging.info(f"Successfully re-loaded note '{self.current_selected_note_title}'.")
                        # self.notify(f"Note '{self.current_selected_note_title}' loaded.", severity="information")
                    else:
                        logging.warning(f"Could not retrieve details for note ID: {note_id} on re-load.")
                        # self.notify(f"Failed to load note ID: {note_id}.", severity="error")
                else:
                    logging.info("No item highlighted in notes list to load.")
                    # self.notify("No note selected in the list to load.", severity="warning")
            except QueryError as e:
                logging.error(f"UI component not found for 'notes-load-selected-button': {e}", exc_info=True)
            except CharactersRAGDBError as e:
                logging.error(f"Database error loading note via button: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error loading note via button: {e}", exc_info=True)
            return

        if button_id == "notes-save-current-button":
            logging.info("Save current note button pressed.")
            await self.save_current_note()  # This method already exists and handles saving
            return

        if button_id == "chat-save-conversation-details-button":
            logging.info("Save conversation details button pressed.")
            if not self.current_chat_conversation_id:
                logging.warning("No active chat conversation ID to save details for.")
                # self.notify("No active conversation to save details for.", severity="warning")
                return

            try:
                title_input = self.query_one("#chat-conversation-title-input", Input)
                new_title = title_input.value
                keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
                new_keywords = keywords_input.text
            except QueryError as e:
                logging.error(f"UI component not found for saving conversation details: {e}", exc_info=True)
                # self.notify("Error accessing UI fields for conversation details.", severity="error")
                return

            conversation_details = None
            db_instance = None
            if self.notes_service:  # notes_service provides access to DB
                try:
                    db_instance = self.notes_service._get_db(self.notes_user_id)  # Accessing protected member
                    conversation_details = db_instance.get_conversation_by_id(self.current_chat_conversation_id)
                except Exception as e:
                    logging.error(f"Failed to get conversation details for ID {self.current_chat_conversation_id}: {e}",
                                  exc_info=True)
                    # self.notify("Error fetching conversation details.", severity="error")
                    return  # Stop if we can't get details

            if not conversation_details:
                logging.error(
                    f"Could not find conversation details for ID {self.current_chat_conversation_id} to save.")
                # self.notify("Active conversation not found in database.", severity="error")
                return

            current_version = conversation_details.get('version')
            if current_version is None:
                logging.error(f"Conversation {self.current_chat_conversation_id} lacks version information.")
                # self.notify("Conversation version information is missing.", severity="error")
                return

            update_data = {}
            if new_title != conversation_details.get('title'):
                update_data['title'] = new_title
            if new_keywords != conversation_details.get('keywords'):  # Assuming 'keywords' can be None
                update_data['keywords'] = new_keywords

            if not self.current_chat_conversation_id:
                logging.warning("Save Conversation Details: No active chat conversation ID.")
                self.notify("No active conversation selected.", severity="warning", timeout=3)
                return

            conversation_id = self.current_chat_conversation_id
            logging.info(f"Saving details for conversation ID: {conversation_id}")

            try:
                title_input = self.query_one("#chat-conversation-title-input", Input)
                keywords_input_widget = self.query_one("#chat-conversation-keywords-input", TextArea)
            except QueryError as e:
                logging.error(f"Save Conversation Details: UI component not found: {e}", exc_info=True)
                self.notify("Error accessing UI fields.", severity="error", timeout=3)
                return

            new_title = title_input.value.strip()
            new_keywords_str = keywords_input_widget.text.strip()

            if not self.notes_service:
                logging.error("Save Conversation Details: Notes service not available.")
                self.notify("Database service not available.", severity="error", timeout=3)
                return

            db = self.notes_service._get_db(self.notes_user_id)

            try:
                conv_details = db.get_conversation_by_id(conversation_id)
                if not conv_details:
                    logging.error(f"Save Conversation Details: Conversation {conversation_id} not found in DB.")
                    self.notify("Error: Conversation not found.", severity="error", timeout=3)
                    return

                current_version = conv_details.get('version')
                if current_version is None:
                    logging.error(f"Save Conversation Details: Conversation {conversation_id} is missing version information.")
                    self.notify("Error: Conversation version missing.", severity="error", timeout=3)
                    return

                title_changed = False
                if new_title != conv_details.get('title'):
                    logging.debug(f"Attempting title update for {conversation_id} from '{conv_details.get('title')}' to '{new_title}' at version {current_version}.")
                    db.update_conversation(conversation_id, {'title': new_title}, current_version)
                    current_version += 1  # Increment version locally for subsequent keyword ops
                    title_changed = True
                    logging.info(f"Title updated for conversation {conversation_id}. New version: {current_version}")
                    try:
                        self.query_one(TitleBar).update_title(f"Chat - {new_title}")
                        logging.debug(f"TitleBar updated to: Chat - {new_title}")
                    except QueryError:
                        logging.error("Failed to update TitleBar after title save.")

                # Keywords Update
                all_db_keywords_list = db.get_keywords_for_conversation(conversation_id)
                db_system_keywords_set = {kw['keyword'] for kw in all_db_keywords_list if kw['keyword'].startswith("__")}
                db_user_keywords_map = {kw['keyword']: kw['id'] for kw in all_db_keywords_list if not kw['keyword'].startswith("__")}
                db_user_keywords_set = set(db_user_keywords_map.keys())

                ui_user_keywords_set = {kw.strip() for kw in new_keywords_str.split(',') if kw.strip() and not kw.strip().startswith("__")}

                keywords_to_add = ui_user_keywords_set - db_user_keywords_set
                keywords_to_remove = db_user_keywords_set - ui_user_keywords_set

                keywords_changed = False

                for keyword_text in keywords_to_add:
                    keyword_detail = db.get_keyword_by_text(keyword_text)
                    keyword_id_to_link = None
                    if not keyword_detail:
                        added_kw_id = db.add_keyword(keyword_text) # add_keyword returns int ID
                        if isinstance(added_kw_id, int):
                           keyword_id_to_link = added_kw_id
                        else:
                            logging.error(f"Failed to add keyword '{keyword_text}', received: {added_kw_id}")
                            continue # skip if add_keyword failed
                    else:
                        keyword_id_to_link = keyword_detail['id']

                    if keyword_id_to_link:
                        db.link_conversation_to_keyword(conversation_id, keyword_id_to_link)
                        logging.debug(f"Linked keyword '{keyword_text}' (ID: {keyword_id_to_link}) to conversation {conversation_id}")
                        keywords_changed = True

                for keyword_text_to_remove in keywords_to_remove:
                    keyword_id_to_unlink = db_user_keywords_map.get(keyword_text_to_remove)
                    if keyword_id_to_unlink:
                        db.unlink_conversation_from_keyword(conversation_id, keyword_id_to_unlink)
                        logging.debug(f"Unlinked keyword '{keyword_text_to_remove}' (ID: {keyword_id_to_unlink}) from conversation {conversation_id}")
                        keywords_changed = True

                if title_changed or keywords_changed:
                    self.notify("Conversation details saved!", severity="information", timeout=3)
                    logging.info(f"Details saved for conversation {conversation_id}. Title changed: {title_changed}, Keywords changed: {keywords_changed}")
                    # Refresh keywords in UI to reflect any changes (e.g. from add_keyword if casing was off)
                    final_db_keywords = db.get_keywords_for_conversation(conversation_id)
                    final_user_keywords = [kw['keyword'] for kw in final_db_keywords if not kw['keyword'].startswith("__")]
                    keywords_input_widget.text = ", ".join(final_user_keywords)
                else:
                    self.notify("No changes to save.", severity="information", timeout=2)
                    logging.info(f"No changes detected to save for conversation {conversation_id}.")

            except ConflictError as e:
                logging.error(f"Conflict saving conversation details for {conversation_id}: {e}", exc_info=True)
                self.notify(f"Save conflict: {e}. Please reload.", severity="error", timeout=5)
            except CharactersRAGDBError as e:
                logging.error(f"DB error saving conversation details for {conversation_id}: {e}", exc_info=True)
                self.notify("Database error saving details.", severity="error", timeout=3)
            except Exception as e:
                logging.error(f"Unexpected error saving conversation details for {conversation_id}: {e}", exc_info=True)
                self.notify("Unexpected error saving details.", severity="error", timeout=3)
            return

        if button_id == "chat-conversation-load-selected-button":
            logging.info("Load selected chat button pressed.")
            try:
                results_list_view = self.query_one("#chat-conversation-search-results-list", ListView)
                highlighted_item = results_list_view.highlighted_child

                if highlighted_item and hasattr(highlighted_item, 'conversation_id'):
                    loaded_conversation_id = highlighted_item.conversation_id
                    logging.info(f"Loading conversation ID: {loaded_conversation_id}")

                    self.current_chat_conversation_id = loaded_conversation_id  # Update reactive variable

                    # Fetch full conversation details (title, keywords might already be on item)
                    db = self.notes_service._get_db(self.notes_user_id)
                    conv_details = db.get_conversation_by_id(loaded_conversation_id)

                    if not conv_details:
                        logging.error(f"Failed to fetch details for conversation ID: {loaded_conversation_id}")
                        # self.notify(f"Error: Could not load details for chat {loaded_conversation_id}.", severity="error")
                        return

                    # Populate title and keywords fields
                    try:
                        title_input = self.query_one("#chat-conversation-title-input", Input)
                        title_input.value = conv_details.get('title', '')
                        logging.debug(f"Set #chat-conversation-title-input to: {conv_details.get('title', '')}")
                    except QueryError:
                        logging.error("Failed to find #chat-conversation-title-input to update.")

                    try:
                        uuid_display = self.query_one("#chat-conversation-uuid-display", Input)
                        uuid_display.value = loaded_conversation_id
                        logging.debug(f"Set #chat-conversation-uuid-display to: {loaded_conversation_id}")
                    except QueryError:
                        logging.error("Failed to find #chat-conversation-uuid-display to update.")

                    try:
                        keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
                        all_keywords_list = db.get_keywords_for_conversation(loaded_conversation_id)
                        visible_keywords = [kw['keyword'] for kw in all_keywords_list if not kw['keyword'].startswith("__")]
                        keywords_input.text = ", ".join(visible_keywords)
                        logging.debug(f"Set #chat-conversation-keywords-input to: {', '.join(visible_keywords)}")
                    except CharactersRAGDBError as e_kw_db:
                        logging.error(f"Database error fetching keywords for conversation {loaded_conversation_id}: {e_kw_db}", exc_info=True)
                        if 'keywords_input' in locals(): keywords_input.text = "Error loading keywords."
                    except QueryError:
                        logging.error("Failed to find #chat-conversation-keywords-input to update.")
                    except Exception as e_kw_gen:
                        logging.error(f"Unexpected error processing keywords for conversation {loaded_conversation_id}: {e_kw_gen}", exc_info=True)
                        if 'keywords_input' in locals(): keywords_input.text = "Unexpected error loading keywords."


                    # Update the main TitleBar
                    try:
                        title_bar = self.query_one(TitleBar)
                        title_bar.update_title(f"Chat - {conv_details.get('title', 'Untitled Conversation')}")
                        logging.debug(f"Updated TitleBar title for conversation {loaded_conversation_id}")
                    except QueryError:
                        logging.error("Failed to find TitleBar to update.")
                    except Exception as e_title_bar:
                        logging.error(f"Unexpected error updating TitleBar: {e_title_bar}", exc_info=True)
                    # Clear existing messages from chat log
                    chat_log = self.query_one("#chat-log", VerticalScroll)
                    await chat_log.remove_children()
                    logging.debug(f"Cleared #chat-log for new conversation {loaded_conversation_id}.")

                    # Fetch and display messages for this conversation
                    messages = db.get_messages_for_conversation(loaded_conversation_id, order_by_timestamp="ASC",
                                                                limit=1000)  # Adjust limit as needed
                    logging.debug(f"Fetched {len(messages)} messages for conversation {loaded_conversation_id}.")

                    for msg_data in messages:
                        # Ensure image_data is handled correctly (it might be None or bytes)
                        image_data_for_widget = msg_data.get('image_data')
                        # logging.debug(f"Message {msg_data['id']} has image_data type: {type(image_data_for_widget)}")

                        chat_message_widget = ChatMessage(
                            message=msg_data['content'],
                            role=msg_data['sender'],
                            timestamp=msg_data.get('timestamp'),  # Ensure timestamp is passed if available
                            image_data=image_data_for_widget,  # Pass image_data
                            image_mime_type=msg_data.get('image_mime_type'),  # Pass mime_type
                            message_id=msg_data['id']  # Pass message_id
                        )
                        await chat_log.mount(chat_message_widget)

                    chat_log.scroll_end(animate=False)  # Scroll to the latest message

                    # Focus the main chat input area
                    self.query_one("#chat-input", TextArea).focus()
                    logging.info(f"Successfully loaded conversation {loaded_conversation_id} into chat view.")
                    # self.notify(f"Chat '{conv_details.get('title', 'Untitled')}' loaded.", severity="information")

                else:
                    logging.info("No conversation selected in the list to load.")
                    # self.notify("No chat selected to load.", severity="warning")
            except QueryError as e:
                logging.error(f"UI component not found for loading chat: {e}", exc_info=True)
                # self.notify("Error accessing UI for loading chat.", severity="error")
            except CharactersRAGDBError as e:
                logging.error(f"Database error loading chat: {e}", exc_info=True)
                # self.notify("Database error loading chat.", severity="error")
            except Exception as e:
                logging.error(f"Unexpected error loading chat: {e}", exc_info=True)
                # self.notify("Unexpected error loading chat.", severity="error")
            return

        # --- Copy Logs Button ---
        if button_id == "copy-logs-button":
            logging.info("Copy logs button pressed.")
            try:
                log_widget = self.query_one("#app-log-display", RichLog)
                if log_widget.lines:
                    all_log_text_parts = []
                    for i, line_item in enumerate(log_widget.lines):
                        if isinstance(line_item, Strip):
                            # The .text property of a Strip object returns its plain text content
                            all_log_text_parts.append(line_item.text)
                        else:
                            # This case should ideally not be hit if RichLog.lines behaves as expected
                            # (i.e., only contains Strip objects from wrapped Text).
                            logging.warning(
                                f"Item {i} in RichLog.lines is of unexpected type: {type(line_item)}. "
                                f"Falling back to str() for this line: {repr(line_item)}"
                            )
                            all_log_text_parts.append(str(line_item))

                    all_log_text = "\n".join(all_log_text_parts)

                    self.copy_to_clipboard(all_log_text)
                    self.notify(
                        "Logs copied to clipboard!",
                        title="Clipboard",
                        severity="information",
                        timeout=4
                    )
                    logging.debug(
                        f"Copied {len(log_widget.lines)} lines ({len(all_log_text)} chars) from RichLog to clipboard.")
                else:
                    self.notify("Log is empty, nothing to copy.", title="Clipboard", severity="warning", timeout=4)
            except QueryError:
                self.notify("Log widget not found. Cannot copy.", title="Error", severity="error", timeout=4)
                logging.error("Could not find #app-log-display to copy logs.")
            except AttributeError as ae:  # Specifically catch if .text is missing on an unexpected object
                self.notify(f"Error processing log line: {str(ae)}", title="Error", severity="error", timeout=6)
                logging.error(f"AttributeError while processing RichLog lines: {ae}", exc_info=True)
            except Exception as e:
                self.notify(f"Error copying logs: {str(e)}", title="Error", severity="error", timeout=6)
                logging.error(f"Failed to copy logs: {e}", exc_info=True)
            return
        # --- End of New Button Functionalities ---

        if button_id == "notes-save-button":  # Existing save button in the notes control area
            await self.save_current_note()
            return

        if button_id == "notes-delete-button":  # Existing delete button
            if not self.notes_service or not self.current_selected_note_id or self.current_selected_note_version is None:
                logging.warning("No note selected to delete.")
                # self.notify("No note selected to delete.", severity="warning")
                return

            # Basic confirmation (logging only, no UI dialog for this subtask)
            logging.info(
                f"Attempting to delete note ID: {self.current_selected_note_id}, Version: {self.current_selected_note_version}")

            # TODO: Implement a proper confirmation dialog here in a future step.
            # confirmed = await self.app.push_screen_wait(ConfirmDeleteDialog(self.current_selected_note_title or "this note"))
            # if not confirmed: return

            try:
                success = self.notes_service.soft_delete_note(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id,
                    expected_version=self.current_selected_note_version
                )
                if success:
                    logging.info(f"Note {self.current_selected_note_id} soft-deleted successfully.")
                    # self.notify("Note deleted.", severity="information")

                    # Clear selection and UI
                    self.current_selected_note_id = None
                    self.current_selected_note_version = None
                    self.current_selected_note_title = ""  # Update reactive
                    self.current_selected_note_content = ""  # Update reactive

                    self.query_one("#notes-editor-area", TextArea).text = ""
                    self.query_one("#notes-title-input", Input).value = ""
                    self.query_one("#notes-keywords-area", TextArea).text = ""  # Clear keywords too

                    await self.load_and_display_notes()  # Refresh list in left sidebar
                else:
                    # This path should ideally not be reached if soft_delete_note raises exceptions on failure.
                    logging.warning(
                        f"notes_service.soft_delete_note for {self.current_selected_note_id} returned False.")
                    # self.notify("Failed to delete note (unknown reason).", severity="error")

            except ConflictError as e:
                logging.error(f"Conflict deleting note {self.current_selected_note_id}: {e}", exc_info=True)
                # self.notify(f"Delete conflict: {e}. Note may have been changed or deleted by another process.", severity="error")
                # Potentially refresh list if conflict occurred
                await self.load_and_display_notes()
            except CharactersRAGDBError as e:
                logging.error(f"Database error deleting note {self.current_selected_note_id}: {e}", exc_info=True)
                # self.notify("Error deleting note from database.", severity="error")
            except QueryError as e:
                logging.error(f"UI component not found while deleting note: {e}", exc_info=True)
                # self.notify("UI error while deleting note.", severity="error")
            except Exception as e:
                logging.error(f"Unexpected error deleting note {self.current_selected_note_id}: {e}", exc_info=True)
                # self.notify("Unexpected error deleting note.", severity="error")
            return

        if button_id == "notes-save-keywords-button":
            if not self.notes_service or not self.current_selected_note_id:
                logging.warning("No note selected or service unavailable. Cannot save keywords.")
                # self.notify("No note selected to save keywords for.", severity="warning")
                return

            try:
                keywords_area = self.query_one("#notes-keywords-area", TextArea)
                input_keyword_texts = {kw.strip().lower() for kw in keywords_area.text.split(',') if kw.strip()}

                logging.info(
                    f"Attempting to save keywords for note {self.current_selected_note_id}. Input: {input_keyword_texts}")

                # Get existing keyword links for the note
                existing_linked_keywords_data = self.notes_service.get_keywords_for_note(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id
                )
                existing_linked_keyword_map = {kw['keyword'].lower(): kw['id'] for kw in existing_linked_keywords_data}

                processed_keyword_ids = set()

                # Process input keywords: get/create IDs
                for kw_text in input_keyword_texts:
                    if not kw_text: continue  # Skip empty strings after split/strip

                    keyword_detail = self.notes_service.get_keyword_by_text(self.notes_user_id, kw_text)
                    if not keyword_detail:  # Keyword doesn't exist globally for this user
                        logging.debug(f"Keyword '{kw_text}' not found globally, creating it.")
                        new_kw_id = self.notes_service.add_keyword(self.notes_user_id, kw_text)
                        if new_kw_id is None:  # Should not happen if add_keyword raises on error
                            logging.error(f"Failed to create new keyword '{kw_text}', skipping.")
                            continue
                        processed_keyword_ids.add(new_kw_id)
                        logging.info(f"Created new keyword '{kw_text}' with ID {new_kw_id}.")
                    else:  # Keyword exists globally
                        processed_keyword_ids.add(keyword_detail['id'])

                # Link new keywords
                for kw_id in processed_keyword_ids:
                    # Check if this keyword_id is among those already linked (by comparing IDs, not text)
                    is_already_linked = any(
                        existing_kw_data['id'] == kw_id for existing_kw_data in existing_linked_keywords_data)
                    if not is_already_linked:
                        self.notes_service.link_note_to_keyword(
                            user_id=self.notes_user_id,
                            note_id=self.current_selected_note_id,
                            keyword_id=kw_id
                        )
                        logging.debug(f"Linked keyword ID {kw_id} to note {self.current_selected_note_id}")

                # Unlink keywords that were removed
                for existing_kw_text, existing_kw_id in existing_linked_keyword_map.items():
                    if existing_kw_text not in input_keyword_texts:  # Compare by lowercased text
                        self.notes_service.unlink_note_from_keyword(
                            user_id=self.notes_user_id,
                            note_id=self.current_selected_note_id,
                            keyword_id=existing_kw_id
                        )
                        logging.debug(
                            f"Unlinked keyword ID {existing_kw_id} ('{existing_kw_text}') from note {self.current_selected_note_id}")

                # Refresh the displayed keywords
                refreshed_keywords_data = self.notes_service.get_keywords_for_note(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id
                )
                keywords_area.text = ", ".join([kw['keyword'] for kw in refreshed_keywords_data])
                # self.notify("Keywords saved successfully!", severity="information")
                logging.info(f"Keywords for note {self.current_selected_note_id} updated and refreshed.")

            except CharactersRAGDBError as e:
                logging.error(f"Database error saving keywords for note {self.current_selected_note_id}: {e}",
                              exc_info=True)
                # self.notify("Error saving keywords.", severity="error")
            except QueryError as e:
                logging.error(f"UI component #notes-keywords-area not found: {e}", exc_info=True)
                # self.notify("UI error while saving keywords.", severity="error")
            except Exception as e:
                logging.error(f"Unexpected error saving keywords for note {self.current_selected_note_id}: {e}",
                              exc_info=True)
                # self.notify("Unexpected error saving keywords.", severity="error")
            return

        # --- New Chat Button ---
        elif event.button.id == "chat-new-conversation-button":
            logging.info("New Chat button pressed.")
            try:
                new_chat_title = f"Untitled_Chat-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
                logging.debug(f"Generated new chat title: {new_chat_title}")

                if not self.notes_service:
                    logging.error("Notes service is not available. Cannot create new chat.")
                    # self.notify("Database service not available.", severity="error") # If notifier is set up
                    return

                db = self.notes_service._get_db(self.notes_user_id)
                new_conversation_id = None
                try:
                    new_conversation_id = db.add_conversation({'title': new_chat_title, 'character_id': None})
                    if not new_conversation_id:
                        logging.error("db.add_conversation returned None for new chat.")
                        # self.notify("Failed to create new conversation in database.", severity="error")
                        return
                    logging.info(f"New conversation created with ID: {new_conversation_id} and title: {new_chat_title}")
                except ConflictError as e:
                    logging.error(f"Conflict error creating new conversation: {e}", exc_info=True)
                    # self.notify("Failed to create conversation: Conflict.", severity="error")
                    return
                except CharactersRAGDBError as e:
                    logging.error(f"Database error creating new conversation: {e}", exc_info=True)
                    # self.notify("Database error creating conversation.", severity="error")
                    return
                except Exception as e:
                    logging.error(f"Unexpected error creating new conversation: {e}", exc_info=True)
                    # self.notify("Unexpected error creating conversation.", severity="error")
                    return

                self.current_chat_conversation_id = new_conversation_id
                logging.debug(f"Set current_chat_conversation_id to: {new_conversation_id}")

                system_keywords = ["__regular_chat", "user_chat"]
                for keyword_text in system_keywords:
                    keyword_id = None
                    try:
                        keyword_details = db.get_keyword_by_text(keyword_text)
                        if keyword_details and 'id' in keyword_details:
                            keyword_id = keyword_details['id']
                            logging.debug(f"Found existing keyword '{keyword_text}' with ID: {keyword_id}")
                        else:
                            # add_keyword in ChaChaNotes_DB handles undeletion or creation.
                            # It's expected to return an int ID.
                            keyword_id_result = db.add_keyword(keyword_text) # Pass only keyword_text
                            if isinstance(keyword_id_result, int):
                                keyword_id = keyword_id_result
                                logging.info(f"Added/Undeleted keyword '{keyword_text}' with ID: {keyword_id}")
                            else: # Should not happen if add_keyword works as expected
                                logging.error(f"Failed to add/get keyword '{keyword_text}'. Result: {keyword_id_result}")
                                continue # Skip linking this keyword
                    except CharactersRAGDBError as e:
                        logging.error(f"Database error getting/adding keyword '{keyword_text}': {e}", exc_info=True)
                        continue # Skip linking this keyword
                    except Exception as e:
                        logging.error(f"Unexpected error processing keyword '{keyword_text}': {e}", exc_info=True)
                        continue # Skip linking this keyword

                    if keyword_id:
                        try:
                            db.link_conversation_to_keyword(new_conversation_id, keyword_id)
                            logging.debug(f"Linked keyword '{keyword_text}' (ID: {keyword_id}) to conversation {new_conversation_id}")
                        except CharactersRAGDBError as e:
                            logging.error(f"Database error linking keyword '{keyword_text}' to conversation {new_conversation_id}: {e}", exc_info=True)
                        except Exception as e:
                            logging.error(f"Unexpected error linking keyword '{keyword_text}': {e}", exc_info=True)

                # Update UI elements
                try:
                    title_input_widget = self.query_one("#chat-conversation-title-input", Input)
                    title_input_widget.value = new_chat_title
                    logging.debug(f"Set #chat-conversation-title-input to: {new_chat_title}")
                except QueryError:
                    logging.error("Failed to find #chat-conversation-title-input to update.")

                try:
                    keywords_input_widget = self.query_one("#chat-conversation-keywords-input", TextArea)
                    keywords_input_widget.text = "" # System keywords are hidden
                    logging.debug("Cleared #chat-conversation-keywords-input.")
                except QueryError:
                    logging.error("Failed to find #chat-conversation-keywords-input to clear.")

                try:
                    uuid_display_widget = self.query_one("#chat-conversation-uuid-display", Input)
                    uuid_display_widget.value = new_conversation_id
                    logging.debug(f"Set #chat-conversation-uuid-display to: {new_conversation_id}")
                except QueryError:
                    logging.error("Failed to find #chat-conversation-uuid-display to update.")

                try:
                    chat_log_widget = self.query_one("#chat-log", VerticalScroll)
                    await chat_log_widget.remove_children()
                    logging.debug("Cleared #chat-log.")
                except QueryError:
                    logging.error("Failed to find #chat-log to clear messages.")

                try:
                    title_bar_widget = self.query_one(TitleBar)
                    title_bar_widget.update_title(f"Chat - {new_chat_title}")
                    logging.debug(f"Updated TitleBar title to: Chat - {new_chat_title}")
                except QueryError:
                    logging.error("Failed to find TitleBar to update title.")

                try:
                    chat_input_widget = self.query_one("#chat-input", TextArea)
                    chat_input_widget.focus()
                    logging.debug("Focused #chat-input.")
                except QueryError:
                    logging.error("Failed to find #chat-input to focus.")

                logging.info(f"Successfully initialized new chat session: {new_conversation_id}")

            except Exception as e: # Catch-all for unexpected issues during the "New Chat" process
                logging.error(f"An unexpected error occurred during 'New Chat' button processing: {e}", exc_info=True)
                # self.notify("An unexpected error occurred while starting a new chat.", severity="error")
            return

        # --- Send Message ---
        if button_id and button_id.startswith("send-"):
            chat_id_part = button_id.replace("send-", "")
            prefix = chat_id_part
            logging.info(f"Send button pressed for '{chat_id_part}'")

            if prefix == "chat":
                logging.debug(f"SENDBUTTON: Querying widgets for prefix '{prefix}'.")
                try:
                    # Check sidebar
                    chat_sidebar_check = self.query_one("#chat-sidebar")
                    logging.debug(f"SENDBUTTON: #chat-sidebar.is_mounted={chat_sidebar_check.is_mounted}, .display={chat_sidebar_check.display}")
                    # Try to query #chat-top-p from sidebar
                    # Ensure the ID is correct, e.g., f"#{prefix}-top-p" if that's how it's constructed
                    chat_top_p_from_sidebar = chat_sidebar_check.query_one(f"#{prefix}-top-p") # Using prefix here
                    logging.debug(f"SENDBUTTON: #{prefix}-top-p from sidebar.is_mounted={chat_top_p_from_sidebar.is_mounted}, .display={chat_top_p_from_sidebar.display}")
                except QueryError as qe_diag_sidebar:
                    logging.debug(f"SENDBUTTON: Diagnostic query for #chat-sidebar or #{prefix}-top-p from it failed: {str(qe_diag_sidebar)}")
                try:
                    # Try to query #chat-top-p directly from app
                    # Ensure the ID is correct, e.g., f"#{prefix}-top-p"
                    chat_top_p_direct_check = self.query_one(f"#{prefix}-top-p") # Using prefix here
                    logging.debug(f"SENDBUTTON: #{prefix}-top-p direct query OK. .is_mounted={chat_top_p_direct_check.is_mounted}, .display={chat_top_p_direct_check.display}")
                except QueryError as qe_diag_direct:
                    logging.debug(f"SENDBUTTON: Diagnostic query for #{prefix}-top-p direct failed: {str(qe_diag_direct)}")

            # --- Query Widgets ---
            try:
                text_area = self.query_one(f"#{prefix}-input", TextArea)
                chat_container = self.query_one(f"#{prefix}-log", VerticalScroll)
                provider_widget = self.query_one(f"#{prefix}-api-provider",
                                                 Select)  # e.g., #chat-api-provider or #conversations_characters-api-provider
                model_widget = self.query_one(f"#{prefix}-api-model", Select)
                system_prompt_widget = self.query_one(f"#{prefix}-system-prompt", TextArea)
                temp_widget = self.query_one(f"#{prefix}-temperature", Input)
                top_p_widget = self.query_one(f"#{prefix}-top-p", Input)
                min_p_widget = self.query_one(f"#{prefix}-min-p", Input)
                top_k_widget = self.query_one(f"#{prefix}-top-k", Input)

                # --- Query "Full Chat Settings" widgets ---
                llm_max_tokens_widget = self.query_one(f"#{prefix}-llm-max-tokens", Input)
                llm_seed_widget = self.query_one(f"#{prefix}-llm-seed", Input)
                llm_stop_widget = self.query_one(f"#{prefix}-llm-stop", Input)
                llm_response_format_widget = self.query_one(f"#{prefix}-llm-response-format", Select)
                llm_n_widget = self.query_one(f"#{prefix}-llm-n", Input)
                llm_user_identifier_widget = self.query_one(f"#{prefix}-llm-user-identifier", Input)
                llm_logprobs_widget = self.query_one(f"#{prefix}-llm-logprobs", Checkbox)
                llm_top_logprobs_widget = self.query_one(f"#{prefix}-llm-top-logprobs", Input)
                llm_logit_bias_widget = self.query_one(f"#{prefix}-llm-logit-bias", TextArea)
                llm_presence_penalty_widget = self.query_one(f"#{prefix}-llm-presence-penalty", Input)
                llm_frequency_penalty_widget = self.query_one(f"#{prefix}-llm-frequency-penalty", Input)
                llm_tools_widget = self.query_one(f"#{prefix}-llm-tools", TextArea)
                llm_tool_choice_widget = self.query_one(f"#{prefix}-llm-tool-choice", Input)
            except QueryError as e:
                logging.error(f"Send Button: Could not find UI widgets for '{prefix}': {e}")
                # Optionally mount an error message in the chat_container
                await chat_container.mount(
                    ChatMessage(f"Internal Error: Missing UI elements for {prefix}.", role="AI", classes="-error"))
                return
            except Exception as e:
                logging.error(f"Send Button: Unexpected error querying widgets for '{prefix}': {e}")
                await chat_container.mount(
                    ChatMessage("Internal Error: Could not query UI elements.", role="AI", classes="-error"))
                return

            # --- Get Values ---
            message = text_area.text.strip()
            if not message:
                try:
                    # look at the very last ChatMessage in the container
                    last_msg_widget: ChatMessage | None = None
                    for last_msg_widget in reversed(chat_container.query(ChatMessage)):
                        # break on the first real message (skip system banners etc. if any)
                        if last_msg_widget.role in ("User", "AI"):
                            break

                    if last_msg_widget and last_msg_widget.role == "User":
                        # reuse its text as the message to send
                        message = last_msg_widget.message_text
                        # DON'T mount a duplicate user bubble later
                        reuse_last_user_bubble = True
                    else:
                        reuse_last_user_bubble = False
                except Exception as exc:
                    logging.error("Failed to inspect last message: %s", exc, exc_info=True)
                    reuse_last_user_bubble = False
            else:
                reuse_last_user_bubble = False
            selected_provider = str(provider_widget.value) if provider_widget.value else None
            selected_model = str(model_widget.value) if model_widget.value else None
            system_prompt = system_prompt_widget.text
            temperature = self._safe_float(temp_widget.value, 0.7, "temperature")
            top_p = self._safe_float(top_p_widget.value, 0.95, "top_p")
            min_p = self._safe_float(min_p_widget.value, 0.05, "min_p")
            top_k = self._safe_int(top_k_widget.value, 50, "top_k")
            custom_prompt = ""
            should_stream = False  # Placeholder, actual streaming logic might differ

            # --- Retrieve and process "Full Chat Settings" values ---
            llm_max_tokens_value = self._safe_int(llm_max_tokens_widget.value, 1024, "llm_max_tokens")
            llm_seed_value = self._safe_int(llm_seed_widget.value, None, "llm_seed")  # API handles None if not set
            llm_stop_value = llm_stop_widget.value.split(',') if llm_stop_widget.value.strip() else None
            # llm_response_format_widget.value is guaranteed by allow_blank=False in settings_sidebar.py
            llm_response_format_value = {"type": str(llm_response_format_widget.value)}
            llm_n_value = self._safe_int(llm_n_widget.value, 1, "llm_n")
            llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
            llm_logprobs_value = llm_logprobs_widget.value # Boolean
            llm_top_logprobs_value = self._safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs") # Default 0 if logprobs is False
            llm_presence_penalty_value = self._safe_float(llm_presence_penalty_widget.value, 0.0, "llm_presence_penalty")
            llm_frequency_penalty_value = self._safe_float(llm_frequency_penalty_widget.value, 0.0, "llm_frequency_penalty")
            llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None

            try:
                llm_logit_bias_text = llm_logit_bias_widget.text.strip()
                llm_logit_bias_value = json.loads(llm_logit_bias_text) if llm_logit_bias_text else None
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in llm_logit_bias: '{llm_logit_bias_widget.text}'")
                await chat_container.mount(ChatMessage("Error: Invalid JSON in LLM Logit Bias setting. Parameter not used.", role="AI", classes="-error"))
                llm_logit_bias_value = None # Default to None if JSON is invalid

            try:
                llm_tools_text = llm_tools_widget.text.strip()
                llm_tools_value = json.loads(llm_tools_text) if llm_tools_text else None
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in llm_tools: '{llm_tools_widget.text}'")
                await chat_container.mount(ChatMessage("Error: Invalid JSON in LLM Tools setting. Parameter not used.", role="AI", classes="-error"))
                llm_tools_value = None # Default to None if JSON is invalid

            # --- Basic Validation ---
            if not message:
                logging.debug("Empty message and no reusable user bubble in '%s'.", prefix)
                text_area.focus()
                return
            if not selected_provider: await chat_container.mount(
                ChatMessage("Please select an API Provider.", role="AI", classes="-error")); return
            if not selected_model: await chat_container.mount(
                ChatMessage("Please select a Model.", role="AI", classes="-error")); return
            # Check if API functions loaded (check the flag)
            if not API_IMPORTS_SUCCESSFUL:
                await chat_container.mount(
                    ChatMessage("Error: Core API functions failed to load. Cannot send message.", role="AI",
                                classes="-error"))
                logging.error("Attempted to send message, but API imports failed.")
                return

            # --- Build History ---
            chat_history = []  # This will be a list of dictionaries
            try:
                message_widgets = chat_container.query(ChatMessage)
                for msg_widget in message_widgets:
                    # Only include User and AI messages that are complete
                    if msg_widget.role in ("User", "AI") and msg_widget.generation_complete:
                        # Use 'assistant' for AI role if that's what your backend/API expects
                        # Or map 'AI' to 'assistant' if needed in Chat_Functions.py
                        role_for_api = "assistant" if msg_widget.role == "AI" else "user"
                        chat_history.append({"role": role_for_api, "content": msg_widget.message_text})

                logging.debug(f"Built chat history with {len(chat_history)} messages (alternating roles).")
            except Exception as e:
                logging.error(f"Failed to build chat history: {e}", exc_info=True)
                await chat_container.mount(
                    ChatMessage("Internal Error: Could not retrieve chat history.", role="AI", classes="-error"))
                return

            # --- Mount User Message ---
            if not reuse_last_user_bubble:
                user_msg_widget = ChatMessage(message, role="User")
                await chat_container.mount(user_msg_widget)

            # --- Conversation ID Management (Conceptual for new chat) ---
            if prefix == "chat" and not self.current_chat_conversation_id:
                # This is likely the first message in a new chat.
                # FIXME - Generate Chat Conversation ID Here
                # Example: self.current_chat_conversation_id = self.notes_service._get_db(self.notes_user_id)._generate_uuid()
                #          (if creating before API call)
                # For this task, the save button relies on it being set.
                # We also need to load/clear the title/keywords input fields.
                logging.info(
                    f"New chat detected or conversation ID not set. Current ID: {self.current_chat_conversation_id}")
                try:
                    title_input = self.query_one("#chat-conversation-title-input", Input)
                    title_input.value = ""  # Clear for new chat
                    keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
                    keywords_input.text = ""  # Clear for new chat
                    # Potentially update TitleBar if a new chat implies a generic title
                    title_bar = self.query_one(TitleBar)
                    title_bar.reset_title()
                except QueryError:
                    logging.error("Could not clear title/keyword inputs for new chat.")

            chat_container.scroll_end(animate=True)
            text_area.clear()
            text_area.focus()

            # --- DEBUG Check for Config loading ---
            logging.debug(f"SENDBUTTON: self.app_config keys: {list(self.app_config.keys())}")
            if "api_settings" in self.app_config:
                logging.debug(
                    f"SENDBUTTON: self.app_config['api_settings'] keys: {list(self.app_config['api_settings'].keys())}")
                if "openai" in self.app_config["api_settings"]:
                    logging.debug(
                        f"SENDBUTTON: self.app_config['api_settings']['openai'] content: {self.app_config['api_settings']['openai']}")
                else:
                    logging.debug(f"SENDBUTTON: 'openai' key NOT FOUND in self.app_config['api_settings']")
            else:
                logging.debug(f"SENDBUTTON: 'api_settings' key NOT FOUND in self.app_config")
            # --- Prepare and Dispatch API Call via Worker calling chat_wrapper ---
            # Fetch API key (adjust based on your actual key management)
            api_key_for_call = None
            config_key_found = False
            env_key_found = False

            # 1. Get provider-specific settings from the loaded config
            provider_settings_key = selected_provider.lower()  # e.g., "openai", "anthropic"
            # Access the already loaded config (self.app_config or load_settings() again if needed)
            # Assuming self.app_config holds the merged config dictionary
            provider_settings = self.app_config.get("api_settings", {}).get(provider_settings_key, {})

            direct_config_key_checked = False
            direct_config_key_empty = False

            if provider_settings:
                if "api_key" in provider_settings:
                    direct_config_key_checked = True
                    config_api_key = provider_settings.get("api_key", "").strip()
                    if config_api_key:
                        api_key_for_call = config_api_key
                        logging.debug(
                            f"Using API key for '{selected_provider}' from config file field [api_settings.{provider_settings_key}].api_key.")
                    else:
                        direct_config_key_empty = True # Mark that 'api_key' was present but empty
                        logging.debug(
                            f"Config field [api_settings.{provider_settings_key}].api_key for '{selected_provider}' is present but empty.")

                if not api_key_for_call: # If not found via direct 'api_key' field or field was empty/missing
                    env_var_name = provider_settings.get("api_key_env_var", "").strip()
                    if env_var_name:
                        env_api_key = os.environ.get(env_var_name, "").strip()
                        if env_api_key:
                            api_key_for_call = env_api_key
                            env_key_found = True
                            logging.debug(
                                f"Using API key for '{selected_provider}' from environment variable '{env_var_name}' (specified in config).")
                        else:
                            logging.debug(
                                f"Environment variable '{env_var_name}' for '{selected_provider}' not found or empty.")
                    else:
                        logging.debug(f"No 'api_key_env_var' specified for '{selected_provider}' in config.")
            else:
                logging.warning(
                    f"No [api_settings.{provider_settings_key}] section found in config for '{selected_provider}'. Cannot check for configured API key or ENV variable name.")

            # 4. Handle case where no key was found (neither config nor ENV)
            if not api_key_for_call:
                logging.warning(
                    f"API Key for '{selected_provider}' not found in config file or specified environment variable.")
                # Define known cloud providers requiring keys
                providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq",
                                           "Cohere", "OpenRouter", "HuggingFace", "DeepSeek"]  # Add any others
                # Check if the selected provider requires a key
                if selected_provider in providers_requiring_key:
                    logging.error(f"API call aborted: API Key for required provider '{selected_provider}' is missing.")

                    error_message_parts = [f"API Key for {selected_provider} could not be found."]

                    if provider_settings:
                        config_field_checked_msg = ""
                        if direct_config_key_checked:
                            if direct_config_key_empty:
                                config_field_checked_msg = f"The 'api_key' field in your config (under [api_settings.{provider_settings_key}]) was found but is empty."
                            # If it was checked and NOT empty, api_key_for_call would be set, so we wouldn't be here.
                        else: # Direct 'api_key' field was not present in provider_settings
                            config_field_checked_msg = f"The 'api_key' field was not found in your config under [api_settings.{provider_settings_key}]."
                        error_message_parts.append(config_field_checked_msg)

                        env_var_name_from_config = provider_settings.get("api_key_env_var", "").strip()
                        if env_var_name_from_config:
                            error_message_parts.append(f"The environment variable '{env_var_name_from_config}' (specified in config) was also checked and was not found or is empty.")
                            error_message_parts.append(f"\nPlease set the 'api_key' field in your config or the '{env_var_name_from_config}' environment variable.")
                        else: # No api_key_env_var configured for this provider
                            error_message_parts.append(f"No 'api_key_env_var' was specified in your config for this provider.")
                            error_message_parts.append(f"\nPlease set the 'api_key' field in your config under [api_settings.{provider_settings_key}].")
                    else: # No provider_settings section at all for this provider
                        error_message_parts.append(
                            f"\nThe configuration section [api_settings.{provider_settings_key}] for {selected_provider} is missing entirely from your config file."
                        )
                        error_message_parts.append(
                            f"Please add this section and specify how to obtain the API key (e.g., using an 'api_key' field or an 'api_key_env_var' field)."
                        )

                    error_message_markup = (
                        f"API Key for {selected_provider} is missing.\n\n"
                        "Please add it to your config file under:\n"
                        f"\\[api_settings.{provider_settings_key}\\]\n"  # Python sees \\ -> \, Rich sees \[
                        "api_key = \"YOUR_KEY\"\n\n"
                        "Or set the environment variable specified in 'api_key_env_var'."
                    )
                    await chat_container.mount(
                        ChatMessage(Text.from_markup(error_message_markup),
                                    role="AI",
                                    classes="-error"
                        )
                    )
                    # Remove the placeholder "thinking" message if it exists
                    if self.current_ai_message_widget and self.current_ai_message_widget.is_mounted:
                        await self.current_ai_message_widget.remove()
                        self.current_ai_message_widget = None
                    return  # Stop processing
                else:
                    # Assume it's a local model or one not needing a key
                    logging.info(
                        f"Proceeding without API key for provider '{selected_provider}' (assumed local/no key required).")
                    api_key_for_call = None  # Explicitly ensure it's None

            # --- Mount Placeholder AI Message ---
            #ai_placeholder_widget = ChatMessage(message="AI thinking...", role="AI", generation_complete=False)
            ai_placeholder_widget = ChatMessage(
                message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}",
                role="AI",
                generation_complete=False
            )
            await chat_container.mount(ai_placeholder_widget)
            chat_container.scroll_end(animate=False)
            self.current_ai_message_widget = ai_placeholder_widget  # Store reference

            # --- Define Worker Target using chat_wrapper ---
            # Use a lambda to capture current values and pass them to the wrapper
            # Note: We pass the UI parameter values directly. chat_wrapper calls chat(), which calls chat_api_call(),
            # and chat_api_call handles the mapping (e.g., top_p vs maxp vs topp).
            worker_target = lambda: self.chat_wrapper(
                message=message,
                history=chat_history,
                # --- Pass other necessary params ---
                api_endpoint=selected_provider,
                api_key=api_key_for_call,
                custom_prompt=custom_prompt,  # Pass the UI custom prompt
                temperature=temperature,
                system_message=system_prompt,
                streaming=should_stream,
                minp=min_p,
                model=selected_model,
                topp=top_p,
                topk=top_k,
                # --- Pass new "Full Chat Settings" values ---
                llm_max_tokens=llm_max_tokens_value,
                llm_seed=llm_seed_value,
                llm_stop=llm_stop_value,
                llm_response_format=llm_response_format_value,
                llm_n=llm_n_value,
                llm_user_identifier=llm_user_identifier_value,
                llm_logprobs=llm_logprobs_value,
                llm_top_logprobs=llm_top_logprobs_value,
                llm_logit_bias=llm_logit_bias_value,
                llm_presence_penalty=llm_presence_penalty_value,
                llm_frequency_penalty=llm_frequency_penalty_value,
                llm_tools=llm_tools_value,
                llm_tool_choice=llm_tool_choice_value,
                # --- Existing parameters for chatdict etc. ---
                media_content={}, # Placeholder for now
                selected_parts=[], # Placeholder for now
                chatdict_entries=None, # Placeholder for now
                max_tokens=500,  # This is the existing chatdict max_tokens
                strategy="sorted_evenly" # Default or get from config/UI
            )

            # --- Run Worker ---
            logging.debug(f"Running worker 'API_Call_{prefix}' to execute chat_wrapper")
            self.run_worker(
                worker_target,
                name=f"API_Call_{prefix}",
                group="api_calls",
                thread=True,  # Run in a separate thread
                description=f"Calling {selected_provider}"
            )
            return  # Finished handling send button

        # --- Handle Action Buttons inside ChatMessage ---
        button_classes = button.classes
        action_widget: Optional[ChatMessage] = None
        node: Optional[DOMNode] = button
        while node is not None:
            if isinstance(node, ChatMessage): action_widget = node; break
            node = node.parent

        if action_widget:
            message_text = action_widget.message_text
            message_role = action_widget.role

            # ───────────── existing branches above this point … ─────────────
            if "edit-button" in button_classes:
                logging.info(
                    "Action: Edit clicked for %s message: '%s…'",
                    message_role,
                    str(action_widget.message_text)[:50],  # Use str() for safety on message_text
                )
                is_editing = getattr(action_widget, "_editing", False)

                if not is_editing:
                    # ── START EDITING ────────────────────────────────────────────────
                    static_text: Static = action_widget.query_one(".message-text", Static)
                    current_renderable = static_text.renderable
                    current_text = ""
                    if isinstance(current_renderable, Text):
                        current_text = current_renderable.plain
                    elif isinstance(current_renderable, str):
                        current_text = current_renderable
                    else:  # Fallback for other types, could be Segment, etc.
                        current_text = str(current_renderable)

                    static_text.display = False
                    editor = TextArea(
                        text=current_text,
                        id="edit-area",
                        classes="edit-area",
                    )
                    editor.styles.width = "100%"
                    await action_widget.mount(editor, before=static_text)
                    editor.focus()
                    action_widget._editing = True
                    button.label = "Save Edit"  # Changed to "Save Edit" for clarity
                    logging.debug("Editing started.")
                else:
                    # ── STOP EDITING & SAVE ─────────────────────────────────────────
                    try:
                        editor: TextArea = action_widget.query_one("#edit-area", TextArea)
                        new_text = editor.text  # TextArea.text is already plain
                    except QueryError:
                        logging.error("Edit TextArea not found when stopping edit.")
                        # Restore original text if editor not found, make sure it's plain or escaped
                        original_renderable = action_widget.message_text  # This should be Text or str
                        if isinstance(original_renderable, Text):
                            new_text = original_renderable.plain
                        else:
                            new_text = str(original_renderable)  # Fallback
                        # new_text = action_widget.message_text.plain if isinstance(action_widget.message_text, Text) else str(action_widget.message_text)

                    try:
                        await editor.remove()
                    except Exception:  # Catch if editor was already removed or not found
                        pass

                    static_text: Static = action_widget.query_one(".message-text", Static)

                    # --- THIS IS THE CRITICAL FIX ---
                    # Ensure the text is escaped before updating the Static widget,
                    # if the Static widget is expected to parse markup.
                    from rich.markup import escape as escape_markup
                    escaped_new_text = escape_markup(new_text)
                    static_text.update(escaped_new_text)
                    # --- END CRITICAL FIX ---

                    static_text.display = True
                    action_widget.message_text = new_text  # Store the *plain* unescaped text internally
                    action_widget._editing = False
                    #button.label = "✏️"
                    button.label = get_char(EMOJI_EDIT, FALLBACK_EDIT)
                    button.label = get_char(EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT)
                    logging.debug("Editing finished. New length: %d", len(new_text))

            elif "copy-button" in button_classes:
                logging.info(
                    "Action: Copy clicked for %s message: '%s…'",
                    message_role,
                    message_text[:50],
                )

                copied = False
                try:
                    # ── Textual ≥ 0.57 ────────────────────────────────────────────────
                    if hasattr(self, "copy_to_clipboard"):
                        self.copy_to_clipboard(message_text)
                        copied = True

                    # ── Early Textual builds (local clipboard only) ──────────────────
                    elif hasattr(self, "set_clipboard"):  # legacy helper
                        self.set_clipboard(message_text)
                        copied = True

                    # ── Fallback to pyperclip (optional dependency) ──────────────────
                    else:
                        import importlib.util
                        if importlib.util.find_spec("pyperclip"):
                            import pyperclip  # type: ignore
                            pyperclip.copy(message_text)
                            copied = True
                        else:
                            logging.warning("pyperclip not available; clipboard copy skipped.")

                except Exception as exc:  # noqa: BLE001
                    logging.error("Clipboard action failed: %s", exc, exc_info=True)

                # ── user feedback ─────────────────────────────────────────────────────
                if copied:
                    #button.label = "✅Copied"
                    #self.set_timer(1.5, lambda: setattr(button, "label", "📋"))
                    button.label = get_char(EMOJI_COPIED, FALLBACK_COPIED) + "Copied"
                    self.set_timer(1.5, lambda: setattr(button, "label", get_char(EMOJI_COPY, FALLBACK_COPY)))

            elif "speak-button" in button_classes:
                logging.info(f"Action: Speak clicked for {message_role} message: '{message_text[:50]}...'")
                try:
                    # Query for Static specifically
                    text_widget = action_widget.query_one(".message-text", Static)
                    text_widget.update(Text(f"[SPEAKING...] {message_text}"))
                except QueryError:
                    logging.error("Could not find .message-text Static widget for speaking placeholder.")

            elif "thumb-up-button" in button_classes:
                logging.info(f"Action: Thumb Up clicked for {message_role} message.")
                button.label = "👍(OK)"  # Provide visual feedback

            elif "thumb-down-button" in button_classes:
                logging.info(f"Action: Thumb Down clicked for {message_role} message.")
                button.label = "👎(OK)"  # Provide visual feedback

            elif "delete-button" in button_classes:
                logging.info(
                    "Action: Delete clicked for %s message: '%s…'",
                    message_role,
                    message_text[:50],
                )
                try:
                    await action_widget.remove()  # removes the ChatMessage from the DOM
                    # If this was the AI placeholder being deleted, clear the reference
                    if action_widget is self.current_ai_message_widget:
                        self.current_ai_message_widget = None
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed to delete message: %s", exc, exc_info=True)

            # FIXME - lol 1shot idk
            # ── REGENERATE an AI message ──────────────────────────────────────────
            elif "regenerate-button" in button_classes and message_role == "AI":
                logging.info(f"Action: Regenerate clicked for AI message ID: {action_widget.message_id_internal}")

                prefix = "chat"  # Assuming regeneration only happens in the main chat tab for now
                chat_container: Optional[VerticalScroll] = None
                try:
                    chat_container = self.query_one(f"#{prefix}-log", VerticalScroll)
                except QueryError:
                    logging.error(f"Regenerate: Could not find chat container #{prefix}-log. Aborting.")
                    self.notify("Error: Chat log not found for regeneration.", severity="error")
                    return

                # 1. Build history up to (but not including) the message being regenerated
                history_for_regeneration = []
                widgets_to_remove = []
                found_target_ai_message = False

                # Iterate through existing ChatMessage widgets in the chat_log
                # We need to be careful modifying the list while iterating, so collect widgets to remove.
                all_message_widgets_in_log = list(chat_container.query(ChatMessage))

                for msg_widget in all_message_widgets_in_log:
                    if msg_widget is action_widget:
                        found_target_ai_message = True
                        widgets_to_remove.append(msg_widget)  # Add the AI message itself to remove
                        continue  # Don't add it to history

                    if found_target_ai_message:
                        # If we've found the target, all subsequent messages also need to be removed
                        widgets_to_remove.append(msg_widget)
                    else:
                        # This message is before the one we're regenerating
                        if msg_widget.role in ("User", "AI") and msg_widget.generation_complete:
                            role_for_api = "assistant" if msg_widget.role == "AI" else "user"
                            # Ensure content is the plain text, not Rich Text object for history
                            content_for_history = msg_widget.message_text
                            history_for_regeneration.append({"role": role_for_api, "content": content_for_history})

                if not history_for_regeneration:
                    logging.warning("Regenerate: No history found before the target AI message. Cannot regenerate.")
                    self.notify("Cannot regenerate: No preceding messages found.", severity="warning")
                    # Optionally, remove the AI message if it's the first one and regenerate makes no sense
                    # await action_widget.remove()
                    return

                logging.debug(
                    f"History for regeneration (count: {len(history_for_regeneration)}): {history_for_regeneration}")

                # 2. Remove the target AI message and subsequent messages from UI
                for widget_to_remove in widgets_to_remove:
                    logging.debug(f"Regenerate: Removing widget {widget_to_remove.id} (Role: {widget_to_remove.role})")
                    await widget_to_remove.remove()

                # Ensure the current_ai_message_widget reference is cleared if it was one of the removed ones
                if self.current_ai_message_widget in widgets_to_remove:
                    self.current_ai_message_widget = None

                # 3. Fetch current chat settings (similar to send-chat button logic)
                try:
                    provider_widget = self.query_one(f"#{prefix}-api-provider", Select)
                    model_widget = self.query_one(f"#{prefix}-api-model", Select)
                    system_prompt_widget = self.query_one(f"#{prefix}-system-prompt", TextArea)
                    temp_widget = self.query_one(f"#{prefix}-temperature", Input)
                    top_p_widget = self.query_one(f"#{prefix}-top-p", Input)
                    min_p_widget = self.query_one(f"#{prefix}-min-p", Input)
                    top_k_widget = self.query_one(f"#{prefix}-top-k", Input)
                    # --- Query "Full Chat Settings" widgets ---
                    llm_max_tokens_widget = self.query_one(f"#{prefix}-llm-max-tokens", Input)
                    llm_seed_widget = self.query_one(f"#{prefix}-llm-seed", Input)
                    llm_stop_widget = self.query_one(f"#{prefix}-llm-stop", Input)
                    llm_response_format_widget = self.query_one(f"#{prefix}-llm-response-format", Select)
                    llm_n_widget = self.query_one(f"#{prefix}-llm-n", Input)
                    llm_user_identifier_widget = self.query_one(f"#{prefix}-llm-user-identifier", Input)
                    llm_logprobs_widget = self.query_one(f"#{prefix}-llm-logprobs", Checkbox)
                    llm_top_logprobs_widget = self.query_one(f"#{prefix}-llm-top-logprobs", Input)
                    llm_logit_bias_widget = self.query_one(f"#{prefix}-llm-logit-bias", TextArea)
                    llm_presence_penalty_widget = self.query_one(f"#{prefix}-llm-presence-penalty", Input)
                    llm_frequency_penalty_widget = self.query_one(f"#{prefix}-llm-frequency-penalty", Input)
                    llm_tools_widget = self.query_one(f"#{prefix}-llm-tools", TextArea)
                    llm_tool_choice_widget = self.query_one(f"#{prefix}-llm-tool-choice", Input)
                except QueryError as e:
                    logging.error(f"Regenerate: Could not find UI settings widgets for '{prefix}': {e}")
                    await chat_container.mount(
                        ChatMessage(
                            Text.from_markup("[bold red]Internal Error:[/]\nMissing UI settings for regeneration."),
                            role="System", classes="-error"))
                    return

                selected_provider = str(provider_widget.value) if provider_widget.value else None
                selected_model = str(model_widget.value) if model_widget.value else None
                system_prompt = system_prompt_widget.text
                temperature = self._safe_float(temp_widget.value, 0.7, "temperature")
                top_p = self._safe_float(top_p_widget.value, 0.95, "top_p")
                min_p = self._safe_float(min_p_widget.value, 0.05, "min_p")
                top_k = self._safe_int(top_k_widget.value, 50, "top_k")

                llm_max_tokens_value = self._safe_int(llm_max_tokens_widget.value, 1024, "llm_max_tokens")
                llm_seed_value = self._safe_int(llm_seed_widget.value, None, "llm_seed")
                llm_stop_value = llm_stop_widget.value.split(',') if llm_stop_widget.value.strip() else None
                llm_response_format_value = {"type": str(llm_response_format_widget.value)}
                llm_n_value = self._safe_int(llm_n_widget.value, 1, "llm_n")
                llm_user_identifier_value = llm_user_identifier_widget.value.strip() or None
                llm_logprobs_value = llm_logprobs_widget.value
                llm_top_logprobs_value = self._safe_int(llm_top_logprobs_widget.value, 0, "llm_top_logprobs")
                llm_presence_penalty_value = self._safe_float(llm_presence_penalty_widget.value, 0.0,
                                                              "llm_presence_penalty")
                llm_frequency_penalty_value = self._safe_float(llm_frequency_penalty_widget.value, 0.0,
                                                               "llm_frequency_penalty")
                llm_tool_choice_value = llm_tool_choice_widget.value.strip() or None
                try:
                    llm_logit_bias_text = llm_logit_bias_widget.text.strip()
                    llm_logit_bias_value = json.loads(llm_logit_bias_text) if llm_logit_bias_text else None
                except json.JSONDecodeError:
                    llm_logit_bias_value = None
                try:
                    llm_tools_text = llm_tools_widget.text.strip()
                    llm_tools_value = json.loads(llm_tools_text) if llm_tools_text else None
                except json.JSONDecodeError:
                    llm_tools_value = None

                if not selected_provider or not selected_model:
                    logging.warning("Regenerate: Provider or model not selected.")
                    await chat_container.mount(
                        ChatMessage(
                            Text.from_markup("[bold red]Error:[/]\nPlease select provider and model for regeneration."),
                            role="System", classes="-error"))
                    return

                # API Key (copied from your send-chat logic, ensure it's robust)
                api_key_for_call = None
                provider_settings_key = selected_provider.lower()
                provider_config_settings = self.app_config.get("api_settings", {}).get(provider_settings_key, {})
                if provider_config_settings.get("api_key"):
                    api_key_for_call = provider_config_settings["api_key"]
                elif provider_config_settings.get("api_key_env_var"):
                    api_key_for_call = os.environ.get(provider_config_settings["api_key_env_var"])

                # Re-check API key presence for cloud providers (copied from send logic)
                providers_requiring_key = ["OpenAI", "Anthropic", "Google", "MistralAI", "Groq", "Cohere", "OpenRouter",
                                           "HuggingFace", "DeepSeek"]
                if selected_provider in providers_requiring_key and not api_key_for_call:
                    logging.error(
                        f"Regenerate aborted: API Key for required provider '{selected_provider}' is missing.")
                    await chat_container.mount(ChatMessage(
                        Text.from_markup(f"[bold red]API Key for {selected_provider} is missing for regeneration.[/]"),
                        role="System", classes="-error"))
                    return

                # 4. Mount placeholder and dispatch worker
                ai_placeholder_widget = ChatMessage(
                    message=f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)} (Regenerating...)",
                    role="AI",
                    generation_complete=False
                )
                await chat_container.mount(ai_placeholder_widget)
                chat_container.scroll_end(animate=False)
                self.current_ai_message_widget = ai_placeholder_widget

                # For regeneration, the 'message' to chat_wrapper is effectively the context ending
                # with the last user message. The history *is* that context.
                # chat_wrapper's 'message' param might need to be empty or None if history is primary.
                # Let's pass an empty message as the "new" input, and the history.
                worker_target_regen = lambda: self.chat_wrapper(
                    message="",  # No new user message, we are regenerating based on history
                    history=history_for_regeneration,  # The crucial part
                    api_endpoint=selected_provider,
                    api_key=api_key_for_call,
                    custom_prompt="",  # Not typically used for regeneration directly, baked into history/system_prompt
                    temperature=temperature,
                    system_message=system_prompt,  # Use current system prompt
                    streaming=False,  # Or get from UI if you have a streaming checkbox
                    minp=min_p,
                    model=selected_model,
                    topp=top_p,
                    topk=top_k,
                    llm_max_tokens=llm_max_tokens_value,
                    llm_seed=llm_seed_value,
                    llm_stop=llm_stop_value,
                    llm_response_format=llm_response_format_value,
                    llm_n=llm_n_value,
                    llm_user_identifier=llm_user_identifier_value,
                    llm_logprobs=llm_logprobs_value,
                    llm_top_logprobs=llm_top_logprobs_value,
                    llm_logit_bias=llm_logit_bias_value,
                    llm_presence_penalty=llm_presence_penalty_value,
                    llm_frequency_penalty=llm_frequency_penalty_value,
                    llm_tools=llm_tools_value,
                    llm_tool_choice=llm_tool_choice_value,
                    media_content={}, selected_parts=[], chatdict_entries=None,
                    max_tokens=500, strategy="sorted_evenly"  # Existing chatdict params
                )

                logging.debug(f"Regenerate: Running worker 'API_Call_{prefix}_regenerate' to execute chat_wrapper")
                self.run_worker(
                    worker_target_regen,
                    name=f"API_Call_{prefix}_regenerate",  # Slightly different name for clarity
                    group="api_calls",
                    thread=True,
                    description=f"Regenerating for {selected_provider}"
                )
                return  # Finished handling regenerate

        else:
            # This handles buttons not inside a ChatMessage or unhandled IDs
            if not button_id.startswith("tab-"):  # Avoid logging tab clicks as warnings
                print(f">>> DEBUG: Button '{button_id}' didn't match tab or send prefixes.")
                logging.warning(f"Button pressed with unhandled ID or context: {button_id}")

    def chat_wrapper(self, message, history, api_endpoint, api_key, custom_prompt, temperature,
                     system_message, streaming, minp, model, topp, topk,
                     # --- New "Full Chat Settings" parameters ---
                     llm_max_tokens: Optional[int],
                     llm_seed: Optional[int],
                     llm_stop: Optional[List[str]], # Corrected based on processing: list of strings
                     llm_response_format: Optional[Dict[str, str]], # Passed as dict {"type": "text"}
                     llm_n: Optional[int],
                     llm_user_identifier: Optional[str],
                     llm_logprobs: Optional[bool],
                     llm_top_logprobs: Optional[int],
                     llm_logit_bias: Optional[Dict[str, float]], # JSON parsed to dict
                     llm_presence_penalty: Optional[float],
                     llm_frequency_penalty: Optional[float],
                     llm_tools: Optional[List[Dict[str, Any]]], # JSON parsed to list of dicts
                     llm_tool_choice: Optional[Union[str, Dict[str, Any]]], # String or JSON parsed to dict
                     # --- Existing parameters ---
                     media_content, selected_parts, chatdict_entries, max_tokens, strategy):
        """
        This method runs in the worker thread and calls the main chat logic.
        It receives parameters captured by the lambda in on_button_pressed.
        """
        logging.debug(f"chat_wrapper executing for endpoint '{api_endpoint}'")
        try:
            # Call the imported chat function from Chat_Functions.py
            result = chat(
                message=message,
                history=history,
                media_content=media_content,
                selected_parts=selected_parts,
                api_endpoint=api_endpoint,
                api_key=api_key,
                custom_prompt=custom_prompt,  # Pass custom_prompt from UI
                temperature=temperature,
                system_message=system_message,
                streaming=streaming,
                minp=minp,
                model=model,
                topp=topp,
                topk=topk,
                # --- Pass new "Full Chat Settings" to chat() ---
                llm_max_tokens=llm_max_tokens,
                llm_seed=llm_seed,
                llm_stop=llm_stop,
                llm_response_format=llm_response_format,  # Pass as dict
                llm_n=llm_n,
                llm_user_identifier=llm_user_identifier,
                llm_logprobs=llm_logprobs,
                llm_top_logprobs=llm_top_logprobs,
                llm_logit_bias=llm_logit_bias,
                llm_presence_penalty=llm_presence_penalty,
                llm_frequency_penalty=llm_frequency_penalty,
                llm_tools=llm_tools,
                llm_tool_choice=llm_tool_choice,
                # --- Existing parameters for chat() ---
                chatdict_entries=chatdict_entries,
                max_tokens=max_tokens, # This is for chatdict context
                strategy=strategy
            )
            logging.debug(f"chat_wrapper finished for '{api_endpoint}'. Result type: {type(result)}")
            return result
        except Exception as e:
            logging.exception(f"Error inside chat_wrapper for endpoint {api_endpoint}: {e}")
            # Return a formatted error string that on_worker_state_changed can display
            return f"[bold red]Error during chat processing:[/]\n{escape(str(e))}"

    # --- Handle worker completion ---
    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle completion / failure of a background API worker."""
        worker_name = event.worker.name or "Unknown Worker"
        logging.debug("Worker '%s' state changed to %s", worker_name, event.state)

        if not worker_name.startswith("API_Call_"):
            return

        prefix = worker_name.replace("API_Call_", "")
        ai_message_widget = self.current_ai_message_widget  # Stored reference to the "AI thinking..." widget

        # --- Gracefully handle missing placeholder widget ---
        if ai_message_widget is None or not ai_message_widget.is_mounted:
            logging.warning(
                "Worker '%s' finished, but its AI placeholder widget is missing or not mounted. Cannot update directly.",
                worker_name,
                getattr(self.current_ai_message_widget, 'id', 'N/A') if self.current_ai_message_widget else 'N/A'
            )
            # Attempt to log a generic error to the main chat container if the placeholder is gone
            try:
                chat_container_fallback: VerticalScroll = self.query_one(f"#{prefix}-log", VerticalScroll)
                error_msg_text_str = f"[bold red]Error: AI response for worker '{worker_name}' received, but its display widget was missing. Check logs.[/]"
                error_widget_fallback = ChatMessage(error_msg_text_str, role="System",
                                                    classes="-error")
                await chat_container_fallback.mount(error_widget_fallback)
                chat_container_fallback.scroll_end(animate=False)
            except QueryError:
                logging.error("Fallback: Could not find chat container '#%s-log' to report missing AI placeholder.",
                              prefix)
            except Exception as e_fallback:
                logging.error(f"Fallback: Error reporting missing AI placeholder: {e_fallback}", exc_info=True)

            self.current_ai_message_widget = None  # Ensure it's cleared
            return  # Exit if placeholder is gone

        try:
            chat_container: VerticalScroll = self.query_one(f"#{prefix}-log", VerticalScroll)
            static_text_widget = ai_message_widget.query_one(".message-text", Static)

            if event.state is WorkerState.SUCCESS:
                result = event.worker.result
                is_streaming_result = isinstance(result, Generator)  # Check if it's a generator for streaming

                # Clear the "AI thinking..." text from the placeholder before updating with actual content
                if ai_message_widget.message_text.strip() == f"AI {get_char(EMOJI_THINKING, FALLBACK_THINKING)}":
                    ai_message_widget.message_text = ""  # Update internal state
                    static_text_widget.update("")  # Clear UI part

                if is_streaming_result:
                    logging.info("API call (%s) returned a generator – streaming.", prefix)

                    async def process_stream() -> None:
                        full_original_text = ""
                        if not ai_message_widget or not ai_message_widget.is_mounted:
                            logging.warning(f"Stream processing for '{prefix}' aborted: AI widget no longer mounted.")
                            return
                        try:
                            stream_static_text_widget = ai_message_widget.query_one(".message-text", Static)
                        except QueryError:
                            logging.error(
                                f"Stream processing for '{prefix}' aborted: .message-text Static widget not found in AI widget.")
                            if ai_message_widget.is_mounted: ai_message_widget.mark_generation_complete()  # Still mark complete
                            self.current_ai_message_widget = None  # Clear reference
                            return

                        try:
                            async for chunk in result:  # type: ignore[misc]
                                text_chunk_original = str(chunk)  # Keep original for internal storage
                                full_original_text += text_chunk_original
                                logging.debug(f"STREAM CHUNK for '{prefix}': {text_chunk_original!r}")

                                text_chunk_escaped_for_display = escape_markup(text_chunk_original)

                                current_display_renderable = stream_static_text_widget.renderable
                                if isinstance(current_display_renderable, Text):
                                    # Append escaped chunk to existing Text plain content
                                    new_text_obj = Text(
                                        current_display_renderable.plain + text_chunk_escaped_for_display, end="")
                                    stream_static_text_widget.update(new_text_obj)
                                else:  # If it was a string or other renderable, concatenate escaped chunk
                                    existing_plain = str(current_display_renderable)
                                    stream_static_text_widget.update(existing_plain + text_chunk_escaped_for_display)

                                # Update internal message_text with the ORIGINAL, UNESCAPED chunk
                                ai_message_widget.message_text += text_chunk_original

                                if chat_container.is_mounted:
                                    chat_container.scroll_end(animate=False, duration=0.05)

                            ai_message_widget.mark_generation_complete()
                            logging.info(
                                f"Stream finished for '{prefix}' (Original length: {len(full_original_text)} chars). Full original text: {full_original_text!r}")

                        except Exception as exc_stream_inner:
                            logging.exception("Stream failure for worker '%s': %s", worker_name, exc_stream_inner)
                            if ai_message_widget.is_mounted and stream_static_text_widget.is_mounted:
                                current_text_plain = ai_message_widget.message_text  # Get internally stored original text
                                error_message_display = Text.from_markup(
                                    escape_markup(current_text_plain) + "\n[bold red]Error during stream.[/]")
                                stream_static_text_widget.update(error_message_display)
                            ai_message_widget.mark_generation_complete()
                        finally:
                            self.current_ai_message_widget = None  # Clear reference
                            if self.is_mounted:
                                try:
                                    self.query_one(f"#{prefix}-input", TextArea).focus()
                                except QueryError:
                                    pass

                    # FIXME - Enable streaming processing
                    # Start the stream processing task
                    self.run_task(process_stream(), name=f"stream_processor_{prefix}", group="streams")

                else:  # Non-streaming result
                    worker_result_str = str(result) if result is not None else ""
                    logging.debug(f"NON-STREAMING RESULT for '{prefix}': {worker_result_str!r}")
                    final_display_text_obj: Text  # Will hold the rich.text.Text object for display
                    # original_text_for_storage will hold the raw string (plain or markup) from the worker
                    # This is useful if ChatMessage needs to store the original form for other purposes (e.g. copy)
                    original_text_for_storage: str

                    if isinstance(result, dict):  # Standard OpenAI non-streaming success
                        try:
                            original_text_for_storage = result['choices'][0]['message']['content']
                            # Assume the content from a successful API call is plain text
                            final_display_text_obj = Text(original_text_for_storage)
                        except (KeyError, IndexError, TypeError) as e:
                            logging.error(f"Error parsing non-streaming dict result: {e}. Resp: {result}",
                                          exc_info=True)
                            original_text_for_storage = "[AI: Error parsing successful response structure.]"
                            # This error message for UI can use markup
                            final_display_text_obj = Text.from_markup(
                                f"[bold red]{escape_markup(original_text_for_storage)}[/]")

                    elif isinstance(result, str):
                        original_text_for_storage = result  # Store the raw string result
                        # Check the source/type of string error
                        if result.startswith("[bold red]Error during chat processing:[/]"):
                            # This result is from chat_wrapper and is already Rich markup
                            final_display_text_obj = Text.from_markup(result)
                        elif result.startswith("An error occurred in the chat function:"):
                            # This result is a plain string error message from the chat()function.
                            # It might contain special characters like '[', which should be displayed literally.
                            # Using Text() constructor achieves this.
                            final_display_text_obj = Text(result)
                        else:
                            # Assume any other string result is plain text (e.g., from a successful non-streaming API)
                            # and should be displayed literally.
                            final_display_text_obj = Text(result)

                    elif result is None:
                        original_text_for_storage = "[AI: Error – No response received from API.]"
                        final_display_text_obj = Text.from_markup(
                            f"[bold red]{escape_markup(original_text_for_storage)}[/]")

                    else:  # Fallback for truly unexpected result types
                        logging.error(f"Unexpected result type from API: {type(result)}. Content: {result!r}")
                        original_text_for_storage = f"[Error: Unexpected result type ({type(result).__name__}) from API.]"
                        final_display_text_obj = Text.from_markup(
                            f"[bold red]{escape_markup(original_text_for_storage)}[/]")
                    # Update the ChatMessage's internal text state (e.g., for copying raw content)
                    # This depends on how ChatMessage is implemented. If its message_text reactive property
                    # triggers its own display update, ensure it handles raw vs. Text objects correctly or
                    # assign final_display_text_obj to it. For now, we assume it stores raw.
                    ai_message_widget.message_text = original_text_for_storage
                    # Update the Static widget with the prepared Text object
                    static_text_widget.update(final_display_text_obj)
                    ai_message_widget.mark_generation_complete()
                    self.current_ai_message_widget = None  # Clear reference

            elif event.state is WorkerState.ERROR:
                error_from_worker = event.worker.error
                logging.error("Worker '%s' failed.", worker_name, exc_info=error_from_worker)

                error_message_str = f"AI Error: Processing failed.\nDetails: {str(error_from_worker)}"
                escaped_error_for_display = escape_markup(error_message_str)  # Escape the whole thing

                ai_message_widget.message_text = error_message_str  # Store original error for internal
                static_text_widget.update(escaped_error_for_display)  # Update UI with escaped version
                ai_message_widget.mark_generation_complete()
                self.current_ai_message_widget = None  # Clear reference

            # Common cleanup/actions after success or error (if placeholder was valid)
            if chat_container.is_mounted:
                chat_container.scroll_end(animate=True)
            if self.is_mounted:  # Check if app itself is still mounted
                try:
                    self.query_one(f"#{prefix}-input", TextArea).focus()
                except QueryError:
                    pass  # Input area might be gone

        except QueryError as qe:
            logging.error("QueryError in on_worker_state_changed for '%s': %s. Widget might have been removed.",
                          worker_name, qe, exc_info=True)
            if self.current_ai_message_widget and self.current_ai_message_widget.is_mounted:
                try:
                    await self.current_ai_message_widget.remove()
                except Exception:
                    pass
            self.current_ai_message_widget = None
        except Exception as exc:  # Catch-all for other unexpected errors in this handler
            logging.exception("Unexpected error in on_worker_state_changed for worker '%s': %s", worker_name, exc)
            if ai_message_widget and ai_message_widget.is_mounted:  # Check again
                try:
                    static_widget_unexp_err = ai_message_widget.query_one(".message-text", Static)
                    # Escape any content from the exception itself if included
                    error_update_text_unexp = Text.from_markup(
                        f"[bold red]Internal error handling AI response:[/]\n{escape_markup(str(exc))}")
                    static_widget_unexp_err.update(error_update_text_unexp)
                    ai_message_widget.mark_generation_complete()
                except Exception as e_unexp_final:
                    logging.error(
                        f"Further error updating with unexpected error in on_worker_state_changed: {e_unexp_final}")
            self.current_ai_message_widget = None

    async def load_and_display_notes(self) -> None:
        """Loads notes from the database and populates the left sidebar list."""
        if not self.notes_service:
            logging.error("Notes service not available, cannot load notes.")
            # Optionally, display an error in the UI
            return
        try:
            notes_list = self.notes_service.list_notes(user_id=self.notes_user_id, limit=200)  # Increased limit
            sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            await sidebar_left.populate_notes_list(notes_list)
            logging.info(f"Loaded {len(notes_list)} notes into the sidebar.")
        except CharactersRAGDBError as e:
            logging.error(f"Database error loading notes: {e}", exc_info=True)
        except QueryError:
            logging.error("Failed to find notes-sidebar-left to populate notes.")
        except Exception as e:
            logging.error(f"Unexpected error loading notes: {e}", exc_info=True)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handles selecting a note from the list in the left sidebar."""
        # Check if the event comes from the notes list view
        if event.list_view.id == "notes-list-view":
            if not self.notes_service:
                logging.error("Notes service not available, cannot load selected note.")
                return

            selected_item = event.item  # This is the ListItem
            if selected_item and hasattr(selected_item, 'note_id') and hasattr(selected_item, 'note_version'):
                note_id = selected_item.note_id
                note_version = selected_item.note_version
                logging.info(f"Note selected: ID={note_id}, Version={note_version}")

                try:
                    note_details = self.notes_service.get_note_by_id(
                        user_id=self.notes_user_id,
                        note_id=note_id
                    )
                    if note_details:
                        self.current_selected_note_id = note_id
                        # Important: Use the version from the freshly loaded details for consistency
                        self.current_selected_note_version = note_details.get('version')
                        self.current_selected_note_title = note_details.get('title')
                        self.current_selected_note_content = note_details.get('content', "")

                        # Update UI elements
                        editor = self.query_one("#notes-editor-area", TextArea)
                        editor.text = self.current_selected_note_content
                        # Ensure editor is focused if you want immediate typing
                        # editor.focus()

                        title_input = self.query_one("#notes-title-input", Input)
                        title_input.value = self.current_selected_note_title or ""  # Handle None title

                        # Clear and update keywords area
                        try:
                            keywords_area = self.query_one("#notes-keywords-area", TextArea)
                            keywords_for_note = self.notes_service.get_keywords_for_note(
                                user_id=self.notes_user_id,
                                note_id=note_id  # note_id is available from the selection logic
                            )

                            if keywords_for_note:
                                keywords_str = ", ".join([kw['keyword'] for kw in keywords_for_note])
                                keywords_area.text = keywords_str
                                logging.info(
                                    f"Displayed {len(keywords_for_note)} keywords for note {note_id}: '{keywords_str}'")
                            else:
                                keywords_area.text = ""  # Clear if no keywords
                                logging.info(f"No keywords found for note {note_id}.")

                        except CharactersRAGDBError as e:
                            logging.error(f"Database error loading keywords for note {note_id}: {e}", exc_info=True)
                            if 'keywords_area' in locals(): keywords_area.text = "Error loading keywords."
                        except QueryError as e:
                            logging.error(f"UI component #notes-keywords-area not found: {e}", exc_info=True)
                        except Exception as e:
                            logging.error(f"Unexpected error loading keywords for note {note_id}: {e}", exc_info=True)
                            if 'keywords_area' in locals(): keywords_area.text = "Unexpected error loading keywords."

                        logging.info(f"Loaded note '{self.current_selected_note_title}' into editor.")
                    else:
                        logging.warning(f"Could not retrieve details for note ID: {note_id}")
                        self.current_selected_note_id = None  # Clear selection
                        self.current_selected_note_version = None
                        self.current_selected_note_title = None
                        self.current_selected_note_content = ""
                        self.query_one("#notes-editor-area", TextArea).text = ""
                        self.query_one("#notes-title-input", Input).value = ""
                        # Ensure keywords area is also cleared if note details fail to load
                        try:
                            self.query_one("#notes-keywords-area", TextArea).text = ""
                        except QueryError:
                            logging.error("Failed to query #notes-keywords-area for clearing on note load failure.")


                except CharactersRAGDBError as e:
                    logging.error(f"Database error loading note {note_id}: {e}", exc_info=True)
                except QueryError as e:
                    logging.error(f"UI component not found while loading note: {e}", exc_info=True)
                except Exception as e:
                    logging.error(f"Unexpected error loading note {note_id}: {e}", exc_info=True)
        # Pass the event to the superclass if you have a base class that handles it
        # Or handle other ListViews if any
        # super().on_list_view_selected(event) # If applicable

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "notes-search-input":
            search_term = event.value.strip()
            logging.debug(f"Search term entered for notes: '{search_term}'")

            if not self.notes_service:
                logging.error("Notes service not available for search.")
                return

            try:
                sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
                if not search_term:  # If search term is empty, load all notes
                    await self.load_and_display_notes()
                else:
                    # Optional: Add a small delay or minimum character count if desired
                    # if len(search_term) < 3: # Example: search only if term is 3+ chars
                    #     # Clear list or keep existing if term is too short but not empty
                    #     await sidebar_left.populate_notes_list([]) # Clear if desired
                    #     return

                    notes_list = self.notes_service.search_notes(
                        user_id=self.notes_user_id,
                        search_term=search_term,
                        limit=200  # Or a suitable limit for search results
                    )
                    await sidebar_left.populate_notes_list(notes_list)
                    logging.info(f"Found {len(notes_list)} notes for search term '{search_term}'.")

            except CharactersRAGDBError as e:
                logging.error(f"Database error searching notes for '{search_term}': {e}", exc_info=True)
                # Optionally, display error in UI or clear list
                # try:
                #     sidebar_left = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
                #     await sidebar_left.populate_notes_list([]) # Clear list on error
                # except QueryError: pass # Sidebar might not be available
            except QueryError as e:
                logging.error(f"UI component not found during note search: {e}", exc_info=True)
            except Exception as e:
                logging.error(f"Unexpected error during note search: {e}", exc_info=True)
            return  # Ensure this event is handled here


        elif event.input.id == "chat-conversation-search-bar":
            # Cancel any existing timer for this search
            if self._conversation_search_timer:
                self._conversation_search_timer.stop()
            # Start a new timer to call _perform_conversation_search after 0.5 seconds
            self._conversation_search_timer = self.set_timer(
                0.5,
                self._perform_conversation_search  # Pass the coroutine directly
            )
            return
        elif event.input.id == "conv-char-search-input":
            # Cancel any existing timer for this search
            if self._conv_char_search_timer:
                self._conv_char_search_timer.stop()
            # Start a new timer to call _perform_conv_char_search after 0.5 seconds
            self._conv_char_search_timer = self.set_timer(
                0.5,
                self._perform_conv_char_search  # Pass the coroutine directly
            )
            return

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes for conversation search filters."""
        checkbox_id = event.checkbox.id
        logging.debug(f"Checkbox '{checkbox_id}' changed to {event.value}")

        if checkbox_id == "chat-conversation-search-all-characters-checkbox":
            try:
                char_filter_select = self.query_one("#chat-conversation-search-character-filter-select", Select)
                if event.value is True:  # "All Characters" is checked
                    char_filter_select.disabled = True
                    char_filter_select.value = Select.BLANK  # Clear selection
                    logging.debug("Disabled character filter select and cleared its value.")
                else:  # "All Characters" is unchecked
                    char_filter_select.disabled = False
                    logging.debug("Enabled character filter select.")
                # Trigger a new search based on the changed filter
                await self._perform_conversation_search()
            except QueryError as e:
                logging.error(f"Error accessing character filter select: {e}", exc_info=True)

        elif checkbox_id == "chat-conversation-search-include-character-checkbox":
            # This checkbox's state will be read by _perform_conversation_search
            # Trigger a new search based on the changed filter
            await self._perform_conversation_search()

    async def _perform_conversation_search(self) -> None:
        """Performs conversation search based on current UI filter states and populates the ListView."""
        logging.debug("Performing conversation search...")
        try:
            search_bar = self.query_one("#chat-conversation-search-bar", Input)
            search_term = search_bar.value.strip()

            include_char_chats_checkbox = self.query_one("#chat-conversation-search-include-character-checkbox",
                                                         Checkbox)
            include_character_chats = include_char_chats_checkbox.value

            all_chars_checkbox = self.query_one("#chat-conversation-search-all-characters-checkbox", Checkbox)
            search_all_characters = all_chars_checkbox.value

            char_filter_select = self.query_one("#chat-conversation-search-character-filter-select", Select)
            selected_character_id = char_filter_select.value if not char_filter_select.disabled and char_filter_select.value != Select.BLANK else None

            results_list_view = self.query_one("#chat-conversation-search-results-list", ListView)
            await results_list_view.clear()

            if not self.notes_service:
                logging.error("Notes service not available for conversation search.")
                # Optionally, show a message in the ListView
                await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
                return

            db = self.notes_service._get_db(self.notes_user_id)

            # Refined search logic based on checkbox states
            conversations = []
            if not include_character_chats:
                # Search only for conversations with character_id IS NULL
                # FIXME
                # This requires db.search_conversations_by_title to support a new flag/mode
                # For now, log this case and skip results, or implement a temporary client-side filter if feasible
                logging.info("Search limited to non-character chats (feature requires DB method update).")
                # As a placeholder, we can fetch all and filter, but this is inefficient.
                # temp_conversations = db.search_conversations_by_title(title_query=search_term, character_id=None, limit=1000)
                # conversations = [conv for conv in temp_conversations if conv.get('character_id') is None]
                # For now, let's just return empty or a message.
                await results_list_view.append(ListItem(Label("Filtering non-character chats needs DB update.")))

            elif search_all_characters or selected_character_id is None:
                # Search all conversations (character_id can be anything or NULL)
                logging.debug(f"Searching all conversations for term: '{search_term}'")
                conversations = db.search_conversations_by_title(title_query=search_term, character_id=None, limit=100)
            else:  # Specific character selected
                logging.debug(
                    f"Searching conversations for term: '{search_term}', character_id: {selected_character_id}")
                conversations = db.search_conversations_by_title(title_query=search_term,
                                                                 character_id=selected_character_id, limit=100)

            if not conversations:
                await results_list_view.append(ListItem(Label(
                    "No conversations found." if search_term or selected_character_id else "Enter search term or select character.")))

            for conv in conversations:
                title = conv.get('title') or f"Chat ID: {conv['id'][:8]}..."
                item = ListItem(Label(title))
                item.conversation_id = conv['id']  # Store ID for loading
                item.conversation_title = conv.get('title')  # Store for potential use
                item.conversation_keywords = conv.get('keywords')  # Store for potential use
                await results_list_view.append(item)
            logging.info(f"Conversation search yielded {len(conversations)} results.")

        except QueryError as e:
            logging.error(f"UI component not found during conversation search: {e}", exc_info=True)
            if 'results_list_view' in locals():  # Check if listview was queried
                try:
                    await results_list_view.append(ListItem(Label("Error: UI component missing.")))
                except Exception:
                    pass  # Avoid error in error handling
        except CharactersRAGDBError as e:
            logging.error(f"Database error during conversation search: {e}", exc_info=True)
            if 'results_list_view' in locals():
                try:
                    await results_list_view.append(ListItem(Label("Error: Database search failed.")))
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Unexpected error during conversation search: {e}", exc_info=True)
            if 'results_list_view' in locals():
                try:
                    await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))
                except Exception:
                    pass

    async def _perform_conv_char_search(self) -> None:
        """Performs conversation search for the Conversations & Characters tab, filtering by selected character and search term."""
        logging.debug("Performing Conversations & Characters search...")
        try:
            search_input = self.query_one("#conv-char-search-input", Input)
            search_term = search_input.value.strip()

            char_select_widget = self.query_one("#conv-char-character-select", Select)
            selected_character_id = char_select_widget.value
            if selected_character_id == Select.BLANK: # Treat BLANK as None for filtering
                selected_character_id = None

            results_list_view = self.query_one("#conv-char-search-results-list", ListView)
            await results_list_view.clear()

            if not self.notes_service:
                logging.error("Notes service not available for ConvChar search.")
                await results_list_view.append(ListItem(Label("Error: Notes service unavailable.")))
                return

            db = self.notes_service._get_db(self.notes_user_id)
            conversations = []

            if selected_character_id:
                logging.debug(f"Filtering for character ID: {selected_character_id}")
                if search_term:
                    logging.debug(f"Searching with term '{search_term}' for character ID {selected_character_id}")
                    conversations = db.search_conversations_by_title(
                        title_query=search_term, character_id=selected_character_id, limit=200
                    )
                else:
                    logging.debug(f"Getting all conversations for character ID {selected_character_id}")
                    conversations = db.get_conversations_for_character(
                        character_id=selected_character_id, limit=200
                    )
            else: # No specific character selected
                logging.debug(f"No character selected. Searching globally with term: '{search_term}'")
                # If search_term is empty, this will list all conversations up to the limit.
                conversations = db.search_conversations_by_title(
                    title_query=search_term, character_id=None, limit=200
                )

            if not conversations:
                if not search_term and not selected_character_id:
                    await results_list_view.append(ListItem(Label("Enter search term or select a character.")))
                else:
                    await results_list_view.append(ListItem(Label("No items found matching your criteria.")))
            else:
                character_name_prefix_for_filter = ""
                if selected_character_id: # A specific character was used for the search filter
                    try:
                        char_info = db.get_character_card_by_id(selected_character_id)
                        if char_info and char_info.get('name'):
                            character_name_prefix_for_filter = f"[{char_info['name']}] "
                    except Exception as e:
                        logging.warning(f"Could not fetch name for selected character ID {selected_character_id}: {e}", exc_info=True)

                for conv in conversations:
                    base_title = conv.get('title') or f"Conversation ID: {conv['id'][:8]}..."
                    display_title = base_title

                    if character_name_prefix_for_filter: # Specific character was selected for search
                        display_title = f"{character_name_prefix_for_filter}{base_title}"
                    else: # Global search, try to find character for each conversation
                        char_id_for_conv = conv.get('character_id')
                        if char_id_for_conv:
                            try:
                                char_details = db.get_character_card_by_id(char_id_for_conv)
                                if char_details and char_details.get('name'):
                                    display_title = f"[{char_details['name']}] {base_title}"
                            except Exception as e_char:
                                logging.warning(f"Could not fetch character name for conversation {conv['id']}, char ID {char_id_for_conv}: {e_char}", exc_info=True)

                    item = ListItem(Label(display_title))
                    item.details = conv  # Store all conversation details
                    await results_list_view.append(item)

            logging.info(f"ConvChar search (Term: '{search_term}', CharID: {selected_character_id}) yielded {len(conversations)} results.")

        except QueryError as e:
            logging.error(f"UI component not found during ConvChar search: {e}", exc_info=True)
            if 'results_list_view' in locals(): # Check if listview was queried before error
                try:
                    await results_list_view.clear() # Clear previous results
                    await results_list_view.append(ListItem(Label("Error: UI component missing. Cannot perform search.")))
                except Exception: pass # Avoid error in error handling
        except CharactersRAGDBError as e:
            logging.error(f"Database error during ConvChar search: {e}", exc_info=True)
            if 'results_list_view' in locals():
                try:
                    await results_list_view.clear()
                    await results_list_view.append(ListItem(Label("Error: Database search failed.")))
                except Exception: pass
        except Exception as e:
            logging.error(f"Unexpected error during ConvChar search: {e}", exc_info=True)
            if 'results_list_view' in locals():
                try:
                    await results_list_view.clear()
                    await results_list_view.append(ListItem(Label("Error: Unexpected search failure.")))
                except Exception: pass

    # --- Helper methods ---
    def _safe_float(self, value: str, default: float, name: str) -> float:
        if not value: return default
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Invalid {name} '{value}', using {default}")
            return default

    def _safe_int(self, value: str, default: int, name: str) -> int:
        if not value: return default
        try:
            return int(value)
        except ValueError:
            logging.warning(f"Invalid {name} '{value}', using {default}")
            return default

    def _get_api_name(self, provider: str, endpoints: dict) -> Optional[str]:
        # Map provider names (case-insensitive keys from config/UI) to endpoint keys in config.toml
        # Ensure these keys match your config.toml [api_endpoints] section
        provider_key_map = {
            "llama_cpp": "llama_cpp",
            "Ollama": "Ollama",  # Assuming key in config is "Ollama"
            "Oobabooga": "Oobabooga",
            "koboldcpp": "koboldcpp",
            "vllm": "vllm",
            "Custom": "Custom",
            "Custom-2": "Custom_2",
            # Add other mappings if needed (TabbyAPI, Aphrodite?)
        }
        endpoint_key = provider_key_map.get(provider)  # Case-sensitive lookup based on UI value
        if endpoint_key:
            url = endpoints.get(endpoint_key)  # Case-sensitive lookup in config dict
            if url:
                logging.debug(f"Using API endpoint '{url}' for provider '{provider}' (key: '{endpoint_key}')")
                return url
            else:
                logging.warning(
                    f"URL key '{endpoint_key}' for provider '{provider}' missing in config [api_endpoints].")
                return None
        else:
            # Cloud providers or those not needing a specific URL here
            logging.debug(f"No specific endpoint URL key configured for provider '{provider}'.")
            return None

    def watch_chat_api_provider_value(self, new_value: Optional[str]) -> None:
        """Watcher triggered when the value of the #chat-api-provider Select changes."""
        # Use print for immediate console feedback
        print(f"\n--- WATCHER TRIGGERED ---")
        print(f"Widget ID: chat-api-provider")
        print(f"New Value Observed: {new_value!r}")
        if new_value is None or new_value == Select.BLANK:
            print(f"Watcher: Value is blank or None. Clearing model options.")
            self._update_model_select("chat", [])  # Pass empty list
            return

        # --- Re-introduce the model update logic here ---
        print(f"Watcher: Updating models for provider '{new_value}'")
        print(f"Watcher: Available Provider Keys: {list(self.providers_models.keys())}")
        models = self.providers_models.get(new_value, [])  # Get models for the new provider key
        print(f"Watcher: Models retrieved: {models}")
        self._update_model_select("chat", models)  # Call helper
        print(f"--- WATCHER END (chat-api-provider) ---\n")

    def watch_character_api_provider_value(self, new_value: Optional[str]) -> None:
        """Watcher triggered when the value of the #character-api-provider Select changes."""
        # Use print for immediate console feedback
        print(f"\n--- WATCHER TRIGGERED ---")
        print(f"Widget ID: character-api-provider")
        print(f"New Value Observed: {new_value!r}")
        if new_value is None or new_value == Select.BLANK:
            print(f"Watcher: Value is blank or None. Clearing model options.")
            self._update_model_select("character", [])
            return

        # --- Re-introduce the model update logic here ---
        print(f"Watcher: Updating models for provider '{new_value}'")
        print(f"Watcher: Available Provider Keys: {list(self.providers_models.keys())}")
        models = self.providers_models.get(new_value, [])
        print(f"Watcher: Models retrieved: {models}")
        self._update_model_select(TAB_CONV_CHAR, models)
        print(f"--- WATCHER END (character-api-provider) ---\n")

    # --- ADD HELPER METHOD to update model select ---
    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        """Helper function to update the model Select options and value."""
        model_select_id = f"#{id_prefix}-api-model"
        print(f"Helper _update_model_select: Targeting '{model_select_id}' with models: {models}")
        try:
            model_select = self.query_one(model_select_id, Select)
        except QueryError:
            print(f"Helper ERROR: Cannot find model select '{model_select_id}'")
            logging.error(f"Helper ERROR: Cannot find model select '{model_select_id}'")
            return

        new_model_options = [(model, model) for model in models]
        print(f"Helper: Setting model options: {new_model_options}")
        try:
            model_select.set_options(new_model_options)
        except Exception as e:
            print(f"Helper ERROR setting options: {e}")
            logging.error(f"Helper ERROR setting options for {model_select_id}: {e}")
            model_select.set_options([])  # Clear on error
            model_select.prompt = "Error"
            model_select.value = None
            return

        # Set value (e.g., to first model or a default)
        model_to_set = None
        if models:
            # You could try and get the default from config here if needed
            # default_model_for_provider = get_setting("api_settings", f"{new_value}.model") # Example
            model_to_set = models[0]  # Simple default: first in list
            print(f"Helper: Setting value to first model: {model_to_set!r}")
        else:
            print(f"Helper: No models, clearing value.")

        try:
            # Explicitly set the value *after* setting options
            # This might be necessary if set_options clears the value
            model_select.value = model_to_set
            print(f">>> HELPER: Model select value AFTER explicit set: {model_select.value!r}")
        except Exception as e:
            print(f">>> HELPER ERROR setting value: {e}")
            logging.error(f"Helper ERROR setting value for {model_select_id}: {e}")

        model_select.prompt = "Select Model..." if models else "No models available"
        print(f"Helper: Model select value after update: {model_select.value!r}")

    async def _populate_conv_char_character_select(self) -> None:
        logging.info("Attempting to populate 'conv-char-character-select' dropdown.")
        if not self.notes_service:
            logging.error("Notes service not available, cannot populate character select for ConvChar tab.")
            return

        try:
            db = self.notes_service._get_db(self.notes_user_id) # Accessing protected member
            # Fetch a reasonable number of characters, assuming names are unique or selection handles it.
            character_cards = db.list_character_cards(limit=1000)

            options = []
            if character_cards:
                options = [(char['name'], char['id']) for char in character_cards if char.get('name') and char.get('id') is not None]

            char_select_widget = self.query_one("#conv-char-character-select", Select)

            if options:
                char_select_widget.set_options(options)
                char_select_widget.value = options[0][1] # Set default to the first character's ID
                logging.info(f"Populated #conv-char-character-select with {len(options)} characters. Default: {options[0][0]} (ID: {options[0][1]})")
            else:
                # Use a list of tuples for set_options, even for a single placeholder
                char_select_widget.set_options([("No characters found", Select.BLANK)])
                char_select_widget.value = Select.BLANK # Ensure value is blank if no options
                char_select_widget.prompt = "No characters available" # Update prompt
                logging.info("No characters found to populate #conv-char-character-select.")

        except QueryError as e:
            logging.error(f"Failed to find #conv-char-character-select widget: {e}", exc_info=True)
        except CharactersRAGDBError as e:
            logging.error(f"Database error populating #conv-char-character-select: {e}", exc_info=True)
            # Optionally update the select prompt to show an error
            try:
                # Ensure char_select_widget is defined or query again if necessary
                # This code block might be entered if char_select_widget was queried successfully before the DB error
                char_select_widget = self.query_one("#conv-char-character-select", Select)
                char_select_widget.prompt = "Error loading characters"
                char_select_widget.set_options([("Error loading", Select.BLANK)])
                char_select_widget.value = Select.BLANK
            except QueryError:
                pass # Widget itself not found
        except Exception as e:
            logging.error(f"Unexpected error populating #conv-char-character-select: {e}", exc_info=True)
            try:
                # Similar to above, try to update the widget to show an error state
                char_select_widget = self.query_one("#conv-char-character-select", Select)
                char_select_widget.prompt = "Error loading characters"
                char_select_widget.set_options([("Error loading", Select.BLANK)])
                char_select_widget.value = Select.BLANK
            except QueryError:
                pass


# --- Main execution block ---
if __name__ == "__main__":
    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w") as f:
                f.write(CONFIG_TOML_CONTENT)  # Write the example content
    except Exception as e:
        logging.error(f"Could not ensure creation of default config file: {e}", exc_info=True)

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS definition ---
    # (Keep your CSS content here, make sure IDs match widgets)
    css_content = """
Screen { layout: vertical; }
Header { dock: top; height: 1; background: $accent-darken-1; }
Footer { dock: bottom; height: 1; background: $accent-darken-1; }
#tabs { dock: top; height: 3; background: $background; padding: 0 1; }
#tabs Button { width: 1fr; height: 100%; border: none; background: $panel; color: $text-muted; }
#tabs Button:hover { background: $panel-lighten-1; color: $text; }
#tabs Button.-active { background: $accent; color: $text; text-style: bold; border: none; }
#content { height: 1fr; width: 100%; }

/* Base style for ALL windows. The watcher will set display: True/False */
.window {
    height: 100%;
    width: 100%;
    layout: horizontal; /* Or vertical if needed by default */
    overflow: hidden;
}

/* REMOVED .hidden class */

.placeholder-window { align: center middle; background: $panel; }

/* Sidebar Styling */
.sidebar { width: 35; background: $boost; padding: 1 2; border-right: thick $background-darken-1; height: 100%; overflow-y: auto; overflow-x: hidden; }
.sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
.sidebar-label { margin-top: 1; text-style: bold; }
.sidebar-input { width: 100%; margin-bottom: 1; }
.sidebar-textarea { width: 100%; height: 5; border: round $surface; margin-bottom: 1; }
.sidebar Select { width: 100%; margin-bottom: 1; }

/* --- Chat Window specific layouts --- */
#chat-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr;
}
/* VerticalScroll for chat messages */
#chat-log {
    height: 1fr; /* Takes remaining space */
    width: 100%;
    /* border: round $surface; Optional: Add border to scroll area */
    padding: 0 1; /* Padding around messages */
}

/* Input area styling (shared by chat and character) */
#chat-input-area, #character-input-area {
    height: auto;    /* Allow height to adjust */
    max-height: 12;  /* Limit growth */
    width: 100%;
    align: left top; /* Align children to top-left */
    padding: 1; /* Consistent padding */
    border-top: round $surface;
}
/* Input widget styling (shared) */
.chat-input { /* Targets TextArea */
    width: 1fr;
    height: auto;      /* Allow height to adjust */
    max-height: 100%;  /* Don't overflow parent */
    margin-right: 1; /* Space before button */
    border: round $surface;
}
/* Send button styling (shared) */
.send-button { /* Targets Button */
    width: 10;
    height: 3; /* Fixed height for consistency */
    /* align-self: stretch; REMOVED */
    margin-top: 0;
}

/* --- Character Chat Window specific layouts --- */
#character-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr;
}
#character-top-area {
    height: 1fr; /* Top area takes remaining vertical space */
    width: 100%;
    layout: horizontal;
    margin-bottom: 1;
}
/* Log when next to portrait (Still RichLog here) */
#character-top-area > #character-log { /* Target by ID is safer */
    margin: 0 1 0 0;
    height: 100%;
    margin-bottom: 0; /* Override base margin */
    border: round $surface; /* Added border back for RichLog */
    padding: 0 1; /* Added padding back for RichLog */
    width: 1fr; /* Ensure it takes space */
}
/* Portrait styling */
#character-portrait {
    width: 25;
    height: 100%;
    border: round $surface;
    padding: 1;
    margin: 0;
    overflow: hidden;
    align: center top;
}

/* Logs Window */
#logs-window { padding: 0; border: none; height: 100%; width: 100%; }
#app-log-display { border: none; height: 1fr; width: 1fr; margin: 0; padding: 1; }

/* --- ChatMessage Styling --- */
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
    height: 1; /* Ensure header is minimal height */
}
.message-text {
    padding: 1; /* Padding around the text itself */
    width: 100%;
    height: auto;
}
.message-actions {
    height: auto;
    width: 100%;
    padding: 1; /* Add padding around buttons */
    /* Use a VALID border type */
    border-top: solid $surface-lighten-1; /* CHANGED thin to solid */
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
/* Initially hide AI actions until generation is complete */
ChatMessage.-ai .message-actions.-generating {
    display: none;
}
/* microphone button – same box as Send but subdued colour */
.mic-button {
    width: 3;
    height: 3;
    margin-right: 1;           /* gap before Send */
    border: none;
    background: $surface-darken-1;
    color: $text-muted;
}
.mic-button:hover {
    background: $surface;
    color: $text;
}
.sidebar-toggle {
    width: 3;
    height: 3;
    margin-right: 1;
    border: none;
    background: $surface-darken-1;
    color: $text;
}
.sidebar-toggle:hover {
    background: $surface;
}

/* collapsed side-bar; width zero and no border */
.sidebar.collapsed {
    width: 0 !important;
    border-right: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none;          /* ensures it doesn’t grab focus */
}
#app-titlebar {
    dock: top;
    height: 1;                 /* single line */
    background: $accent;       /* or any colour */
    color: $text;
    text-align: center;
    text-style: bold;
    padding: 0 1;
}
    """

    # --- CSS File Handling ---
    try:
        css_file = Path(TldwCli.CSS_PATH)
        if not css_file.is_file():
            css_file.parent.mkdir(parents=True, exist_ok=True)
            with open(css_file, "w") as f: f.write(css_content)
            logging.info(f"Created default CSS file: {css_file}")
    except Exception as e:
        logging.error(f"Error handling CSS file '{TldwCli.CSS_PATH}': {e}", exc_info=True)

    # --- Run the App ---
    logging.info("Starting Textual App...")
    # Pass the loaded config to the App instance
    loguru_logger.info("--- INSTANTIATING TldwCli ---")
    app = TldwCli()
    loguru_logger.info("--- INSTANTIATED TldwCli ---")
    loguru_logger.info("--- CALLING app.run() ---")
    try:
        app.run()
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
