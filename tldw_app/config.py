# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
# Imports
import configparser
import tomllib
import logging
import os
from pathlib import Path
import toml
from typing import Dict, Any, List, Optional

from loguru import logger

#
# Third-Party Imports
## No third-party imports in this file
# Local Imports
#
#######################################################################################################################
#
# Functions:

# --- Default Configuration ---
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tldw_cli" / "config.toml"

# Define this list at the module level for clarity
DEFAULT_SUPPORTED_API_PROVIDERS_LIST = [
    "anthropic", "cohere", "deepseek", "google", "groq", "huggingface",
    "mistral", "openai", "openrouter", "llama.cpp", "kobold", "ollama",
    "ooba", "tabbyapi", "vllm", "local-llm", "custom-openai-api", "custom-openai-api-2"
]

DEFAULT_CONFIG = {
    "general": {"default_tab": "chat", "log_level": "INFO"},
    "logging": {
        "log_filename": "tldw_cli_app.log",
        "file_log_level": "INFO",
        "log_max_bytes": 10 * 1024 * 1024, # 10 MB
        "log_backup_count": 5
    },
    "database": {
        "path": "~/.local/share/tldw_cli/tldw_cli_data.db"
    },
    "api_endpoints": { # This section might be for base URLs if not in api_settings
        "Ollama": "http://localhost:11434",
    },
    "api_defaults": { # New section for API related global defaults
        "default_llm_provider": "openai", # This will be the ultimate fallback
        "supported_providers": DEFAULT_SUPPORTED_API_PROVIDERS_LIST
    },
    # *** Keep default providers EMPTY as discussed before - let user config define it ***
    "providers": {},
    "chat_defaults": {
        # Use a provider guaranteed by DEFAULT_CONFIG if file load fails completely
        "provider": "OpenAI", "model": "gpt-4o",
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7, "top_p": 1.0, "min_p": 0.0, "top_k": 0,
    },
    "character_defaults": {
        # Use a fallback provider if file load fails
        "provider": "Ollama", "model": "llama3:latest", # Changed fallback
        "system_prompt": "You are a helpful character.",
        "temperature": 0.8, "top_p": 1.0, "min_p": 0.0, "top_k": 0,
    },
    # api_settings will be populated by user's config.toml; no defaults here needed for structure
    "api_settings": {}
}

# --- Configuration Path ---
def get_config_path() -> Path:
    env_path = os.environ.get("TLDW_CLI_CONFIG_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        logging.debug(f"Using config path from TLDW_CLI_CONFIG_PATH: {path}")
        return path
    config_dir = Path.home() / ".config" / "tldw_cli"
    default_path = config_dir / "config.toml"
    logging.debug(f"Using default config path: {default_path}")
    return default_path

# --- Global Cache ---
_APP_CONFIG: Optional[Dict[str, Any]] = None

# --- Loading Function ---
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration, merges with defaults, handles errors, caches."""
    global _APP_CONFIG
    # Use print for initial check as logging might not be ready
    print(f"--- load_config: Top - _APP_CONFIG is None? {_APP_CONFIG is None} ---")
    if _APP_CONFIG is not None:
        print(f"--- load_config: RETURNING CACHED CONFIG ---")
        # logging.debug("Returning cached config.") # Log is fine here
        return _APP_CONFIG

    if config_path is None:
        config_path = get_config_path()

    print(f"--- load_config: Checking path: {config_path.resolve()}")
    print(f"--- load_config: Does path exist? {config_path.exists()}")

    # Deep copy defaults FIRST
    config = {k: v.copy() if isinstance(v, dict) else (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    logging.info(f"Initialized config with DEFAULTS. Default providers: {config.get('providers')}")

    try:
        print(f"--- load_config: Before mkdir parent: {config_path.parent} ---")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- load_config: After mkdir parent ---")
        logging.info(f"Ensured parent directory exists: {config_path.parent}")

        if config_path.exists():
            print(f"--- load_config: INSIDE if config_path.exists() ---")
            logging.info(f"Config file FOUND at {config_path}")
            user_config = None
            try:
                print(f"--- load_config: Entering inner try block ---")
                print(f"--- load_config: Attempting 'with open...' ---")
                with open(config_path, "rb") as f:
                    print(f"--- load_config: File opened. Attempting tomllib.load... ---")
                    user_config = tomllib.load(f)
                    print(f"--- load_config: tomllib.load SUCCEEDED ---")

                if user_config:
                    logging.info(f"Successfully parsed TOML file.")
                    raw_providers = user_config.get("providers")
                    print(f"--- load_config: Raw [providers] section: {raw_providers} ---")

                    # --- Merge Logic ---
                    logging.debug("Starting merge process...")
                    for section, section_config in user_config.items():
                        if section in config and isinstance(config[section], dict) and isinstance(section_config, dict):
                            if section == "providers" or section == "api_settings" or section == "api_defaults": # Sections to replace or deep merge carefully
                                # For api_settings, we want to merge provider by provider
                                if section == "api_settings":
                                    for prov_key, prov_config in section_config.items():
                                        if prov_key not in config[section]:
                                            config[section][prov_key] = {}
                                        if isinstance(config[section].get(prov_key), dict) and isinstance(prov_config, dict):
                                             config[section][prov_key].update(prov_config)
                                        else:
                                            config[section][prov_key] = prov_config # Overwrite if types don't match for update
                                    logging.info(f"MERGE: Deep-merged '{section}' section from user config.")
                                else: # For 'providers' and 'api_defaults', user config replaces default section.
                                    config[section] = section_config
                                    logging.info(f"MERGE: Replaced entire '{section}' section from user config.")
                            else: # Default merge for other dict sections
                                config[section].update(section_config)
                                logging.info(f"MERGE: Updated '{section}' section.")
                        else: # Add/overwrite non-dict sections or new sections
                            config[section] = section_config
                            logging.info(f"MERGE: Set/added section '{section}' from user config.")
                    logging.debug("Finished merge process.")
                    # --- End Merge Logic ---

                else:
                    print(f"--- load_config: WARNING - tomllib.load did not return data ---")
                    logging.warning("Parsed TOML file but result was empty/None.")

            except tomllib.TOMLDecodeError as e:
                print(f"--- load_config: EXCEPTION - TOMLDecodeError: {e} ---")
                logging.error(f"!!! TOML Decode Error in {config_path}: {e}", exc_info=True)
                logging.warning("Using (initial) default configuration values due to TOML error.")
            except Exception as e:
                print(f"--- load_config: EXCEPTION - General Exception during parse/merge: {e} ---")
                logging.error(f"!!! Unexpected error processing config file {config_path}: {e}", exc_info=True)
                logging.warning("Using (initial) default configuration values due to loading error.")
        else:
            logging.warning(f"Config file NOT FOUND at {config_path}. Using defaults and attempting to create.")
            # Default creation logic (keep as is)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    toml.dump(DEFAULT_CONFIG, f) # Dump the basic default structure
                logging.info(f"Created default configuration file at: {config_path}")
            except ImportError:
                 logging.warning("`toml` library not found. Cannot write default config file.")
                 with open(config_path, "w", encoding="utf-8") as f:
                     f.write("# Config file for tldw_cli. Install 'toml' or create manually.\n")
            except Exception as e:
                logging.error(f"Failed to create default config file at {config_path}: {e}", exc_info=True)

    except OSError as e:
        logging.error(f"OS error accessing config directory or file {config_path}: {e}", exc_info=True)
        logging.warning("Using (initial) default configuration values due to OS error.")
    except Exception as e:
        logging.error(f"General error during config loading setup for {config_path}: {e}", exc_info=True)
        logging.warning("Using (initial) default configuration values due to unexpected error.")

    # Log final state and cache
    merged_providers = config.get("providers")
    logging.info(f"FINAL_CONFIG_STATE: Final merged [providers] section keys: {list(merged_providers.keys()) if merged_providers else 'None'}")
    _APP_CONFIG = config
    logging.debug(f"Configuration loaded and cached. Sections={list(_APP_CONFIG.keys())}")
    return _APP_CONFIG


# --- Convenience Access Functions ---
def get_setting(section: str, key: str, default: Any = None) -> Any:
    config = load_config()
    return config.get(section, {}).get(key, default)

# --- New/Modified Config Functions ---

def get_default_llm_provider() -> str:
    """
    Determines the default LLM provider.
    Priority:
    1. Environment variable "DEFAULT_LLM_PROVIDER".
    2. Value from config file: [api_defaults].default_llm_provider.
    3. Hardcoded default in DEFAULT_CONFIG.
    """
    env_provider = os.getenv("DEFAULT_LLM_PROVIDER")
    if env_provider:
        logging.debug(f"Using default LLM provider from env var DEFAULT_LLM_PROVIDER: {env_provider}")
        return env_provider

    # get_setting will fetch from user's toml or DEFAULT_CONFIG's default.
    # The ultimate fallback "openai" is passed to get_setting.
    config_provider = get_setting("api_defaults", "default_llm_provider", DEFAULT_CONFIG["api_defaults"]["default_llm_provider"])
    logging.debug(f"Using default LLM provider from config: {config_provider}")
    return config_provider


def get_supported_api_providers() -> List[str]:
    """
    Returns the list of supported API provider names.
    Priority:
    1. List from config file: [api_defaults].supported_providers.
    2. Default list defined in DEFAULT_CONFIG.
    """
    # get_setting will fetch from user's toml or DEFAULT_CONFIG's default.
    # The ultimate fallback is DEFAULT_SUPPORTED_API_PROVIDERS_LIST.
    providers_list = get_setting("api_defaults", "supported_providers", DEFAULT_CONFIG["api_defaults"]["supported_providers"])

    if not providers_list or not isinstance(providers_list, list):
        logging.warning("Supported providers list is invalid or empty in config, falling back to internal default.")
        return DEFAULT_SUPPORTED_API_PROVIDERS_LIST.copy() # Return a copy

    return providers_list


# Mapping from canonical provider names (used in Literal) to keys in [api_settings] if they differ.
# This helps keep the Literal names user-friendly while allowing different TOML keys if needed.
PROVIDER_NAME_TO_API_SETTINGS_KEY_MAP = {
    "mistral": "mistralai",           # e.g., Literal uses "mistral", config.toml has [api_settings.mistralai]
    "custom-openai-api": "custom",    # Literal uses "custom-openai-api", config.toml has [api_settings.custom]
    "custom-openai-api-2": "custom_2",# Literal uses "custom-openai-api-2", config.toml has [api_settings.custom_2]
    "kobold": "koboldcpp",            # Literal uses "kobold", config.toml has [api_settings.koboldcpp]
    "ooba": "oobabooga",              # Literal uses "ooba", config.toml has [api_settings.oobabooga]
    # "llama.cpp" is handled by normalization: llama.cpp -> llama_cpp
    # Add other mappings here if `DEFAULT_SUPPORTED_API_PROVIDERS_LIST` names
    # don't directly map to `api_settings.<key>` after basic normalization.
}

def get_api_key(provider_name: str) -> Optional[str]:
    """
    Retrieves the API key for a given provider.
    Checks environment variables specified in config, then direct 'api_key' in config.
    Also checks generic environment variable names (e.g., OPENAI_API_KEY).
    """
    config = load_config()

    # Normalize the input provider_name and check mapping for the settings key
    base_normalized_name = provider_name.lower().replace('-', '_').replace('.', '_')
    settings_key = PROVIDER_NAME_TO_API_SETTINGS_KEY_MAP.get(provider_name, base_normalized_name)

    provider_settings = config.get("api_settings", {}).get(settings_key, {})

    if not provider_settings and settings_key != base_normalized_name: # Try base normalization if mapped key yielded no settings
        provider_settings = config.get("api_settings", {}).get(base_normalized_name, {})


    # 1. Try fetching from environment variable specified in config (e.g., api_settings.openai.api_key_env_var)
    env_var_name_from_config = provider_settings.get("api_key_env_var")
    if env_var_name_from_config:
        api_key = os.getenv(env_var_name_from_config)
        if api_key:
            logging.debug(f"API key for '{provider_name}' (settings key '{settings_key}') found in env var '{env_var_name_from_config}'.")
            return api_key

    # 2. Try fetching from 'api_key' field directly in config (e.g., api_settings.openai.api_key)
    api_key_direct = provider_settings.get("api_key")
    if api_key_direct:
        logging.debug(f"API key for '{provider_name}' (settings key '{settings_key}') found directly in config.")
        return api_key_direct

    # 3. Fallback: Check a generic environment variable format (e.g., PROVIDER_NAME_API_KEY)
    #    This matches the original logic in chat_request_schemas.py
    generic_env_var_name = f"{provider_name.upper().replace('.', '_').replace('-', '_')}_API_KEY"
    api_key_generic = os.getenv(generic_env_var_name)
    if api_key_generic:
        logging.debug(f"API key for '{provider_name}' found in generic env var '{generic_env_var_name}'.")
        return api_key_generic

    logging.debug(f"API key for '{provider_name}' (settings key '{settings_key}') not found in specified env var, direct config, or generic env var.")
    return None

def get_providers_and_models() -> Dict[str, List[str]]:
    """
    Loads provider and model configuration, validates it, and returns
    a dictionary of valid providers mapped to their list of models.
    """
    config = load_config()
    providers_data = config.get("providers", {})

    valid_providers: Dict[str, List[str]] = {}

    if isinstance(providers_data, dict):
        for provider, models in providers_data.items():
            if isinstance(models, list) and all(isinstance(m, str) for m in models):
                valid_providers[provider] = models
            else:
                logging.warning(f"Invalid model list for provider '{provider}' in loaded config's [providers] section. Value: {models!r}. Skipping.")
    else:
        logging.error(f"Loaded 'providers' section is not a dictionary: {providers_data!r}. No provider/model data available from [providers] section.")

    logging.debug(f"get_providers_and_models returning providers: {list(valid_providers.keys())}")

    return valid_providers

# (get_database_path and get_log_file_path remain the same)
def get_database_path() -> Path:
    config = load_config() # Ensures config is loaded
    db_path_str = get_setting("database", "path", DEFAULT_CONFIG["database"]["path"])
    db_path = Path(db_path_str).expanduser().resolve()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create database directory {db_path.parent}: {e}", exc_info=True)
    return db_path

def get_log_file_path() -> Path:
    # Ensure database path (and thus its parent dir) is resolved first, as log might go there.
    db_parent_dir = get_database_path().parent
    log_filename = get_setting("logging", "log_filename", DEFAULT_CONFIG["logging"]["log_filename"])
    return db_parent_dir / log_filename



# --- Helper Function (Optional but can keep dictionary creation clean) ---
def load_settings():
    """Loads all settings from environment variables or defaults into a dictionary."""

    # Determine Actual Project Root based on the location of this file
    # config.py is in project_root/tldw_server_api/app/core/config.py
    # ACTUAL_PROJECT_ROOT will be /project_root/
    ACTUAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    logger.info(f"Determined ACTUAL_PROJECT_ROOT for database paths: {ACTUAL_PROJECT_ROOT}")

    # Base directory for all user-specific data: ACTUAL_PROJECT_ROOT/user_databases/
    default_user_data_base_dir = ACTUAL_PROJECT_ROOT / "user_databases"
    user_data_base_dir_str = os.getenv("USER_DB_BASE_DIR", str(default_user_data_base_dir.resolve()))
    user_data_base_dir = Path(user_data_base_dir_str)

    # Main/central SQLite database: ACTUAL_PROJECT_ROOT/user_databases/databases/tldw.db
    # FIXME
    #default_main_db_path = (ACTUAL_PROJECT_ROOT / "user_databases" / f"{single_user_username}" / "tldw.db").resolve()
    #default_database_url = f"sqlite:///{default_main_db_path}"
    #database_url = os.getenv("DATABASE_URL", default_database_url)

    users_db_configured = os.getenv("USERS_DB_ENABLED", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Load comprehensive configurations (API keys, embedding settings, etc.)
    comprehensive_config = {}
    try:
        comprehensive_config = load_and_log_configs() # This function is already defined in your provided code
        if comprehensive_config is None:
            logger.error("Failed to load comprehensive_config, will use fallbacks for some settings.")
            comprehensive_config = {} # Ensure it's a dict to avoid errors on .get()
    except Exception as e:
        logger.error(f"Error loading comprehensive_config: {e}", exc_info=True)
        comprehensive_config = {}


    config_dict = {
        # General App
        "LOG_LEVEL": log_level,
        "PROJECT_ROOT": ACTUAL_PROJECT_ROOT, # Centralized project root definition

        # Merge relevant parts from comprehensive_config
        # Embedding Config
        "EMBEDDING_CONFIG": comprehensive_config.get("embedding_config", {
            'embedding_provider': 'openai', # Fallback defaults
            'embedding_model': 'text-embedding-3-small',
            'onnx_model_path': "./Models/onnx_models/text-embedding-3-small.onnx",
            'model_dir': "./Models",
            'embedding_api_url': "http://localhost:8080/v1/embeddings",
            'embedding_api_key': '',
            'chunk_size': 400,
            'chunk_overlap': 200
        }),
        # Add other configs from comprehensive_config as needed
        "OPENAI_API_KEY": comprehensive_config.get("openai_api", {}).get("api_key", os.getenv("OPENAI_API_KEY")),
        # You can continue to merge other specific keys or whole sections
        "COMPREHENSIVE_CONFIG_RAW": comprehensive_config # Store the raw one if needed elsewhere
    }

    # Create necessary directories if they don't exist
    # Ensure main SQLite database directory exists
    # FIXME

    # Ensure USER_DB_BASE_DIR exists (base for user-specific SQLite and ChromaDB)
    # FIXME

    return config_dict


def load_comprehensive_config():
    current_file_path = Path(__file__).resolve()
    # Correct project_root calculation:
    # __file__ is .../tldw_Server_API/app/core/config.py
    # .parent -> .../app/core
    # .parent.parent -> .../app
    # .parent.parent.parent -> .../tldw_Server_API (This is the project root)
    project_root = current_file_path.parent.parent.parent

    config_path_obj = project_root / 'Config_Files' / 'config.txt'

    logger.info(f"Attempting to load comprehensive config from: {str(config_path_obj)}")

    if not config_path_obj.exists():
        logger.error(f"Config file not found at {str(config_path_obj)}")
        raise FileNotFoundError(f"Config file not found at {str(config_path_obj)}")

    config_parser = configparser.ConfigParser()
    try:
        config_parser.read(config_path_obj)  # configparser can read Path objects directly
    except configparser.Error as e:
        logger.error(f"Error parsing config file {str(config_path_obj)}: {e}", exc_info=True)
        raise  # Re-raise the parsing error to be caught by load_and_log_configs

    logger.info(f"load_comprehensive_config(): Sections found in config: {config_parser.sections()}")
    return config_parser

def load_and_log_configs():
    logger.debug("load_and_log_configs(): Loading and logging configurations...")
    try:
        # The 'config' variable below should be the result from load_comprehensive_config()
        config_parser_object = load_comprehensive_config()

        # This check might be redundant if load_comprehensive_config always raises on critical failure
        if config_parser_object is None:
            logger.error("Comprehensive config object is None, cannot proceed")  # Changed to logger
            return None
        # API Keys
        anthropic_api_key = config_parser_object.get('API', 'anthropic_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = config_parser_object.get('API', 'cohere_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = config_parser_object.get('API', 'groq_api_key', fallback=None)
        # logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = config_parser_object.get('API', 'openai_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = config_parser_object.get('API', 'huggingface_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = config_parser_object.get('API', 'openrouter_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        deepseek_api_key = config_parser_object.get('API', 'deepseek_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        mistral_api_key = config_parser_object.get('API', 'mistral_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")

        google_api_key = config_parser_object.get('API', 'google_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")

        elevenlabs_api_key = config_parser_object.get('API', 'elevenlabs_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded elevenlabs API Key: {elevenlabs_api_key[:5]}...{elevenlabs_api_key[-5:] if elevenlabs_api_key else None}")

        # LLM API Settings - streaming / temperature / top_p / min_p
        # Anthropic
        anthropic_api_key = config_parser_object.get('API', 'anthropic_api_key', fallback=None)
        anthropic_model = config_parser_object.get('API', 'anthropic_model', fallback='claude-3-5-sonnet-20240620')
        anthropic_streaming = config_parser_object.get('API', 'anthropic_streaming', fallback='False')
        anthropic_temperature = config_parser_object.get('API', 'anthropic_temperature', fallback='0.7')
        anthropic_top_p = config_parser_object.get('API', 'anthropic_top_p', fallback='0.95')
        anthropic_top_k = config_parser_object.get('API', 'anthropic_top_k', fallback='100')
        anthropic_max_tokens = config_parser_object.get('API', 'anthropic_max_tokens', fallback='4096')
        anthropic_api_timeout = config_parser_object.get('API', 'anthropic_api_timeout', fallback='90')
        anthropic_api_retries = config_parser_object.get('API', 'anthropic_api_retry', fallback='3')
        anthropic_api_retry_delay = config_parser_object.get('API', 'anthropic_api_retry_delay', fallback='5')

        # Cohere
        cohere_streaming = config_parser_object.get('API', 'cohere_streaming', fallback='False')
        cohere_temperature = config_parser_object.get('API', 'cohere_temperature', fallback='0.7')
        cohere_max_p = config_parser_object.get('API', 'cohere_max_p', fallback='0.95')
        cohere_top_k = config_parser_object.get('API', 'cohere_top_k', fallback='100')
        cohere_model = config_parser_object.get('API', 'cohere_model', fallback='command-r-plus')
        cohere_max_tokens = config_parser_object.get('API', 'cohere_max_tokens', fallback='4096')
        cohere_api_timeout = config_parser_object.get('API', 'cohere_api_timeout', fallback='90')
        cohere_api_retries = config_parser_object.get('API', 'cohere_api_retry', fallback='3')
        cohere_api_retry_delay = config_parser_object.get('API', 'cohere_api_retry_delay', fallback='5')

        # Deepseek
        deepseek_streaming = config_parser_object.get('API', 'deepseek_streaming', fallback='False')
        deepseek_temperature = config_parser_object.get('API', 'deepseek_temperature', fallback='0.7')
        deepseek_top_p = config_parser_object.get('API', 'deepseek_top_p', fallback='0.95')
        deepseek_min_p = config_parser_object.get('API', 'deepseek_min_p', fallback='0.05')
        deepseek_model = config_parser_object.get('API', 'deepseek_model', fallback='deepseek-chat')
        deepseek_max_tokens = config_parser_object.get('API', 'deepseek_max_tokens', fallback='4096')
        deepseek_api_timeout = config_parser_object.get('API', 'deepseek_api_timeout', fallback='90')
        deepseek_api_retries = config_parser_object.get('API', 'deepseek_api_retry', fallback='3')
        deepseek_api_retry_delay = config_parser_object.get('API', 'deepseek_api_retry_delay', fallback='5')

        # Groq
        groq_model = config_parser_object.get('API', 'groq_model', fallback='llama3-70b-8192')
        groq_streaming = config_parser_object.get('API', 'groq_streaming', fallback='False')
        groq_temperature = config_parser_object.get('API', 'groq_temperature', fallback='0.7')
        groq_top_p = config_parser_object.get('API', 'groq_top_p', fallback='0.95')
        groq_max_tokens = config_parser_object.get('API', 'groq_max_tokens', fallback='4096')
        groq_api_timeout = config_parser_object.get('API', 'groq_api_timeout', fallback='90')
        groq_api_retries = config_parser_object.get('API', 'groq_api_retry', fallback='3')
        groq_api_retry_delay = config_parser_object.get('API', 'groq_api_retry_delay', fallback='5')

        # Google
        google_model = config_parser_object.get('API', 'google_model', fallback='gemini-1.5-pro')
        google_streaming = config_parser_object.get('API', 'google_streaming', fallback='False')
        google_temperature = config_parser_object.get('API', 'google_temperature', fallback='0.7')
        google_top_p = config_parser_object.get('API', 'google_top_p', fallback='0.95')
        google_min_p = config_parser_object.get('API', 'google_min_p', fallback='0.05')
        google_max_tokens = config_parser_object.get('API', 'google_max_tokens', fallback='4096')
        google_api_timeout = config_parser_object.get('API', 'google_api_timeout', fallback='90')
        google_api_retries = config_parser_object.get('API', 'google_api_retry', fallback='3')
        google_api_retry_delay = config_parser_object.get('API', 'google_api_retry_delay', fallback='5')

        # HuggingFace
        huggingface_use_router_url_format = config_parser_object.getboolean('API', 'huggingface_use_router_url_format', fallback=False)
        huggingface_router_base_url = config_parser_object.get('API', 'huggingface_router_base_url', fallback='https://router.huggingface.co/hf-inference')
        huggingface_api_base_url = config_parser_object.get('API', 'huggingface_api_base_url', fallback='https://router.huggingface.co/hf-inference/models')
        huggingface_model = config_parser_object.get('API', 'huggingface_model', fallback='/Qwen/Qwen3-235B-A22B')
        huggingface_streaming = config_parser_object.get('API', 'huggingface_streaming', fallback='False')
        huggingface_temperature = config_parser_object.get('API', 'huggingface_temperature', fallback='0.7')
        huggingface_top_p = config_parser_object.get('API', 'huggingface_top_p', fallback='0.95')
        huggingface_min_p = config_parser_object.get('API', 'huggingface_min_p', fallback='0.05')
        huggingface_max_tokens = config_parser_object.get('API', 'huggingface_max_tokens', fallback='4096')
        huggingface_api_timeout = config_parser_object.get('API', 'huggingface_api_timeout', fallback='90')
        huggingface_api_retries = config_parser_object.get('API', 'huggingface_api_retry', fallback='3')
        huggingface_api_retry_delay = config_parser_object.get('API', 'huggingface_api_retry_delay', fallback='5')

        # Mistral
        mistral_model = config_parser_object.get('API', 'mistral_model', fallback='mistral-large-latest')
        mistral_streaming = config_parser_object.get('API', 'mistral_streaming', fallback='False')
        mistral_temperature = config_parser_object.get('API', 'mistral_temperature', fallback='0.7')
        mistral_top_p = config_parser_object.get('API', 'mistral_top_p', fallback='0.95')
        mistral_max_tokens = config_parser_object.get('API', 'mistral_max_tokens', fallback='4096')
        mistral_api_timeout = config_parser_object.get('API', 'mistral_api_timeout', fallback='90')
        mistral_api_retries = config_parser_object.get('API', 'mistral_api_retry', fallback='3')
        mistral_api_retry_delay = config_parser_object.get('API', 'mistral_api_retry_delay', fallback='5')

        # OpenAI
        openai_model = config_parser_object.get('API', 'openai_model', fallback='gpt-4o')
        openai_streaming = config_parser_object.get('API', 'openai_streaming', fallback='False')
        openai_temperature = config_parser_object.get('API', 'openai_temperature', fallback='0.7')
        openai_top_p = config_parser_object.get('API', 'openai_top_p', fallback='0.95')
        openai_max_tokens = config_parser_object.get('API', 'openai_max_tokens', fallback='4096')
        openai_api_timeout = config_parser_object.get('API', 'openai_api_timeout', fallback='90')
        openai_api_retries = config_parser_object.get('API', 'openai_api_retry', fallback='3')
        openai_api_retry_delay = config_parser_object.get('API', 'openai_api_retry_delay', fallback='5')

        # OpenRouter
        openrouter_model = config_parser_object.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')
        openrouter_streaming = config_parser_object.get('API', 'openrouter_streaming', fallback='False')
        openrouter_temperature = config_parser_object.get('API', 'openrouter_temperature', fallback='0.7')
        openrouter_top_p = config_parser_object.get('API', 'openrouter_top_p', fallback='0.95')
        openrouter_min_p = config_parser_object.get('API', 'openrouter_min_p', fallback='0.05')
        openrouter_top_k = config_parser_object.get('API', 'openrouter_top_k', fallback='100')
        openrouter_max_tokens = config_parser_object.get('API', 'openrouter_max_tokens', fallback='4096')
        openrouter_api_timeout = config_parser_object.get('API', 'openrouter_api_timeout', fallback='90')
        openrouter_api_retries = config_parser_object.get('API', 'openrouter_api_retry', fallback='3')
        openrouter_api_retry_delay = config_parser_object.get('API', 'openrouter_api_retry_delay', fallback='5')

        # Logging Checks for model loads
        # logging.debug(f"Loaded Anthropic Model: {anthropic_model}")
        # logging.debug(f"Loaded Cohere Model: {cohere_model}")
        # logging.debug(f"Loaded Groq Model: {groq_model}")
        # logging.debug(f"Loaded OpenAI Model: {openai_model}")
        # logging.debug(f"Loaded HuggingFace Model: {huggingface_model}")
        # logging.debug(f"Loaded OpenRouter Model: {openrouter_model}")
        # logging.debug(f"Loaded Deepseek Model: {deepseek_model}")
        # logging.debug(f"Loaded Mistral Model: {mistral_model}")

        # Local-Models
        kobold_api_ip = config_parser_object.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        kobold_openai_api_IP = config_parser_object.get('Local-API', 'kobold_openai_api_IP', fallback='http://127.0.0.1:5001/v1/chat/completions')
        kobold_api_key = config_parser_object.get('Local-API', 'kobold_api_key', fallback='')
        kobold_streaming = config_parser_object.get('Local-API', 'kobold_streaming', fallback='False')
        kobold_temperature = config_parser_object.get('Local-API', 'kobold_temperature', fallback='0.7')
        kobold_top_p = config_parser_object.get('Local-API', 'kobold_top_p', fallback='0.95')
        kobold_top_k = config_parser_object.get('Local-API', 'kobold_top_k', fallback='100')
        kobold_max_tokens = config_parser_object.get('Local-API', 'kobold_max_tokens', fallback='4096')
        kobold_api_timeout = config_parser_object.get('Local-API', 'kobold_api_timeout', fallback='90')
        kobold_api_retries = config_parser_object.get('Local-API', 'kobold_api_retry', fallback='3')
        kobold_api_retry_delay = config_parser_object.get('Local-API', 'kobold_api_retry_delay', fallback='5')

        llama_api_IP = config_parser_object.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        llama_api_key = config_parser_object.get('Local-API', 'llama_api_key', fallback='')
        llama_streaming = config_parser_object.get('Local-API', 'llama_streaming', fallback='False')
        llama_temperature = config_parser_object.get('Local-API', 'llama_temperature', fallback='0.7')
        llama_top_p = config_parser_object.get('Local-API', 'llama_top_p', fallback='0.95')
        llama_min_p = config_parser_object.get('Local-API', 'llama_min_p', fallback='0.05')
        llama_top_k = config_parser_object.get('Local-API', 'llama_top_k', fallback='100')
        llama_max_tokens = config_parser_object.get('Local-API', 'llama_max_tokens', fallback='4096')
        llama_api_timeout = config_parser_object.get('Local-API', 'llama_api_timeout', fallback='90')
        llama_api_retries = config_parser_object.get('Local-API', 'llama_api_retry', fallback='3')
        llama_api_retry_delay = config_parser_object.get('Local-API', 'llama_api_retry_delay', fallback='5')

        ooba_api_IP = config_parser_object.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
        ooba_api_key = config_parser_object.get('Local-API', 'ooba_api_key', fallback='')
        ooba_streaming = config_parser_object.get('Local-API', 'ooba_streaming', fallback='False')
        ooba_temperature = config_parser_object.get('Local-API', 'ooba_temperature', fallback='0.7')
        ooba_top_p = config_parser_object.get('Local-API', 'ooba_top_p', fallback='0.95')
        ooba_min_p = config_parser_object.get('Local-API', 'ooba_min_p', fallback='0.05')
        ooba_top_k = config_parser_object.get('Local-API', 'ooba_top_k', fallback='100')
        ooba_max_tokens = config_parser_object.get('Local-API', 'ooba_max_tokens', fallback='4096')
        ooba_api_timeout = config_parser_object.get('Local-API', 'ooba_api_timeout', fallback='90')
        ooba_api_retries = config_parser_object.get('Local-API', 'ooba_api_retry', fallback='3')
        ooba_api_retry_delay = config_parser_object.get('Local-API', 'ooba_api_retry_delay', fallback='5')

        tabby_api_IP = config_parser_object.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        tabby_api_key = config_parser_object.get('Local-API', 'tabby_api_key', fallback=None)
        tabby_model = config_parser_object.get('models', 'tabby_model', fallback=None)
        tabby_streaming = config_parser_object.get('Local-API', 'tabby_streaming', fallback='False')
        tabby_temperature = config_parser_object.get('Local-API', 'tabby_temperature', fallback='0.7')
        tabby_top_p = config_parser_object.get('Local-API', 'tabby_top_p', fallback='0.95')
        tabby_top_k = config_parser_object.get('Local-API', 'tabby_top_k', fallback='100')
        tabby_min_p = config_parser_object.get('Local-API', 'tabby_min_p', fallback='0.05')
        tabby_max_tokens = config_parser_object.get('Local-API', 'tabby_max_tokens', fallback='4096')
        tabby_api_timeout = config_parser_object.get('Local-API', 'tabby_api_timeout', fallback='90')
        tabby_api_retries = config_parser_object.get('Local-API', 'tabby_api_retry', fallback='3')
        tabby_api_retry_delay = config_parser_object.get('Local-API', 'tabby_api_retry_delay', fallback='5')

        vllm_api_url = config_parser_object.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
        vllm_api_key = config_parser_object.get('Local-API', 'vllm_api_key', fallback=None)
        vllm_model = config_parser_object.get('Local-API', 'vllm_model', fallback=None)
        vllm_streaming = config_parser_object.get('Local-API', 'vllm_streaming', fallback='False')
        vllm_temperature = config_parser_object.get('Local-API', 'vllm_temperature', fallback='0.7')
        vllm_top_p = config_parser_object.get('Local-API', 'vllm_top_p', fallback='0.95')
        vllm_top_k = config_parser_object.get('Local-API', 'vllm_top_k', fallback='100')
        vllm_min_p = config_parser_object.get('Local-API', 'vllm_min_p', fallback='0.05')
        vllm_max_tokens = config_parser_object.get('Local-API', 'vllm_max_tokens', fallback='4096')
        vllm_api_timeout = config_parser_object.get('Local-API', 'vllm_api_timeout', fallback='90')
        vllm_api_retries = config_parser_object.get('Local-API', 'vllm_api_retry', fallback='3')
        vllm_api_retry_delay = config_parser_object.get('Local-API', 'vllm_api_retry_delay', fallback='5')

        ollama_api_url = config_parser_object.get('Local-API', 'ollama_api_IP', fallback='http://127.0.0.1:11434/api/generate')
        ollama_api_key = config_parser_object.get('Local-API', 'ollama_api_key', fallback=None)
        ollama_model = config_parser_object.get('Local-API', 'ollama_model', fallback=None)
        ollama_streaming = config_parser_object.get('Local-API', 'ollama_streaming', fallback='False')
        ollama_temperature = config_parser_object.get('Local-API', 'ollama_temperature', fallback='0.7')
        ollama_top_p = config_parser_object.get('Local-API', 'ollama_top_p', fallback='0.95')
        ollama_max_tokens = config_parser_object.get('Local-API', 'ollama_max_tokens', fallback='4096')
        ollama_api_timeout = config_parser_object.get('Local-API', 'ollama_api_timeout', fallback='90')
        ollama_api_retries = config_parser_object.get('Local-API', 'ollama_api_retry', fallback='3')
        ollama_api_retry_delay = config_parser_object.get('Local-API', 'ollama_api_retry_delay', fallback='5')

        aphrodite_api_url = config_parser_object.get('Local-API', 'aphrodite_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        aphrodite_api_key = config_parser_object.get('Local-API', 'aphrodite_api_key', fallback='')
        aphrodite_model = config_parser_object.get('Local-API', 'aphrodite_model', fallback='')
        aphrodite_max_tokens = config_parser_object.get('Local-API', 'aphrodite_max_tokens', fallback='4096')
        aphrodite_streaming = config_parser_object.get('Local-API', 'aphrodite_streaming', fallback='False')
        aphrodite_api_timeout = config_parser_object.get('Local-API', 'llama_api_timeout', fallback='90')
        aphrodite_api_retries = config_parser_object.get('Local-API', 'aphrodite_api_retry', fallback='3')
        aphrodite_api_retry_delay = config_parser_object.get('Local-API', 'aphrodite_api_retry_delay', fallback='5')

        custom_openai_api_key = config_parser_object.get('API', 'custom_openai_api_key', fallback=None)
        custom_openai_api_ip = config_parser_object.get('API', 'custom_openai_api_ip', fallback=None)
        custom_openai_api_model = config_parser_object.get('API', 'custom_openai_api_model', fallback=None)
        custom_openai_api_streaming = config_parser_object.get('API', 'custom_openai_api_streaming', fallback='False')
        custom_openai_api_temperature = config_parser_object.get('API', 'custom_openai_api_temperature', fallback='0.7')
        custom_openai_api_top_p = config_parser_object.get('API', 'custom_openai_api_top_p', fallback='0.95')
        custom_openai_api_min_p = config_parser_object.get('API', 'custom_openai_api_top_k', fallback='100')
        custom_openai_api_max_tokens = config_parser_object.get('API', 'custom_openai_api_max_tokens', fallback='4096')
        custom_openai_api_timeout = config_parser_object.get('API', 'custom_openai_api_timeout', fallback='90')
        custom_openai_api_retries = config_parser_object.get('API', 'custom_openai_api_retry', fallback='3')
        custom_openai_api_retry_delay = config_parser_object.get('API', 'custom_openai_api_retry_delay', fallback='5')

        # 2nd Custom OpenAI API
        custom_openai2_api_key = config_parser_object.get('API', 'custom_openai2_api_key', fallback=None)
        custom_openai2_api_ip = config_parser_object.get('API', 'custom_openai2_api_ip', fallback=None)
        custom_openai2_api_model = config_parser_object.get('API', 'custom_openai2_api_model', fallback=None)
        custom_openai2_api_streaming = config_parser_object.get('API', 'custom_openai2_api_streaming', fallback='False')
        custom_openai2_api_temperature = config_parser_object.get('API', 'custom_openai2_api_temperature', fallback='0.7')
        custom_openai2_api_top_p = config_parser_object.get('API', 'custom_openai_api2_top_p', fallback='0.95')
        custom_openai2_api_min_p = config_parser_object.get('API', 'custom_openai_api2_top_k', fallback='100')
        custom_openai2_api_max_tokens = config_parser_object.get('API', 'custom_openai2_api_max_tokens', fallback='4096')
        custom_openai2_api_timeout = config_parser_object.get('API', 'custom_openai2_api_timeout', fallback='90')
        custom_openai2_api_retries = config_parser_object.get('API', 'custom_openai2_api_retry', fallback='3')
        custom_openai2_api_retry_delay = config_parser_object.get('API', 'custom_openai2_api_retry_delay', fallback='5')

        # Logging Checks for Local API IP loads
        # logging.debug(f"Loaded Kobold API IP: {kobold_api_ip}")
        # logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        # logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        # logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        # logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve default API choices from the configuration file
        default_api = config_parser_object.get('API', 'default_api', fallback='openai')

        # Retrieve LLM API settings from the configuration file
        local_api_retries = config_parser_object.get('Local-API', 'Settings', fallback='3')
        local_api_retry_delay = config_parser_object.get('Local-API', 'local_api_retry_delay', fallback='5')

        # Retrieve output paths from the configuration file
        output_path = config_parser_object.get('Paths', 'output_path', fallback='results')
        logger.trace(f"Output path set to: {output_path}")

        # Save video transcripts
        save_video_transcripts = config_parser_object.get('Paths', 'save_video_transcripts', fallback='True')

        # Retrieve logging settings from the configuration file
        log_level = config_parser_object.get('Logging', 'log_level', fallback='INFO')
        log_file = config_parser_object.get('Logging', 'log_file', fallback='./Logs/tldw_logs.json')
        log_metrics_file = config_parser_object.get('Logging', 'log_metrics_file', fallback='./Logs/tldw_metrics_logs.json')

        # Retrieve processing choice from the configuration file
        processing_choice = config_parser_object.get('Processing', 'processing_choice', fallback='cpu')
        logger.trace(f"Processing choice set to: {processing_choice}")

        # [Chunking]
        # # Chunking Defaults
        # #
        # # Default Chunking Options for each media type
        chunking_method = config_parser_object.get('Chunking', 'chunking_method', fallback='words')
        chunk_max_size = config_parser_object.get('Chunking', 'chunk_max_size', fallback='400')
        chunk_overlap = config_parser_object.get('Chunking', 'chunk_overlap', fallback='200')
        adaptive_chunking = config_parser_object.get('Chunking', 'adaptive_chunking', fallback='False')
        chunking_multi_level = config_parser_object.get('Chunking', 'chunking_multi_level', fallback='False')
        chunk_language = config_parser_object.get('Chunking', 'chunk_language', fallback='en')
        #
        # Article Chunking
        article_chunking_method = config_parser_object.get('Chunking', 'article_chunking_method', fallback='words')
        article_chunk_max_size = config_parser_object.get('Chunking', 'article_chunk_max_size', fallback='400')
        article_chunk_overlap = config_parser_object.get('Chunking', 'article_chunk_overlap', fallback='200')
        article_adaptive_chunking = config_parser_object.get('Chunking', 'article_adaptive_chunking', fallback='False')
        article_chunking_multi_level = config_parser_object.get('Chunking', 'article_chunking_multi_level', fallback='False')
        article_language = config_parser_object.get('Chunking', 'article_language', fallback='english')
        #
        # Audio file Chunking
        audio_chunking_method = config_parser_object.get('Chunking', 'audio_chunking_method', fallback='words')
        audio_chunk_max_size = config_parser_object.get('Chunking', 'audio_chunk_max_size', fallback='400')
        audio_chunk_overlap = config_parser_object.get('Chunking', 'audio_chunk_overlap', fallback='200')
        audio_adaptive_chunking = config_parser_object.get('Chunking', 'audio_adaptive_chunking', fallback='False')
        audio_chunking_multi_level = config_parser_object.get('Chunking', 'audio_chunking_multi_level', fallback='False')
        audio_language = config_parser_object.get('Chunking', 'audio_language', fallback='english')
        #
        # Book Chunking
        book_chunking_method = config_parser_object.get('Chunking', 'book_chunking_method', fallback='words')
        book_chunk_max_size = config_parser_object.get('Chunking', 'book_chunk_max_size', fallback='400')
        book_chunk_overlap = config_parser_object.get('Chunking', 'book_chunk_overlap', fallback='200')
        book_adaptive_chunking = config_parser_object.get('Chunking', 'book_adaptive_chunking', fallback='False')
        book_chunking_multi_level = config_parser_object.get('Chunking', 'book_chunking_multi_level', fallback='False')
        book_language = config_parser_object.get('Chunking', 'book_language', fallback='english')
        #
        # Document Chunking
        document_chunking_method = config_parser_object.get('Chunking', 'document_chunking_method', fallback='words')
        document_chunk_max_size = config_parser_object.get('Chunking', 'document_chunk_max_size', fallback='400')
        document_chunk_overlap = config_parser_object.get('Chunking', 'document_chunk_overlap', fallback='200')
        document_adaptive_chunking = config_parser_object.get('Chunking', 'document_adaptive_chunking', fallback='False')
        document_chunking_multi_level = config_parser_object.get('Chunking', 'document_chunking_multi_level', fallback='False')
        document_language = config_parser_object.get('Chunking', 'document_language', fallback='english')
        #
        # Mediawiki Article Chunking
        mediawiki_article_chunking_method = config_parser_object.get('Chunking', 'mediawiki_article_chunking_method', fallback='words')
        mediawiki_article_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_article_chunk_max_size', fallback='400')
        mediawiki_article_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_article_chunk_overlap', fallback='200')
        mediawiki_article_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_article_adaptive_chunking', fallback='False')
        mediawiki_article_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_article_chunking_multi_level', fallback='False')
        mediawiki_article_language = config_parser_object.get('Chunking', 'mediawiki_article_language', fallback='english')
        #
        # Mediawiki Dump Chunking
        mediawiki_dump_chunking_method = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_method', fallback='words')
        mediawiki_dump_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_max_size', fallback='400')
        mediawiki_dump_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_overlap', fallback='200')
        mediawiki_dump_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_dump_adaptive_chunking', fallback='False')
        mediawiki_dump_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_multi_level', fallback='False')
        mediawiki_dump_language = config_parser_object.get('Chunking', 'mediawiki_dump_language', fallback='english')
        #
        # Obsidian Note Chunking
        obsidian_note_chunking_method = config_parser_object.get('Chunking', 'obsidian_note_chunking_method', fallback='words')
        obsidian_note_chunk_max_size = config_parser_object.get('Chunking', 'obsidian_note_chunk_max_size', fallback='400')
        obsidian_note_chunk_overlap = config_parser_object.get('Chunking', 'obsidian_note_chunk_overlap', fallback='200')
        obsidian_note_adaptive_chunking = config_parser_object.get('Chunking', 'obsidian_note_adaptive_chunking', fallback='False')
        obsidian_note_chunking_multi_level = config_parser_object.get('Chunking', 'obsidian_note_chunking_multi_level', fallback='False')
        obsidian_note_language = config_parser_object.get('Chunking', 'obsidian_note_language', fallback='english')
        #
        # Podcast Chunking
        podcast_chunking_method = config_parser_object.get('Chunking', 'podcast_chunking_method', fallback='words')
        podcast_chunk_max_size = config_parser_object.get('Chunking', 'podcast_chunk_max_size', fallback='400')
        podcast_chunk_overlap = config_parser_object.get('Chunking', 'podcast_chunk_overlap', fallback='200')
        podcast_adaptive_chunking = config_parser_object.get('Chunking', 'podcast_adaptive_chunking', fallback='False')
        podcast_chunking_multi_level = config_parser_object.get('Chunking', 'podcast_chunking_multi_level', fallback='False')
        podcast_language = config_parser_object.get('Chunking', 'podcast_language', fallback='english')
        #
        # Text Chunking
        text_chunking_method = config_parser_object.get('Chunking', 'text_chunking_method', fallback='words')
        text_chunk_max_size = config_parser_object.get('Chunking', 'text_chunk_max_size', fallback='400')
        text_chunk_overlap = config_parser_object.get('Chunking', 'text_chunk_overlap', fallback='200')
        text_adaptive_chunking = config_parser_object.get('Chunking', 'text_adaptive_chunking', fallback='False')
        text_chunking_multi_level = config_parser_object.get('Chunking', 'text_chunking_multi_level', fallback='False')
        text_language = config_parser_object.get('Chunking', 'text_language', fallback='english')
        #
        # Video Transcription Chunking
        video_chunking_method = config_parser_object.get('Chunking', 'video_chunking_method', fallback='words')
        video_chunk_max_size = config_parser_object.get('Chunking', 'video_chunk_max_size', fallback='400')
        video_chunk_overlap = config_parser_object.get('Chunking', 'video_chunk_overlap', fallback='200')
        video_adaptive_chunking = config_parser_object.get('Chunking', 'video_adaptive_chunking', fallback='False')
        video_chunking_multi_level = config_parser_object.get('Chunking', 'video_chunking_multi_level', fallback='False')
        video_language = config_parser_object.get('Chunking', 'video_language', fallback='english')
        #
        chunking_types = 'article', 'audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump', 'obsidian_note', 'podcast', 'text', 'video'

        # Retrieve Embedding model settings from the configuration file
        embedding_model = config_parser_object.get('Embeddings', 'embedding_model', fallback='')
        logger.trace(f"Embedding model set to: {embedding_model}")
        embedding_provider = config_parser_object.get('Embeddings', 'embedding_provider', fallback='')
        embedding_model = config_parser_object.get('Embeddings', 'embedding_model', fallback='')
        onnx_model_path = config_parser_object.get('Embeddings', 'onnx_model_path', fallback="./App_Function_Libraries/onnx_models/text-embedding-3-small.onnx")
        model_dir = config_parser_object.get('Embeddings', 'model_dir', fallback="./App_Function_Libraries/onnx_models")
        embedding_api_url = config_parser_object.get('Embeddings', 'embedding_api_url', fallback="http://localhost:8080/v1/embeddings")
        embedding_api_key = config_parser_object.get('Embeddings', 'embedding_api_key', fallback='')
        chunk_size = config_parser_object.get('Embeddings', 'chunk_size', fallback=400)
        overlap = config_parser_object.get('Embeddings', 'overlap', fallback=200)

        # Prompts - FIXME
        prompt_path = config_parser_object.get('Prompts', 'prompt_path', fallback='Databases/prompts.db')

        # Chat Dictionaries
        enable_chat_dictionaries = config_parser_object.get('Chat-Dictionaries', 'enable_chat_dictionaries', fallback='False')
        post_gen_replacement = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement', fallback='False')
        post_gen_replacement_dict = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement_dict', fallback='')
        chat_dict_chat_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_chat_prompts', fallback='')
        chat_dict_rag_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_RAG_prompts', fallback='')
        chat_dict_replacement_strategy = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_replacement_strategy', fallback='character_lore_first')
        chat_dict_max_tokens = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_max_tokens', fallback='1000')
        default_rag_prompt = config_parser_object.get('Chat-Dictionaries', 'default_rag_prompt', fallback='')

        # Auto-Save Values
        save_character_chats = config_parser_object.get('Auto-Save', 'save_character_chats', fallback='False')
        save_rag_chats = config_parser_object.get('Auto-Save', 'save_rag_chats', fallback='False')

        # Local API Timeout
        local_api_timeout = config_parser_object.get('Local-API', 'local_api_timeout', fallback='90')

        # STT Settings
        default_stt_provider = config_parser_object.get('STT-Settings', 'default_stt_provider', fallback='faster_whisper')

        # TTS Settings
        # FIXME
        local_tts_device = config_parser_object.get('TTS-Settings', 'local_tts_device', fallback='cpu')
        default_tts_provider = config_parser_object.get('TTS-Settings', 'default_tts_provider', fallback='openai')
        tts_voice = config_parser_object.get('TTS-Settings', 'default_tts_voice', fallback='shimmer')
        # Open AI TTS
        default_openai_tts_model = config_parser_object.get('TTS-Settings', 'default_openai_tts_model', fallback='tts-1-hd')
        default_openai_tts_voice = config_parser_object.get('TTS-Settings', 'default_openai_tts_voice', fallback='shimmer')
        default_openai_tts_speed = config_parser_object.get('TTS-Settings', 'default_openai_tts_speed', fallback='1')
        default_openai_tts_output_format = config_parser_object.get('TTS-Settings', 'default_openai_tts_output_format', fallback='mp3')
        default_openai_tts_streaming = config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')
        # Google TTS
        # FIXME - FIX THESE DEFAULTS
        default_google_tts_model = config_parser_object.get('TTS-Settings', 'default_google_tts_model', fallback='en')
        default_google_tts_voice = config_parser_object.get('TTS-Settings', 'default_google_tts_voice', fallback='en')
        default_google_tts_speed = config_parser_object.get('TTS-Settings', 'default_google_tts_speed', fallback='1')
        # ElevenLabs TTS
        default_eleven_tts_model = config_parser_object.get('TTS-Settings', 'default_eleven_tts_model', fallback='FIXME')
        default_eleven_tts_voice = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice', fallback='FIXME')
        default_eleven_tts_language_code = config_parser_object.get('TTS-Settings', 'default_eleven_tts_language_code', fallback='FIXME')
        default_eleven_tts_voice_stability = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_stability', fallback='FIXME')
        default_eleven_tts_voice_similiarity_boost = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_similiarity_boost', fallback='FIXME')
        default_eleven_tts_voice_style = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_style', fallback='FIXME')
        default_eleven_tts_voice_use_speaker_boost = config_parser_object.get('TTS-Settings', 'default_eleven_tts_voice_use_speaker_boost', fallback='FIXME')
        default_eleven_tts_output_format = config_parser_object.get('TTS-Settings', 'default_eleven_tts_output_format',
                                                      fallback='mp3_44100_192')
        # AllTalk TTS
        alltalk_api_ip = config_parser_object.get('TTS-Settings', 'alltalk_api_ip', fallback='http://127.0.0.1:7851/v1/audio/speech')
        default_alltalk_tts_model = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_model', fallback='alltalk_model')
        default_alltalk_tts_voice = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_voice', fallback='alloy')
        default_alltalk_tts_speed = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_speed', fallback=1.0)
        default_alltalk_tts_output_format = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_output_format', fallback='mp3')

        # Kokoro TTS
        kokoro_model_path = config_parser_object.get('TTS-Settings', 'kokoro_model_path', fallback='Databases/kokoro_models')
        default_kokoro_tts_model = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_model', fallback='pht')
        default_kokoro_tts_voice = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_voice', fallback='sky')
        default_kokoro_tts_speed = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_speed', fallback=1.0)
        default_kokoro_tts_output_format = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_output_format', fallback='wav')


        # Self-hosted OpenAI API TTS
        default_openai_api_tts_model = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_model', fallback='tts-1-hd')
        default_openai_api_tts_voice = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_voice', fallback='shimmer')
        default_openai_api_tts_speed = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_speed', fallback='1')
        default_openai_api_tts_output_format = config_parser_object.get('TTS-Settings', 'default_openai_tts_api_output_format', fallback='mp3')
        default_openai_api_tts_streaming = config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')


        # Search Engines
        search_provider_default = config_parser_object.get('Search-Engines', 'search_provider_default', fallback='google')
        search_language_query = config_parser_object.get('Search-Engines', 'search_language_query', fallback='en')
        search_language_results = config_parser_object.get('Search-Engines', 'search_language_results', fallback='en')
        search_language_analysis = config_parser_object.get('Search-Engines', 'search_language_analysis', fallback='en')
        search_default_max_queries = 10
        search_enable_subquery = config_parser_object.get('Search-Engines', 'search_enable_subquery', fallback='True')
        search_enable_subquery_count_max = config_parser_object.get('Search-Engines', 'search_enable_subquery_count_max', fallback=5)
        search_result_rerank = config_parser_object.get('Search-Engines', 'search_result_rerank', fallback='True')
        search_result_max = config_parser_object.get('Search-Engines', 'search_result_max', fallback=10)
        search_result_max_per_query = config_parser_object.get('Search-Engines', 'search_result_max_per_query', fallback=10)
        search_result_blacklist = config_parser_object.get('Search-Engines', 'search_result_blacklist', fallback='')
        search_result_display_type = config_parser_object.get('Search-Engines', 'search_result_display_type', fallback='list')
        search_result_display_metadata = config_parser_object.get('Search-Engines', 'search_result_display_metadata', fallback='False')
        search_result_save_to_db = config_parser_object.get('Search-Engines', 'search_result_save_to_db', fallback='True')
        search_result_analysis_tone = config_parser_object.get('Search-Engines', 'search_result_analysis_tone', fallback='')
        relevance_analysis_llm = config_parser_object.get('Search-Engines', 'relevance_analysis_llm', fallback='False')
        final_answer_llm = config_parser_object.get('Search-Engines', 'final_answer_llm', fallback='False')
        # Search Engine Specifics
        baidu_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_baidu', fallback='')
        # Bing Search Settings
        bing_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_bing', fallback='')
        bing_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_bing', fallback='us')
        bing_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_bing', fallback='')
        # Brave Search Settings
        brave_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_regular', fallback='')
        brave_search_ai_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_ai', fallback='')
        brave_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_brave', fallback='us')
        # DuckDuckGo Search Settings
        duckduckgo_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_duckduckgo', fallback='')
        # Google Search Settings
        google_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_google', fallback='')
        google_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_google', fallback='')
        google_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_google', fallback='')
        google_simp_trad_chinese = config_parser_object.get('Search-Engines', 'enable_traditional_chinese', fallback='0')
        limit_google_search_to_country = config_parser_object.get('Search-Engines', 'limit_google_search_to_country', fallback='0')
        google_search_country = config_parser_object.get('Search-Engines', 'google_search_country', fallback='us')
        google_search_country_code = config_parser_object.get('Search-Engines', 'google_search_country_code', fallback='us')
        google_filter_setting = config_parser_object.get('Search-Engines', 'google_filter_setting', fallback='1')
        google_user_geolocation = config_parser_object.get('Search-Engines', 'google_user_geolocation', fallback='')
        google_ui_language = config_parser_object.get('Search-Engines', 'google_ui_language', fallback='en')
        google_limit_search_results_to_language = config_parser_object.get('Search-Engines', 'google_limit_search_results_to_language', fallback='')
        google_default_search_results = config_parser_object.get('Search-Engines', 'google_default_search_results', fallback='10')
        google_safe_search = config_parser_object.get('Search-Engines', 'google_safe_search', fallback='active')
        google_enable_site_search = config_parser_object.get('Search-Engines', 'google_enable_site_search', fallback='0')
        google_site_search_include = config_parser_object.get('Search-Engines', 'google_site_search_include', fallback='')
        google_site_search_exclude = config_parser_object.get('Search-Engines', 'google_site_search_exclude', fallback='')
        google_sort_results_by = config_parser_object.get('Search-Engines', 'google_sort_results_by', fallback='relevance')
        # Kagi Search Settings
        kagi_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_kagi', fallback='')
        # Searx Search Settings
        search_engine_searx_api = config_parser_object.get('Search-Engines', 'search_engine_searx_api', fallback='')
        # Tavily Search Settings
        tavily_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_tavily', fallback='')
        # Yandex Search Settings
        yandex_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_yandex', fallback='')
        yandex_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_yandex', fallback='')

        # Prompts
        sub_question_generation_prompt = config_parser_object.get('Prompts', 'sub_question_generation_prompt', fallback='')
        search_result_relevance_eval_prompt = config_parser_object.get('Prompts', 'search_result_relevance_eval_prompt', fallback='')
        analyze_search_results_prompt = config_parser_object.get('Prompts', 'analyze_search_results_prompt', fallback='')

        # Web Scraper settings
        web_scraper_api_key = config_parser_object.get('Web-Scraper', 'web_scraper_api_key', fallback='')
        web_scraper_api_url = config_parser_object.get('Web-Scraper', 'web_scraper_api_url', fallback='')
        web_scraper_api_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_api_timeout', fallback='90')
        web_scraper_api_retries = config_parser_object.get('Web-Scraper', 'web_scraper_api_retries', fallback='3')
        web_scraper_api_retry_delay = config_parser_object.get('Web-Scraper', 'web_scraper_api_retry_delay', fallback='5')
        web_scraper_retry_count = config_parser_object.get('Web-Scraper', 'web_scraper_retry_count', fallback='3')
        web_scraper_retry_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_retry_timeout', fallback='5')
        web_scraper_stealth_playwright = config_parser_object.get('Web-Scraper', 'web_scraper_stealth_playwright', fallback='False')

        return_dict = {
            'anthropic_api': {
                'api_key': anthropic_api_key,
                'model': anthropic_model,
                'streaming': anthropic_streaming,
                'temperature': anthropic_temperature,
                'top_p': anthropic_top_p,
                'top_k': anthropic_top_k,
                'max_tokens': anthropic_max_tokens,
                'api_timeout': anthropic_api_timeout,
                'api_retries': anthropic_api_retries,
                'api_retry_delay': anthropic_api_retry_delay
            },
            'cohere_api': {
                'api_key': cohere_api_key,
                'model': cohere_model,
                'streaming': cohere_streaming,
                'temperature': cohere_temperature,
                'max_p': cohere_max_p,
                'top_k': cohere_top_k,
                'max_tokens': cohere_max_tokens,
                'api_timeout': cohere_api_timeout,
                'api_retries': cohere_api_retries,
                'api_retry_delay': cohere_api_retry_delay
            },
            'deepseek_api': {
                'api_key': deepseek_api_key,
                'model': deepseek_model,
                'streaming': deepseek_streaming,
                'temperature': deepseek_temperature,
                'top_p': deepseek_top_p,
                'min_p': deepseek_min_p,
                'max_tokens': deepseek_max_tokens,
                'api_timeout': deepseek_api_timeout,
                'api_retries': deepseek_api_retries,
                'api_retry_delay': deepseek_api_retry_delay
            },
            'google_api': {
                'api_key': google_api_key,
                'model': google_model,
                'streaming': google_streaming,
                'temperature': google_temperature,
                'top_p': google_top_p,
                'min_p': google_min_p,
                'max_tokens': google_max_tokens,
                'api_timeout': google_api_timeout,
                'api_retries': google_api_retries,
                'api_retry_delay': google_api_retry_delay
            },
            'groq_api': {
                'api_key': groq_api_key,
                'model': groq_model,
                'streaming': groq_streaming,
                'temperature': groq_temperature,
                'top_p': groq_top_p,
                'max_tokens': groq_max_tokens,
                'api_timeout': groq_api_timeout,
                'api_retries': groq_api_retries,
                'api_retry_delay': groq_api_retry_delay
            },
            'huggingface_api': {
                'huggingface_use_router_url_format': huggingface_use_router_url_format,
                'huggingface_router_base_url': huggingface_router_base_url,
                'api_base_url': huggingface_api_base_url,
                'api_key': huggingface_api_key,
                'model': huggingface_model,
                'streaming': huggingface_streaming,
                'temperature': huggingface_temperature,
                'top_p': huggingface_top_p,
                'min_p': huggingface_min_p,
                'max_tokens': huggingface_max_tokens,
                'api_timeout': huggingface_api_timeout,
                'api_retries': huggingface_api_retries,
                'api_retry_delay': huggingface_api_retry_delay
            },
            'mistral_api': {
                'api_key': mistral_api_key,
                'model': mistral_model,
                'streaming': mistral_streaming,
                'temperature': mistral_temperature,
                'top_p': mistral_top_p,
                'max_tokens': mistral_max_tokens,
                'api_timeout': mistral_api_timeout,
                'api_retries': mistral_api_retries,
                'api_retry_delay': mistral_api_retry_delay
            },
            'openrouter_api': {
                'api_key': openrouter_api_key,
                'model': openrouter_model,
                'streaming': openrouter_streaming,
                'temperature': openrouter_temperature,
                'top_p': openrouter_top_p,
                'min_p': openrouter_min_p,
                'top_k': openrouter_top_k,
                'max_tokens': openrouter_max_tokens,
                'api_timeout': openrouter_api_timeout,
                'api_retries': openrouter_api_retries,
                'api_retry_delay': openrouter_api_retry_delay
            },
            'openai_api': {
                'api_key': openai_api_key,
                'model': openai_model,
                'streaming': openai_streaming,
                'temperature': openai_temperature,
                'top_p': openai_top_p,
                'max_tokens': openai_max_tokens,
                'api_timeout': openai_api_timeout,
                'api_retries': openai_api_retries,
                'api_retry_delay': openai_api_retry_delay
            },
            'elevenlabs_api': {
                'api_key': elevenlabs_api_key,
            },
            'alltalk_api': {
                'api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
            },
            'llama_api': {
                'api_ip': llama_api_IP,
                'api_key': llama_api_key,
                'streaming': llama_streaming,
                'temperature': llama_temperature,
                'top_p': llama_top_p,
                'min_p': llama_min_p,
                'top_k': llama_top_k,
                'max_tokens': llama_max_tokens,
                'api_timeout': llama_api_timeout,
                'api_retries': llama_api_retries,
                'api_retry_delay': llama_api_retry_delay
            },
            'ooba_api': {
                'api_ip': ooba_api_IP,
                'api_key': ooba_api_key,
                'streaming': ooba_streaming,
                'temperature': ooba_temperature,
                'top_p': ooba_top_p,
                'min_p': ooba_min_p,
                'top_k': ooba_top_k,
                'max_tokens': ooba_max_tokens,
                'api_timeout': ooba_api_timeout,
                'api_retries': ooba_api_retries,
                'api_retry_delay': ooba_api_retry_delay
            },
            'kobold_api': {
                'api_ip': kobold_api_ip,
                'api_streaming_ip': kobold_openai_api_IP,
                'api_key': kobold_api_key,
                'streaming': kobold_streaming,
                'temperature': kobold_temperature,
                'top_p': kobold_top_p,
                'top_k': kobold_top_k,
                'max_tokens': kobold_max_tokens,
                'api_timeout': kobold_api_timeout,
                'api_retries': kobold_api_retries,
                'api_retry_delay': kobold_api_retry_delay
            },
            'tabby_api': {
                'api_ip': tabby_api_IP,
                'api_key': tabby_api_key,
                'model': tabby_model,
                'streaming': tabby_streaming,
                'temperature': tabby_temperature,
                'top_p': tabby_top_p,
                'top_k': tabby_top_k,
                'min_p': tabby_min_p,
                'max_tokens': tabby_max_tokens,
                'api_timeout': tabby_api_timeout,
                'api_retries': tabby_api_retries,
                'api_retry_delay': tabby_api_retry_delay
            },
            'vllm_api': {
                'api_ip': vllm_api_url,
                'api_key': vllm_api_key,
                'model': vllm_model,
                'streaming': vllm_streaming,
                'temperature': vllm_temperature,
                'top_p': vllm_top_p,
                'top_k': vllm_top_k,
                'min_p': vllm_min_p,
                'max_tokens': vllm_max_tokens,
                'api_timeout': vllm_api_timeout,
                'api_retries': vllm_api_retries,
                'api_retry_delay': vllm_api_retry_delay
            },
            'ollama_api': {
                'api_url': ollama_api_url,
                'api_key': ollama_api_key,
                'model': ollama_model,
                'streaming': ollama_streaming,
                'temperature': ollama_temperature,
                'top_p': ollama_top_p,
                'max_tokens': ollama_max_tokens,
                'api_timeout': ollama_api_timeout,
                'api_retries': ollama_api_retries,
                'api_retry_delay': ollama_api_retry_delay
            },
            'aphrodite_api': {
                'api_ip': aphrodite_api_url,
                'api_key': aphrodite_api_key,
                'model': aphrodite_model,
                'max_tokens': aphrodite_max_tokens,
                'streaming': aphrodite_streaming,
                'api_timeout': aphrodite_api_timeout,
                'api_retries': aphrodite_api_retries,
                'api_retry_delay': aphrodite_api_retry_delay
            },
            'custom_openai_api': {
                'api_ip': custom_openai_api_ip,
                'api_key': custom_openai_api_key,
                'streaming': custom_openai_api_streaming,
                'model': custom_openai_api_model,
                'temperature': custom_openai_api_temperature,
                'max_tokens': custom_openai_api_max_tokens,
                'top_p': custom_openai_api_top_p,
                'min_p': custom_openai_api_min_p,
                'api_timeout': custom_openai_api_timeout,
                'api_retries': custom_openai_api_retries,
                'api_retry_delay': custom_openai_api_retry_delay
            },
            'custom_openai_api_2': {
                'api_ip': custom_openai2_api_ip,
                'api_key': custom_openai2_api_key,
                'streaming': custom_openai2_api_streaming,
                'model': custom_openai2_api_model,
                'temperature': custom_openai2_api_temperature,
                'max_tokens': custom_openai2_api_max_tokens,
                'top_p': custom_openai2_api_top_p,
                'min_p': custom_openai2_api_min_p,
                'api_timeout': custom_openai2_api_timeout,
                'api_retries': custom_openai2_api_retries,
                'api_retry_delay': custom_openai2_api_retry_delay
            },
            'llm_api_settings': {
                'default_api': default_api,
                'local_api_timeout': local_api_timeout,
                'local_api_retries': local_api_retries,
                'local_api_retry_delay': local_api_retry_delay,
            },
            'output_path': output_path,
            'system_preferences': {
                'save_video_transcripts': save_video_transcripts,
            },
            'processing_choice': processing_choice,
            'chat_dictionaries': {
                'enable_chat_dictionaries': enable_chat_dictionaries,
                'post_gen_replacement': post_gen_replacement,
                'post_gen_replacement_dict': post_gen_replacement_dict,
                'chat_dict_chat_prompts': chat_dict_chat_prompts,
                'chat_dict_RAG_prompts': chat_dict_rag_prompts,
                'chat_dict_replacement_strategy': chat_dict_replacement_strategy,
                'chat_dict_max_tokens': chat_dict_max_tokens,
                'default_rag_prompt': default_rag_prompt
            },
            'chunking_config': {
                'chunking_method': chunking_method,
                'chunk_max_size': chunk_max_size,
                'adaptive_chunking': adaptive_chunking,
                'multi_level': chunking_multi_level,
                'chunk_language': chunk_language,
                'chunk_overlap': chunk_overlap,
                'article_chunking_method': article_chunking_method,
                'article_chunk_max_size': article_chunk_max_size,
                'article_chunk_overlap': article_chunk_overlap,
                'article_adaptive_chunking': article_adaptive_chunking,
                'article_chunking_multi_level': article_chunking_multi_level,
                'article_language': article_language,
                'audio_chunking_method': audio_chunking_method,
                'audio_chunk_max_size': audio_chunk_max_size,
                'audio_chunk_overlap': audio_chunk_overlap,
                'audio_adaptive_chunking': audio_adaptive_chunking,
                'audio_chunking_multi_level': audio_chunking_multi_level,
                'audio_language': audio_language,
                'book_chunking_method': book_chunking_method,
                'book_chunk_max_size': book_chunk_max_size,
                'book_chunk_overlap': book_chunk_overlap,
                'book_adaptive_chunking': book_adaptive_chunking,
                'book_chunking_multi_level': book_chunking_multi_level,
                'book_language': book_language,
                'document_chunking_method': document_chunking_method,
                'document_chunk_max_size': document_chunk_max_size,
                'document_chunk_overlap': document_chunk_overlap,
                'document_adaptive_chunking': document_adaptive_chunking,
                'document_chunking_multi_level': document_chunking_multi_level,
                'document_language': document_language,
                'mediawiki_article_chunking_method': mediawiki_article_chunking_method,
                'mediawiki_article_chunk_max_size': mediawiki_article_chunk_max_size,
                'mediawiki_article_chunk_overlap': mediawiki_article_chunk_overlap,
                'mediawiki_article_adaptive_chunking': mediawiki_article_adaptive_chunking,
                'mediawiki_article_chunking_multi_level': mediawiki_article_chunking_multi_level,
                'mediawiki_article_language': mediawiki_article_language,
                'mediawiki_dump_chunking_method': mediawiki_dump_chunking_method,
                'mediawiki_dump_chunk_max_size': mediawiki_dump_chunk_max_size,
                'mediawiki_dump_chunk_overlap': mediawiki_dump_chunk_overlap,
                'mediawiki_dump_adaptive_chunking': mediawiki_dump_adaptive_chunking,
                'mediawiki_dump_chunking_multi_level': mediawiki_dump_chunking_multi_level,
                'mediawiki_dump_language': mediawiki_dump_language,
                'obsidian_note_chunking_method': obsidian_note_chunking_method,
                'obsidian_note_chunk_max_size': obsidian_note_chunk_max_size,
                'obsidian_note_chunk_overlap': obsidian_note_chunk_overlap,
                'obsidian_note_adaptive_chunking': obsidian_note_adaptive_chunking,
                'obsidian_note_chunking_multi_level': obsidian_note_chunking_multi_level,
                'obsidian_note_language': obsidian_note_language,
                'podcast_chunking_method': podcast_chunking_method,
                'podcast_chunk_max_size': podcast_chunk_max_size,
                'podcast_chunk_overlap': podcast_chunk_overlap,
                'podcast_adaptive_chunking': podcast_adaptive_chunking,
                'podcast_chunking_multi_level': podcast_chunking_multi_level,
                'podcast_language': podcast_language,
                'text_chunking_method': text_chunking_method,
                'text_chunk_max_size': text_chunk_max_size,
                'text_chunk_overlap': text_chunk_overlap,
                'text_adaptive_chunking': text_adaptive_chunking,
                'text_chunking_multi_level': text_chunking_multi_level,
                'text_language': text_language,
                'video_chunking_method': video_chunking_method,
                'video_chunk_max_size': video_chunk_max_size,
                'video_chunk_overlap': video_chunk_overlap,
                'video_adaptive_chunking': video_adaptive_chunking,
                'video_chunking_multi_level': video_chunking_multi_level,
                'video_language': video_language,
            },
            'embedding_config': {
                'embedding_provider': embedding_provider,
                'embedding_model': embedding_model,
                'onnx_model_path': onnx_model_path,
                'model_dir': model_dir,
                'embedding_api_url': embedding_api_url,
                'embedding_api_key': embedding_api_key,
                'chunk_size': chunk_size,
                'chunk_overlap': overlap
            },
            'auto-save': {
                'save_character_chats': save_character_chats,
                'save_rag_chats': save_rag_chats,
            },
            'default_api': default_api,
            'local_api_timeout': local_api_timeout,
            'STT_Settings': {
                'default_stt_provider': default_stt_provider,
            },
            'tts_settings': {
                'default_tts_provider': default_tts_provider,
                'tts_voice': tts_voice,
                'local_tts_device': local_tts_device,
                # OpenAI
                'default_openai_tts_voice': default_openai_tts_voice,
                'default_openai_tts_speed': default_openai_tts_speed,
                'default_openai_tts_model': default_openai_tts_model,
                'default_openai_tts_output_format': default_openai_tts_output_format,
                # Google
                'default_google_tts_model': default_google_tts_model,
                'default_google_tts_voice': default_google_tts_voice,
                'default_google_tts_speed': default_google_tts_speed,
                # ElevenLabs
                'default_eleven_tts_model': default_eleven_tts_model,
                'default_eleven_tts_voice': default_eleven_tts_voice,
                'default_eleven_tts_language_code': default_eleven_tts_language_code,
                'default_eleven_tts_voice_stability': default_eleven_tts_voice_stability,
                'default_eleven_tts_voice_similiarity_boost': default_eleven_tts_voice_similiarity_boost,
                'default_eleven_tts_voice_style': default_eleven_tts_voice_style,
                'default_eleven_tts_voice_use_speaker_boost': default_eleven_tts_voice_use_speaker_boost,
                'default_eleven_tts_output_format': default_eleven_tts_output_format,
                # Open Source / Self-Hosted TTS
                # GPT SoVITS
                # 'default_gpt_tts_model': default_gpt_tts_model,
                # 'default_gpt_tts_voice': default_gpt_tts_voice,
                # 'default_gpt_tts_speed': default_gpt_tts_speed,
                # 'default_gpt_tts_output_format': default_gpt_tts_output_format
                # AllTalk
                'alltalk_api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
                # Kokoro
                'default_kokoro_tts_model': default_kokoro_tts_model,
                'default_kokoro_tts_voice': default_kokoro_tts_voice,
                'default_kokoro_tts_speed': default_kokoro_tts_speed,
                'default_kokoro_tts_output_format': default_kokoro_tts_output_format,
                # Self-hosted OpenAI API
                'default_openai_api_tts_model': default_openai_api_tts_model,
                'default_openai_api_tts_voice': default_openai_api_tts_voice,
                'default_openai_api_tts_speed': default_openai_api_tts_speed,
                'default_openai_api_tts_output_format': default_openai_api_tts_output_format,
                'default_openai_api_tts_streaming': default_openai_api_tts_streaming,
            },
            'search_settings': {
                'default_search_provider': search_provider_default,
                'search_language_query': search_language_query,
                'search_language_results': search_language_results,
                'search_language_analysis': search_language_analysis,
                'search_default_max_queries': search_default_max_queries,
                'search_enable_subquery': search_enable_subquery,
                'search_enable_subquery_count_max': search_enable_subquery_count_max,
                'search_result_rerank': search_result_rerank,
                'search_result_max': search_result_max,
                'search_result_max_per_query': search_result_max_per_query,
                'search_result_blacklist': search_result_blacklist,
                'search_result_display_type': search_result_display_type,
                'search_result_display_metadata': search_result_display_metadata,
                'search_result_save_to_db': search_result_save_to_db,
                'search_result_analysis_tone': search_result_analysis_tone,
                'relevance_analysis_llm': relevance_analysis_llm,
                'final_answer_llm': final_answer_llm,
            },
            'search_engines': {
                'baidu_search_api_key': baidu_search_api_key,
                'bing_search_api_key': bing_search_api_key,
                'bing_country_code': bing_country_code,
                'bing_search_api_url': bing_search_api_url,
                'brave_search_api_key': brave_search_api_key,
                'brave_search_ai_api_key': brave_search_ai_api_key,
                'brave_country_code': brave_country_code,
                'duckduckgo_search_api_key': duckduckgo_search_api_key,
                'google_search_api_url': google_search_api_url,
                'google_search_api_key': google_search_api_key,
                'google_search_engine_id': google_search_engine_id,
                'google_simp_trad_chinese': google_simp_trad_chinese,
                'limit_google_search_to_country': limit_google_search_to_country,
                'google_search_country': google_search_country,
                'google_search_country_code': google_search_country_code,
                'google_search_filter_setting': google_filter_setting,
                'google_user_geolocation': google_user_geolocation,
                'google_ui_language': google_ui_language,
                'google_limit_search_results_to_language': google_limit_search_results_to_language,
                'google_site_search_include': google_site_search_include,
                'google_site_search_exclude': google_site_search_exclude,
                'google_sort_results_by': google_sort_results_by,
                'google_default_search_results': google_default_search_results,
                'google_safe_search': google_safe_search,
                'google_enable_site_search' : google_enable_site_search,
                'kagi_search_api_key': kagi_search_api_key,
                'searx_search_api_url': search_engine_searx_api,
                'tavily_search_api_key': tavily_search_api_key,
                'yandex_search_api_key': yandex_search_api_key,
                'yandex_search_engine_id': yandex_search_engine_id
            },
            'prompts': {
                'sub_question_generation_prompt': sub_question_generation_prompt,
                'search_result_relevance_eval_prompt': search_result_relevance_eval_prompt,
                'analyze_search_results_prompt': analyze_search_results_prompt,
            },
            'web_scraper':{
                'web_scraper_api_key': web_scraper_api_key,
                'web_scraper_api_url': web_scraper_api_url,
                'web_scraper_api_timeout': web_scraper_api_timeout,
                'web_scraper_api_retries': web_scraper_api_retries,
                'web_scraper_api_retry_delay': web_scraper_api_retry_delay,
                'web_scraper_retry_count': web_scraper_retry_count,
                'web_scraper_retry_timeout': web_scraper_retry_timeout,
                'web_scraper_stealth_playwright': web_scraper_stealth_playwright,
            }
        }
        return return_dict
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None


# Global scope in config.py
try:
    loaded_config_data = load_and_log_configs()
    if loaded_config_data is None:  # Add a check here
        logger.critical("Failed to load configuration data at module import. `loaded_config_data` is None.")
        default_api_endpoint = "openai"  # Fallback
    else:
        default_api_endpoint = loaded_config_data.get('default_api', 'openai')  # Use .get() for safety
        logger.info(f"Default API Endpoint (from config.py global scope): {default_api_endpoint}")
except Exception as e:  # Should be less likely to hit this outer if inner one is robust
    logger.error(f"Critical error setting default_api_endpoint in config.py global scope: {str(e)}", exc_info=True)
    default_api_endpoint = "openai"  # Fallback


# --- Global Settings Object ---
# Load the settings when the module is imported
settings = load_settings()



# --- Configuration File Content (for reference or auto-creation) ---
CONFIG_TOML_CONTENT = """
# Configuration for tldw-cli TUI App
# tldw_cli/config.toml
[general]
default_tab = "chat"  # "chat", "character", "logs", "media", "search", "ingest", "stats"
log_level = "DEBUG" # TUI Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL

[logging]
# Log file will be placed in the same directory as the database file specified below.
log_filename = "tldw_cli_app.log"
file_log_level = "INFO" # File Log Level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_max_bytes = 10485760 # 10 MB
log_backup_count = 5

[database]
# Path to the main application data/history database.
# Use ~ for home directory expansion.
path = "~/.local/share/tldw_cli/tldw_cli_data.db"
# Path to user config dir (derived from this in code if needed for other files)
# user_config_dir = "~/.config/tldw_cli" # Example if needed explicitly

[api_endpoints]
# Optional: Specify URLs for local/custom endpoints if they differ from library defaults
# These keys should match the provider names used in the app (adjust if needed)
Ollama = "http://localhost:11434"
Llama_cpp = "http://localhost:8080" # Check if your API provider uses this address
Oobabooga = "http://localhost:5000/api" # Check if your API provider uses this address
KoboldCpp = "http://localhost:5001/api" # Check if your API provider uses this address
vLLM = "http://localhost:8000" # Check if your API provider uses this address
Custom = "http://localhost:1234/v1"
Custom_2 = "http://localhost:5678/v1"
# Add other local URLs if needed

[providers]
# This section primarily lists providers and their *available* models for the UI dropdown.
# Actual default model/settings used for calls are defined in [api_settings.*] or [chat_defaults]/[character_defaults].
OpenAI = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
Anthropic = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
Google = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
MistralAI = ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b"]
Groq = ["llama3-70b-8192", "mixtral-8x7b-32768"]
Cohere = ["command-r-plus", "command-r"]
OpenRouter = ["meta-llama/Llama-3.1-8B-Instruct"] # Example, add more
HuggingFace = ["mistralai/Mixtral-8x7B-Instruct-v0.1"] # Example, add more
DeepSeek = ["deepseek-chat"]
# Local Providers
Ollama = ["llama3:latest", "mistral:latest"] # Use format expected by your Ollama client
Llama_cpp = ["."] # Often model is specified at server start, client might not need it
Oobabooga = ["."] # Often model is specified at server start
KoboldCpp = ["."] # Often model is specified at server start
vLLM = ["."] # Often model is specified at server start
Custom = ["custom-model-alpha", "custom-model-beta"]
Custom_2 = ["custom-model-gamma"]
TabbyAPI = ["tabby-model"]
Aphrodite = ["aphrodite-engine"]
# local-llm = ["."] # Add if you have a specific local-llm provider entry

[api_settings] # Parent section for all API provider specific settings

    # --- Cloud Providers ---
    [api_settings.openai]
    api_key_env_var = "OPENAI_API_KEY"
    # api_key = "" # Less secure fallback - use env var instead
    model = "gpt-4o" # Default model for direct calls (if not overridden)
    temperature = 0.7
    top_p = 1.0 # OpenAI uses top_p (represented as maxp sometimes in UI)
    max_tokens = 4096
    timeout = 60 # seconds
    retries = 3
    retry_delay = 5 # seconds (backoff factor)
    streaming = false

    [api_settings.anthropic]
    api_key_env_var = "ANTHROPIC_API_KEY"
    model = "claude-3-haiku-20240307"
    temperature = 0.7
    top_p = 1.0 # Anthropic uses top_p (represented as topp in UI)
    top_k = 0 # Anthropic specific, 0 or -1 usually disables it
    max_tokens = 4096
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.google]
    api_key_env_var = "GOOGLE_API_KEY"
    model = "gemini-1.5-pro-latest"
    temperature = 0.7
    top_p = 0.9 # Google uses topP (represented as topp in UI)
    top_k = 100 # Google uses topK
    max_tokens = 8192 # Google uses maxOutputTokens
    timeout = 120
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.mistralai] # Matches key in [providers]
    api_key_env_var = "MISTRAL_API_KEY"
    model = "mistral-large-latest"
    temperature = 0.7
    top_p = 1.0 # Mistral uses top_p (represented as topp in UI)
    max_tokens = 4096
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.groq]
    api_key_env_var = "GROQ_API_KEY"
    model = "llama3-70b-8192"
    temperature = 0.7
    top_p = 1.0 # Groq uses top_p (represented as maxp in UI)
    max_tokens = 8192
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.cohere]
    api_key_env_var = "COHERE_API_KEY"
    model = "command-r-plus"
    temperature = 0.3
    top_p = 0.75 # Cohere uses 'p' (represented as topp in UI)
    top_k = 0 # Cohere uses 'k'
    max_tokens = 4096 # Cohere uses max_tokens
    timeout = 90
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.openrouter]
    api_key_env_var = "OPENROUTER_API_KEY"
    model = "meta-llama/Llama-3.1-8B-Instruct"
    temperature = 0.7
    top_p = 1.0 # OpenRouter uses top_p
    top_k = 0   # OpenRouter uses top_k
    min_p = 0.0 # OpenRouter uses min_p
    max_tokens = 4096
    timeout = 120
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.huggingface]
    api_key_env_var = "HUGGINGFACE_API_KEY"
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    temperature = 0.7
    top_p = 1.0 # HF Inference API uses top_p
    top_k = 50  # HF Inference API uses top_k
    max_tokens = 4096 # HF Inf API uses max_tokens / max_new_tokens
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    [api_settings.deepseek]
    api_key_env_var = "DEEPSEEK_API_KEY"
    model = "deepseek-chat"
    temperature = 0.7
    top_p = 1.0 # Deepseek uses top_p (represented as topp in UI)
    max_tokens = 4096
    timeout = 60
    retries = 3
    retry_delay = 5
    streaming = false

    # --- Local Providers ---
    [api_settings.ollama]
    # No API Key usually needed
    api_url = "http://localhost:11434/v1/chat/completions" # Default Ollama OpenAI endpoint
    model = "llama3:latest"
    temperature = 0.7
    top_p = 0.9
    top_k = 40 # Ollama supports top_k via OpenAI endpoint
    # min_p = 0.05 # Ollama OpenAI endpoint doesn't support min_p directly
    max_tokens = 4096
    timeout = 300 # Longer timeout for local models
    retries = 1
    retry_delay = 2
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.llama_cpp] # Matches key in [providers]
    api_key_env_var = "LLAMA_CPP_API_KEY" # If you set one on the server
    # api_key = ""
    api_url = "http://localhost:8080/completion" # llama.cpp /completion endpoint
    model = "" # Often not needed if server serves one model
    temperature = 0.7
    top_p = 0.95
    top_k = 40
    min_p = 0.05
    max_tokens = 4096 # llama.cpp uses n_predict
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.oobabooga] # Matches key in [providers]
    api_key_env_var = "OOBABOOGA_API_KEY" # If API extension needs one
    api_url = "http://localhost:5000/v1/chat/completions" # Ooba OpenAI compatible endpoint
    model = "" # Model loaded in Ooba UI
    temperature = 0.7
    top_p = 0.9
    # top_k = 50 # Check Ooba endpoint docs for OpenAI compatibility params
    # min_p = 0.0
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.koboldcpp] # Matches key in [providers]
    # api_key = "" # Kobold doesn't use keys
    api_url = "http://localhost:5001/api/v1/generate" # Kobold non-streaming API
    # api_streaming_url = "http://localhost:5001/api/v1/stream" # Kobold streaming API (different format)
    model = "" # Model loaded in Kobold UI
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    max_tokens = 4096 # Kobold uses max_context_length / max_length
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false # Kobold streaming is non-standard, handle carefully
    system_prompt = "You are a helpful AI assistant"

    [api_settings.vllm] # Matches key in [providers]
    api_key_env_var = "VLLM_API_KEY" # If served behind auth
    api_url = "http://localhost:8000/v1/chat/completions" # vLLM OpenAI compatible endpoint
    model = "" # Model specified when starting vLLM server
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.custom] # Matches key in [providers]
    api_key_env_var = "CUSTOM_API_KEY"
    api_url = "http://localhost:1234/v1/chat/completions"
    model = "custom-model-alpha"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.custom_2] # Matches key in [providers]
    api_key_env_var = "CUSTOM_2_API_KEY"
    api_url = "http://localhost:5678/v1/chat/completions"
    model = "custom-model-gamma"
    temperature = 0.7
    top_p = 1.0
    top_k = 0
    min_p = 0.0
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 5
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.tabbyapi] # Matches key in [providers]
    api_key_env_var = "TABBYAPI_API_KEY"
    api_url = "http://localhost:8080/v1/chat/completions" # Check TabbyAPI docs for exact URL
    model = "tabby-model" # Model configured in TabbyAPI
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 120
    retries = 2
    retry_delay = 3
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    [api_settings.aphrodite] # Matches key in [providers]
    api_key_env_var = "APHRODITE_API_KEY" # If served behind auth
    api_url = "http://localhost:2242/v1/chat/completions" # Default Aphrodite port
    model = "aphrodite-engine" # Model loaded in Aphrodite
    temperature = 0.7
    top_p = 0.95
    top_k = 50
    min_p = 0.05
    max_tokens = 4096
    timeout = 300
    retries = 1
    retry_delay = 2
    streaming = false
    system_prompt = "You are a helpful AI assistant"

    # [api_settings.local-llm] # If you have a generic local-llm setup
    # api_url = "http://127.0.0.1:8080/v1/chat/completions" # Example LM Studio / Jan
    # model = ""
    # temperature = 0.7
    # ... etc ...

[chat_defaults]
# Default settings specifically for the 'Chat' tab
provider = "Ollama"
model = "llama3:latest"
system_prompt = "You are a helpful AI assistant."
temperature = 0.7
top_p = 0.95
min_p = 0.05
top_k = 50

[character_defaults]
# Default settings specifically for the 'Character' tab
provider = "Anthropic"
model = "claude-3-haiku-20240307" # Make sure this exists in [providers.Anthropic]
system_prompt = "You are roleplaying as a witty pirate captain."
temperature = 0.8
top_p = 0.9
min_p = 0.0 # Check if API supports this
top_k = 100 # Check if API supports this

# --- Sections below are placeholders based on config.txt, integrate as needed ---
# [tts_settings]
# default_provider = "kokoro"
# ...

# [search_settings]
# default_provider = "google"
# ...

# [embedding_settings]
# provider = "openai"
# ...

# [chunking_settings]
# default_method = "words"
# ...
"""

#
# End of tldw_cli/config.py
#######################################################################################################################
