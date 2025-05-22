# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
# Imports
import tomllib
import logging
import os
from pathlib import Path
import toml
from typing import Dict, Any, List, Optional
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
