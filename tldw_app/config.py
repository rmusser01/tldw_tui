# tldw_cli/config.py
# Description: Configuration management for the tldw_cli application.
#
# Imports
import copy
import json
import tomllib
# import logging # Standard logging is replaced by loguru for consistency in the CLI part
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

# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs
SERVER_CLIENT_ID = "SERVER_API_V1"

# --- Path to the CLI's configuration file ---
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tldw_cli" / "config.toml"

# --- Chunking Settings (Default, can be overridden by TOML) ---
global_default_chunk_language = "en"

# --- Default Fallback Configurations (if not found in TOML) ---
# These will be populated from TOML or use these hardcoded dicts as fallbacks.
DEFAULT_APP_TTS_CONFIG = {
    "OPENAI_API_KEY_fallback": "sk-...", # Note: API keys should primarily come from [API] section or ENV
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": "path/to/your/downloaded/kokoro-v0_19.onnx",
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": "path/to/your/downloaded/voices.json",
    "KOKORO_DEVICE_DEFAULT": "cpu", # or "cuda"
    "ELEVENLABS_API_KEY_fallback": "el-...", # Note: API keys should primarily come from [API] section or ENV
    "local_kokoro_default_onnx": {
        "KOKORO_DEVICE": "cuda:0"
    },
    "global_tts_settings": {
        # shared settings
    }
}

DEFAULT_DATABASE_CONFIG = {} # Example, can be populated if needed

DEFAULT_RAG_SEARCH_CONFIG = {
    "fts_top_k": 10,
    "vector_top_k": 10,
    "web_vector_top_k": 10,
    "llm_context_document_limit": 10,
    "chat_context_limit": 10,
}

def load_openai_mappings() -> Dict:
    current_file_path = Path(__file__).resolve()
    api_component_root = current_file_path.parent.parent.parent
    mapping_path = api_component_root / "Config_Files" / "openai_tts_mappings.json"
    logger.info(f"Attempting to load OpenAI TTS mappings from: {str(mapping_path)}")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OpenAI TTS mappings from {mapping_path}: {e}", exc_info=True)
        return {
            "models": {"tts-1": "openai_official_tts-1"},
            "voices": {"alloy": "alloy"}
        }

_openai_mappings = load_openai_mappings()

# This hardcoded mapping can also be moved to TOML or be a fallback for the JSON loaded one
openai_tts_mappings = {
    "models": {
        "tts-1": "openai_official_tts-1",
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_monolingual_v1": "elevenlabs_english_v1",
        "kokoro": "local_kokoro_default_onnx"
    },
    "voices": {
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",
        "RachelEL": "21m00Tcm4TlvDq8ikWAM",
        "k_bella": "af_bella",
        "k_adam" : "am_v0adam"
    }
}
# Update openai_tts_mappings with values from _openai_mappings (JSON file takes precedence)
if _openai_mappings:
    openai_tts_mappings["models"].update(_openai_mappings.get("models", {}))
    openai_tts_mappings["voices"].update(_openai_mappings.get("voices", {}))


def _get_typed_value(data_dict: Dict, key: str, default: Any, target_type: type = str) -> Any:
    """Helper to get value from dict and cast to type, with logging for type errors."""
    value = data_dict.get(key, default)
    if value is default and default is not None : # if value is the default, it's already typed
        return value
    if value is None: # If key is missing and default is None
        return None

    try:
        if target_type == bool:
            if isinstance(value, bool):
                return value
            # For bools from TOML strings (shouldn't happen if TOML is well-formed)
            return str(value).lower() in ['true', '1', 't', 'y', 'yes']
        if target_type == Path:
             return Path(value) if value else default
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Config key '{key}' has value '{value}' which could not be converted to {target_type}. Using default: '{default}'. Error: {e}")
        return default

def load_settings() -> Dict:
    """Loads all settings from TOML config file, environment variables, or defaults into a dictionary."""

    current_file_path = Path(__file__).resolve()
    # config.py is in project_root/tldw_server_api/app/core/config.py
    ACTUAL_PROJECT_ROOT = current_file_path.parent # /project_root/
    APP_COMPONENT_ROOT = current_file_path.parent # /project_root/tldw_server_api/
    logger.info(f"Determined ACTUAL_PROJECT_ROOT for general paths: {ACTUAL_PROJECT_ROOT}")
    logger.info(f"Determined APP_COMPONENT_ROOT for config files: {APP_COMPONENT_ROOT}")

    # --- Load Comprehensive Config from TOML ---
    toml_config_data = {}
    config_toml_path = APP_COMPONENT_ROOT / "Config_Files" / "config.toml" # This is for the server/API part
    logger.info(f"Attempting to load comprehensive TOML config from: {str(config_toml_path)}")
    try:
        with open(config_toml_path, "rb") as f: # TOML library recommends 'rb'
            toml_config_data = toml.load(f)
        logger.info(f"Successfully loaded TOML config from: {str(config_toml_path)}")
    except FileNotFoundError:
        logger.warning(f"TOML Config file not found at {config_toml_path}. Using defaults and environment variables for many settings.")
    except toml.TomlDecodeError as e:
        logger.error(f"Error decoding TOML config file {config_toml_path}: {e}. Using defaults and environment variables.", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {config_toml_path}: {e}", exc_info=True)


    # --- Extract settings from TOML, with fallbacks ---
    # Helper to get values from specific TOML sections
    def get_toml_section(section_name: str) -> Dict:
        return toml_config_data.get(section_name, {})

    api_section = get_toml_section('API')
    local_api_section = get_toml_section('LocalAPI') # Renamed for TOML convention
    paths_section = get_toml_section('Paths')
    logging_section_server = get_toml_section('Logging') # Renamed to avoid conflict with CLI logging section
    processing_section = get_toml_section('Processing')
    chunking_section = get_toml_section('Chunking')
    embeddings_section = get_toml_section('Embeddings')
    # prompts_section = get_toml_section('Prompts') # For specific prompt strings
    chat_dicts_section = get_toml_section('ChatDictionaries')
    auto_save_section = get_toml_section('AutoSave')
    stt_settings_section = get_toml_section('STTSettings')
    tts_settings_section = get_toml_section('TTSSettings')
    search_engines_section = get_toml_section('SearchEngines') # Contains API keys for search engines
    search_settings_section = get_toml_section('SearchSettings') # Contains general search behavior
    web_scraper_section = get_toml_section('WebScraper')
    file_validation_section = get_toml_section('FileValidation')

    # --- Application Mode ---
    single_user_mode_str = os.getenv("APP_MODE", _get_typed_value(processing_section, "app_mode", "single")).lower()
    single_user_mode = single_user_mode_str != "multi"

    # --- Single-User Settings ---
    single_user_fixed_id = int(os.getenv("SINGLE_USER_FIXED_ID", _get_typed_value(processing_section, "single_user_fixed_id", "0", int)))
    single_user_api_key = os.getenv("API_KEY", _get_typed_value(api_section, "single_user_api_key", "default-secret-key-for-single-user"))

    # --- Paths ---
    default_user_data_base_dir = ACTUAL_PROJECT_ROOT / "user_databases"
    user_data_base_dir_str = os.getenv("USER_DB_BASE_DIR", _get_typed_value(paths_section, "user_db_base_dir", str(default_user_data_base_dir.resolve())))
    user_data_base_dir = Path(user_data_base_dir_str)

    default_main_db_path = (user_data_base_dir / str(single_user_fixed_id) / "tldw.db").resolve() # Adjusted to use resolved user_data_base_dir
    if single_user_mode:
         # For single user, db path might be simpler or directly under project root if not user-specific
        default_main_db_path = (ACTUAL_PROJECT_ROOT / "user_databases" / "single_user" / "tldw.db").resolve()


    default_database_url = f"sqlite:///{default_main_db_path}"
    database_url = os.getenv("DATABASE_URL", _get_typed_value(paths_section, "database_url", default_database_url))

    users_db_configured = os.getenv("USERS_DB_ENABLED", _get_typed_value(processing_section, "users_db_enabled", "false", str)).lower() == "true"
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_toml = _get_typed_value(logging_section_server, "log_level", log_level_env, str).upper()


    # --- Load specific configurations from TOML or use defaults ---
    app_tts_config = get_toml_section('AppTTSConfig') # For APP_CONFIG related values
    app_database_config = get_toml_section('AppDatabaseConfig') # For DATABASE_CONFIG
    app_rag_search_config = get_toml_section('AppRAGSearchConfig') # For RAG_SEARCH_CONFIG

    # API Keys (Prioritize ENV, then TOML, then None)
    def get_api_key(toml_key: str, env_var: str, section: Dict = api_section) -> Optional[str]:
        return os.getenv(env_var, section.get(toml_key))

    openai_api_key = get_api_key('openai_api_key', 'OPENAI_API_KEY')
    anthropic_api_key = get_api_key('anthropic_api_key', 'ANTHROPIC_API_KEY')
    cohere_api_key = get_api_key('cohere_api_key', 'COHERE_API_KEY')
    groq_api_key = get_api_key('groq_api_key', 'GROQ_API_KEY')
    huggingface_api_key = get_api_key('huggingface_api_key', 'HUGGINGFACE_API_KEY')
    openrouter_api_key = get_api_key('openrouter_api_key', 'OPENROUTER_API_KEY')
    deepseek_api_key = get_api_key('deepseek_api_key', 'DEEPSEEK_API_KEY')
    mistral_api_key = get_api_key('mistral_api_key', 'MISTRAL_API_KEY')
    google_api_key = get_api_key('google_api_key', 'GOOGLE_API_KEY') # For Google Generative AI
    elevenlabs_api_key = get_api_key('elevenlabs_api_key', 'ELEVENLABS_API_KEY')


    config_dict = {
        # General App
        "APP_MODE_STR": single_user_mode_str,
        "SINGLE_USER_MODE": single_user_mode,
        "LOG_LEVEL": log_level_toml, # This is for the server/API part
        "PROJECT_ROOT": ACTUAL_PROJECT_ROOT,
        "API_COMPONENT_ROOT": APP_COMPONENT_ROOT, # Added for clarity

        # Single User
        "SINGLE_USER_FIXED_ID": single_user_fixed_id,
        "SINGLE_USER_API_KEY": single_user_api_key,

        # Auth
        "DATABASE_URL": database_url,
        "USER_DB_BASE_DIR": user_data_base_dir,
        "USERS_DB_CONFIGURED": users_db_configured,

        # --- Configurations migrated from load_and_log_configs ---
        "anthropic_api": {
            'api_key': anthropic_api_key,
            'model': _get_typed_value(api_section, 'anthropic_model', 'claude-3-5-sonnet-20240620'),
            'streaming': _get_typed_value(api_section, 'anthropic_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'anthropic_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'anthropic_top_p', 0.95, float),
            'top_k': _get_typed_value(api_section, 'anthropic_top_k', 100, int),
            'max_tokens': _get_typed_value(api_section, 'anthropic_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'anthropic_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'anthropic_api_retry', 3, int), # Key name consistency
            'api_retry_delay': _get_typed_value(api_section, 'anthropic_api_retry_delay', 5, int)
        },
        "cohere_api": {
            'api_key': cohere_api_key,
            'model': _get_typed_value(api_section, 'cohere_model', 'command-r-plus'),
            'streaming': _get_typed_value(api_section, 'cohere_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'cohere_temperature', 0.7, float),
            'max_p': _get_typed_value(api_section, 'cohere_max_p', 0.95, float), # Note: check param name, Cohere might use 'p' or 'top_p'
            'top_k': _get_typed_value(api_section, 'cohere_top_k', 100, int),
            'max_tokens': _get_typed_value(api_section, 'cohere_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'cohere_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'cohere_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'cohere_api_retry_delay', 5, int)
        },
        "deepseek_api": {
            'api_key': deepseek_api_key,
            'model': _get_typed_value(api_section, 'deepseek_model', 'deepseek-chat'),
            'streaming': _get_typed_value(api_section, 'deepseek_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'deepseek_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'deepseek_top_p', 0.95, float),
            'min_p': _get_typed_value(api_section, 'deepseek_min_p', 0.05, float),
            'max_tokens': _get_typed_value(api_section, 'deepseek_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'deepseek_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'deepseek_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'deepseek_api_retry_delay', 5, int)
        },
        "google_generative_api": { # Renamed to avoid confusion with Google Search API
            'api_key': google_api_key,
            'model': _get_typed_value(api_section, 'google_model', 'gemini-1.5-pro'),
            'streaming': _get_typed_value(api_section, 'google_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'google_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'google_top_p', 0.95, float),
            'min_p': _get_typed_value(api_section, 'google_min_p', 0.05, float), # Check if 'min_p' is valid for Gemini
            'max_tokens': _get_typed_value(api_section, 'google_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'google_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'google_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'google_api_retry_delay', 5, int)
        },
        "groq_api": {
            'api_key': groq_api_key,
            'model': _get_typed_value(api_section, 'groq_model', 'llama3-70b-8192'),
            'streaming': _get_typed_value(api_section, 'groq_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'groq_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'groq_top_p', 0.95, float),
            'max_tokens': _get_typed_value(api_section, 'groq_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'groq_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'groq_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'groq_api_retry_delay', 5, int)
        },
        "huggingface_api": {
            'api_key': huggingface_api_key,
            'huggingface_use_router_url_format': _get_typed_value(api_section, 'huggingface_use_router_url_format', False, bool),
            'huggingface_router_base_url': _get_typed_value(api_section, 'huggingface_router_base_url', 'https://router.huggingface.co/hf-inference'),
            'api_base_url': _get_typed_value(api_section, 'huggingface_api_base_url', 'https://router.huggingface.co/hf-inference/models'), # Redundant if router_base_url is used for construction
            'model': _get_typed_value(api_section, 'huggingface_model', '/Qwen/Qwen3-235B-A22B'),
            'streaming': _get_typed_value(api_section, 'huggingface_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'huggingface_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'huggingface_top_p', 0.95, float),
            'min_p': _get_typed_value(api_section, 'huggingface_min_p', 0.05, float),
            'max_tokens': _get_typed_value(api_section, 'huggingface_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'huggingface_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'huggingface_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'huggingface_api_retry_delay', 5, int)
        },
        "mistral_api": {
            'api_key': mistral_api_key,
            'model': _get_typed_value(api_section, 'mistral_model', 'mistral-large-latest'),
            'streaming': _get_typed_value(api_section, 'mistral_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'mistral_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'mistral_top_p', 0.95, float),
            'max_tokens': _get_typed_value(api_section, 'mistral_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'mistral_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'mistral_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'mistral_api_retry_delay', 5, int)
        },
        "openrouter_api": {
            'api_key': openrouter_api_key,
            'model': _get_typed_value(api_section, 'openrouter_model', 'microsoft/wizardlm-2-8x22b'),
            'streaming': _get_typed_value(api_section, 'openrouter_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'openrouter_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'openrouter_top_p', 0.95, float),
            'min_p': _get_typed_value(api_section, 'openrouter_min_p', 0.05, float),
            'top_k': _get_typed_value(api_section, 'openrouter_top_k', 100, int),
            'max_tokens': _get_typed_value(api_section, 'openrouter_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'openrouter_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'openrouter_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'openrouter_api_retry_delay', 5, int)
        },
        "openai_api": { # OpenAI specific model params, API key is separate
            'api_key': openai_api_key, # This is now the primary OpenAI API key
            'model': _get_typed_value(api_section, 'openai_model', 'gpt-4o'),
            'streaming': _get_typed_value(api_section, 'openai_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'openai_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'openai_top_p', 0.95, float),
            'max_tokens': _get_typed_value(api_section, 'openai_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'openai_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'openai_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'openai_api_retry_delay', 5, int)
        },
        "elevenlabs_api": { # Primarily for the API key, other settings in TTS
            'api_key': elevenlabs_api_key,
        },
        # Local APIs from LocalAPI section
        "kobold_api": {
            'api_ip': _get_typed_value(local_api_section, 'kobold_api_IP', 'http://127.0.0.1:5000/api/v1/generate'),
            'api_streaming_ip': _get_typed_value(local_api_section, 'kobold_openai_api_IP', 'http://127.0.0.1:5001/v1/chat/completions'),
            'api_key': _get_typed_value(local_api_section, 'kobold_api_key', ''),
            'streaming': _get_typed_value(local_api_section, 'kobold_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'kobold_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'kobold_top_p', 0.95, float),
            'top_k': _get_typed_value(local_api_section, 'kobold_top_k', 100, int),
            'max_tokens': _get_typed_value(local_api_section, 'kobold_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(local_api_section, 'kobold_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'kobold_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'kobold_api_retry_delay', 5, int)
        },
        "llama_cpp_api": { # Renamed for clarity, assuming llama.cpp server
            'api_ip': _get_typed_value(local_api_section, 'llama_api_IP', 'http://127.0.0.1:8080/v1/chat/completions'),
            'api_key': _get_typed_value(local_api_section, 'llama_api_key', ''),
            'streaming': _get_typed_value(local_api_section, 'llama_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'llama_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'llama_top_p', 0.95, float),
            'min_p': _get_typed_value(local_api_section, 'llama_min_p', 0.05, float),
            'top_k': _get_typed_value(local_api_section, 'llama_top_k', 100, int),
            'max_tokens': _get_typed_value(local_api_section, 'llama_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(local_api_section, 'llama_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'llama_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'llama_api_retry_delay', 5, int)
        },
        "ooba_api": {
            'api_ip': _get_typed_value(local_api_section, 'ooba_api_IP', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': _get_typed_value(local_api_section, 'ooba_api_key', ''),
            'streaming': _get_typed_value(local_api_section, 'ooba_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'ooba_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'ooba_top_p', 0.95, float),
            'min_p': _get_typed_value(local_api_section, 'ooba_min_p', 0.05, float),
            'top_k': _get_typed_value(local_api_section, 'ooba_top_k', 100, int),
            'max_tokens': _get_typed_value(local_api_section, 'ooba_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(local_api_section, 'ooba_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'ooba_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'ooba_api_retry_delay', 5, int)
        },
         "tabby_api": {
            'api_ip': _get_typed_value(local_api_section, 'tabby_api_IP', 'http://127.0.0.1:5000/api/v1/generate'),
            'api_key': _get_typed_value(local_api_section, 'tabby_api_key', None),
            'model': _get_typed_value(local_api_section, 'tabby_model', None), # Tabby model might be part of URL or configured in Tabby
            'streaming': _get_typed_value(local_api_section, 'tabby_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'tabby_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'tabby_top_p', 0.95, float),
            'top_k': _get_typed_value(local_api_section, 'tabby_top_k', 100, int),
            'min_p': _get_typed_value(local_api_section, 'tabby_min_p', 0.05, float),
            'max_tokens': _get_typed_value(local_api_section, 'tabby_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(local_api_section, 'tabby_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'tabby_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'tabby_api_retry_delay', 5, int)
        },
        "vllm_api": {
            'api_ip': _get_typed_value(local_api_section, 'vllm_api_IP', 'http://127.0.0.1:5000/v1/chat/completions'), # Corrected key
            'api_key': _get_typed_value(local_api_section, 'vllm_api_key', None),
            'model': _get_typed_value(local_api_section, 'vllm_model', None),
            'streaming': _get_typed_value(local_api_section, 'vllm_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'vllm_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'vllm_top_p', 0.95, float),
            'top_k': _get_typed_value(local_api_section, 'vllm_top_k', 100, int),
            'min_p': _get_typed_value(local_api_section, 'vllm_min_p', 0.05, float),
            'max_tokens': _get_typed_value(local_api_section, 'vllm_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(local_api_section, 'vllm_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'vllm_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'vllm_api_retry_delay', 5, int)
        },
        "ollama_api": {
            'api_url': _get_typed_value(local_api_section, 'ollama_api_IP', 'http://127.0.0.1:11434/api/generate'), # ollama_api_url or IP
            'api_key': _get_typed_value(local_api_section, 'ollama_api_key', None), # Ollama doesn't typically use API keys
            'model': _get_typed_value(local_api_section, 'ollama_model', None),
            'streaming': _get_typed_value(local_api_section, 'ollama_streaming', False, bool),
            'temperature': _get_typed_value(local_api_section, 'ollama_temperature', 0.7, float),
            'top_p': _get_typed_value(local_api_section, 'ollama_top_p', 0.95, float),
            'max_tokens': _get_typed_value(local_api_section, 'ollama_max_tokens', 4096, int), # Ollama might handle max_tokens differently (num_predict)
            'api_timeout': _get_typed_value(local_api_section, 'ollama_api_timeout', 90, int),
            'api_retries': _get_typed_value(local_api_section, 'ollama_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'ollama_api_retry_delay', 5, int)
        },
        "aphrodite_api": {
            'api_ip': _get_typed_value(local_api_section, 'aphrodite_api_IP', 'http://127.0.0.1:8080/v1/chat/completions'),
            'api_key': _get_typed_value(local_api_section, 'aphrodite_api_key', ''),
            'model': _get_typed_value(local_api_section, 'aphrodite_model', ''),
            'max_tokens': _get_typed_value(local_api_section, 'aphrodite_max_tokens', 4096, int),
            'streaming': _get_typed_value(local_api_section, 'aphrodite_streaming', False, bool),
            'api_timeout': _get_typed_value(local_api_section, 'aphrodite_api_timeout', 90, int), # Original used llama_api_timeout
            'api_retries': _get_typed_value(local_api_section, 'aphrodite_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(local_api_section, 'aphrodite_api_retry_delay', 5, int)
        },
        "custom_openai_api": {
            'api_ip': _get_typed_value(api_section, 'custom_openai_api_ip', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': _get_typed_value(api_section, 'custom_openai_api_key', None),
            'model': _get_typed_value(api_section, 'custom_openai_api_model', None),
            'streaming': _get_typed_value(api_section, 'custom_openai_api_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'custom_openai_api_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'custom_openai_api_top_p', 0.95, float),
            'min_p': _get_typed_value(api_section, 'custom_openai_api_min_p', 0.05, float), # Original used top_k, ensure consistency
            'max_tokens': _get_typed_value(api_section, 'custom_openai_api_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'custom_openai_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'custom_openai_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'custom_openai_api_retry_delay', 5, int)
        },
        "custom_openai_api_2": { # Ensure key names are consistent e.g. custom_openai2_api_min_p
            'api_ip': _get_typed_value(api_section, 'custom_openai2_api_ip', 'http://127.0.0.1:5000/v1/chat/completions'),
            'api_key': _get_typed_value(api_section, 'custom_openai2_api_key', None),
            'model': _get_typed_value(api_section, 'custom_openai2_api_model', None),
            'streaming': _get_typed_value(api_section, 'custom_openai2_api_streaming', False, bool),
            'temperature': _get_typed_value(api_section, 'custom_openai2_api_temperature', 0.7, float),
            'top_p': _get_typed_value(api_section, 'custom_openai2_api_top_p', 0.95, float), # original had custom_openai_api2_top_p
            'min_p': _get_typed_value(api_section, 'custom_openai2_api_min_p', 0.05, float), # original had custom_openai_api2_top_k
            'max_tokens': _get_typed_value(api_section, 'custom_openai2_api_max_tokens', 4096, int),
            'api_timeout': _get_typed_value(api_section, 'custom_openai2_api_timeout', 90, int),
            'api_retries': _get_typed_value(api_section, 'custom_openai2_api_retry', 3, int),
            'api_retry_delay': _get_typed_value(api_section, 'custom_openai2_api_retry_delay', 5, int)
        },
        "llm_api_settings": { # General LLM settings
            'default_api': _get_typed_value(api_section, 'default_api', 'openai'),
            'local_api_timeout': _get_typed_value(local_api_section, 'local_api_timeout', 90, int), # Note: this was also in Local-API Settings before
            'local_api_retries': _get_typed_value(local_api_section, 'local_api_retry', 3, int), # Key name consistency
            'local_api_retry_delay': _get_typed_value(local_api_section, 'local_api_retry_delay', 5, int),
        },
        "output_path": _get_typed_value(paths_section, 'output_path', 'results', Path),
        "system_preferences": {
            'save_video_transcripts': _get_typed_value(paths_section, 'save_video_transcripts', True, bool),
        },
        "processing_choice": _get_typed_value(processing_section, 'processing_choice', 'cpu'),

        "chat_dictionaries": {
            'enable_chat_dictionaries': _get_typed_value(chat_dicts_section, 'enable_chat_dictionaries', False, bool),
            'post_gen_replacement': _get_typed_value(chat_dicts_section, 'post_gen_replacement', False, bool),
            'post_gen_replacement_dict': _get_typed_value(chat_dicts_section, 'post_gen_replacement_dict', ''),
            'chat_dict_chat_prompts': _get_typed_value(chat_dicts_section, 'chat_dictionary_chat_prompts', ''),
            'chat_dict_RAG_prompts': _get_typed_value(chat_dicts_section, 'chat_dictionary_RAG_prompts', ''),
            'chat_dict_replacement_strategy': _get_typed_value(chat_dicts_section, 'chat_dictionary_replacement_strategy', 'character_lore_first'),
            'chat_dict_max_tokens': _get_typed_value(chat_dicts_section, 'chat_dictionary_max_tokens', 1000, int),
            'default_rag_prompt': _get_typed_value(chat_dicts_section, 'default_rag_prompt', '')
        },
        "chunking_config": {
            # Global defaults
            'chunking_method': _get_typed_value(chunking_section, 'chunking_method', 'words'),
            'chunk_max_size': _get_typed_value(chunking_section, 'chunk_max_size', 400, int),
            'chunk_overlap': _get_typed_value(chunking_section, 'chunk_overlap', 200, int),
            'adaptive_chunking': _get_typed_value(chunking_section, 'adaptive_chunking', False, bool),
            'multi_level': _get_typed_value(chunking_section, 'chunking_multi_level', False, bool),
            'chunk_language': _get_typed_value(chunking_section, 'chunk_language', global_default_chunk_language), # Use global default
            # Per-type overrides (example for article, repeat for others: audio, book, etc.)
            'article_chunking_method': _get_typed_value(chunking_section, 'article_chunking_method', 'words'),
            'article_chunk_max_size': _get_typed_value(chunking_section, 'article_chunk_max_size', 400, int),
            'article_chunk_overlap': _get_typed_value(chunking_section, 'article_chunk_overlap', 200, int),
            'article_adaptive_chunking': _get_typed_value(chunking_section, 'article_adaptive_chunking', False, bool),
            'article_chunking_multi_level': _get_typed_value(chunking_section,'article_chunking_multi_level', False, bool),
            'article_language': _get_typed_value(chunking_section,'article_language', 'en'),
            'audio_chunking_method': _get_typed_value(chunking_section,'audio_chunking_method', 'words'),
            'audio_chunk_max_size': _get_typed_value(chunking_section,'audio_chunk_max_size', 400, int),
            'audio_chunk_overlap': _get_typed_value(chunking_section,'audio_chunk_overlap', 200, int),
            'audio_adaptive_chunking': _get_typed_value(chunking_section,'audio_adaptive_chunking', False, bool),
            'audio_chunking_multi_level': _get_typed_value(chunking_section,'audio_chunking_multi_level', False, bool),
            'audio_language': _get_typed_value(chunking_section,'audio_language', 'en'),
            'book_chunking_method': _get_typed_value(chunking_section,'book_chunking_method', 'ebook_chunk_by_chapter'),
            'book_chunk_max_size': _get_typed_value(chunking_section,'book_chunk_max_size', 400, int),
            'book_chunk_overlap': _get_typed_value(chunking_section,'book_chunk_overlap', 200, int),
            'book_adaptive_chunking': _get_typed_value(chunking_section,'book_adaptive_chunking', False, bool),
            'book_chunking_multi_level': _get_typed_value(chunking_section,'book_chunking_multi_level', False, bool),
            'book_language': _get_typed_value(chunking_section,'book_language', 'en'),
            'document_chunking_method': _get_typed_value(chunking_section,'document_chunking_method', 'words'),
            'document_chunk_max_size': _get_typed_value(chunking_section,'document_chunk_max_size', 400, int),
            'document_chunk_overlap': _get_typed_value(chunking_section,'document_chunk_overlap', 200, int),
            'document_adaptive_chunking': _get_typed_value(chunking_section,'document_adaptive_chunking', False, bool),
            'document_chunking_multi_level': _get_typed_value(chunking_section,'document_chunking_multi_level', False, bool),
            'document_language': _get_typed_value(chunking_section,'document_language', 'en'),
            'mediawiki_article_chunking_method': _get_typed_value(chunking_section,'mediawiki_article_chunking_method', 'words'),
            'mediawiki_article_chunk_max_size': _get_typed_value(chunking_section,'mediawiki_article_chunk_max_size', 400, int),
            'mediawiki_article_chunk_overlap': _get_typed_value(chunking_section,'mediawiki_article_chunk_overlap', 200, int),
            'mediawiki_article_adaptive_chunking': _get_typed_value(chunking_section,'mediawiki_article_adaptive_chunking', False, bool),
            'mediawiki_article_chunking_multi_level': _get_typed_value(chunking_section,'mediawiki_article_chunking_multi_level', False, bool),
            'mediawiki_article_language': _get_typed_value(chunking_section,'mediawiki_article_language', 'en'),
            'mediawiki_dump_chunking_method': _get_typed_value(chunking_section,'mediawiki_dump_chunking_method', 'words'),
            'mediawiki_dump_chunk_max_size': _get_typed_value(chunking_section,'mediawiki_dump_chunk_max_size', 400, int),
            'mediawiki_dump_chunk_overlap': _get_typed_value(chunking_section,'mediawiki_dump_chunk_overlap', 200, int),
            'mediawiki_dump_adaptive_chunking': _get_typed_value(chunking_section,'mediawiki_dump_adaptive_chunking', False, bool),
            'mediawiki_dump_chunking_multi_level': _get_typed_value(chunking_section,'mediawiki_dump_chunking_multi_level', False, bool),
            'mediawiki_dump_language': _get_typed_value(chunking_section,'mediawiki_dump_language', 'en'),
            'obsidian_note_chunking_method': _get_typed_value(chunking_section,'obsidian_note_chunking_method', 'words'),
            'obsidian_note_chunk_max_size': _get_typed_value(chunking_section,'obsidian_note_chunk_max_size', 400, int),
            'obsidian_note_chunk_overlap': _get_typed_value(chunking_section,'obsidian_note_chunk_overlap', 200, int),
            'obsidian_note_adaptive_chunking': _get_typed_value(chunking_section,'obsidian_note_adaptive_chunking', False, bool),
            'obsidian_note_chunking_multi_level': _get_typed_value(chunking_section,'obsidian_note_chunking_multi_level', False, bool),
            'obsidian_note_language': _get_typed_value(chunking_section,'obsidian_note_language', 'en'),
            'podcast_chunking_method': _get_typed_value(chunking_section,'podcast_chunking_method', 'sentences'),
            'podcast_chunk_max_size': _get_typed_value(chunking_section,'podcast_chunk_max_size', 300, int),
            'podcast_chunk_overlap': _get_typed_value(chunking_section,'podcast_chunk_overlap', 30, int),
            'podcast_adaptive_chunking': _get_typed_value(chunking_section,'podcast_adaptive_chunking', False, bool),
            'podcast_chunking_multi_level': _get_typed_value(chunking_section,'podcast_chunking_multi_level', False, bool),
            'podcast_language': _get_typed_value(chunking_section,'podcast_language', 'en'),
            'text_chunking_method': _get_typed_value(chunking_section,'text_chunking_method', 'words'),
            'text_chunk_max_size': _get_typed_value(chunking_section,'text_chunk_max_size', 400, int),
            'text_chunk_overlap': _get_typed_value(chunking_section,'text_chunk_overlap', 200, int),
            'text_adaptive_chunking': _get_typed_value(chunking_section,'text_adaptive_chunking', False, bool),
            'text_chunking_multi_level': _get_typed_value(chunking_section,'text_chunking_multi_level', False, bool),
            'text_language': _get_typed_value(chunking_section,'text_language', 'en'),
            'video_chunking_method': _get_typed_value(chunking_section,'video_chunking_method', 'words'),
            'video_chunk_max_size': _get_typed_value(chunking_section,'video_chunk_max_size', 400, int),
            'video_chunk_overlap': _get_typed_value(chunking_section,'video_chunk_overlap', 200, int),
            'video_adaptive_chunking': _get_typed_value(chunking_section,'video_adaptive_chunking', False, bool),
            'video_chunking_multi_level': _get_typed_value(chunking_section,'video_chunking_multi_level', False, bool),
            'video_language': _get_typed_value(chunking_section,'video_language', 'en'),
        },
        "embedding_config": {
            'embedding_provider': _get_typed_value(embeddings_section, 'embedding_provider', 'openai'),
            'embedding_model': _get_typed_value(embeddings_section, 'embedding_model', 'text-embedding-3-small'),
            'onnx_model_path': _get_typed_value(embeddings_section, 'onnx_model_path', "./Models/onnx_models/text-embedding-3-small.onnx", Path),
            'model_dir': _get_typed_value(embeddings_section, 'model_dir', "./Models", Path),
            'embedding_api_url': _get_typed_value(embeddings_section, 'embedding_api_url', "http://localhost:8080/v1/embeddings"),
            'embedding_api_key': _get_typed_value(embeddings_section, 'embedding_api_key', ''),
            'chunk_size': _get_typed_value(embeddings_section, 'chunk_size', 400, int), # This was 'chunk_size' in old Embeddings, also in Chunking
            'chunk_overlap': _get_typed_value(embeddings_section, 'overlap', 200, int) # This was 'overlap' in old Embeddings
        },
        "auto_save": {
            'save_character_chats': _get_typed_value(auto_save_section, 'save_character_chats', False, bool),
            'save_rag_chats': _get_typed_value(auto_save_section, 'save_rag_chats', False, bool),
        },
        "STT_settings": { # Corrected key from STT-Settings
            'default_stt_provider': _get_typed_value(stt_settings_section, 'default_stt_provider', 'faster_whisper'),
        },
        "tts_settings": {
            'default_tts_provider': _get_typed_value(tts_settings_section, 'default_tts_provider', 'openai'),
            'tts_voice': _get_typed_value(tts_settings_section, 'default_tts_voice', 'shimmer'), # General default voice
            'local_tts_device': _get_typed_value(tts_settings_section, 'local_tts_device', 'cpu'),
            # OpenAI TTS
            'default_openai_tts_model': _get_typed_value(tts_settings_section, 'default_openai_tts_model', 'tts-1-hd'),
            'default_openai_tts_voice': _get_typed_value(tts_settings_section, 'default_openai_tts_voice', 'shimmer'),
            'default_openai_tts_speed': _get_typed_value(tts_settings_section, 'default_openai_tts_speed', 1.0, float),
            'default_openai_tts_output_format': _get_typed_value(tts_settings_section, 'default_openai_tts_output_format', 'mp3'),
            'default_openai_tts_streaming': _get_typed_value(tts_settings_section, 'default_openai_tts_streaming', False, bool),
             # Google TTS
            'default_google_tts_model': _get_typed_value(tts_settings_section, 'default_google_tts_model', 'en'), # FIXME: Review defaults
            'default_google_tts_voice': _get_typed_value(tts_settings_section, 'default_google_tts_voice', 'en'), # FIXME: Review defaults
            'default_google_tts_speed': _get_typed_value(tts_settings_section, 'default_google_tts_speed', 1.0, float), # FIXME: Review defaults
            # ElevenLabs TTS
            'default_eleven_tts_model': _get_typed_value(tts_settings_section, 'default_eleven_tts_model', 'eleven_multilingual_v2'), # FIXME: Placeholder
            'default_eleven_tts_voice': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice', 'Rachel'), # FIXME: Placeholder
            'default_eleven_tts_language_code': _get_typed_value(tts_settings_section, 'default_eleven_tts_language_code', 'en-US'), # FIXME
            'default_eleven_tts_voice_stability': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_stability', 0.5, float), # FIXME
            'default_eleven_tts_voice_similiarity_boost': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_similiarity_boost', 0.75, float), # FIXME
            'default_eleven_tts_voice_style': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_style', 0.0, float), # FIXME
            'default_eleven_tts_voice_use_speaker_boost': _get_typed_value(tts_settings_section, 'default_eleven_tts_voice_use_speaker_boost', True, bool), # FIXME
            'default_eleven_tts_output_format': _get_typed_value(tts_settings_section, 'default_eleven_tts_output_format', 'mp3_44100_192'),
            # AllTalk TTS (from load_and_log_configs, now integrated)
            'alltalk_api_ip': _get_typed_value(tts_settings_section, 'alltalk_api_ip', 'http://127.0.0.1:7851/v1/audio/speech'),
            'default_alltalk_tts_model': _get_typed_value(tts_settings_section, 'default_alltalk_tts_model', 'alltalk_model'),
            'default_alltalk_tts_voice': _get_typed_value(tts_settings_section, 'default_alltalk_tts_voice', 'alloy'),
            'default_alltalk_tts_speed': _get_typed_value(tts_settings_section, 'default_alltalk_tts_speed', 1.0, float),
            'default_alltalk_tts_output_format': _get_typed_value(tts_settings_section, 'default_alltalk_tts_output_format', 'mp3'),
            # Kokoro TTS
            'kokoro_model_path': _get_typed_value(tts_settings_section, 'kokoro_model_path', 'Databases/kokoro_models', Path),
            'default_kokoro_tts_model': _get_typed_value(tts_settings_section, 'default_kokoro_tts_model', 'pht'),
            'default_kokoro_tts_voice': _get_typed_value(tts_settings_section, 'default_kokoro_tts_voice', 'sky'),
            'default_kokoro_tts_speed': _get_typed_value(tts_settings_section, 'default_kokoro_tts_speed', 1.0, float),
            'default_kokoro_tts_output_format': _get_typed_value(tts_settings_section, 'default_kokoro_tts_output_format', 'wav'),
            # Self-hosted OpenAI API TTS
            'default_openai_api_tts_model': _get_typed_value(tts_settings_section, 'default_openai_api_tts_model', 'tts-1-hd'),
            'default_openai_api_tts_voice': _get_typed_value(tts_settings_section, 'default_openai_api_tts_voice', 'shimmer'),
            'default_openai_api_tts_speed': _get_typed_value(tts_settings_section, 'default_openai_api_tts_speed', 1.0, float), # Was '1' string
            'default_openai_api_tts_output_format': _get_typed_value(tts_settings_section, 'default_openai_api_tts_output_format', 'mp3'), # key was default_openai_tts_api_output_format
            'default_openai_api_tts_streaming': _get_typed_value(tts_settings_section, 'default_openai_api_tts_streaming', False, bool),
        },
        "search_settings_general": { # Renamed from 'search_settings' to avoid conflict with SearchEngines section for keys
            'default_search_provider': _get_typed_value(search_settings_section, 'search_provider_default', 'google'),
            'search_language_query': _get_typed_value(search_settings_section, 'search_language_query', 'en'),
            'search_language_analysis': _get_typed_value(search_settings_section, 'search_language_analysis', 'en'),
            'search_default_max_queries': _get_typed_value(search_settings_section, 'search_default_max_queries', 5, int),
            'search_enable_subquery': _get_typed_value(search_settings_section, 'search_enable_subquery', False, bool),
            'search_enable_subquery_count_max': _get_typed_value(search_settings_section, 'search_enable_subquery_count_max', 3, int),
            'search_result_rerank': _get_typed_value(search_settings_section, 'search_result_rerank', False, bool),
            'search_result_max': _get_typed_value(search_settings_section, 'search_result_max', 10, int),
            'search_result_max_per_query': _get_typed_value(search_settings_section, 'search_result_max_per_query', 10, int),
            'search_result_blacklist': _get_typed_value(search_settings_section, 'search_result_blacklist' , ''),
            'search_result_display_type': _get_typed_value(search_settings_section, 'search_result_display_type' , 'text'),
            'search_result_display_metadata': _get_typed_value(search_settings_section, 'search_result_display_metadata' , True, bool),
            'search_result_save_to_db': _get_typed_value(search_settings_section, 'search_result_save_to_db' , True, bool),
            'search_result_analysis_tone': _get_typed_value(search_settings_section, 'search_result_analysis_tone' , 'neutral'),
            'relevance_analysis_llm': _get_typed_value(search_settings_section, 'relevance_analysis_llm' , 'openai'),
            'final_answer_llm': _get_typed_value(search_settings_section, 'final_answer_llm' , 'openai'),
        },
        "search_engine_specific_settings": {  # API Keys for various search engines from 'SearchEngines' TOML table
            'baidu_search_api_key': _get_typed_value(search_engines_section, 'baidu_search_api_key', ''),
            'bing_country_code': _get_typed_value(search_engines_section, 'bing_country_code', ''),
            'bing_search_api_url': _get_typed_value(search_engines_section, 'bing_search_api_url', ''),
            'brave_country_code': _get_typed_value(search_engines_section, 'brave_country_code', ''),
            'google_search_api_url': _get_typed_value(search_engines_section, 'google_search_api_url', ''),
            'google_search_engine_id': _get_typed_value(search_engines_section, 'google_search_engine_id', ''),
            'google_simp_trad_chinese': _get_typed_value(search_engines_section, 'google_simp_trad_chinese', False, bool),
            'limit_google_search_to_country': _get_typed_value(search_engines_section, 'limit_google_search_to_country', False, bool),
            'google_search_country': _get_typed_value(search_engines_section, 'google_search_country', ''),
            'google_search_country_code': _get_typed_value(search_engines_section, 'google_search_country_code', ''),
            'google_search_filter_setting': _get_typed_value(search_engines_section, 'google_filter_setting', ''),
            'google_user_geolocation': _get_typed_value(search_engines_section, 'google_user_geolocation', False, bool),
            'google_ui_language': _get_typed_value(search_engines_section, 'google_ui_language', ''),
            'google_limit_search_results_to_language': _get_typed_value(search_engines_section, 'google_limit_search_results_to_language', False, bool),
            'google_site_search_include': _get_typed_value(search_engines_section, 'google_site_search_include', ''),
            'google_site_search_exclude': _get_typed_value(search_engines_section, 'google_site_search_exclude', ''),
            'google_sort_results_by': _get_typed_value(search_engines_section, 'google_sort_results_by', ''),
            'google_default_search_results': _get_typed_value(search_engines_section, 'google_default_search_results', 10, int),
            'google_safe_search': _get_typed_value(search_engines_section, 'google_safe_search', False, bool),
            'google_enable_site_search': _get_typed_value(search_engines_section, 'google_enable_site_search', False, bool),
            'yandex_search_engine_id': _get_typed_value(search_engines_section, 'yandex_search_engine_id', ''),
        },
        "search_engines_keys": { # API Keys for various search engines from 'SearchEngines' TOML table
            'baidu_search_api_key': _get_typed_value(search_engines_section, 'search_engine_api_key_baidu', ''),
            'bing_search_api_key': _get_typed_value(search_engines_section, 'search_engine_api_key_bing', ''),
            'brave_search_api_key': _get_typed_value(search_engines_section, 'brave_search_api_key', ''),
            'brave_search_ai_api_key': _get_typed_value(search_engines_section, 'brave_search_ai_api_key', ''),
            'duckduckgo_search_api_key': _get_typed_value(search_engines_section, 'duckduckgo_search_api_key', ''),
            'google_search_api_key': _get_typed_value(search_engines_section, 'google_search_api_key', ''),
            'kagi_search_api_key': _get_typed_value(search_engines_section, 'kagi_search_api_key', ''),
            'searx_search_api_url': _get_typed_value(search_engines_section, 'search_engine_searx_api', ''),
            'tavily_search_api_key': _get_typed_value(search_engines_section, 'tavily_search_api_key', ''),
            'yandex_search_api_key': _get_typed_value(search_engines_section, 'yandex_search_api_key', ''),
        },
        "prompts_strings": { # Specific prompt strings from 'Prompts' TOML table
            'sub_question_generation_prompt': _get_typed_value(get_toml_section('Prompts'), 'sub_question_generation_prompt', ''),
            'search_result_relevance_eval_prompt': _get_typed_value(get_toml_section('Prompts'), 'search_result_relevance_eval_prompt', ''),
            'analyze_search_results_prompt': _get_typed_value(get_toml_section('Prompts'), 'analyze_search_results_prompt', ''),
        },
        "web_scraper_settings": {
            'web_scraper_api_key': _get_typed_value(web_scraper_section, 'web_scraper_api_key', ''),
            'web_scraper_api_url': _get_typed_value(web_scraper_section, 'web_scraper_api_url', ''),
            # ... (all web scraper settings)
        },

        # Configurations from hardcoded dicts (now from TOML or fallback to Python dicts)
        "APP_TTS_CONFIG": {**DEFAULT_APP_TTS_CONFIG, **app_tts_config},
        "APP_DATABASE_CONFIG": {**DEFAULT_DATABASE_CONFIG, **app_database_config},
        "APP_RAG_SEARCH_CONFIG": {**DEFAULT_RAG_SEARCH_CONFIG, **app_rag_search_config},

        "COMPREHENSIVE_CONFIG_RAW": toml_config_data, # Store the raw TOML data if needed
        "OPENAI_API_KEY": openai_api_key, # Top-level convenience access
    }

    # Populate the rest of chunking_config (tedious but necessary)
    chunking_types = ['audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump',
                      'obsidian_note', 'podcast', 'text', 'video']
    for ctype in chunking_types:
        # Use direct defaults from chunking_section or hardcoded fallbacks
        default_method = _get_typed_value(chunking_section, "chunking_method", "words")
        default_max_size = _get_typed_value(chunking_section, "chunk_max_size", 400, int)
        default_overlap = _get_typed_value(chunking_section, "chunk_overlap", 200, int)
        default_adaptive = _get_typed_value(chunking_section, "adaptive_chunking", False, bool)
        default_multi_level = _get_typed_value(chunking_section, "chunking_multi_level", False, bool)
        default_language = _get_typed_value(chunking_section, "chunk_language", global_default_chunk_language)

        # Only set if not already defined in lines 494-562
        if f"{ctype}_chunking_method" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_method"] = _get_typed_value(
                chunking_section, f"{ctype}_chunking_method", default_method)
        if f"{ctype}_chunk_max_size" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_max_size"] = _get_typed_value(
                chunking_section, f"{ctype}_chunk_max_size", default_max_size, int)
        if f"{ctype}_chunk_overlap" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunk_overlap"] = _get_typed_value(
                chunking_section, f"{ctype}_chunk_overlap", default_overlap, int)
        if f"{ctype}_adaptive_chunking" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_adaptive_chunking"] = _get_typed_value(
                chunking_section, f"{ctype}_adaptive_chunking", default_adaptive, bool)
        if f"{ctype}_chunking_multi_level" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_chunking_multi_level"] = _get_typed_value(
                chunking_section, f"{ctype}_chunking_multi_level", default_multi_level, bool)
        if f"{ctype}_language" not in config_dict["chunking_config"]:
            config_dict["chunking_config"][f"{ctype}_language"] = _get_typed_value(
                chunking_section, f"{ctype}_language", default_language)


    # --- Warnings ---

    # Create necessary directories if they don't exist
    # Ensure main SQLite database directory exists
    if config_dict["DATABASE_URL"].startswith("sqlite:///"):
        main_db_file_path_str = config_dict["DATABASE_URL"].replace("sqlite:///", "")
        main_db_file_path = Path(main_db_file_path_str)
        if not main_db_file_path.is_absolute():
             main_db_file_path = ACTUAL_PROJECT_ROOT / main_db_file_path # Ensure absolute if relative
        try:
            main_db_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured main SQLite database directory exists: {main_db_file_path.parent}")
        except Exception as e:
            logger.error(f"Could not create database directory {main_db_file_path.parent}: {e}")


    # Ensure USER_DB_BASE_DIR exists
    try:
        user_data_base_dir.mkdir(parents=True, exist_ok=True) # user_data_base_dir is already a Path object
        logger.info(f"Ensured user data base directory exists: {user_data_base_dir}")
    except Exception as e:
        logger.error(f"Could not create user data base directory {user_data_base_dir}: {e}")


    return config_dict


# --- Global Settings Object ---
settings = load_settings()

# --- Global default_api_endpoint (example of using the new settings) ---
try:
    # Accessing deeply nested key safely
    default_api_endpoint = settings.get('llm_api_settings', {}).get('default_api', 'openai')
    logger.info(f"Default API Endpoint (from config.py global scope): {default_api_endpoint}")
except Exception as e:
    logger.error(f"Critical error setting default_api_endpoint in config.py global scope: {str(e)}", exc_info=True)
    default_api_endpoint = "openai"  # Fallback

# --- Optional: Export individual variables if needed (generally prefer using settings dict) ---
# SINGLE_USER_MODE = settings["SINGLE_USER_MODE"]
# OPENAI_API_KEY = settings["OPENAI_API_KEY"]
# ... etc.

# Make APP_CONFIG, DATABASE_CONFIG, RAG_SEARCH_CONFIG available globally if needed
# These are now loaded from TOML into the `settings` dictionary.
APP_CONFIG = settings.get("APP_TTS_CONFIG", DEFAULT_APP_TTS_CONFIG) # Fallback if not in settings for some reason
DATABASE_CONFIG = settings.get("APP_DATABASE_CONFIG", DEFAULT_DATABASE_CONFIG)
RAG_SEARCH_CONFIG = settings.get("APP_RAG_SEARCH_CONFIG", DEFAULT_RAG_SEARCH_CONFIG)

#######################################################################################################################
# --- CLI User Configuration Section ---
#######################################################################################################################

# --- Configuration File Content (for reference or auto-creation for the CLI) ---
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
"""

# --- Python dictionary representation of the default configuration for the CLI ---
DEFAULT_CONFIG: Dict[str, Any] = tomllib.loads(CONFIG_TOML_CONTENT)


# --- Configuration Loading Logic for the CLI ---
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def deep_merge_dicts(base: Dict, update: Dict) -> Dict:
    """Recursively merges update_dict into base_dict."""
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Loads CLI configuration from TOML file (DEFAULT_CONFIG_PATH),
    falling back to DEFAULT_CONFIG.
    Ensures that all keys from DEFAULT_CONFIG are present.
    If the config file does not exist, it is created with default content.
    Caches the loaded configuration.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    loaded_config = copy.deepcopy(DEFAULT_CONFIG) # Start with defaults

    if not DEFAULT_CONFIG_PATH.exists():
        logger.info(f"CLI Config file not found at {DEFAULT_CONFIG_PATH}. Creating with default values.")
        try:
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(CONFIG_TOML_CONTENT)
            logger.info(f"Created default CLI config file at {DEFAULT_CONFIG_PATH}")
        except OSError as e:
            logger.error(f"Could not create default CLI config file {DEFAULT_CONFIG_PATH}: {e}")
            # loaded_config remains DEFAULT_CONFIG, which is fine.
    else:
        try:
            with open(DEFAULT_CONFIG_PATH, "rb") as f:
                user_config = tomllib.load(f)
            # Deep merge user_config into loaded_config (which is a copy of DEFAULT_CONFIG)
            loaded_config = deep_merge_dicts(loaded_config, user_config)
            logger.info(f"Successfully loaded and merged CLI config from {DEFAULT_CONFIG_PATH}")
        except tomllib.TOMLDecodeError as e: # Corrected from TomlDecodeError
            logger.error(f"Error decoding CLI TOML config file {DEFAULT_CONFIG_PATH}: {e}. Using default configuration.", exc_info=True)
            # loaded_config remains as DEFAULT_CONFIG merged with potentially partial user_config if deep_merge_dicts ran before error
            # For safety, reset to pure defaults if major decode error.
            # However, deep_merge_dicts happens *after* successful tomllib.load(), so this path means user_config is bad.
            # loaded_config at this point would still be the initial deepcopy(DEFAULT_CONFIG).
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading CLI config {DEFAULT_CONFIG_PATH}: {e}. Using default configuration.", exc_info=True)
            # loaded_config remains as DEFAULT_CONFIG

    _CONFIG_CACHE = loaded_config
    return _CONFIG_CACHE


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """Gets a specific setting from the loaded CLI configuration."""
    config = load_config()
    # If default is not None, it implies we expect the key, and default acts as a true fallback.
    # If default is None, tomllib might have loaded a key with value None, which is valid.
    if default is not None:
        return config.get(section, {}).get(key, default)
    else:
        return config.get(section, {}).get(key)


# --- Function to get providers and their models for the CLI ---
def get_providers_and_models() -> Dict[str, List[str]]:
    """
    Loads provider and model configuration from the [providers] section of CLI config,
    validates it, and returns a dictionary of valid providers mapped
    to their list of models.
    """
    config = load_config()
    providers_data = config.get("providers", {})

    valid_providers: Dict[str, List[str]] = {}

    if isinstance(providers_data, dict):
        for provider, models in providers_data.items():
            if isinstance(models, list) and all(isinstance(m, str) for m in models):
                valid_providers[provider] = models
            else:
                logger.warning(
                    f"Invalid model list for provider '{provider}' in CLI config's [providers] section. "
                    f"Value: {models!r}. Expected a list of strings. Skipping."
                )
    else:
        logger.error(
            f"CLI Config's 'providers' section is not a dictionary: {providers_data!r}. "
            f"No provider/model data available from [providers] section."
        )
        if not isinstance(DEFAULT_CONFIG.get("providers"), dict):
             logger.error("DEFAULT_CONFIG 'providers' section is also not a dict. This is a bug in DEFAULT_CONFIG.")
             return {}

    logger.debug(f"get_providers_and_models (CLI) returning providers: {list(valid_providers.keys())}")
    return valid_providers

# --- Database and Log File Path for the CLI (uses CLI config) ---
def get_database_path() -> Path:
    config = load_config()
    default_db_path_str = DEFAULT_CONFIG.get("database", {}).get("path", "~/.local/share/tldw_cli/tldw_cli_data.db")
    db_path_str = get_setting("database", "path", default_db_path_str)

    db_path = Path(db_path_str).expanduser().resolve()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create CLI database directory {db_path.parent}: {e}", exc_info=True)
    return db_path


def get_log_file_path() -> Path:
    # Ensure database path (and thus its parent dir) is resolved first, as log might go there.
    db_parent_dir = get_database_path().parent

    default_log_filename = DEFAULT_CONFIG.get("logging", {}).get("log_filename", "tldw_cli_app.log")
    log_filename = get_setting("logging", "log_filename", default_log_filename)

    return db_parent_dir / log_filename

#
# End of tldw_cli/config.py
#######################################################################################################################
