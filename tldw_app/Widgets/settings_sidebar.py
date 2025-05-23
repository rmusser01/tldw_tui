# settings_sidebar.py
# Description: settings sidebar widget
#
# Imports
#
# 3rd-Party Imports
import logging

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Select, TextArea, Input, Collapsible, Button, Checkbox, ListView
#
# Local Imports
from ..config import get_cli_providers_and_models

#
#######################################################################################################################
#
# Functions:

# Sidebar visual constants ---------------------------------------------------
SIDEBAR_WIDTH = "30%"


def create_settings_sidebar(id_prefix: str, config: dict) -> ComposeResult:
    """Yield the widgets for the settings sidebar.

    The sidebar is divided into four collapsible groups:
        1. General & Chat Settings  – existing controls
        2. Character Chat Settings – placeholders for character‑specific UI
        3. Media Settings          – placeholders for media configuration
        4. Search & Tools Settings – placeholders for search / tool options
    """

    with VerticalScroll(id=f"{id_prefix}-sidebar", classes="sidebar"):
        # -------------------------------------------------------------------
        # Retrieve defaults / provider information (used in Collapsible #1)
        # -------------------------------------------------------------------
        defaults = config.get(f"{id_prefix}_defaults", config.get("chat_defaults", {}))
        providers_models = get_cli_providers_and_models()
        logging.info(
            "Sidebar %s: Received providers_models. Count: %d. Keys: %s",
            id_prefix,
            len(providers_models),
            list(providers_models.keys()),
        )

        available_providers = list(providers_models.keys())
        default_provider: str = defaults.get(
            "provider", available_providers[0] if available_providers else ""
        )
        default_model: str = defaults.get("model", "")
        default_system_prompt: str = defaults.get("system_prompt", "")
        default_temp = str(defaults.get("temperature", 0.7))
        default_top_p = str(defaults.get("top_p", 0.95))
        default_min_p = str(defaults.get("min_p", 0.05))
        default_top_k = str(defaults.get("top_k", 50))

        # -------------------------------------------------------------------
        # Sidebar title (always visible)
        # -------------------------------------------------------------------
        yield Static("Settings", classes="sidebar-title")

        # ===================================================================
        # 1. General & Chat Settings – existing controls
        # ===================================================================
        with Collapsible(title="Current Chat Settings", collapsed=True):
            yield Static(
                "Inference Endpoints & \nService Providers", classes="sidebar-label"
            )
            provider_options = [(provider, provider) for provider in available_providers]
            yield Select(
                options=provider_options,
                prompt="Select Provider…",
                allow_blank=False,
                id=f"{id_prefix}-api-provider",
                value=default_provider,
            )

            # ----------------------------- Model ---------------------------
            yield Static("Model", classes="sidebar-label")
            initial_models = providers_models.get(default_provider, [])
            model_options = [(model, model) for model in initial_models]
            current_model_value = (
                default_model if default_model in initial_models else (initial_models[0] if initial_models else None)
            )
            yield Select(
                options=model_options,
                prompt="Select Model…",
                allow_blank=True,
                id=f"{id_prefix}-api-model",
                value=current_model_value,
            )

            # ------------------ Remaining numeric / text inputs ------------
            yield Static(
                "API Key (Set in config/env)",
                classes="sidebar-label",
                id=f"{id_prefix}-api-key-placeholder",
            )
            yield Static("System prompt", classes="sidebar-label")
            yield TextArea(
                id=f"{id_prefix}-system-prompt",
                text=default_system_prompt,
                classes="sidebar-textarea",
            )
            yield Static("Temperature", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 0.7",
                id=f"{id_prefix}-temperature",
                value=default_temp,
                classes="sidebar-input",
            )
            yield Static("Top‑P", classes="sidebar-label")
            yield Input(
                placeholder="0.0 to 1.0",
                id=f"{id_prefix}-top-p",
                value=default_top_p,
                classes="sidebar-input",
            )
            yield Static("Min‑P", classes="sidebar-label")
            yield Input(
                placeholder="0.0 to 1.0",
                id=f"{id_prefix}-min-p",
                value=default_min_p,
                classes="sidebar-input",
            )
            yield Static("Top‑K", classes="sidebar-label")
            yield Input(
                placeholder="e.g., 50",
                id=f"{id_prefix}-top-k",
                value=default_top_k,
                classes="sidebar-input",
            )

            if id_prefix == "chat": # Moved Conversation Details content here
                yield Static("Current Conversation Title:", classes="sidebar-label")
                yield Input(
                    id=f"{id_prefix}-conversation-title-input",
                    placeholder="Enter conversation title...",
                    classes="sidebar-input"
                )
                yield Static("Conversation Keywords:", classes="sidebar-label")
                yield TextArea(
                    id=f"{id_prefix}-conversation-keywords-input",
                    classes="sidebar-textarea"
                )
                yield Button(
                    "Save Title & Keywords",
                    id=f"{id_prefix}-save-conversation-details-button",
                    variant="primary",
                    classes="sidebar-button"
                )

        # ===================================================================
        # NEW: Prompts (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":
            with Collapsible(title="Prompts", collapsed=True):
                yield Static("Prompt management UI placeholder")

        # ===================================================================
        # NEW: Saved Conversations (only for chat tab)
        # ===================================================================
        if id_prefix == "chat":  # Assuming "chat" is the id_prefix for the main chat tab
            # NEW: Saved Conversations (only for chat tab)
            with Collapsible(title="Saved Conversations", collapsed=True):
                yield Input(
                    id=f"{id_prefix}-conversation-search-bar",
                    placeholder="Search all chats...",
                    classes="sidebar-input"
                )
                yield Checkbox(
                    "Include Character Chats",
                    id=f"{id_prefix}-conversation-search-include-character-checkbox"
                    # value=False by default for Checkbox
                )
                yield Select(
                    [],  # Empty options initially
                    id=f"{id_prefix}-conversation-search-character-filter-select",
                    allow_blank=True,  # User can select nothing to clear filter
                    prompt="Filter by Character...",
                    classes="sidebar-select"  # Assuming a general class for selects or use default
                )
                yield Checkbox(
                    "All Characters",
                    id=f"{id_prefix}-conversation-search-all-characters-checkbox",
                    value=True  # Default to True
                )
                yield ListView(
                    id=f"{id_prefix}-conversation-search-results-list",
                    classes="sidebar-listview"  # Add specific styling if needed
                )
                # Set initial height for ListView via styles property if not handled by class
                # Example: self.query_one(f"#{id_prefix}-conversation-search-results-list", ListView).styles.height = 10
                yield Button(
                    "Load Selected Chat",
                    id=f"{id_prefix}-conversation-load-selected-button",
                    variant="default",  # Or "primary"
                    classes="sidebar-button"  # Use existing class or new one
                )

        # ===================================================================
        # 3. Media Settings – placeholders
        # ===================================================================
        with Collapsible(title="Media Settings", collapsed=True):
            yield Static("Media settings will go here (placeholder)", classes="sidebar-placeholder")

        # ===================================================================
        # 4. Search & Tools Settings – placeholders
        # ===================================================================
        with Collapsible(title="Search & Tools Settings", collapsed=True):
            yield Static(
                "Search & tools configuration will go here (placeholder)",
                classes="sidebar-placeholder",
            )

        # ===================================================================
        # 5. System Settings – placeholders
        # ===================================================================
        with Collapsible(title="Partial System Settings", collapsed=True):
            yield Static(
                "some key system settings will go here (placeholder)",
                classes="sidebar-placeholder",
            )

#
# End of settings_sidebar.py
#######################################################################################################################
