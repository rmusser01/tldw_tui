"""llm_management_events_ollama.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **Ollama** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates Ollama-specific logic from the main llm_management_events.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual.containers import Container
from textual.css.query import QueryError
from textual.widgets import Input, TextArea, RichLog

from .llm_management_events import _make_path_update_callback
from ..Third_Party.textual_fspicker import FileOpen, Filters

if TYPE_CHECKING:
    from ..app import TldwCli

__all__ = [
    # ─── Ollama ───────────────────────────────────────────────────────────────
    "handle_ollama_nav_button_pressed",
    "handle_ollama_list_models_button_pressed",
    "handle_ollama_show_model_button_pressed",
    "handle_ollama_delete_model_button_pressed",
    "handle_ollama_copy_model_button_pressed",
    "handle_ollama_pull_model_button_pressed",
    "handle_ollama_create_model_button_pressed",
    "handle_ollama_browse_modelfile_button_pressed",
    "handle_ollama_push_model_button_pressed",
    "handle_ollama_embeddings_button_pressed",
    "handle_ollama_ps_button_pressed",
]

###############################################################################
# ─── Ollama UI helpers ──────────────────────────────────────────────────────
###############################################################################


async def handle_ollama_nav_button_pressed(app: "TldwCli") -> None:
    """Handle the Ollama navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        view_areas = content_pane.query(".llm-view-area")

        for view in view_areas:
            if view.id:  # Only hide if it has an ID
                logger.debug(f"Hiding view #{view.id}")
                view.styles.display = "none"
            else: # pragma: no cover
                logger.warning("Found a .llm-view-area without an ID, not hiding it.")

        ollama_view = app.query_one("#llm-view-ollama", Container)
        logger.debug(f"Showing view #{ollama_view.id}")
        ollama_view.styles.display = "block"

        # Clear/initialize Ollama specific fields
        try:
            # Input fields for actions
            app.query_one("#ollama-show-model-name", Input).value = ""
            app.query_one("#ollama-delete-model-name", Input).value = ""
            app.query_one("#ollama-copy-source-model", Input).value = ""
            app.query_one("#ollama-copy-destination-model", Input).value = ""
            app.query_one("#ollama-pull-model-name", Input).value = ""
            app.query_one("#ollama-create-model-name", Input).value = ""
            app.query_one("#ollama-create-modelfile-path", Input).value = ""
            app.query_one("#ollama-push-model-name", Input).value = ""
            app.query_one("#ollama-embeddings-model-name", Input).value = ""
            app.query_one("#ollama-embeddings-prompt", Input).value = ""

            # Output TextAreas
            app.query_one("#ollama-list-models-output", TextArea).clear()
            app.query_one("#ollama-show-model-output", TextArea).clear()
            app.query_one("#ollama-embeddings-output", TextArea).clear()
            app.query_one("#ollama-ps-output", TextArea).clear()

            # Main log output
            log_output = app.query_one("#ollama-log-output", RichLog)
            log_output.clear()
            log_output.write("Switched to Ollama view. Output log cleared.")

        except QueryError as qe: # pragma: no cover
            logger.warning(f"Ollama UI clear: Could not find one or more UI elements during view switch: {qe}")
            app.notify("Warning: Some Ollama UI elements might not have been reset properly.", severity="warning")

        logger.info("Switched to Ollama view and cleared/initialized fields.")
        # app.notify("Switched to Ollama view.") # Optional notification

    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to Ollama view: Could not find required UI elements.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while switching to Ollama view.", severity="error")


async def handle_ollama_list_models_button_pressed(app: "TldwCli") -> None:
    """Handles the 'List Models' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'List Models' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        log_output_widget = app.query_one("#ollama-list-models-output", TextArea) # Changed to specific output
        # general_log_widget = app.query_one("#ollama-log-output", RichLog) # For general messages

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        log_output_widget.clear()
        # general_log_widget.write(f"Attempting to list models from: {base_url}")
        # Placeholder for API call:
        log_output_widget.write(f"Placeholder: Called ollama_list_local_models with base_url: {base_url}\nOutput will appear here.")
        app.notify("Listing Ollama models (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_list_models_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for listing models.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_list_models_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while listing Ollama models.", severity="error")


async def handle_ollama_show_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Show Model Info' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Show Model Info' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-show-model-name", Input)
        log_output_widget = app.query_one("#ollama-show-model-output", TextArea)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to show info.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.clear()
        # general_log_widget.write(f"Attempting to show info for model: {model_name} from {base_url}")
        # Placeholder for API call:
        log_output_widget.write(f"Placeholder: Called ollama_model_info for model: {model_name} at base_url: {base_url}\nDetails will appear here.")
        app.notify(f"Fetching info for {model_name} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_show_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for showing model info.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_show_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while showing model info.", severity="error")


async def handle_ollama_delete_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Delete Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Delete Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-delete-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog) # General log for delete

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to delete.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.write(f"Attempting to delete model: {model_name} from {base_url}")
        # Placeholder for API call:
        log_output_widget.write(f"Placeholder: Called ollama_delete_model for model: {model_name} at base_url: {base_url}")
        app.notify(f"Deleting model {model_name} (placeholder)...")
        # After actual deletion, you might want to refresh the model list automatically
        # await handle_ollama_list_models_button_pressed(app)
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_delete_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for deleting model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_delete_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while deleting model.", severity="error")


async def handle_ollama_copy_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Copy Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Copy Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        source_model_input = app.query_one("#ollama-copy-source-model", Input)
        dest_model_input = app.query_one("#ollama-copy-destination-model", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        source_model = source_model_input.value.strip()
        dest_model = dest_model_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not source_model:
            app.notify("Source model name is required for copy.", severity="error")
            source_model_input.focus()
            return
        if not dest_model:
            app.notify("Destination model name is required for copy.", severity="error")
            dest_model_input.focus()
            return

        log_output_widget.write(f"Attempting to copy model: {source_model} to {dest_model} from {base_url}")
        # Placeholder for API call:
        log_output_widget.write(f"Placeholder: Called ollama_copy_model for source: {source_model}, destination: {dest_model} at base_url: {base_url}")
        app.notify(f"Copying model {source_model} to {dest_model} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_copy_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for copying model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_copy_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while copying model.", severity="error")


async def handle_ollama_pull_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Pull Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Pull Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-pull-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to pull.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.write(f"Attempting to pull model: {model_name} from {base_url}")
        # Placeholder for API call (will use stream_log_callback):
        log_output_widget.write(f"Placeholder: Called ollama_pull_model for model: {model_name} at base_url: {base_url}. Streaming logs here...")
        app.notify(f"Pulling model {model_name} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_pull_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for pulling model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_pull_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while pulling model.", severity="error")


async def handle_ollama_browse_modelfile_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Browse for Modelfile' button press for Ollama create model."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Browse for Modelfile' button pressed.")

    # No specific filters for "Modelfile" by extension, so allow all files or common text files.
    # Users should know what a Modelfile looks like.
    modelfile_filters = Filters(
        ("Modelfiles (Modelfile, *.txt)", lambda p: p.name.lower() == "modelfile" or p.suffix.lower() == ".txt"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.cwd()), # Start in current working directory or user's preferred location
            title="Select Modelfile",
            filters=modelfile_filters,
        ),
        callback=_make_path_update_callback(app, "ollama-create-modelfile-path"),
    )


async def handle_ollama_create_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Create Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Create Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-create-model-name", Input)
        modelfile_path_input = app.query_one("#ollama-create-modelfile-path", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()
        modelfile_path = modelfile_path_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("New model name is required for creation.", severity="error")
            model_name_input.focus()
            return
        if not modelfile_path:
            app.notify("Path to Modelfile is required for creation.", severity="error")
            # modelfile_path_input.focus() # This is read-only, so focus the browse button indirectly or notify.
            app.notify("Use 'Browse for Modelfile' to select a file.", severity="information")
            return
        if not Path(modelfile_path).is_file():
            app.notify(f"Modelfile not found at: {modelfile_path}", severity="error")
            # modelfile_path_input.focus()
            return


        log_output_widget.write(f"Attempting to create model: {model_name} using Modelfile: {modelfile_path} from {base_url}")
        # Placeholder for API call (will use stream_log_callback):
        log_output_widget.write(f"Placeholder: Called ollama_create_model for model: {model_name}, path: {modelfile_path} at base_url: {base_url}. Streaming logs here...")
        app.notify(f"Creating model {model_name} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_create_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for creating model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_create_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while creating model.", severity="error")


async def handle_ollama_push_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Push Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Push Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-push-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to push.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.write(f"Attempting to push model: {model_name} from {base_url}")
        # Placeholder for API call (will use stream_log_callback):
        log_output_widget.write(f"Placeholder: Called ollama_push_model for model: {model_name} at base_url: {base_url}. Streaming logs here...")
        app.notify(f"Pushing model {model_name} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_push_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for pushing model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_push_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while pushing model.", severity="error")


async def handle_ollama_embeddings_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Generate Embeddings' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Generate Embeddings' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-embeddings-model-name", Input)
        prompt_input = app.query_one("#ollama-embeddings-prompt", Input)
        embeddings_output_widget = app.query_one("#ollama-embeddings-output", TextArea)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)


        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()
        prompt = prompt_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required for embeddings.", severity="error")
            model_name_input.focus()
            return
        if not prompt:
            app.notify("Prompt is required for embeddings.", severity="error")
            prompt_input.focus()
            return

        embeddings_output_widget.clear()
        # general_log_widget.write(f"Attempting to generate embeddings for model: {model_name} with prompt: '{prompt[:30]}...' from {base_url}")
        # Placeholder for API call:
        embeddings_output_widget.write(f"Placeholder: Called ollama_generate_embeddings for model: {model_name}, prompt: '{prompt}' at base_url: {base_url}\nEmbeddings will appear here.")
        app.notify(f"Generating embeddings with {model_name} (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_embeddings_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for generating embeddings.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_embeddings_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while generating embeddings.", severity="error")


async def handle_ollama_ps_button_pressed(app: "TldwCli") -> None:
    """Handles the 'List Running Models (ps)' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'List Running Models (ps)' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        ps_output_widget = app.query_one("#ollama-ps-output", TextArea)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        ps_output_widget.clear()
        # general_log_widget.write(f"Attempting to list running models (ps) from: {base_url}")
        # Placeholder for API call:
        ps_output_widget.write(f"Placeholder: Called ollama_list_running_models at base_url: {base_url}\nOutput will appear here.")
        app.notify("Listing running Ollama models (ps) (placeholder)...")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_ps_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for listing running models.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_ps_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while listing running models.", severity="error")

#
# End of llm_management_events_ollama.py
########################################################################################################################
