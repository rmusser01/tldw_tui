# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Static, Button, Input, RichLog, Label, TextArea
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class LLMManagementWindow(Container):
    """
    Container for the LLM Management Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        # Use a unique ID for this window if it's also set in app.py, e.g., "llm_management-window"
        # The id passed from app.py during instantiation will take precedence if set there.
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def on_mount(self) -> None:
        self.app_instance.loguru_logger.debug("LLMManagementWindow.on_mount called")
        # try:
        #     content_pane = self.query_one("#llm-content-pane", Container)
        #     view_areas = content_pane.query(".llm-view-area")
        #     if not view_areas:
        #         self.app_instance.loguru_logger.warning("LLMManagementWindow.on_mount: No .llm-view-area found in #llm-content-pane.")
        #         return
        #
        #     for view in view_areas:
        #         if view.id: # Only hide if it has an ID (sanity check)
        #             self.app_instance.loguru_logger.debug(f"LLMManagementWindow.on_mount: Hiding view #{view.id}")
        #             view.styles.display = "none"
        #         else:
        #             self.app_instance.loguru_logger.warning("LLMManagementWindow.on_mount: Found a .llm-view-area without an ID, not hiding it.")
        # except QueryError as e:
        #     self.app_instance.loguru_logger.error(f"LLMManagementWindow.on_mount: QueryError: {e}", exc_info=True)
        # except Exception as e:
        #     self.app_instance.loguru_logger.error(f"LLMManagementWindow.on_mount: Unexpected error: {e}", exc_info=True)
        pass

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="llm-nav-pane", classes="llm-nav-pane"):
            yield Static("LLM Options", classes="sidebar-title")
            yield Button("Llama.cpp", id="llm-nav-llama-cpp", classes="llm-nav-button")
            yield Button("Llamafile", id="llm-nav-llamafile", classes="llm-nav-button")
            yield Button("Ollama", id="llm-nav-ollama", classes="llm-nav-button")
            yield Button("vLLM", id="llm-nav-vllm", classes="llm-nav-button")
            yield Button("Transformers", id="llm-nav-transformers", classes="llm-nav-button")
            yield Button("MLX-LM", id="llm-nav-mlx-lm", classes="llm-nav-button")
            yield Button("Local Models", id="llm-nav-local-models", classes="llm-nav-button")
            yield Button("Download Models", id="llm-nav-download-models", classes="llm-nav-button")

        with Container(id="llm-content-pane", classes="llm-content-pane"):
            with Container(id="llm-view-llama-cpp", classes="llm-view-area"):
                yield Label("Executable Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamacpp-exec-path", placeholder="/path/to/llama.cpp/server")
                    yield Button("Browse", id="llamacpp-browse-exec-button", classes="browse_button")
                yield Label("Model Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamacpp-model-path", placeholder="/path/to/model.gguf")
                    yield Button("Browse", id="llamacpp-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="llamacpp-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="llamacpp-port", value="8001")
                yield Label("Additional Arguments:", classes="label")
                yield Input(id="llamacpp-additional-args", placeholder="e.g., --n-gpu-layers 1")
                with Container(classes="button_container"):
                    yield Button("Start Server", id="llamacpp-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="llamacpp-stop-server-button", classes="action_button")
                yield RichLog(id="llamacpp-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-llamafile", classes="llm-view-area"):
                yield Label("Llamafile Executable Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamafile-exec-path", placeholder="/path/to/llamafile_executable")
                    yield Button("Browse", id="llamafile-browse-exec-button", classes="browse_button")
                yield Label("Llamafile Model Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamafile-model-path", placeholder="/path/to/model.gguf")
                    yield Button("Browse", id="llamafile-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="llamafile-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="llamafile-port", value="8000")
                yield Label("Additional Arguments:", classes="label")
                yield TextArea(id="llamafile-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark") # Ensure TextArea is imported
                with Container(classes="button_container"):
                    yield Button("Start Server", id="llamafile-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="llamafile-stop-server-button", classes="action_button")
                yield RichLog(id="llamafile-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-vllm", classes="llm-view-area"):
                yield Label("Python Interpreter Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-python-path", value="python", placeholder="e.g., /path/to/venv/bin/python")
                    yield Button("Browse", id="vllm-browse-python-button", classes="browse_button")
                yield Label("Model Path (or HuggingFace Repo ID):", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-model-path", placeholder="e.g., /path/to/model or HuggingFaceName/ModelName")
                    yield Button("Browse", id="vllm-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="vllm-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="vllm-port", value="8000")
                yield Label("Additional Arguments:", classes="label")
                yield TextArea(id="vllm-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark") # Ensure TextArea is imported
                with Container(classes="button_container"):
                    yield Button("Start Server", id="vllm-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="vllm-stop-server-button", classes="action_button")
                yield RichLog(id="vllm-log-output", classes="log_output", wrap=True, highlight=True)
            yield Container(
                Static("Transformers Library Management Area - Content Coming Soon!"),
                id="llm-view-transformers",
                classes="llm-view-area",
            )
            yield Container(
                Static("Local Model Management Area - Content Coming Soon!"),
                id="llm-view-local-models",
                classes="llm-view-area",
            )
            yield Container(
                Static("Model Download Area - Content Coming Soon!"),
                id="llm-view-download-models",
                classes="llm-view-area",
            )
            with Container(id="llm-view-mlx-lm", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("MLX Model Path (HuggingFace ID or local path):", classes="label")
                    yield Input(id="mlx-model-path", placeholder="e.g., mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX", classes="input_field")
                    yield Label("Host:", classes="label")
                    yield Input(id="mlx-host", value="127.0.0.1", classes="input_field")
                    yield Label("Port:", classes="label")
                    yield Input(id="mlx-port", value="8080", classes="input_field")
                    yield Label("Additional Server Arguments:", classes="label")
                    yield TextArea(id="mlx-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark")
                    with Container(classes="button_container"):
                        yield Button("Start MLX Server", id="mlx-start-server-button", classes="action_button")
                        yield Button("Stop MLX Server", id="mlx-stop-server-button", classes="action_button")
                    yield RichLog(id="mlx-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-ollama", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("Ollama Server URL:", classes="label")
                    yield Input(id="ollama-server-url", value="http://localhost:11434", classes="input_field")

                    # List Models
                    yield Label("List Models:", classes="label section_label")
                    with Container(classes="action_container"):
                        yield Button("List Models", id="ollama-list-models-button", classes="action_button")
                    yield TextArea(id="ollama-list-models-output", read_only=True, classes="output_textarea")

                    # Show Model Info
                    yield Label("Show Model Information:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-show-model-name", placeholder="Model name (e.g., llama2)", classes="input_field_short")
                        yield Button("Show Info", id="ollama-show-model-button", classes="action_button_short")
                    yield TextArea(id="ollama-show-model-output", read_only=True, classes="output_textarea")

                    # Delete Model
                    yield Label("Delete Model:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-delete-model-name", placeholder="Model name to delete", classes="input_field_short")
                        yield Button("Delete Model", id="ollama-delete-model-button", classes="action_button_short delete_button")
                    # Output for delete will go to the main ollama-log-output

                    # Copy Model
                    yield Label("Copy Model:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-copy-source-model", placeholder="Source model name", classes="input_field_short")
                        yield Input(id="ollama-copy-destination-model", placeholder="New model name", classes="input_field_short")
                        yield Button("Copy Model", id="ollama-copy-model-button", classes="action_button_short")
                    # Output for copy will go to the main ollama-log-output

                    # Pull Model
                    yield Label("Pull Model:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-pull-model-name", placeholder="Model name (e.g., llama2)", classes="input_field_short")
                        yield Button("Pull Model", id="ollama-pull-model-button", classes="action_button_short")
                    # Progress/output for pull will go to the main ollama-log-output (or a dedicated area if needed later)

                    # Create Model
                    yield Label("Create Model (from Modelfile):", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-create-model-name", placeholder="Name for new model", classes="input_field_short")
                        yield Button("Browse for Modelfile", id="ollama-browse-modelfile-button", classes="browse_button_short") # Consider styling consistency
                    yield Input(id="ollama-create-modelfile-path", placeholder="Path to Modelfile will appear here", disabled=True, classes="input_field_long") # To display selected path
                    yield Button("Create Model", id="ollama-create-model-button", classes="action_button")
                    # Output for create will go to the main ollama-log-output

                    # Push Model
                    yield Label("Push Model:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-push-model-name", placeholder="Model name to push", classes="input_field_short")
                        # Potentially add an input for destination if not defaulting or configured elsewhere
                        yield Button("Push Model", id="ollama-push-model-button", classes="action_button_short")
                    # Progress/output for push will go to the main ollama-log-output

                    # Generate Embeddings
                    yield Label("Generate Embeddings:", classes="label section_label")
                    with Container(classes="input_action_container"):
                        yield Input(id="ollama-embeddings-model-name", placeholder="Model name for embeddings", classes="input_field_short")
                        yield Input(id="ollama-embeddings-prompt", placeholder="Prompt for embeddings", classes="input_field_long") # Longer input for prompt
                        yield Button("Generate Embeddings", id="ollama-embeddings-button", classes="action_button_short")
                    yield TextArea(id="ollama-embeddings-output", read_only=True, classes="output_textarea_small") # Smaller text area for embeddings

                    # Running Models (ps)
                    yield Label("List Running Models (ps):", classes="label section_label")
                    with Container(classes="action_container"):
                        yield Button("List Running Models", id="ollama-ps-button", classes="action_button")
                    yield TextArea(id="ollama-ps-output", read_only=True, classes="output_textarea")

                    # General Log Output
                    yield Label("Ollama Log Output:", classes="label section_label")
                    yield RichLog(id="ollama-log-output", classes="log_output", wrap=True, highlight=True)

#
# End of LLM_Management_Window.py
#######################################################################################################################
