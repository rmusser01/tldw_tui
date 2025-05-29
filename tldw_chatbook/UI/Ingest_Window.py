# tldw_chatbook/UI/Ingest_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.widgets import Static, Button, Input, Select, Checkbox, TextArea, Label, RadioSet, RadioButton, Collapsible
#
# Local Imports
from ..tldw_api.schemas import MediaType, ChunkMethod, PdfEngine  # Import Enums
from ..Event_Handlers.ingest_events import (  # Import constants for option container IDs
    TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID,
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID,
    TLDW_API_MEDIAWIKI_OPTIONS_ID
)

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class IngestWindow(Container):
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose_tldw_api_form(self) -> ComposeResult:
        """Composes the form for 'Ingest Media via tldw API'."""
        # Get default API URL from app config
        default_api_url = self.app_instance.app_config.get("tldw_api", {}).get("base_url", "http://127.0.0.1:8000")

        # Get available API providers for analysis from app config
        # This assumes your main app_config has a structure like:
        # "api_settings": { "openai": {"model": "gpt-4"}, "anthropic": { ... } }
        # Or use self.app_instance.providers_models if that's more appropriate
        analysis_api_providers = list(self.app_instance.app_config.get("api_settings", {}).keys())
        analysis_provider_options = [(name, name) for name in analysis_api_providers if name]
        if not analysis_provider_options:
            analysis_provider_options = [("No Providers Configured", Select.BLANK)]

        with VerticalScroll(classes="ingest-form-scrollable"):
            yield Static("TLDW API Configuration", classes="sidebar-title")
            yield Label("API Endpoint URL:")
            yield Input(default_api_url, id="tldw-api-endpoint-url", placeholder="http://localhost:8000")

            yield Label("Authentication Method:")
            yield Select(
                [
                    ("Token from Config", "config_token"),
                    ("Custom Token", "custom_token"),
                    # ("Environment Variable", "env_var_token") # Future: add if needed
                ],
                prompt="Select Auth Method...",
                id="tldw-api-auth-method",
                value="config_token"  # Default
            )
            yield Label("Custom Auth Token:", id="tldw-api-custom-token-label", classes="hidden")  # Hidden by default
            yield Input(
                "",
                id="tldw-api-custom-token",
                placeholder="Enter custom Bearer token",
                password=True,
                classes="hidden"  # Hidden by default
            )

            yield Static("Media Details & Processing Options", classes="sidebar-title")
            yield Label("Media Type to Process:")
            # Use values from the MediaType Literal for options
            media_type_options = [(mt, mt) for mt in MediaType.__args__]
            yield Select(media_type_options, prompt="Select Media Type...", id="tldw-api-media-type")

            # --- Common Input Fields ---
            yield Label("Media URLs (one per line):")
            yield TextArea(id="tldw-api-urls", language="plain_text", classes="ingest-textarea-small")
            yield Label(
                "Local File Paths (one per line, if API supports local path references or for client-side upload):")
            yield TextArea(id="tldw-api-local-files", language="plain_text", classes="ingest-textarea-small")

            with Horizontal(classes="ingest-form-row"):
                with Vertical(classes="ingest-form-col"):
                    yield Label("Title (Optional):")
                    yield Input(id="tldw-api-title", placeholder="Optional title override")
                with Vertical(classes="ingest-form-col"):
                    yield Label("Author (Optional):")
                    yield Input(id="tldw-api-author", placeholder="Optional author override")

            yield Label("Keywords (comma-separated):")
            yield TextArea(id="tldw-api-keywords", classes="ingest-textarea-small")

            # --- Common Processing Options ---
            yield Label("Custom Prompt (for analysis):")
            yield TextArea(id="tldw-api-custom-prompt", classes="ingest-textarea-medium")
            yield Label("System Prompt (for analysis):")
            yield TextArea(id="tldw-api-system-prompt", classes="ingest-textarea-medium")
            yield Checkbox("Perform Analysis (e.g., Summarization)", True, id="tldw-api-perform-analysis")
            yield Label("Analysis API Provider (if analysis enabled):")
            yield Select(analysis_provider_options, id="tldw-api-analysis-api-name",
                         prompt="Select API for Analysis...")

            # --- Common Chunking Options ---
            with Collapsible(title="Chunking Options", collapsed=True, id="tldw-api-chunking-collapsible"):
                yield Checkbox("Perform Chunking", True, id="tldw-api-perform-chunking")
                yield Label("Chunking Method:")
                chunk_method_options = [(cm, cm) for cm in ChunkMethod.__args__]
                yield Select(chunk_method_options, id="tldw-api-chunk-method", prompt="Default (per type)")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Size:")
                        yield Input("500", id="tldw-api-chunk-size", type="integer")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Chunk Overlap:")
                        yield Input("200", id="tldw-api-chunk-overlap", type="integer")
                yield Label("Chunk Language (e.g., 'en', optional):")
                yield Input(id="tldw-api-chunk-lang", placeholder="Defaults to media language")
                yield Checkbox("Use Adaptive Chunking", False, id="tldw-api-adaptive-chunking")
                yield Checkbox("Use Multi-level Chunking", False, id="tldw-api-multi-level-chunking")
                yield Label("Custom Chapter Pattern (Regex, optional):")
                yield Input(id="tldw-api-custom-chapter-pattern", placeholder="e.g., ^Chapter\\s+\\d+")

            # --- Common Analysis Options ---
            with Collapsible(title="Advanced Analysis Options", collapsed=True,
                             id="tldw-api-analysis-opts-collapsible"):
                yield Checkbox("Summarize Recursively (if chunked)", False, id="tldw-api-summarize-recursively")
                yield Checkbox("Perform Rolling Summarization", False, id="tldw-api-perform-rolling-summarization")
                # Add more analysis options here as needed

            # --- Media-Type Specific Option Containers (initially hidden) ---
            # Video Options
            with Container(id=TLDW_API_VIDEO_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("Video Specific Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-whisper-large-v3-turbo-ct2", id="tldw-api-video-transcription-model")
                yield Label("Transcription Language (e.g., 'en'):")
                yield Input("en", id="tldw-api-video-transcription-language")
                yield Checkbox("Enable Speaker Diarization", False, id="tldw-api-video-diarize")
                yield Checkbox("Include Timestamps in Transcription", True, id="tldw-api-video-timestamp")
                yield Checkbox("Enable VAD (Voice Activity Detection)", False, id="tldw-api-video-vad")
                yield Checkbox("Perform Confabulation Check of Analysis", False, id="tldw-api-video-confab-check")
                with Horizontal(classes="ingest-form-row"):
                    with Vertical(classes="ingest-form-col"):
                        yield Label("Start Time (HH:MM:SS or secs):")
                        yield Input(id="tldw-api-video-start-time", placeholder="Optional")
                    with Vertical(classes="ingest-form-col"):
                        yield Label("End Time (HH:MM:SS or secs):")
                        yield Input(id="tldw-api-video-end-time", placeholder="Optional")

            # Audio Options (Example structure, add fields as needed)
            with Container(id=TLDW_API_AUDIO_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("Audio Specific Options", classes="sidebar-title")
                yield Label("Transcription Model:")
                yield Input("deepdml/faster-distil-whisper-large-v3.5", id="tldw-api-audio-transcription-model")
                # Add other audio specific fields from ProcessAudioRequest: lang, diarize, timestamp, vad, confab

            # PDF Options
            with Container(id=TLDW_API_PDF_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("PDF Specific Options", classes="sidebar-title")
                pdf_engine_options = [(engine, engine) for engine in PdfEngine.__args__]
                yield Label("PDF Parsing Engine:")
                yield Select(pdf_engine_options, id="tldw-api-pdf-engine", value="pymupdf4llm")

            # Ebook Options
            with Container(id=TLDW_API_EBOOK_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("Ebook Specific Options", classes="sidebar-title")
                ebook_extraction_options = [("filtered", "filtered"), ("markdown", "markdown"), ("basic", "basic")]
                yield Label("Ebook Extraction Method:")
                yield Select(ebook_extraction_options, id="tldw-api-ebook-extraction-method", value="filtered")
                # Ebook chunk_method defaults to ebook_chapters in schema, no specific UI override here unless complex

            # Document Options
            with Container(id=TLDW_API_DOCUMENT_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("Document Specific Options", classes="sidebar-title")
                # Document chunk_method defaults to sentences in schema
                # Add specific document parsing options if any

            # XML Options
            with Container(id=TLDW_API_XML_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("XML Specific Options", classes="sidebar-title")
                yield Checkbox("Auto Summarize XML Content", False, id="tldw-api-xml-auto-summarize")

            # MediaWiki Dump Options
            with Container(id=TLDW_API_MEDIAWIKI_OPTIONS_ID, classes="tldw-api-media-specific-options hidden"):
                yield Static("MediaWiki Dump Specific Options", classes="sidebar-title")
                yield Label("Wiki Name (for identification):")
                yield Input(id="tldw-api-mediawiki-wiki-name", placeholder="e.g., my_wiki_backup")
                yield Label("Namespaces (comma-sep IDs, optional):")
                yield Input(id="tldw-api-mediawiki-namespaces", placeholder="e.g., 0,14")
                yield Checkbox("Skip Redirect Pages", True, id="tldw-api-mediawiki-skip-redirects")

            yield Static("Local Database Options", classes="sidebar-title")
            yield Checkbox("Overwrite if media exists in local DB", False, id="tldw-api-overwrite-db")

            yield Button("Submit to TLDW API", id="tldw-api-submit", variant="primary", classes="ingest-submit-button")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ingest-nav-pane", classes="ingest-nav-pane"):
            yield Static("Ingestion Methods", classes="sidebar-title")
            yield Button("Ingest Prompts", id="ingest-nav-prompts", classes="ingest-nav-button")
            yield Button("Ingest Characters", id="ingest-nav-characters", classes="ingest-nav-button")
            yield Button("Ingest Media (Local)", id="ingest-nav-media",
                         classes="ingest-nav-button")  # Renamed for clarity
            yield Button("Ingest Notes", id="ingest-nav-notes", classes="ingest-nav-button")
            yield Button("Ingest Media via tldw API", id="ingest-nav-tldw-api",
                         classes="ingest-nav-button")  # Changed ID slightly

        with Container(id="ingest-content-pane", classes="ingest-content-pane"):
            yield Container(
                Static("Prompt Ingestion Area - Content Coming Soon!"),
                id="ingest-view-prompts",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Character Ingestion Area - Content Coming Soon!"),
                id="ingest-view-characters",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Local Media Ingestion Area - Content Coming Soon!"),  # For direct local processing
                id="ingest-view-media",
                classes="ingest-view-area",
            )
            yield Container(
                Static("Note Ingestion Area - Content Coming Soon!"),
                id="ingest-view-notes",
                classes="ingest-view-area",
            )
            # New container for tldw API form
            with Container(id="ingest-view-tldw-api", classes="ingest-view-area"):
                yield from self.compose_tldw_api_form()

#
# End of Logs_Window.py
#######################################################################################################################
