# tldw_chatbook/Event_Handlers/ingest_events.py
#
#
# Imports
import json
import yaml  # Add yaml import if not already there
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Callable
#
# 3rd-party Libraries
from loguru import logger
from textual.widgets import Select, Input, TextArea, Checkbox, Label, Static, Markdown, ListItem, \
    ListView, Collapsible
from textual.css.query import QueryError
from textual.containers import Container, VerticalScroll
#
# Local Imports
import tldw_chatbook.Event_Handlers.conv_char_events as ccp_handlers
from . import chat_events as chat_handlers
from .chat_events import populate_chat_conversation_character_filter_select
from ..tldw_api import (
    TLDWAPIClient, ProcessVideoRequest, ProcessAudioRequest,
    APIConnectionError, APIRequestError, APIResponseError, AuthenticationError,
    MediaItemProcessResult, ProcessedMediaWikiPage, BatchMediaProcessResponse,
    ProcessPDFRequest, ProcessEbookRequest, ProcessDocumentRequest,
    ProcessXMLRequest, ProcessMediaWikiRequest
)
# Prompts Interop (existing)
from ..Prompt_Management.Prompts_Interop import (
    parse_yaml_prompts_from_content, parse_json_prompts_from_content,
    parse_markdown_prompts_from_content, parse_txt_prompts_from_content,
    is_initialized as prompts_db_initialized,  # Renamed for clarity
    import_prompts_from_files, _get_file_type as _get_prompt_file_type  # Renamed for clarity
)
from ..DB.ChaChaNotes_DB import ConflictError as ChaChaConflictError, CharactersRAGDBError
# Character Chat Lib for parsing and importing character cards
from ..Character_Chat import Character_Chat_Lib as ccl
from ..DB.ChaChaNotes_DB import ConflictError as ChaChaConflictError  # For character import conflict
from ..Third_Party.textual_fspicker import Filters, FileOpen
#
if TYPE_CHECKING:
    from ..app import TldwCli
########################################################################################################################
#
# Functions:

# --- Prompt Ingest Constants (existing) ---
MAX_PROMPT_PREVIEWS = 10
PROMPT_FILE_FILTERS = Filters(
    ("Markdown", lambda p: p.suffix.lower() == ".md"),
    ("JSON", lambda p: p.suffix.lower() == ".json"),
    ("YAML", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("Text", lambda p: p.suffix.lower() == ".txt"),
    ("All Supported", lambda p: p.suffix.lower() in (".md", ".json", ".yaml", ".yml", ".txt")),
    ("All Files", lambda _: True),
)

# --- Character Ingest Constants ---
MAX_CHARACTER_PREVIEWS = 5  # Show fewer character previews as they can be larger
CHARACTER_FILE_FILTERS = Filters(
    ("Character Cards (JSON, YAML, PNG, WebP, MD)",
     lambda p: p.suffix.lower() in (".json", ".yaml", ".yml", ".png", ".webp", ".md")),
    ("JSON (*.json)", lambda p: p.suffix.lower() == ".json"),
    ("YAML (*.yaml, *.yml)", lambda p: p.suffix.lower() in (".yaml", ".yml")),
    ("PNG (*.png)", lambda p: p.suffix.lower() == ".png"),
    ("WebP (*.webp)", lambda p: p.suffix.lower() == ".webp"),
    ("Markdown (*.md)", lambda p: p.suffix.lower() == ".md"),
    ("All Files", lambda _: True),
)

# --- Notes Ingest Constants ---
MAX_NOTE_PREVIEWS = 10
NOTE_FILE_FILTERS = Filters(
    ("JSON Notes (*.json)", lambda p: p.suffix.lower() == ".json"),
    # ("Markdown Notes (*.md)", lambda p: p.suffix.lower() == ".md"), # Example for future
    ("All Files", lambda _: True),
)

def _truncate_text(text: Optional[str], max_len: int) -> str:
    """
    Truncates a string to a maximum length, adding ellipsis if truncated.
    Returns 'N/A' if the input text is None or empty.
    """
    if not text: # Handles None or empty string
        return "N/A"
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


# --- Character Preview Functions (NEW) ---
async def _update_character_preview_display(app: 'TldwCli') -> None:
    """Updates the character preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-characters-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_characters_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no characters found.",
                       id="ingest-characters-preview-placeholder"))
            return

        num_to_display = len(app.parsed_characters_for_preview)
        chars_to_show = app.parsed_characters_for_preview[:MAX_CHARACTER_PREVIEWS]

        for idx, char_data in enumerate(chars_to_show):
            name = char_data.get("name", f"Unnamed Character {idx + 1}")
            description = _truncate_text(char_data.get("description"), 150)
            creator = char_data.get("creator", "N/A")
            # Add more fields as relevant for previewing a character

            md_content = f"""### {name}
**Creator:** {creator}
**Description:**
```text
{description}
```
---
"""
            # For PNG/WebP, you might not show much textual preview other than filename or basic metadata if extracted.
            # For JSON/YAML/MD, you can parse more.
            # This is a simplified preview.
            if "error" in char_data:  # If parsing produced an error message
                md_content = f"""### Error parsing {char_data.get("filename", "file")}
 ```text
 {char_data["error"]}
 ```
 ---
 """
            await preview_area.mount(
                Markdown(md_content, classes="prompt-preview-item"))  # Reusing class, can make specific

        if num_to_display > MAX_CHARACTER_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_CHARACTER_PREVIEWS} more characters loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for character preview update: {e}")
        app.notify("Error updating character preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating character preview: {e}", exc_info=True)
        app.notify("Unexpected error during character preview update.", severity="error")


def _parse_single_character_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """
    Parses a single character file for preview.
    Returns a list containing one dict (or an error dict).
    """
    logger.debug(f"Parsing character file for preview: {file_path}")
    # For preview, we might not need the full DB interaction of ccl.load_character_card_from_file
    # We primarily need 'name' and maybe 'description'.
    # ccl.load_character_card_from_file handles JSON, YAML, MD, PNG, WebP.
    # It returns a dictionary of the character card data.

    # Minimal DB for ccl.load_character_card_from_file to work if it expects one
    # For preview, we might not want to pass a real DB.
    # Let's assume ccl.load_character_card_from_file can work without a db for simple parsing,
    # or we make a lightweight parser.
    # For simplicity, let's try calling it. If it strictly needs a DB, this part needs adjustment.

    # Placeholder: In a real scenario, ccl.load_character_card_from_file might need a dummy DB object
    # or a refactor to separate parsing from DB saving.
    # For now, we'll attempt a simplified parsing for preview.

    preview_data = {"filename": file_path.name}
    file_suffix = file_path.suffix.lower()

    try:
        if file_suffix in (".json", ".yaml", ".yml", ".md"):
            # ccl.load_character_card_from_file can take a path and parse these
            # It doesn't strictly need a DB for parsing if the file doesn't refer to external DB lookups during parse.
            # Let's assume it's primarily about loading the structure.
            # This function is in Character_Chat_Lib.py
            # `load_character_card_from_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:`
            char_dict = ccl.load_character_card_from_file(str(file_path))  # No DB passed
            if char_dict:
                preview_data.update(char_dict)
                # Ensure 'name' is present for valid preview item
                if not preview_data.get("name"):
                    preview_data["name"] = file_path.stem  # Fallback to filename without ext
            else:
                preview_data["error"] = f"Could not parse character data from {file_path.name}."
                preview_data["name"] = f"Error: {file_path.name}"

        elif file_suffix in (".png", ".webp"):
            # For images, ccl.load_character_card_from_file tries to extract metadata.
            char_dict = ccl.load_character_card_from_file(str(file_path))
            if char_dict:
                preview_data.update(char_dict)
                if not preview_data.get("name"):
                    preview_data["name"] = file_path.stem
            else:
                preview_data["name"] = file_path.name  # Just show filename
                preview_data["description"] = "Image file (binary data not shown in preview)"
        else:
            preview_data["error"] = f"Unsupported file type for character preview: {file_path.name}"
            preview_data["name"] = f"Error: {file_path.name}"

        return [preview_data]

    except Exception as e:
        logger.error(f"Error parsing character file {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error previewing {file_path.name}.", severity="error")
        return [{"filename": file_path.name, "name": f"Error: {file_path.name}", "error": str(e)}]


async def _handle_character_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """Callback for character file selection."""
    if selected_path:
        logger.info(f"Character file selected via dialog: {selected_path}")
        if selected_path in app.selected_character_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the character selection.", severity="warning")
            return

        app.selected_character_files_for_import.append(selected_path)
        app.last_character_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-characters-selected-files-list", ListView)

            # Check if the list view contains only the "No files selected." placeholder
            # This is safer than assuming it's always the first child.
            placeholder_exists = False
            if list_view.children:  # Check if there are any children
                first_child = list_view.children[0]
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label_of_first_item = first_child.children[0]
                    if isinstance(first_label_of_first_item, Label):
                        # Convert Label's renderable (Rich Text) to plain string for comparison
                        if str(first_label_of_first_item.renderable).strip() == "No files selected.":
                            placeholder_exists = True

            if placeholder_exists:
                await list_view.clear()
                logger.debug("Cleared 'No files selected.' placeholder from character list.")

            await list_view.append(ListItem(Label(str(selected_path))))
            logger.debug(f"Appended '{selected_path}' to character list view.")

        except QueryError:
            logger.error("Could not find #ingest-characters-selected-files-list ListView to update.")
        except Exception as e_lv:
            logger.error(f"Error updating character list view: {e_lv}", exc_info=True)

        parsed_chars_from_file = _parse_single_character_file_for_preview(selected_path, app)
        app.parsed_characters_for_preview.extend(parsed_chars_from_file)

        await _update_character_preview_display(app)
    else:
        logger.info("Character file selection cancelled.")
        app.notify("File selection cancelled.")


# --- Prompt Ingest Handlers (existing, ensure they use renamed constants/functions if any) ---
async def handle_ingest_prompts_select_file_button_pressed(app: 'TldwCli') -> None:
    logger.debug("Select Prompt File(s) button pressed. Opening file dialog.")
    current_dir = app.last_prompt_import_dir or Path(".")
    await app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Prompt File (.md, .json, .yaml, .txt)",
            filters=PROMPT_FILE_FILTERS
        ),
        lambda path: app.call_after_refresh(lambda: _handle_prompt_file_selected_callback(app, path))
        # path type here is Optional[Path]
    )



# --- Character Ingest Handlers (NEW) ---
async def handle_ingest_characters_select_file_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Select Character File(s)' button press."""
    logger.debug("Select Character File(s) button pressed. Opening file dialog.")
    current_dir = app.last_character_import_dir or Path(".")  # Use new state var

    await app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Character File (.json, .yaml, .png, .webp, .md)",
            filters=CHARACTER_FILE_FILTERS
        ),
        lambda path: app.call_after_refresh(lambda: _handle_character_file_selected_callback(app, path))
        # path type here is Optional[Path]
    )


async def handle_ingest_characters_clear_files_button_pressed(app: 'TldwCli') -> None:
    """Handles 'Clear Selection' for character import."""
    logger.info("Clearing selected character files and preview.")
    app.selected_character_files_for_import.clear()
    app.parsed_characters_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-characters-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-characters-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-characters-preview-placeholder"))

        status_area = app.query_one("#ingest-character-import-status-area", TextArea)
        status_area.clear()
        app.notify("Character selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing character selection: {e}")
        app.notify("Error clearing character UI.", severity="error")


async def handle_ingest_characters_import_now_button_pressed(app: 'TldwCli') -> None:
    """Handles 'Import Selected Characters Now' button press."""
    logger.info("Import Selected Character Files Now button pressed.")

    if not app.selected_character_files_for_import:
        app.notify("No character files selected to import.", severity="warning")
        return

    if not app.notes_service:  # Character cards are stored via NotesService (ChaChaNotesDB)
        msg = "Notes/Character database service is not initialized. Cannot import characters."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting character import.")
        return

    try:
        status_area = app.query_one("#ingest-character-import-status-area", TextArea)
        status_area.clear()  # Clear previous status
        status_area.load_text("Starting character import process...\n")  # Use load_text to set initial content
    except QueryError:
        logger.error("Could not find #ingest-character-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    app.notify("Importing characters...")

    db = app.notes_service._get_db(app.notes_user_id)

    async def import_worker_char():
        results = []
        for file_path in app.selected_character_files_for_import:
            try:
                # Ensure file_path is a string for ccl.import_and_save_character_from_file
                char_id = ccl.import_and_save_character_from_file(db, str(file_path))
                if char_id is not None:
                    char_name = file_path.stem
                    try:
                        card_data = ccl.load_character_card_from_file(str(file_path))
                        if card_data and card_data.get("name"):
                            char_name = card_data.get("name")
                    except Exception:
                        pass

                    results.append({
                        "file_path": str(file_path),
                        "character_name": char_name,
                        "status": "success",
                        "message": f"Character imported successfully. ID: {char_id}",
                        "char_id": char_id
                    })
                else:
                    results.append({
                        "file_path": str(file_path),
                        "character_name": file_path.stem,
                        "status": "failure",
                        "message": "Failed to import (see logs for details)."
                    })
            except ChaChaConflictError as ce:
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "conflict",
                    "message": str(ce)
                })
            except ImportError as ie:
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "failure",
                    "message": f"Import error: {ie}. A required library might be missing."
                })
            except Exception as e:
                logger.error(f"Error importing character from {file_path}: {e}", exc_info=True)
                results.append({
                    "file_path": str(file_path),
                    "character_name": file_path.stem,
                    "status": "failure",
                    "message": f"Unexpected error: {type(e).__name__}"
                })
        return results

    def on_import_success_char(results: List[Dict[str, Any]]):
        log_text_parts = ["Character import process finished.\n\nResults:\n"]  # Renamed to avoid conflict
        successful_imports = 0
        failed_imports = 0
        for res in results:
            status = res.get("status", "unknown")
            file_path_str = res.get("file_path", "N/A")
            char_name = res.get("character_name", "N/A")
            message = res.get("message", "")

            log_text_parts.append(f"File: {Path(file_path_str).name}\n")
            log_text_parts.append(f"  Character: '{char_name}'\n")
            log_text_parts.append(f"  Status: {status.upper()}\n")
            if message:
                log_text_parts.append(f"  Message: {message}\n")
            log_text_parts.append("-" * 30 + "\n")

            if status == "success":
                successful_imports += 1
            else:
                failed_imports += 1

        summary = f"\nSummary: {successful_imports} characters imported, {failed_imports} failed/conflicts."
        log_text_parts.append(summary)

        try:
            status_area_widget = app.query_one("#ingest-character-import-status-area", TextArea)
            status_area_widget.load_text("".join(log_text_parts))  # Use load_text for the final result
        except QueryError:
            logger.error("Could not find #ingest-character-import-status-area to update with results.")

        app.notify(f"Character import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(summary)

        app.call_later(populate_chat_conversation_character_filter_select, app)
        app.call_later(ccp_handlers.populate_ccp_character_select, app)

    def on_import_failure_char(error: Exception):
        logger.error(f"Character import worker failed critically: {error}", exc_info=True)
        try:
            status_area_widget = app.query_one("#ingest-character-import-status-area", TextArea)
            # Append error to existing text or load new text
            current_text = status_area_widget.text
            status_area_widget.load_text(
                current_text + f"\nCharacter import process failed critically: {error}\nCheck logs.\n")
        except QueryError:
            logger.error("Could not find #ingest-character-import-status-area to report critical failure.")

        app.notify(f"Character import CRITICALLY failed: {error}", severity="error", timeout=10)

    app.run_worker(
        import_worker_char,
        name="character_import_worker",
        group="file_operations",
        description="Importing selected character files."
    )


async def _update_prompt_preview_display(app: 'TldwCli') -> None:
    """Updates the prompt preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_prompts_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no prompts found.", id="ingest-prompts-preview-placeholder"))
            return

        num_to_display = len(app.parsed_prompts_for_preview)
        prompts_to_show = app.parsed_prompts_for_preview[:MAX_PROMPT_PREVIEWS]

        for idx, prompt_data in enumerate(prompts_to_show):
            name = prompt_data.get("name", f"Unnamed Prompt {idx + 1}")
            author = prompt_data.get("author", "N/A")
            details = _truncate_text(prompt_data.get("details"), 150)
            system_prompt = _truncate_text(prompt_data.get("system_prompt"), 200)
            user_prompt = _truncate_text(prompt_data.get("user_prompt"), 200)
            keywords_list = prompt_data.get("keywords", [])
            keywords = ", ".join(keywords_list) if keywords_list else "N/A"

            md_content = f"""### {name}
**Author:** {author}
**Keywords:** {keywords}

**Details:**
```text
{details}
```

**System Prompt:**
```text
{system_prompt}
```

**User Prompt:**
```text
{user_prompt}
```
---
"""
            await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

        if num_to_display > MAX_PROMPT_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_PROMPT_PREVIEWS} more prompts loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for prompt preview update: {e}")
        app.notify("Error updating prompt preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating prompt preview: {e}", exc_info=True)
        app.notify("Unexpected error during preview update.", severity="error")


def _parse_single_prompt_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """Parses a single prompt file and returns a list of prompt data dicts."""
    file_type = _get_prompt_file_type(file_path)  # Use helper from interop
    if not file_type:
        logger.warning(f"Unsupported file type for preview: {file_path}")
        return [{"name": f"Error: Unsupported type {file_path.name}",
                 "details": "File type not recognized for prompt import."}]

    parser_map: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
        "json": parse_json_prompts_from_content,
        "yaml": parse_yaml_prompts_from_content,
        "markdown": parse_markdown_prompts_from_content,
        "txt": parse_txt_prompts_from_content,
    }
    parser = parser_map.get(file_type)
    if not parser:
        logger.error(f"No parser found for file type {file_type} (preview)")
        return [
            {"name": f"Error: No parser for {file_path.name}", "details": f"File type '{file_type}' has no parser."}]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = parser(content)
        if not parsed:  # If parser returns empty list (e.g. empty file or no valid prompts)
            logger.info(f"No prompts found in {file_path.name} by parser for preview.")
            # Not necessarily an error, could be an empty file.
            # Return an empty list, or a specific message if preferred.
            return []
        return parsed
    except RuntimeError as e:
        logger.error(f"Parser dependency missing for {file_path}: {e}")
        app_ref.notify(f"Cannot preview {file_path.name}: Required library missing ({e}).", severity="error", timeout=7)
        return [{"name": f"Error processing {file_path.name}", "details": str(e)}]
    except ValueError as e:
        logger.error(f"Failed to parse {file_path} for preview: {e}")
        app_ref.notify(f"Error parsing {file_path.name}: Invalid format.", severity="warning", timeout=7)
        return [{"name": f"Error parsing {file_path.name}", "details": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error reading {file_path.name}.", severity="error", timeout=7)
        return [{"name": f"Error reading {file_path.name}", "details": str(e)}]


async def _handle_prompt_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """
    Callback function executed after the FileOpen dialog for prompt selection returns.
    """
    if selected_path:
        logger.info(f"Prompt file selected via dialog: {selected_path}")
        if selected_path in app.selected_prompt_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the selection.", severity="warning")
            return

        app.selected_prompt_files_for_import.append(selected_path)
        app.last_prompt_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)

            placeholder_exists = False
            if list_view.children:  # Check if there are any children
                first_child = list_view.children[0]
                # Ensure the first child is a ListItem and it has children (the Label)
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label_of_first_item = first_child.children[0]
                    if isinstance(first_label_of_first_item, Label):
                        # Convert Label's renderable (Rich Text) to plain string for comparison
                        if str(first_label_of_first_item.renderable).strip() == "No files selected.":
                            placeholder_exists = True

            if placeholder_exists:
                await list_view.clear()
                logger.debug("Cleared 'No files selected.' placeholder from prompt list.")

            await list_view.append(ListItem(Label(str(selected_path))))
            logger.debug(f"Appended '{selected_path}' to prompt list view.")

        except QueryError:
            logger.error("Could not find #ingest-prompts-selected-files-list ListView to update.")
        except Exception as e_lv:
            logger.error(f"Error updating prompt list view: {e_lv}", exc_info=True)

        # Parse this file and add to overall preview list
        parsed_prompts_from_file = _parse_single_prompt_file_for_preview(selected_path, app)
        app.parsed_prompts_for_preview.extend(parsed_prompts_from_file)

        await _update_prompt_preview_display(app)  # Update the preview display
    else:
        logger.info("Prompt file selection cancelled by user.")
        app.notify("File selection cancelled.")


async def handle_ingest_prompts_clear_files_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Clear Selection' button press for prompt import."""
    logger.info("Clearing selected prompt files and preview.")
    app.selected_prompt_files_for_import.clear()
    app.parsed_prompts_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))

        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.clear()
        app.notify("Selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing prompt selection: {e}")
        app.notify("Error clearing UI.", severity="error")


async def handle_ingest_prompts_import_now_button_pressed(app: 'TldwCli') -> None:
    """Handles the 'Import Selected Files Now' button press."""
    logger.info("Import Selected Prompt Files Now button pressed.")

    if not app.selected_prompt_files_for_import:
        app.notify("No prompt files selected to import.", severity="warning")
        return

    if not prompts_db_initialized():
        msg = "Prompts database is not initialized. Cannot import."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting import.")
        return

    try:
        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.text = ""
        status_area.text = "Starting import process...\n"
    except QueryError:
        logger.error("Could not find #prompt-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    app.notify("Importing prompts... This may take a moment.")

    # The worker function itself remains the same
    async def import_worker_target():  # Renamed to avoid confusion with Worker class
        logger.info("--- import_worker_target (Prompts) RUNNING ---")
        try:
            results = import_prompts_from_files(app.selected_prompt_files_for_import)
            logger.info(f"--- import_worker_target (Prompts) FINISHED, results count: {len(results)} ---")
            return results  # Return the results
        except Exception as e_worker:
            logger.error(f"Exception inside import_worker_target (Prompts): {e_worker}", exc_info=True)
            # To signal an error to the worker system, you should re-raise the exception
            # or return a specific error indicator if you want to handle it differently
            # in on_worker_state_changed. For now, re-raising is simpler.
            raise e_worker

    # Define the functions that will handle success and failure,
    # these will be called by your app's on_worker_state_changed handler.
    # We pass the worker_name to identify which worker completed.

    def process_prompt_import_success(results: List[Dict[str, Any]], worker_name: str):
        if worker_name != "prompt_import_worker":  # Ensure this is for the correct worker
            return

        logger.info(f"--- process_prompt_import_success CALLED for worker: {worker_name} ---")
        logger.debug(f"Import results received: {results}")

        log_text_parts = ["Import process finished.\n\nResults:\n"]
        successful_imports = 0
        failed_imports = 0

        if not results:
            log_text_parts.append("No results returned from import worker.\n")
            logger.warning("process_prompt_import_success: Received empty results list.")
        else:
            for res_idx, res in enumerate(results):
                logger.debug(f"Processing result item {res_idx}: {res}")
                status = res.get("status", "unknown")
                file_path_str = res.get("file_path", "N/A")
                prompt_name = res.get("prompt_name", "N/A")
                message = res.get("message", "")

                log_text_parts.append(f"File: {Path(file_path_str).name}\n")
                if prompt_name and prompt_name != "N/A":
                    log_text_parts.append(f"  Prompt: '{prompt_name}'\n")
                log_text_parts.append(f"  Status: {status.upper()}\n")
                if message:
                    log_text_parts.append(f"  Message: {message}\n")
                log_text_parts.append("-" * 30 + "\n")

                if status == "success":
                    successful_imports += 1
                else:
                    failed_imports += 1

        summary = f"\nSummary: {successful_imports} prompts imported successfully, {failed_imports} failed."
        log_text_parts.append(summary)
        final_log_text_to_display = "".join(log_text_parts)
        logger.debug(f"Final text for status_area:\n{final_log_text_to_display}")

        try:
            status_area_cb = app.query_one("#prompt-import-status-area", TextArea)
            logger.info("Successfully queried #prompt-import-status-area in process_prompt_import_success.")
            status_area_cb.load_text(final_log_text_to_display)
            logger.info("Called load_text on #prompt-import-status-area.")
            status_area_cb.refresh(layout=True)
            logger.info("Called refresh() on status_area_cb.")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_success.")
        except Exception as e_load_text:
            logger.error(f"Error during status_area_cb.load_text in process_prompt_import_success: {e_load_text}",
                         exc_info=True)

        app.notify(f"Prompt import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(f"Prompt import summary: {summary.strip()}")

        app.call_later(ccp_handlers.populate_ccp_prompts_list_view, app)
        app.call_later(chat_handlers.handle_chat_sidebar_prompt_search_changed, app, "")
        logger.info("--- process_prompt_import_success FINISHED ---")

    def process_prompt_import_failure(error: Exception, worker_name: str):
        if worker_name != "prompt_import_worker":
            return

        logger.error(f"--- process_prompt_import_failure CALLED for worker {worker_name}: {error} ---", exc_info=True)
        try:
            status_area_cb_fail = app.query_one("#prompt-import-status-area", TextArea)
            current_text = status_area_cb_fail.text
            status_area_cb_fail.load_text(
                current_text + f"\nImport process failed critically: {error}\nCheck logs for details.\n")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_failure.")
        app.notify(f"Prompt import failed: {str(error)[:100]}", severity="error", timeout=10)

    # Store these handlers on the app instance temporarily or pass them via a different mechanism
    # For simplicity here, we'll assume app.py's on_worker_state_changed will call them.
    # A more robust way is to make these methods of a class or use a dispatch dictionary in app.py.
    app.prompt_import_success_handler = process_prompt_import_success
    app.prompt_import_failure_handler = process_prompt_import_failure

    # Run the worker
    app.run_worker(
        import_worker_target,  # The async callable
        name="prompt_import_worker",  # Crucial for identifying the worker later
        group="file_operations",
        description="Importing selected prompt files."
        # No on_success or on_failure here
    )


# --- TLDW API Form Handlers ---
async def handle_tldw_api_auth_method_changed(app: 'TldwCli', event_value: str) -> None:
    # ... (implementation as you provided)
    logger.debug(f"TLDW API Auth method changed to: {event_value}")
    try:
        custom_token_input = app.query_one("#tldw-api-custom-token", Input)
        custom_token_label = app.query_one("#tldw-api-custom-token-label", Label)
        if event_value == "custom_token":
            custom_token_input.display = True
            custom_token_label.display = True
            custom_token_input.focus()
        else:
            custom_token_input.display = False
            custom_token_label.display = False
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth method change: {e}")

# --- TLDW API Form Specific Option Containers (IDs) ---
TLDW_API_VIDEO_OPTIONS_ID = "tldw-api-video-options"
TLDW_API_AUDIO_OPTIONS_ID = "tldw-api-audio-options"
TLDW_API_PDF_OPTIONS_ID = "tldw-api-pdf-options"
TLDW_API_EBOOK_OPTIONS_ID = "tldw-api-ebook-options"
TLDW_API_DOCUMENT_OPTIONS_ID = "tldw-api-document-options"
TLDW_API_XML_OPTIONS_ID = "tldw-api-xml-options"
TLDW_API_MEDIAWIKI_OPTIONS_ID = "tldw-api-mediawiki-options"

ALL_TLDW_API_OPTION_CONTAINERS = [
    TLDW_API_VIDEO_OPTIONS_ID, TLDW_API_AUDIO_OPTIONS_ID, TLDW_API_PDF_OPTIONS_ID,
    TLDW_API_EBOOK_OPTIONS_ID, TLDW_API_DOCUMENT_OPTIONS_ID, TLDW_API_XML_OPTIONS_ID,
    TLDW_API_MEDIAWIKI_OPTIONS_ID
]

async def handle_tldw_api_media_type_changed(app: 'TldwCli', event_value: str) -> None:
    """Shows/hides media type specific option containers."""
    logger.debug(f"TLDW API Media Type changed to: {event_value}")
    try:
        # Hide all specific option containers first
        for container_id in ALL_TLDW_API_OPTION_CONTAINERS:
            try:
                app.query_one(f"#{container_id}", Container).display = False
            except QueryError:
                pass # Container might not exist if not composed yet, or for all types

        # Show the relevant one
        target_container_id = f"tldw-api-{event_value.lower().replace('_', '-')}-options"
        if event_value: # If a type is selected
            try:
                container_to_show = app.query_one(f"#{target_container_id}", Container)
                container_to_show.display = True
                logger.info(f"Displaying options container: {target_container_id}")
            except QueryError:
                logger.warning(f"Options container #{target_container_id} not found for media type {event_value}.")

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API media type change: {e}")
    except Exception as ex:
        logger.error(f"Unexpected error handling media type change: {ex}", exc_info=True)


def _collect_common_form_data(app: 'TldwCli') -> Dict[str, Any]:
    """Collects common data fields from the TLDW API form."""
    data = {}
    current_field_id_for_error = "Unknown Field" # Keep track of which field was being processed
    try:
        current_field_id_for_error = "#tldw-api-urls"
        data["urls"] = [url.strip() for url in app.query_one("#tldw-api-urls", TextArea).text.splitlines() if url.strip()]

        current_field_id_for_error = "#tldw-api-local-files"
        data["local_files"] = [fp.strip() for fp in app.query_one("#tldw-api-local-files", TextArea).text.splitlines() if fp.strip()]

        current_field_id_for_error = "#tldw-api-title"
        data["title"] = app.query_one("#tldw-api-title", Input).value or None

        current_field_id_for_error = "#tldw-api-author"
        data["author"] = app.query_one("#tldw-api-author", Input).value or None

        current_field_id_for_error = "#tldw-api-keywords"
        data["keywords_str"] = app.query_one("#tldw-api-keywords", TextArea).text

        current_field_id_for_error = "#tldw-api-custom-prompt"
        data["custom_prompt"] = app.query_one("#tldw-api-custom-prompt", TextArea).text or None

        current_field_id_for_error = "#tldw-api-system-prompt"
        data["system_prompt"] = app.query_one("#tldw-api-system-prompt", TextArea).text or None

        current_field_id_for_error = "#tldw-api-perform-analysis"
        data["perform_analysis"] = app.query_one("#tldw-api-perform-analysis", Checkbox).value

        current_field_id_for_error = "#tldw-api-overwrite-db"
        data["overwrite_existing_db"] = app.query_one("#tldw-api-overwrite-db", Checkbox).value

        current_field_id_for_error = "#tldw-api-perform-chunking"
        data["perform_chunking"] = app.query_one("#tldw-api-perform-chunking", Checkbox).value

        current_field_id_for_error = "#tldw-api-chunk-method"
        chunk_method_select = app.query_one("#tldw-api-chunk-method", Select)
        data["chunk_method"] = chunk_method_select.value if chunk_method_select.value != Select.BLANK else None

        current_field_id_for_error = "#tldw-api-chunk-size"
        data["chunk_size"] = int(app.query_one("#tldw-api-chunk-size", Input).value or "500")

        current_field_id_for_error = "#tldw-api-chunk-overlap"
        data["chunk_overlap"] = int(app.query_one("#tldw-api-chunk-overlap", Input).value or "200")

        current_field_id_for_error = "#tldw-api-chunk-lang"
        data["chunk_language"] = app.query_one("#tldw-api-chunk-lang", Input).value or None

        current_field_id_for_error = "#tldw-api-adaptive-chunking"
        data["use_adaptive_chunking"] = app.query_one("#tldw-api-adaptive-chunking", Checkbox).value

        current_field_id_for_error = "#tldw-api-multi-level-chunking"
        data["use_multi_level_chunking"] = app.query_one("#tldw-api-multi-level-chunking", Checkbox).value

        current_field_id_for_error = "#tldw-api-custom-chapter-pattern"
        data["custom_chapter_pattern"] = app.query_one("#tldw-api-custom-chapter-pattern", Input).value or None

        current_field_id_for_error = "#tldw-api-analysis-api-name"
        analysis_api_select = app.query_one("#tldw-api-analysis-api-name", Select)
        data["api_name"] = analysis_api_select.value if analysis_api_select.value != Select.BLANK else None

        current_field_id_for_error = "#tldw-api-summarize-recursively"
        data["summarize_recursively"] = app.query_one("#tldw-api-summarize-recursively", Checkbox).value

        current_field_id_for_error = "#tldw-api-perform-rolling-summarization"
        data["perform_rolling_summarization"] = app.query_one("#tldw-api-perform-rolling-summarization", Checkbox).value

    except QueryError as e:
        # Log the specific query that failed if possible, or the last attempted field ID
        logger.error(f"Error querying TLDW API form field (around {current_field_id_for_error}): {e}")
        # The QueryError 'e' itself will contain the selector string that failed.
        app.notify(f"Error: Missing form field. Details: {e}", severity="error")
        raise # Re-raise to stop further processing
    except ValueError as e: # For int() conversion errors
        logger.error(f"Error converting TLDW API form field value (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Invalid value in form field (around {current_field_id_for_error}). Check numbers.", severity="error")
        raise # Re-raise
    return data


def _collect_video_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessVideoRequest:
    current_field_id_for_error = "Unknown Video Field"
    try:
        current_field_id_for_error = "#tldw-api-video-transcription-model"
        common_data["transcription_model"] = app.query_one("#tldw-api-video-transcription-model",
                                                           Input).value or "deepdml/faster-whisper-large-v3-turbo-ct2"

        current_field_id_for_error = "#tldw-api-video-transcription-language"
        common_data["transcription_language"] = app.query_one("#tldw-api-video-transcription-language",
                                                              Input).value or "en"

        current_field_id_for_error = "#tldw-api-video-diarize"
        common_data["diarize"] = app.query_one("#tldw-api-video-diarize", Checkbox).value

        current_field_id_for_error = "#tldw-api-video-timestamp"
        common_data["timestamp_option"] = app.query_one("#tldw-api-video-timestamp", Checkbox).value

        current_field_id_for_error = "#tldw-api-video-vad"
        common_data["vad_use"] = app.query_one("#tldw-api-video-vad", Checkbox).value

        current_field_id_for_error = "#tldw-api-video-confab-check"
        common_data["perform_confabulation_check_of_analysis"] = app.query_one("#tldw-api-video-confab-check",
                                                                               Checkbox).value

        current_field_id_for_error = "#tldw-api-video-start-time"
        common_data["start_time"] = app.query_one("#tldw-api-video-start-time", Input).value or None

        current_field_id_for_error = "#tldw-api-video-end-time"
        common_data["end_time"] = app.query_one("#tldw-api-video-end-time", Input).value or None

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]

        return ProcessVideoRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying video-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing video form field. Details: {e}", severity="error")
        raise
    except ValueError as e:
        logger.error(
            f"Error converting video-specific TLDW API form field value (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Invalid value in video form field (around {current_field_id_for_error}).", severity="error")
        raise

def _collect_audio_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessAudioRequest:
    current_field_id_for_error = "Unknown Audio Field"
    try:
        current_field_id_for_error = "#tldw-api-audio-transcription-model"
        common_data["transcription_model"] = app.query_one("#tldw-api-audio-transcription-model", Input).value or "deepdml/faster-distil-whisper-large-v3.5"
        # other audio specific fields...
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessAudioRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying audio-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing audio form field. Details: {e}", severity="error")
        raise


def _collect_pdf_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessPDFRequest:
    current_field_id_for_error = "Unknown PDF Field"
    try:
        current_field_id_for_error = "#tldw-api-pdf-engine"
        pdf_engine_select = app.query_one("#tldw-api-pdf-engine", Select)
        common_data["pdf_parsing_engine"] = pdf_engine_select.value if pdf_engine_select.value != Select.BLANK else "pymupdf4llm"

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessPDFRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying PDF-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing PDF form field. Details: {e}", severity="error")
        raise

def _collect_ebook_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessEbookRequest:
    current_field_id_for_error = "Unknown Ebook Field"
    try:
        current_field_id_for_error = "#tldw-api-ebook-extraction-method"
        extraction_method_select = app.query_one("#tldw-api-ebook-extraction-method", Select)
        common_data["extraction_method"] = extraction_method_select.value if extraction_method_select.value != Select.BLANK else "filtered"

        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessEbookRequest(**common_data)
    except QueryError as e:
        logger.error(f"Error querying Ebook-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing Ebook form field. Details: {e}", severity="error")
        raise

def _collect_document_specific_data(app: 'TldwCli', common_data: Dict[str, Any]) -> ProcessDocumentRequest:
    # No document-specific fields in UI yet, so it's just converting common_data
    try:
        common_data["keywords"] = [k.strip() for k in common_data.pop("keywords_str", "").split(',') if k.strip()]
        return ProcessDocumentRequest(**common_data)
    except Exception as e: # Catch potential Pydantic validation errors if common_data is bad
        logger.error(f"Error creating ProcessDocumentRequest: {e}")
        app.notify("Error: Could not prepare document request data.", severity="error")
        raise

def _collect_xml_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any]) -> ProcessXMLRequest:
    # XML request model is different, collects specific fields ignoring most common_data
    data = {}
    current_field_id_for_error = "Unknown XML Field"
    try:
        data["title"] = common_api_data.get("title")
        data["author"] = common_api_data.get("author")
        data["keywords"] = [k.strip() for k in common_api_data.get("keywords_str", "").split(',') if k.strip()]
        data["system_prompt"] = common_api_data.get("system_prompt")
        data["custom_prompt"] = common_api_data.get("custom_prompt")
        data["api_name"] = common_api_data.get("api_name")
        data["api_key"] = common_api_data.get("api_key") # From auth_token

        current_field_id_for_error = "#tldw-api-xml-auto-summarize"
        data["auto_summarize"] = app.query_one("#tldw-api-xml-auto-summarize", Checkbox).value
        return ProcessXMLRequest(**data)
    except QueryError as e:
        logger.error(f"Error querying XML-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing XML form field. Details: {e}", severity="error")
        raise

def _collect_mediawiki_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any]) -> ProcessMediaWikiRequest:
    data = {}
    current_field_id_for_error = "Unknown MediaWiki Field"
    try:
        current_field_id_for_error = "#tldw-api-mediawiki-wiki-name"
        data["wiki_name"] = app.query_one("#tldw-api-mediawiki-wiki-name", Input).value or "default_wiki"
        current_field_id_for_error = "#tldw-api-mediawiki-namespaces"
        data["namespaces_str"] = app.query_one("#tldw-api-mediawiki-namespaces", Input).value or None
        current_field_id_for_error = "#tldw-api-mediawiki-skip-redirects"
        data["skip_redirects"] = app.query_one("#tldw-api-mediawiki-skip-redirects", Checkbox).value
        data["chunk_max_size"] = common_api_data.get("chunk_size", 1000) # Use common chunk size
        # api_name_vector_db and api_key_vector_db are not collected from UI for now
        return ProcessMediaWikiRequest(**data)
    except QueryError as e:
        logger.error(f"Error querying MediaWiki-specific TLDW API form field (around {current_field_id_for_error}): {e}")
        app.notify(f"Error: Missing MediaWiki form field. Details: {e}", severity="error")
        raise


async def handle_tldw_api_submit_button_pressed(app: 'TldwCli') -> None:
    logger.info("TLDW API Submit button pressed.")
    app.notify("Processing request via tldw API...")

    # 1. Get Endpoint URL and Auth
    try:
        endpoint_url_input = app.query_one("#tldw-api-endpoint-url", Input)
        auth_method_select = app.query_one("#tldw-api-auth-method", Select)
        media_type_select = app.query_one("#tldw-api-media-type", Select)

        endpoint_url = endpoint_url_input.value.strip()
        auth_method = auth_method_select.value
        selected_media_type = media_type_select.value

        if not endpoint_url:
            app.notify("API Endpoint URL is required.", severity="error")
            endpoint_url_input.focus()
            return
        if auth_method == Select.BLANK:
            app.notify("Please select an Authentication Method.", severity="error")
            auth_method_select.focus()
            return
        if selected_media_type == Select.BLANK:
            app.notify("Please select a Media Type to process.", severity="error")
            media_type_select.focus()
            return

        auth_token: Optional[str] = None
        if auth_method == "custom_token":
            custom_token_input = app.query_one("#tldw-api-custom-token", Input)
            auth_token = custom_token_input.value.strip()
            if not auth_token:
                app.notify("Custom Auth Token is required for selected method.", severity="error")
                custom_token_input.focus()
                return
        elif auth_method == "config_token":
            auth_token = app.app_config.get("tldw_api", {}).get("auth_token_config")
            if not auth_token:
                app.notify("Auth Token not found in tldw_api.auth_token_config. Please configure or use custom.", severity="error")
                return
        # Add more auth methods like ENV VAR here if needed

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API submission: {e}")
        app.notify(f"Error: Missing required UI field: {e.widget.id if e.widget else 'Unknown'}", severity="error")
        return

    # 2. Collect Form Data and Create Request Model
    request_model: Optional[Any] = None
    local_file_paths: Optional[List[str]] = None
    try:
        common_data = _collect_common_form_data(app)
        local_file_paths = common_data.pop("local_files", []) # Extract local files
        common_data["api_key"] = auth_token # Pass the resolved token as api_key for the request model

        if selected_media_type == "video":
            request_model = _collect_video_specific_data(app, common_data)
        elif selected_media_type == "audio":
            request_model = _collect_audio_specific_data(app, common_data)
        elif selected_media_type == "pdf":
            request_model = _collect_pdf_specific_data(app, common_data)
        elif selected_media_type == "ebook":
            request_model = _collect_ebook_specific_data(app, common_data)
        elif selected_media_type == "document":
            request_model = _collect_document_specific_data(app, common_data)
        elif selected_media_type == "xml":
            # XML has a different request structure, pass common_data for it to pick relevant fields
            request_model = _collect_xml_specific_data(app, common_data)
        elif selected_media_type == "mediawiki_dump":
            # MediaWiki also has a different request structure
            request_model = _collect_mediawiki_specific_data(app, common_data)
        # Add elif for ProcessPDFRequest, ProcessEbookRequest, etc.
        # Example for PDF:
        # elif selected_media_type == "pdf":
        #     specific_pdf_data = {} # Collect PDF specific fields
        #     request_model = ProcessPDFRequest(**common_data, **specific_pdf_data)
        else:
            app.notify(f"Media type '{selected_media_type}' not yet supported by this client form.", severity="warning")
            return

    except QueryError: # Already handled by app.notify in collectors
        return
    except ValueError: # Already handled
        return
    except Exception as e:
        logger.error(f"Error preparing request model for TLDW API: {e}", exc_info=True)
        app.notify("Error: Could not prepare data for API request.", severity="error")
        return

    if not request_model:
        app.notify("Failed to create request model.", severity="error")
        return

    # Ensure URLs and local_file_paths are not both empty if they are the primary inputs
    if not request_model.urls and not local_file_paths:
        app.notify("Please provide at least one URL or one local file path.", severity="warning")
        try:
            app.query_one("#tldw-api-urls", TextArea).focus()
        except QueryError: pass
        return


    # 3. Initialize API Client and Run Worker
    api_client = TLDWAPIClient(base_url=endpoint_url, token=auth_token) # Token for client, api_key in model for server
    overwrite_db = common_data.get("overwrite_existing_db", False) # Get the DB overwrite flag

    async def process_media_worker():
        nonlocal request_model # Allow modification for XML/MediaWiki
        try:
            if selected_media_type == "video":
                return await api_client.process_video(request_model, local_file_paths)
            elif selected_media_type == "audio":
                return await api_client.process_audio(request_model, local_file_paths)
            elif selected_media_type == "pdf":
                return await api_client.process_pdf(request_model, local_file_paths)
            elif selected_media_type == "ebook":
                return await api_client.process_ebook(request_model, local_file_paths)
            elif selected_media_type == "document":
                return await api_client.process_document(request_model, local_file_paths)
            elif selected_media_type == "xml":
                if not local_file_paths: raise ValueError("XML processing requires a local file path.")
                return await api_client.process_xml(request_model, local_file_paths[0])  # XML takes single path
            elif selected_media_type == "mediawiki_dump":
                if not local_file_paths: raise ValueError("MediaWiki processing requires a local file path.")
                # For streaming, the worker should yield, not return directly.
                # This example shows how to initiate and collect, actual handling of stream in on_success would differ.
                results = []
                async for item in api_client.process_mediawiki_dump(request_model, local_file_paths[
                    0]):  # request_model is ProcessMediaWikiRequest
                    results.append(item)  # Collect all streamed items
                return results  # Return collected list for on_success
            else:
                raise NotImplementedError(f"Client-side processing for {selected_media_type} not implemented.")
        finally:
            await api_client.close()

    def on_worker_success(response_data: Any): # Type hint can be Union of BatchMediaProcessResponse, etc.
        app.notify("TLDW API request successful. Ingesting results...", timeout=3)
        logger.info(f"TLDW API Response: {response_data}")

        if not app.media_db:
            logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
            app.notify("Error: Local media database not available.", severity="error")
            return

        processed_count = 0
        error_count = 0

        # Handle different response types
        results_to_ingest: List[MediaItemProcessResult] = []
        if isinstance(response_data, BatchMediaProcessResponse):  # Pydantic model
            results_to_ingest = response_data.results
        elif isinstance(response_data, dict) and "results" in response_data:  # Raw dict from XML perhaps
            # This handles BatchProcessXMLResponse implicitly if results are compatible
            if "processed_count" in response_data:  # Looks like BatchMediaProcessResponse or BatchProcessXMLResponse
                raw_results = response_data.get("results", [])
                for item_dict in raw_results:
                    # Try to coerce into MediaItemProcessResult, might need specific mapping for XML
                    # For now, assume XML result items can be mostly mapped.
                    results_to_ingest.append(MediaItemProcessResult(**item_dict))

        elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
            # MediaWiki dump (if collected into a list by worker)
            for mw_page in response_data:
                if mw_page.status == "Error":
                    error_count +=1
                    logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                    continue
                # Adapt ProcessedMediaWikiPage to MediaItemProcessResult structure for ingestion
                results_to_ingest.append(MediaItemProcessResult(
                    status="Success", # Assume success if no error status
                    input_ref=mw_page.input_ref or mw_page.title,
                    processing_source=mw_page.title, # or another identifier
                    media_type="mediawiki_article", # or "mediawiki_page"
                    metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace, "revision_id": mw_page.revision_id, "timestamp": mw_page.timestamp},
                    content=mw_page.content,
                    chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if mw_page.chunks else None, # Simplified chunk adaptation
                    # analysis, summary, etc. might not be directly available from MediaWiki processing
                ))
        else:
            logger.error(f"Unexpected TLDW API response data type: {type(response_data)}. Cannot ingest.")
            app.notify("Error: Received unexpected data format from API.", severity="error")
            return
        # Add elif for XML if it returns a single ProcessXMLResponseItem or similar

        for item_result in results_to_ingest:
            if item_result.status == "Success":
                try:
                    # Prepare data for add_media_with_keywords
                    # Keywords: API response might not have 'keywords'. Use originally submitted ones if available.
                    # For simplicity, let's assume API response's metadata *might* have keywords.
                    api_keywords = item_result.metadata.get("keywords", []) if item_result.metadata else []
                    if isinstance(api_keywords, str): # If server returns comma-sep string
                        api_keywords = [k.strip() for k in api_keywords.split(',') if k.strip()]

                    # Chunks for UnvectorizedMediaChunks
                    # Ensure chunks are in the format: [{'text': str, 'start_char': int, ...}, ...]
                    unvectorized_chunks_to_save = None
                    if item_result.chunks:
                        unvectorized_chunks_to_save = []
                        for idx, chunk_item in enumerate(item_result.chunks):
                            if isinstance(chunk_item, dict) and "text" in chunk_item:
                                unvectorized_chunks_to_save.append({
                                    "text": chunk_item.get("text"),
                                    "start_char": chunk_item.get("metadata", {}).get("start_char"), # Assuming metadata structure
                                    "end_char": chunk_item.get("metadata", {}).get("end_char"),
                                    "chunk_type": chunk_item.get("metadata", {}).get("type", selected_media_type),
                                    "metadata": chunk_item.get("metadata", {}) # Store full chunk metadata
                                })
                            elif isinstance(chunk_item, str): # If chunks are just strings
                                 unvectorized_chunks_to_save.append({"text": chunk_item})
                            else:
                                logger.warning(f"Skipping malformed chunk item: {chunk_item}")


                    media_id, media_uuid, msg = app.media_db.add_media_with_keywords(
                        url=item_result.input_ref, # Original URL/filename
                        title=item_result.metadata.get("title", item_result.input_ref) if item_result.metadata else item_result.input_ref,
                        media_type=item_result.media_type,
                        content=item_result.content or item_result.transcript, # Use transcript if content is empty
                        keywords=api_keywords or (request_model.keywords if hasattr(request_model, "keywords") else []), # Fallback to request
                        prompt=request_model.custom_prompt if hasattr(request_model, "custom_prompt") else None, # From original request
                        analysis_content=item_result.analysis or item_result.summary,
                        transcription_model=item_result.analysis_details.get("transcription_model") if item_result.analysis_details else (request_model.transcription_model if hasattr(request_model, "transcription_model") else None),
                        author=item_result.metadata.get("author") if item_result.metadata else (request_model.author if hasattr(request_model, "author") else None),
                        # ingestion_date: use current time,
                        overwrite=overwrite_db, # Use the specific DB overwrite flag
                        chunks=unvectorized_chunks_to_save # Pass prepared chunks
                    )
                    if media_id:
                        logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB. Media ID: {media_id}. Message: {msg}")
                        processed_count += 1
                    else:
                        logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB. Message: {msg}")
                        error_count += 1
                except Exception as e_ingest:
                    logger.error(f"Error ingesting item '{item_result.input_ref}' into local DB: {e_ingest}", exc_info=True)
                    error_count += 1
            else:
                logger.error(f"API processing error for '{item_result.input_ref}': {item_result.error}")
                error_count += 1

        final_msg = f"Ingestion complete. Processed: {processed_count}, Errors: {error_count}."
        app.notify(final_msg, severity="information" if error_count == 0 else "warning", timeout=5)

    def on_worker_failure(error: Exception):
        logger.error(f"TLDW API request worker failed: {error}", exc_info=True)
        if isinstance(error, APIResponseError):
            app.notify(f"API Error {error.status_code}: {str(error)[:200]}", severity="error", timeout=8)
        elif isinstance(error, (APIConnectionError, APIRequestError, AuthenticationError)):
            app.notify(f"API Client Error: {str(error)[:200]}", severity="error", timeout=8)
        else:
            app.notify(f"TLDW API processing failed: {str(error)[:200]}", severity="error", timeout=8)

    app.run_worker(
        process_media_worker,
        name="tldw_api_media_processing",
        group="api_calls",
        description="Processing media via TLDW API"
    )


async def _update_note_preview_display(app: 'TldwCli') -> None:
    """Updates the note preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-notes-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_notes_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no notes found.", id="ingest-notes-preview-placeholder"))
            return

        num_to_display = len(app.parsed_notes_for_preview)
        notes_to_show = app.parsed_notes_for_preview[:MAX_NOTE_PREVIEWS]

        for idx, note_data in enumerate(notes_to_show):
            title = note_data.get("title", f"Untitled Note {idx + 1}")
            content_preview = _truncate_text(note_data.get("content"), 200)

            md_content = f"""### {title}
{content_preview}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

"""
            if "error" in note_data:
                md_content = f"""### Error parsing {note_data.get("filename", "file")}
                {note_data["error"]}
                IGNORE_WHEN_COPYING_START
                content_copy
                download
                Use code with caution.
                Text
                IGNORE_WHEN_COPYING_END                
                """
                await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

            if num_to_display > MAX_NOTE_PREVIEWS:
                    await preview_area.mount(
                        Static(f"...and {num_to_display - MAX_NOTE_PREVIEWS} more notes loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for note preview update: {e}")
        app.notify("Error updating note preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating note preview: {e}", exc_info=True)
        app.notify("Unexpected error during note preview update.", severity="error")


def _parse_single_note_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """
    Parses a single note file (JSON) for preview.
    Returns a list of note data dicts.
    """
    logger.debug(f"Parsing note file for preview: {file_path}")
    preview_notes = []
    file_suffix = file_path.suffix.lower()

    if file_suffix == ".json":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            data = json.loads(content)

            if isinstance(data, dict):  # Single note object
                if "title" in data and "content" in data:
                    preview_notes.append(
                        {"filename": file_path.name, "title": data.get("title"), "content": data.get("content")})
                else:
                    preview_notes.append({"filename": file_path.name, "title": f"Error: {file_path.name}",
                                          "error": "JSON object missing 'title' or 'content'."})
            elif isinstance(data, list):  # Array of note objects
                for item in data:
                    if isinstance(item, dict) and "title" in item and "content" in item:
                        preview_notes.append(
                            {"filename": file_path.name, "title": item.get("title"), "content": item.get("content")})
                    else:
                        logger.warning(f"Skipping invalid note item in array from {file_path.name}: {item}")
            else:
                preview_notes.append({"filename": file_path.name, "title": f"Error: {file_path.name}",
                                      "error": "JSON content is not a valid note object or array of note objects."})
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path.name}: {e}")
            preview_notes.append(
                {"filename": file_path.name, "title": f"Error: {file_path.name}", "error": f"Invalid JSON: {e}"})
        except Exception as e:
            logger.error(f"Error parsing note file {file_path} for preview: {e}", exc_info=True)
            preview_notes.append({"filename": file_path.name, "title": f"Error: {file_path.name}", "error": str(e)})
    else:
        preview_notes.append({"filename": file_path.name, "title": f"Error: {file_path.name}",
                              "error": f"Unsupported file type for note preview: {file_suffix}"})

    if not preview_notes:  # If parsing yielded nothing (e.g. empty JSON array)
        preview_notes.append({"filename": file_path.name, "title": file_path.name, "content": "No notes found in file."})

    return preview_notes

async def _handle_note_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """Callback for note file selection."""
    if selected_path:
        logger.info(f"Note file selected via dialog: {selected_path}")
        if selected_path in app.selected_note_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the note selection.", severity="warning")
            return

        app.selected_note_files_for_import.append(selected_path)
        app.last_note_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-notes-selected-files-list", ListView)
            placeholder_exists = False
            if list_view.children:
                first_child = list_view.children[0]
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label = first_child.children[0]
                    if isinstance(first_label, Label) and str(first_label.renderable).strip() == "No files selected.":
                        placeholder_exists = True
            if placeholder_exists:
                await list_view.clear()
            await list_view.append(ListItem(Label(str(selected_path))))
        except QueryError:
            logger.error("Could not find #ingest-notes-selected-files-list ListView to update.")

        parsed_notes_from_file = _parse_single_note_file_for_preview(selected_path, app)
        app.parsed_notes_for_preview.extend(parsed_notes_from_file)

        await _update_note_preview_display(app)
    else:
        logger.info("Note file selection cancelled.")
        app.notify("File selection cancelled.")


# --- Notes Ingest Handlers (NEW) ---
async def handle_ingest_notes_select_file_button_pressed(app: 'TldwCli') -> None:
    logger.debug("Select Notes File(s) button pressed. Opening file dialog.")
    current_dir = app.last_note_import_dir or Path(".")

    def post_file_open_action(selected_file_path: Optional[Path]) -> None:
        """This function matches the expected callback signature for push_screen more directly."""
        # It's okay for this outer function to be synchronous if all it does
        # is schedule an async task via call_after_refresh.
        if selected_file_path is not None: # Or however you want to handle None path
            # The lambda passed to call_after_refresh captures selected_file_path
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, selected_file_path))
        else:
            # Handle the case where selection was cancelled (path is None)
            app.call_after_refresh(lambda: _handle_note_file_selected_callback(app, None))
    # The screen you're pushing
    file_open_screen = FileOpen(
        location=str(current_dir),
        title="Select Notes File (.json)",
        filters=NOTE_FILE_FILTERS
    )
    # Push the screen with the defined callback
    # await app.push_screen(file_open_screen, post_file_open_action) # This should work
    # If you need to call an async method from a sync context (like a button press handler that isn't async itself)
    # and push_screen itself needs to be awaited, then the button handler must be async.
    # Your handle_ingest_notes_select_file_button_pressed is already async, so this is fine:
    await app.push_screen(file_open_screen, post_file_open_action)


async def handle_ingest_notes_clear_files_button_pressed(app: 'TldwCli') -> None:
    """Handles 'Clear Selection' for note import."""
    logger.info("Clearing selected note files and preview.")
    app.selected_note_files_for_import.clear()
    app.parsed_notes_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-notes-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-notes-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-notes-preview-placeholder"))

        status_area = app.query_one("#ingest-notes-import-status-area", TextArea)
        status_area.clear()
        app.notify("Note selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing note selection: {e}")
        app.notify("Error clearing note UI.", severity="error")


async def handle_ingest_notes_import_now_button_pressed(app: 'TldwCli') -> None:
    """Handles 'Import Selected Notes Now' button press."""
    logger.info("Import Selected Note Files Now button pressed.")

    if not app.selected_note_files_for_import:
        app.notify("No note files selected to import.", severity="warning")
        return

    if not app.notes_service:
        msg = "Notes database service is not initialized. Cannot import notes."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting note import.")
        return

    try:
        # Use query_one to get the widget directly or raise QueryError
        status_area = app.query_one("#ingest-notes-import-status-area", TextArea)
    except QueryError:
        logger.error("Could not find #ingest-notes-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    status_area.text = ""  # Clear the TextArea
    status_area.text = "Starting note import process...\n"  # Set initial text
    app.notify("Importing notes...")

    user_id = app.notes_user_id

    async def import_worker_notes():
        results = []
        for file_path in app.selected_note_files_for_import:
            notes_in_file = _parse_single_note_file_for_preview(file_path, app)
            for note_data in notes_in_file:
                if "error" in note_data or not note_data.get("title") or not note_data.get("content"):
                    results.append({
                        "file_path": str(file_path),
                        "note_title": note_data.get("title", file_path.stem),
                        "status": "failure",
                        "message": note_data.get("error", "Missing title or content.")
                    })
                    continue
                try:
                    note_id = app.notes_service.add_note(
                        user_id=user_id,
                        title=note_data["title"],
                        content=note_data["content"]
                    )
                    results.append({
                        "file_path": str(file_path),
                        "note_title": note_data["title"],
                        "status": "success",
                        "message": f"Note imported successfully. ID: {note_id}",
                        "note_id": note_id
                    })
                except (ChaChaConflictError, CharactersRAGDBError, ValueError) as e:
                    logger.error(f"Error importing note '{note_data['title']}' from {file_path}: {e}", exc_info=True)
                    results.append({
                        "file_path": str(file_path),
                        "note_title": note_data["title"],
                        "status": "failure",
                        "message": f"DB/Input error: {type(e).__name__} - {str(e)[:100]}"
                    })
                except Exception as e:
                    logger.error(f"Unexpected error importing note '{note_data['title']}' from {file_path}: {e}",
                                 exc_info=True)
                    results.append({
                        "file_path": str(file_path),
                        "note_title": note_data["title"],
                        "status": "failure",
                        "message": f"Unexpected error: {type(e).__name__}"
                    })
        return results

    def on_import_success_notes(results: List[Dict[str, Any]]):
        log_text_parts = ["Note import process finished.\n\nResults:\n"]  # Renamed to avoid conflict
        successful_imports = 0
        failed_imports = 0
        for res in results:
            status = res.get("status", "unknown")
            file_path_str = res.get("file_path", "N/A")
            note_title = res.get("note_title", "N/A")
            message = res.get("message", "")

            log_text_parts.append(f"File: {Path(file_path_str).name} (Note: '{note_title}')\n")
            log_text_parts.append(f"  Status: {status.upper()}\n")
            if message:
                log_text_parts.append(f"  Message: {message}\n")
            log_text_parts.append("-" * 30 + "\n")

            if status == "success":
                successful_imports += 1
            else:
                failed_imports += 1

        summary = f"\nSummary: {successful_imports} notes imported, {failed_imports} failed."
        log_text_parts.append(summary)

        try:
            status_area_cb = app.query_one("#ingest-notes-import-status-area", TextArea)
            status_area_cb.load_text("".join(log_text_parts))
        except QueryError:
            logger.error("Failed to find #ingest-notes-import-status-area in on_import_success_notes.")

        app.notify(f"Note import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(summary)

        #app.call_later(load_and_display_notes_handler, app)
        app.call_later(app.refresh_notes_tab_after_ingest)
        try:
            # Make sure to query the collapsible before creating the Toggled event instance
            chat_notes_collapsible_widget = app.query_one("#chat-notes-collapsible", Collapsible)
            app.call_later(app.on_chat_notes_collapsible_toggle, Collapsible.Toggled(chat_notes_collapsible_widget))
        except QueryError:
            logger.error("Failed to find #chat-notes-collapsible widget for refresh after note import.")

    def on_import_failure_notes(error: Exception):
        logger.error(f"Note import worker failed critically: {error}", exc_info=True)
        try:
            status_area_cb_fail = app.query_one("#ingest-notes-import-status-area", TextArea)
            current_text = status_area_cb_fail.text
            status_area_cb_fail.load_text(
                current_text + f"\nNote import process failed critically: {error}\nCheck logs.\n")
        except QueryError:
            logger.error("Failed to find #ingest-notes-import-status-area in on_import_failure_notes.")
        app.notify(f"Note import CRITICALLY failed: {error}", severity="error", timeout=10)

    app.run_worker(
        import_worker_notes,
        name="note_import_worker",
        group="file_operations",
        description="Importing selected note files."
    )
#
# End of tldw_chatbook/Event_Handlers/ingest_events.py
#######################################################################################################################