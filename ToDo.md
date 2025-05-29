# ToDo for tldw-cli


### ToDo List
- [ ] Manual for use
- [ ] Add tests
- [ ] Add examples of use
- **General UI/UX**
  - [ ] Setup Emoji support/handling/detection if the user's terminal supports it
  - [ ] Remove the 'speak' buttons
  - 
- **Chat Functionality**
  - [ ] Support for uploading files in chat
- **Character Chat Functionality**
  - [ ] Add support for multiple files
- **Media Endpoint Functionality**
  - [ ] Add support for media endpoint
  - [ ] Add support for searching media endpoint for specific media
  - [ ] Add support for reviewing full details of found items/media items
  - [ ] Add support for ingestion of media files into the tldw `/media/add` endpoint
  - [ ] Add support for viewing versions of media files
  - [ ] Add support for modifying or deleting versions of media files
  - [ ] Add support for processing of media files without remote ingestion.
- **RAG Search Endpoint Functionality**
  - [ ] Add support for RAG search endpoint
  - [ ] Add support for searching against RAG endpoint
- **Stats & Logs Functionality**
  - [ ] Add support for local usage statistics
    - This should capture total tokens used, tokens per endpoint/API, and tokens per character
    - Also things like cost per endpoint/API, and cost per character (maybe? low priority)
  - [ ] Add support for logging of usage
    - This should capture at least the same information as the stats, but more
    - So requests, responses, errors, etc.
    - Also a way to file bug reports if one is encountered. (maybe from main menu?)
- **Local DB Functionality**
  - [ ] Allow for ingestion of media files that have been processed by the tldw API (process-* endpoints)
  - [ ] Allow for editing/modifications/deletion of locally stored media files/character cards/chats


https://github.com/paulrobello/parllama
https://github.com/Toy-97/Chat-WebUI



UX Improvement Suggestions for tldw-cli Application

Date: July 23, 2024 Prepared by: Jules (AI Software Engineering Agent)

1. Introduction * This report outlines proposals for improving the User Interface (UI) and User Experience (UX) of the tldw-cli application. The suggestions are based on an analysis of the application's Python codebase (primarily tldw_app/app.py, widget files, and CSS). The goal is to enhance usability, clarity, and efficiency for you.

2. General UI/UX Principles Applied * Clarity: The UI should be easy to understand, and the purpose of different elements should be clear. * Consistency: Similar UI elements should behave similarly and be styled consistently. * Efficiency: You should be able to accomplish tasks with minimal effort and steps. * Feedback: The application should provide clear feedback for your actions. * Error Prevention & Handling: Minimize the chance of errors and provide helpful messages when errors occur.

3. Specific Improvement Proposals

**3.1. Tab Structure and Purpose**
    *   **Issue**: Significant functional overlap exists between `TAB_CHAT` and `TAB_CPP`, particularly in managing and searching for conversations. This can lead to your confusion regarding where to perform these actions. Additionally, placeholder tabs (`TAB_MEDIA`, `TAB_SEARCH`, `TAB_INGEST`, `TAB_STATS`) add clutter to the interface without providing current functionality.
    *   **Proposals**:
        *   **Primary Recommendation: Merge `TAB_CHAT` and `TAB_CPP` into a single "Conversations" Tab.**
            *   This unified tab would adopt a three-pane layout:
                *   Left Pane: List of all conversations (searchable/filterable), button to start a "New Chat".
                *   Center Pane: Message history of the selected/active conversation, including the chat input area.
                *   Right Pane: Contextual details (editable title, keywords, associated character info), chat-specific settings (e.g., system prompt for this chat), and export options.
            *   Global LLM settings (provider, model, default parameters) should be relocated to a dedicated "Application Settings" tab or a modal dialog.
        *   **Alternative (If Merging is Not Feasible): Clearly Differentiate `TAB_CHAT` and `TAB_CONV_CHAR`.**
            *   `TAB_CHAT`: Focus on active/new chats. Its left sidebar (`settings_sidebar`) should only contain LLM settings. No conversation history browsing.
            *   `TAB_CONV_CHAR`: Dedicated to archived conversations and character management, retaining its three-pane layout for detailed browsing and management.
        *   **Placeholder Tabs**: Remove `TAB_MEDIA`, `TAB_SEARCH`, `TAB_INGEST`, `TAB_STATS` from the default tab bar. List them in a "Help" or "About" section under "Future Features," or make their visibility configurable in `config.toml` (hidden by default).
    *   **Justification**: Merging or clearly differentiating conversation-related tabs will create a more intuitive workflow, reduce redundancy, and simplify the main navigation. Removing placeholder tabs declutters the UI.

**3.2. Sidebar Usage and Content**
    *   **Issue**: The Chat tab's left sidebar (`settings_sidebar` with `id_prefix="chat"`) is overloaded with diverse functionalities. The Notes tab sidebars exhibit redundant buttons and a fragmented save mechanism. A general issue is the excessive default width (`70` units) for all sidebars, which cramps main content areas.
    *   **Proposals**:
        *   **Chat Tab Left Sidebar (`settings_sidebar` for "chat")**:
            *   Relocate global LLM settings (provider, model, parameters) to a dedicated app settings area/modal.
            *   "Current Conversation Details" (title, keywords) should be part of the main chat context (e.g., in the right pane of the proposed merged "Conversations" tab).
            *   Remove the "Saved Conversations" section; this functionality belongs to the left pane of the merged "Conversations" tab.
        *   **Notes Tab Sidebars (`NotesSidebarLeft`, `NotesSidebarRight`)**:
            *   In `NotesSidebarLeft`, consolidate all note actions (Create New, Load, Save, Delete) within the "Notes Actions" `Collapsible`. Remove redundant standalone "New Note" and "Delete Selected Note" buttons.
            *   Unify the "Save Keywords" button (from `NotesSidebarRight`) with the main "Save Current Note" action to persist all note changes (title, content, keywords) simultaneously.
        *   **General Sidebar Width**: In `tldw_cli.tcss`, reduce the default width for `.sidebar` and `#chat-right-sidebar` from `70` to a more appropriate `35` to `45` units.
    *   **Justification**: These changes will improve the information hierarchy within sidebars, reduce visual clutter, make main content areas more spacious and usable, and streamline your actions like saving notes.

**3.3. Layout of Multi-Pane Tabs**
    *   **Issue**: Multi-pane layouts in `TAB_CONV_CHAR` and `TAB_NOTES` can feel cramped and overwhelming, especially with the current wide sidebar definitions.
    *   **Proposals**:
        *   **`TAB_CONV_CHAR` (or merged "Conversations" tab)**:
            *   Maintain the three-pane layout (Navigation | Content | Details) as it's powerful for management tasks.
            *   Crucially, implement the sidebar width reduction (e.g., side panes to 25-30% or fixed 35-45 units).
            *   Use `Collapsible` sections within the right pane if its content grows, to maintain organization.
        *   **`TAB_NOTES`**:
            *   After significantly reducing sidebar widths, further improve by adopting a two-pane layout.
            *   **Recommended**: Remove `NotesSidebarRight`. Integrate its content (note title `Input`, keywords `TextArea`) into `NotesSidebarLeft`, perhaps below the notes list or in a dedicated collapsible section for "Selected Note Details."
            *   This results in a clearer layout: Left Pane (Note List & Details/Actions) and Center Pane (Note Editor).
    *   **Justification**: Optimizing layouts for the terminal's typically constrained horizontal space is key. These changes aim to provide more room for primary content (chat messages, note editing) while keeping necessary controls accessible.

**3.4. UI Consistency**
    *   **Issue**: Minor inconsistencies observed in sidebar toggle iconography, search input behavior, button styling/phrasing, default states of collapsibles, and user feedback mechanisms.
    *   **Proposals**:
        *   **Sidebar Toggles**: Standardize toggle icons (e.g., "☰" for most, "👤" for character sidebar is fine). Aim for consistent placement logic relative to the sidebar they control.
        *   **Search Functionality**: Apply debouncing to search inputs where appropriate (already done for conversations, consider for notes if search becomes intensive). Standardize placeholder text (e.g., "Search [item type]...") and the presentation of "no results" messages.
        *   **Buttons**:
            *   Strive for consistent sizing for common action buttons in sidebars/control areas, unless specific context demands otherwise.
            *   Standardize button labels for similar actions (e.g., prefer "New Note" over "Create New Note" if "New" is used elsewhere for creation).
            *   Continue consistent use of semantic variants (`primary` for confirmation, `success` for positive actions, `error` for destructive actions).
        *   **Collapsible Sections**: Ensure the default collapsed/expanded state is logical and consistent (e.g., less-used options collapsed by default).
        *   **Error Messages & Notifications**: Consistently use `app.notify()` for non-blocking information, warnings, or success confirmations. For errors displayed directly in the UI (like in `ChatMessage`), maintain a standard, clear format (e.g., `[bold red]Error: ...[/]`).
    *   **Justification**: A consistent UI is more intuitive, easier to learn, and feels more polished and predictable for you.

4. Summary of Key Recommendations * Merge TAB_CHAT and TAB_CONV_CHAR: Create a single, unified "Conversations" tab for all chat-related activities. * Overhaul Chat Tab's Left Sidebar: Relocate global LLM settings and remove duplicated conversation management features. * Reduce Sidebar Widths: Change default sidebar width from 70 to ~35-45 units. * Streamline Notes Tab: Convert to a two-pane layout (List & Details | Editor) by consolidating elements from the right sidebar into the left. * Systematically Enhance UI Consistency: Focus on toggles, search, buttons, collapsibles, and user feedback.

5. Next Steps (for implementation) * Discuss these proposals with the project stakeholders and development team. * Prioritize the suggested changes based on their potential impact on usability and the effort required for implementation. * Implement changes iteratively, ideally gathering your feedback at various stages to validate improvements.
