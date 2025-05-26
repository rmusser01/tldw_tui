# Constants.py
# Description: Constants for the application
#
# Imports
#
# 3rd-Party Imports
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Constants ---
TAB_CHAT = "chat"
TAB_CONV_CHAR = "conversations_characters"
TAB_MEDIA = "media"
TAB_METRICS = "metrics"
TAB_NOTES = "notes"
TAB_SEARCH = "search"
TAB_INGEST = "ingest"
TAB_LOGS = "logs"
TAB_STATS = "stats"
ALL_TABS = [TAB_CHAT, TAB_CONV_CHAR, TAB_INGEST, TAB_LOGS, TAB_MEDIA, TAB_METRICS, TAB_NOTES, TAB_SEARCH, TAB_STATS]



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

.placeholder-window { align: center middle; background: $panel; }

/* Sidebar Styling */
/* Generic .sidebar (used by #chat-sidebar and potentially others) */
.sidebar {
    /* width: 70;  <-- REMOVE fixed width */
    width: 25%; /* <-- CHANGE to percentage (adjust 20% to 35% as needed) */
    min-width: 20; /* <-- ADD a minimum width to prevent it becoming unusable */
    max-width: 80; /* <-- ADD a maximum width (optional) */
    background: $boost;
    padding: 1 2;
    border-right: thick $background-darken-1;
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
}
/* Collapsed state for the existing left sidebar */
.sidebar.collapsed {
    width: 0 !important;
    min-width: 0 !important; /* Ensure min-width is also 0 */
    border-right: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none; /* ensures it doesn’t grab focus */
}

/* Right sidebar (character-sidebar) */
#character-sidebar {
    dock: right;
    /* width: 70;   <-- REMOVE fixed width */
    width: 25%;  /* <-- CHANGE to percentage (match .sidebar or use a different one) */
    min-width: 20; /* <-- ADD a minimum width */
    max-width: 80; /* <-- ADD a maximum width (optional) */
    background: $boost;
    padding: 1 2;
    border-left: thick $background-darken-1; /* Border on the left */
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Collapsed state for the new right sidebar */
#character-sidebar.collapsed {
    width: 0 !important;
    min-width: 0 !important; /* Ensure min-width is also 0 */
    border-left: none !important;
    padding: 0 !important;
    overflow: hidden !important;
    display: none; /* Ensures it doesn't take space or grab focus */
}

/* Common sidebar elements */
.sidebar-title { text-style: bold underline; margin-bottom: 1; width: 100%; text-align: center; }
.sidebar-label { margin-top: 1; text-style: bold; }
.sidebar-input { width: 100%; margin-bottom: 1; }
.sidebar-textarea { width: 100%; border: round $surface; margin-bottom: 1; }
.sidebar Select { width: 100%; margin-bottom: 1; }

/* --- Chat Window specific layouts --- */
#chat-main-content {
    layout: vertical;
    height: 100%;
    width: 1fr; /* This is KEY - it takes up the remaining horizontal space */
}
/* VerticalScroll for chat messages */
#chat-log {
    height: 1fr; /* Takes remaining space */
    width: 100%;
    /* border: round $surface; Optional: Add border to scroll area */
    padding: 0 1; /* Padding around messages */
}

/* Input area styling (shared by chat and character) */
#chat-input-area, #conv-char-input-area { /* Updated from #character-input-area */
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
    width: 5;
    height: 3; /* Fixed height for consistency */
    /* align-self: stretch; REMOVED */
    margin-top: 0;
}

/* --- Conversations & Characters Window specific layouts (previously Character Chat) --- */
/* Main container for the three-pane layout */
#conversations_characters-window {
    layout: horizontal; /* Crucial for side-by-side panes */
    /* Ensure it takes full height if not already by .window */
    height: 100%;
}

/* Left Pane Styling */
.cc-left-pane {
    width: 25%; /* Keep 25% or 30% - adjust as needed */
    min-width: 20; /* ADD a minimum width */
    height: 100%;
    background: $boost;
    padding: 1;
    border-right: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Center Pane Styling */
.cc-center-pane {
    width: 1fr; /* Takes remaining space */
    height: 100%;
    padding: 1;
    overflow-y: auto; /* For conversation history */
}

/* Right Pane Styling */
.cc-right-pane {
    width: 25%; /* Keep 25% or 30% - adjust as needed */
    min-width: 20; /* ADD a minimum width */
    height: 100%;
    background: $boost;
    padding: 1;
    border-left: thick $background-darken-1;
    overflow-y: auto;
    overflow-x: hidden;
}

/* General styles for elements within these panes (can reuse/adapt from .sidebar styles) */
.cc-left-pane Input, .cc-right-pane Input {
    width: 100%; margin-bottom: 1;
}
.cc-left-pane ListView {
    height: 1fr; /* Make ListView take available space */
    margin-bottom: 1;
    border: round $surface;
}
.cc-left-pane Button, .cc-right_pane Button { /* Typo Fixed */
    width: 100%;
    margin-bottom: 1;
}

/* Specific title style for panes */
.pane-title {
    text-style: bold;
    margin-bottom: 1;
    text-align: center;
    width: 100%; /* Ensure it spans width for centering */
}

/* Specific style for keywords TextArea in the right pane */
.conv-char-keywords-textarea {
    height: 5; /* Example height */
    width: 100%;
    margin-bottom: 1;
    border: round $surface; /* Re-apply border if not inherited */
}

/* Specific style for the "Export Options" label */
.export-label {
    margin-top: 2; /* Add some space above export options */
}


/* Old styles for #conv-char-main-content, #conv-char-top-area etc. are removed */
/* as the structure within #conversations_characters-window is now different. */
/* Portrait styling - if still needed, would be part of a specific pane's content now */
/* #conv-char-portrait {
    width: 25;
    height: 100%;
    border: round $surface;
    padding: 1;
    margin: 0;
    overflow: hidden;
    align: center top;
}

/* Logs Window adjustments */
#logs-window {
    layout: vertical; /* Override .window's default horizontal layout for this specific window */
    /* The rest of your #logs-window styles (padding, border, height, width) are fine */
    /* E.g., if you had: padding: 0; border: none; height: 100%; width: 100%; those are okay. */
}
#app-log-display {
    border: none;
    height: 1fr;    /* RichLog takes most of the vertical space */
    width: 100%;    /* RichLog takes full width in the vertical layout */
    margin: 0;
    padding: 1;     /* Your existing padding is good */
}

/* Style for the new "Copy All Logs" button */
.logs-action-button {
    width: 100%;     /* Button takes full width */
    height: 3;       /* A standard button height */
    margin-top: 1;   /* Add some space between RichLog and the button */
    /* dock: bottom; /* Optional: If you want it always pinned to the very bottom.
                       If omitted, it will just flow after the RichLog in the vertical layout.
                       For simplicity, let's omit it for now. */
}
/* old #logs-window { padding: 0; border: none; height: 100%; width: 100%; }
#app-log-display { border: none; height: 1fr; width: 1fr; margin: 0; padding: 1; }
*/

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
    width: 1;
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
    width: 2;                /* tiny square */
    height: 3;
    /* margin-right: 1; Removed default margin, apply specific below */
    border: none;
    background: $surface-darken-1;
    color: $text;
}
.sidebar-toggle:hover { background: $surface; }

/* Specific margins for sidebar toggles based on position */
#toggle-chat-sidebar {
    margin-right: 1; /* Original toggle on the left of input area */
}

#toggle-character-sidebar {
    margin-left: 1; /* New toggle on the right of input area */
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

/* Reduce height of Collapsible headers */
Collapsible > .collapsible--header {
    height: 2;
}

.chat-system-prompt-styling {
    width: 100%;
    height: auto;
    min-height: 3;
    max-height: 10; /* Limit height */
    border: round $surface;
    margin-bottom: 1;
}
    """





#
# End of Constants.py
########################################################################################################################
