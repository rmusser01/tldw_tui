# themes.py
from textual.theme import Theme
from textual.color import Color


# Helper to convert string dicts to Theme objects
def create_theme_from_dict(name: str, theme_dict: dict) -> Theme:
    theme_args = {"name": name}
    for key, value in theme_dict.items():
        if key == "dark":
            theme_args[key] = bool(value)
        # All other color keys are assumed to be color strings
        elif key in [
            "primary", "secondary", "accent", "warning", "error", "success",
            "background", "surface", "panel", "foreground",
        ]:
            try:
                # Ensure value is a string before parsing, though it should be from the dict
                theme_args[key] = Color.parse(str(value))
            except Exception as e:
                print(f"Warning: Could not parse color '{value}' for key '{key}' in theme '{name}'. Error: {e}")
                # Fallback to a default color or skip if parsing fails
                # For example, Color.parse("red") or continue
        else:  # For any other variables Textual's Theme constructor might support (e.g., 'variables' dict)
            theme_args[key] = value
    return Theme(**theme_args)


RAW_THEMES_DATA = {
    # It's good practice to ensure your custom theme names are unique
    # and don't clash with potential future built-in Textual themes.
    "my_custom_dark": {
        "primary": "#004578",
        "secondary": "#006FB3",
        "accent": "#0099FF",
        "warning": "#FFD700",
        "error": "#FF0000",
        "success": "#008000",
        "background": "#1E1E1E",
        "surface": "#2C2C2C",
        "panel": "#252525",
        "foreground": "#FFFFFF",
        "dark": True,
    },
    "solarized_dark_inspired": {
        "background": "#002b36",
        "surface": "#073642",
        "panel": "#073642",
        "primary": "#268bd2",
        "secondary": "#2aa198",
        "accent": "#b58900",
        "foreground": "#839496",
        "warning": "#cb4b16",
        "error": "#dc322f",
        "success": "#859900",
        "dark": True,
    },
    "paper_light_inspired": {
        "background": "#EEEEEE",
        "surface": "#FFFFFF",
        "panel": "#FFFFFF",
        "primary": "#AF8700",
        "secondary": "#D75F00",
        "accent": "#005F87",
        "foreground": "#444444",
        "warning": "#D75F00",
        "error": "#D70000",
        "success": "#008700",
        "dark": False,
    },
    # Add more themes here
}

MY_THEMES = {
    name: create_theme_from_dict(name, data)
    for name, data in RAW_THEMES_DATA.items()
}


# 1. Theme: "Classic Terminal" (Retro Green)
classic_terminal_green_theme = Theme(
    name="classic_terminal_green",
    primary="#33FF33",       # Bright Green (main interactive elements)
    secondary="#228B22",     # Forest Green (secondary elements, e.g., button backgrounds)
    accent="#55FF55",        # Lighter Green (borders, focus highlights)
    foreground="#33FF33",    # Bright Green (main text)
    background="#0A0A0A",    # Very dark gray (main background)
    surface="#1A1A1A",       # Darker gray (input backgrounds, component surfaces)
    panel="#0A0A0A",         # Very dark gray (panel backgrounds, same as main)
    success="#55FF55",       # Brighter Green
    warning="#FFBF00",       # Amber
    error="#FF3333",         # Classic Red
    dark=True,
    variables={
        "footer-key-foreground": "#55FF55",
        "input-selection-background": "#228B22 50%", # Forest Green with 50% alpha
    },
)

# 2. Theme: "Modern Dark" (Inspired by Dracula)
modern_dark_dracula_theme = Theme(
    name="modern_dark_dracula",
    primary="#ff79c6",       # Pink (primary actions)
    secondary="#bd93f9",     # Purple (secondary actions, input borders)
    accent="#8be9fd",        # Cyan (focus, highlights)
    foreground="#f8f8f2",    # Light gray/off-white
    background="#282a36",    # Dark purplish-blue
    surface="#44475a",       # Lighter background shade (buttons)
    panel="#20222b",         # Even darker (sidebars, data tables)
    success="#50fa7b",       # Green
    warning="#f1fa8c",       # Yellow
    error="#ff5555",         # Red (Dracula's red)
    dark=True,
    variables={
        "footer-key-foreground": "#8be9fd",   # Cyan
        "input-selection-background": "#bd93f9 40%", # Purple with 40% alpha
        "text-muted": "#6272a4",             # Lighter purple/gray for comments/dimmed text
    },
)

# 3. Theme: "Paper Light"
paper_light_theme = Theme(
    name="paper_light",
    primary="#5B4636",       # Sepia (primary actions)
    secondary="#AACCFF",     # Soft blue (secondary actions, hover states)
    accent="#AACCFF",        # Soft blue (focus, highlights)
    foreground="#333333",    # Dark gray (main text)
    background="#FEFEFA",    # Off-white/very light beige
    surface="#E0E0E0",       # Light gray (buttons)
    panel="#F8F8F4",         # Slightly off-white (panels, distinct from pure white inputs)
    success="#A8D8B0",       # Muted green
    warning="#EBCB8B",       # Standard warning yellow, fits light theme
    error="#BF616A",         # Standard error red, muted enough
    dark=False,
    variables={
        "footer-key-foreground": "#5B4636",   # Sepia
        "input-selection-background": "#AACCFF 50%", # Soft blue with 50% alpha
        "text-muted": "#777777",             # Lighter gray for subtle text
    },
)

# 4. Theme: "High Contrast Accessibility" (Yellow on Black)
high_contrast_yellow_black_theme = Theme(
    name="high_contrast_yellow_black",
    primary="#FFFFFF",       # White (major interactive elements, borders)
    secondary="#FFFF00",     # Bright Yellow (button text, secondary highlight)
    accent="#FFFFFF",        # White (focus outline, borders)
    foreground="#FFFF00",    # Bright Yellow
    background="#000000",    # Black
    surface="#333333",       # Dark Gray (button background)
    panel="#1A1A1A",         # Very Dark Gray (input background, panels)
    success="#00FF00",       # Bright Green
    warning="#FFFF00",       # Bright Yellow (main theme color for warning)
    error="#FF0000",         # Bright Red
    dark=True,
    variables={
        "footer-key-foreground": "#FFFFFF",   # White
        "input-selection-background": "#FFFFFF 30%", # White with 30% alpha
    },
)

# 5. Theme: "Ocean Depths"
ocean_depths_theme = Theme(
    name="ocean_depths",
    primary="#FF7F50",       # Coral (accent buttons, primary actions)
    secondary="#75E6DA",     # Light teal/aqua (hover, input focus)
    accent="#189AB4",        # Brighter teal (borders)
    foreground="#E0FFFF",    # Light Cyan/Off-White
    background="#0A2342",    # Deep dark blue
    surface="#005073",       # Medium dark blue/teal (buttons)
    panel="#001B33",         # Very dark blue (inputs, panels)
    success="#66CDAA",       # Medium Aquamarine (fits theme)
    warning="#F0E68C",       # Sandy Beige
    error="#FF6347",         # Tomato (reddish coral)
    dark=True,
    variables={
        "footer-key-foreground": "#75E6DA",         # Light teal/aqua
        "input-selection-background": "#FF7F50 40%", # Coral with 40% alpha
        "text-muted": "#ADD8E6",                   # Light blue
        "statusbar-background": "#F0E68C",         # Sandy beige (for specific components like status bar)
        "statusbar-foreground": "#0A2342",         # Deep dark blue (text on status bar)
    },
)

# 6. Theme: "Solarized Dark"
solarized_dark_theme = Theme(
    name="solarized_dark",
    primary="#268bd2",       # blue
    secondary="#6c71c4",     # violet
    accent="#2aa198",        # cyan
    foreground="#839496",    # base0
    background="#002b36",    # base03
    surface="#073642",       # base02 (buttons, inputs)
    panel="#073642",         # base02 (larger panels)
    success="#859900",       # green
    warning="#b58900",       # yellow
    error="#dc322f",         # red
    dark=True,
    variables={
        "footer-key-foreground": "#2aa198",         # cyan
        "input-selection-background": "#268bd2 40%", # blue with 40% alpha
        "text-muted": "#586e75",                   # base01 (dimmed text)
        "text-highlight": "#93a1a1",               # base1 (more important text)
    },
)

# 6. Theme: "Solarized Light"
solarized_light_theme = Theme(
    name="solarized_light",
    primary="#268bd2",       # blue
    secondary="#6c71c4",     # violet
    accent="#2aa198",        # cyan
    foreground="#657b83",    # base00
    background="#fdf6e3",    # base3
    surface="#eee8d5",       # base2 (buttons, inputs)
    panel="#eee8d5",         # base2 (larger panels)
    success="#859900",       # green
    warning="#b58900",       # yellow
    error="#dc322f",         # red
    dark=False,
    variables={
        "footer-key-foreground": "#2aa198",         # cyan
        "input-selection-background": "#268bd2 40%", # blue with 40% alpha
        "text-muted": "#93a1a1",                   # base1 (dimmed text for light theme)
        "text-highlight": "#586e75",               # base01 (more important text for light theme)
    },
)

# 7. Theme: "Monokai Pro"
monokai_pro_theme = Theme(
    name="monokai_pro",
    primary="#A6E22E",       # Green (action buttons)
    secondary="#FF6188",     # Pink (special buttons)
    accent="#AE81FF",        # Purple (default button border, hover)
    foreground="#FCFCFA",    # Off-white
    background="#1E1D20",    # Very dark gray/black
    surface="#4A454D",       # Darker mid-tone (buttons)
    panel="#2D2A2E",         # Slightly lighter dark (inputs, panels)
    success="#A6E22E",       # Green
    warning="#FD971F",       # Orange
    error="#FF6188",         # Pink (often used for errors in Monokai)
    dark=True,
    variables={
        "footer-key-foreground": "#66D9EF",         # Blue
        "input-selection-background": "#AE81FF 40%", # Purple with 40% alpha
        "text-muted": "#75715E",                   # Gray (comments)
        "log-view-background": "#272822",          # Custom for specific widget example
        "log-view-foreground": "#E6DB74",          # Custom for specific widget example
    },
)

# 8. Theme: "Gruvbox Dark"
gruvbox_dark_theme = Theme(
    name="gruvbox_dark",
    primary="#fabd2f",       # Bright Yellow (prominent accent, hover)
    secondary="#b8bb26",     # Bright Green (confirm buttons)
    accent="#83a598",        # Bright Blue (input borders)
    foreground="#fbf1c7",    # fg0 (pale yellow)
    background="#1d2021",    # bg0_h (dark gray)
    surface="#3c3836",       # Darker bg shade (buttons)
    panel="#282828",         # Slightly lighter bg (inputs, panels)
    success="#b8bb26",       # Bright Green
    warning="#fabd2f",       # Bright Yellow
    error="#fb4934",         # Bright Red
    dark=True,
    variables={
        "footer-key-foreground": "#83a598",         # Bright Blue
        "input-selection-background": "#fabd2f 40%", # Bright Yellow with 40% alpha
        "text-muted": "#a89984",                   # Gray
    },
)

# 8. Theme: "Gruvbox Light"
gruvbox_light_theme = Theme(
    name="gruvbox_light",
    primary="#d65d0e",       # Dark Orange (prominent accent)
    secondary="#458588",     # Dark Blue
    accent="#b16286",        # Dark Purple
    foreground="#3c3836",    # Dark gray text
    background="#fbf1c7",    # Pale yellow background
    surface="#ebdbb2",       # Lighter bg shade (buttons)
    panel="#f9f5d7",         # Even lighter or same as background (panels)
    success="#98971a",       # Dark Green
    warning="#d79921",       # Dark Yellow
    error="#cc241d",         # Dark Red
    dark=False,
    variables={
        "footer-key-foreground": "#458588",         # Dark Blue
        "input-selection-background": "#d65d0e 40%", # Dark Orange with 40% alpha
        "text-muted": "#7c6f64",                   # Gray for light background
    },
)

# 9. Theme: "Cyberpunk Neon"
cyberpunk_neon_theme = Theme(
    name="cyberpunk_neon",
    primary="#FF00FF",       # Hot Pink (button color, border)
    secondary="#39FF14",     # Neon Green (secondary actions, input focus)
    accent="#FEFE22",        # Laser Lemon (highlights, contrast border)
    foreground="#00FFFF",    # Electric Blue
    background="#0D0221",    # Deep space blue/purple
    surface="#240046",       # Dark Purple (button background)
    panel="#101010",         # Dark Gray/Black (inputs, panels)
    success="#39FF14",       # Neon Green
    warning="#FEFE22",       # Laser Lemon
    error="#FF1D58",         # Neon Red/Pink (distinct error color)
    dark=True,
    variables={
        "footer-key-foreground": "#FEFE22",         # Laser Lemon
        "input-selection-background": "#FF00FF 40%", # Hot Pink with 40% alpha
        "text-highlight": "#FEFE22",               # Laser Lemon for highlighted text
    },
)

# 10. Theme: "Earthy Nature Tones"
earthy_nature_theme = Theme(
    name="earthy_nature",
    primary="#CD853F",       # Peru (Muted Orange/Brown, action buttons)
    secondary="#808000",     # Olive (button hover)
    accent="#E2725B",        # Terracotta (input focus)
    foreground="#F5F5DC",    # Beige (Off-White)
    background="#2F4F2F",    # Forest Green
    surface="#556B2F",       # Dark Olive Green (button background)
    panel="#4A4034",         # Dark Taupe (inputs, panels)
    success="#8FBC8F",       # Dark Sea Green
    warning="#DAA520",       # Goldenrod
    error="#B22222",         # Firebrick
    dark=True,
    variables={
        "footer-key-foreground": "#E2725B",         # Terracotta
        "input-selection-background": "#CD853F 40%", # Peru with 40% alpha
        "text-muted": "#BDB76B",                   # Dark Khaki
        "tree-control-background": "#3A322A",      # Darker Brown for tree view specific (example)
    },
)

# 11. Theme: "Pastel Dreams"
pastel_dreams_theme = Theme(
    name="pastel_dreams",
    primary="#FFDFD3",       # Baby Pink (default button background)
    secondary="#D4F0E0",     # Mint Green (confirm button background)
    accent="#A8DFFF",        # Brighter Sky Blue (input focus)
    foreground="#A094B7",    # Soft Gray-Purple
    background="#F5F0FF",    # Light Lavender White
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#E6DFF2",         # Slightly darker lavender (sidebars, panels)
    success="#D4F0E0",       # Mint Green
    warning="#FFE6B3",       # Soft Pastel Yellow
    error="#FFB6C1",         # Light Pink (for errors)
    dark=False,
    variables={
        "text-title": "#796A93",             # Darker Soft Purple for titles
        "input-border-default": "#C9EBFB",   # Light Sky Blue for input border
        "button-default-foreground": "#BF8A7E", # Muted Rose for default button text
        "button-confirm-foreground": "#7BAA8F", # Muted Mint for confirm button text
    },
)

# 12. Theme: "Sweet Sorbet"
sweet_sorbet_theme = Theme(
    name="sweet_sorbet",
    primary="#FFFACD",       # Pastel Lemon (default button)
    secondary="#FFB6C1",     # Raspberry Pink (special action button)
    accent="#CFF0C0",        # Slightly darker lime (input focus)
    foreground="#7D7068",    # Warm Gray
    background="#FFF0E5",    # Peachy Cream
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#FFF0E5",         # Peachy Cream (panels, same as background or slightly off)
    success="#E0FFD1",       # Light Lime
    warning="#FFEEAA",       # Soft Lemon Yellow
    error="#FFB6C1",         # Raspberry Pink (can serve as error too)
    dark=False,
    variables={
        "input-border-default": "#E0FFD1",      # Light Lime for input border
        "button-default-foreground": "#B8A26B", # Muted Gold for default button text
        "button-special-foreground": "#996570", # Muted Dark Pink for special button text
        "progressbar-background": "#FFE0CC",    # Light peach track for progress bar
        "progressbar-color": "#FFB6C1",         # Raspberry fill for progress bar
    },
)

# 13. Theme: "Cloudy Day"
cloudy_day_theme = Theme(
    name="cloudy_day",
    primary="#B0C4DE",       # Powder Blue (navigation buttons, input focus)
    secondary="#C5D0E6",     # Light Periwinkle (hover states)
    accent="#B0C4DE",        # Powder Blue (matches primary for focus)
    foreground="#5A6470",    # Cool Dark Gray
    background="#E0E8F0",    # Light Blue-Gray
    surface="#FAFAFA",       # Soft White (default button backgrounds, component surfaces)
    panel="#F0F4F8",         # Slightly lighter blue-gray for panels
    success="#B2D8B2",       # Soft Green
    warning="#F0E6A2",       # Soft Yellow
    error="#DEB0B0",         # Soft Red
    dark=False,
    variables={
        "text-muted": "#778899",               # Light Slate Gray for info text
        "input-background": "#FFFFFF",         # White for input backgrounds
        "input-border-default": "#D0D8E0",     # Default input border
        "button-default-background": "#FAFAFA",# Soft white for default buttons
        "button-default-foreground": "#5A6470",# Cool dark gray for default button text
        "button-navigation-foreground": "#42505E", # Text for navigation buttons
    },
)

# 14. Theme: "Kawaii Candy"
kawaii_candy_theme = Theme(
    name="kawaii_candy",
    primary="#E6E6FA",       # Lavender (default buttons)
    secondary="#FF69B4",     # Hot Pink (super important buttons)
    accent="#FFFFE0",        # Sunny Yellow (input focus)
    foreground="#754C59",    # Dark Magenta-Brown
    background="#FFEFF2",    # Light Pink
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#FFF5F7",         # Very Light Pink (panels)
    success="#7FFFD4",       # Aquamarine (can be success)
    warning="#FFFFE0",       # Sunny Yellow
    error="#FF69B4",         # Hot Pink (can be error)
    dark=False,
    variables={
        "input-border-default": "#7FFFD4",      # Aquamarine for input border
        "button-default-foreground": "#5C5C7D", # Default button text color
        "markdown-h1-color": "#FF69B4",
        "markdown-h2-color": "#754C59",
        "markdown-link-color": "#7FFFD4",
    },
)

# 15. Theme: "Bunny Fluff"
bunny_fluff_theme = Theme(
    name="bunny_fluff",
    primary="#F5F5DC",       # Light Beige (default buttons)
    secondary="#FCEEED",     # Very Light Rose (hover, focus accent)
    accent="#FCEEED",        # Very Light Rose (input focus)
    foreground="#8D8580",    # Soft Warm Gray
    background="#FAF7F5",    # Off-White
    surface="#FFFFFF",       # White (input backgrounds)
    panel="#F0EBE8",         # Slightly darker for footer/panels
    success="#E8F5E8",       # Very Light Green
    warning="#FFFDE8",       # Very Light Yellow
    error="#FDE8E8",         # Very Light Red/Rose
    dark=False,
    variables={
        "input-border-default": "#EAE2DC",      # Very light warm gray for input border
        "button-default-foreground": "#7A736E", # Default button text color
        "footer-background": "#F0EBE8",
        "footer-foreground": "#8D8580",
    },
)

# 16. Theme: "Neon Sunset Drive"
neon_sunset_drive_theme = Theme(
    name="neon_sunset_drive",
    primary="#008080",       # Teal (action buttons)
    secondary="#FF8C00",     # Sunset Orange (secondary interactive elements, borders)
    accent="#FFD700",        # Golden Yellow (focus, highlights)
    foreground="#FF00FF",    # Electric Pink
    background="#2A0A4A",    # Deep Indigo/Purple
    surface="#4B0082",       # Indigo (default button background)
    panel="#1A0433",         # Darker Purple (input backgrounds)
    success="#00CC66",       # Bright Neon Green
    warning="#FFD700",       # Golden Yellow
    error="#FF3333",         # Bright Neon Red
    dark=True,
    variables={
        "text-secondary-info": "#00FFFF",      # Cyan for secondary info text
        "input-border-default": "#00FFFF",     # Cyan for input border
        "button-default-foreground": "#FFD700",# Golden Yellow for default button text
        "button-action-foreground": "#FFFFFF", # White for action button text
    },
)

# 17. Theme: "Palm Mall"
palm_mall_theme = Theme(
    name="palm_mall",
    primary="#FF69B4",       # Neon Pink (prominent interactive, focus)
    secondary="#FFFFAA",     # Light Yellow (alternative accent)
    accent="#FF69B4",        # Neon Pink (input focus, matches primary)
    foreground="#4B0082",    # Dark Purple
    background="#B4E1E7",    # Washed-out Teal/Blue
    surface="#FFDAE9",       # Light Pink (default button background)
    panel="#FFFFFF",         # White (input backgrounds)
    success="#A7F0A7",       # Pastel Green
    warning="#FFFFAA",       # Light Yellow
    error="#FF80A0",         # Slightly deeper pink for errors
    dark=False,
    variables={
        "input-border-default": "#FFFFAA",      # Light Yellow for input border
        "button-default-foreground": "#C71585", # Medium Violet Red for default button text
        "header-background": "#FFB6C1",
        "header-foreground": "#4B0082",
        "statusbar-background": "#4B0082",
        "statusbar-foreground": "#B0E0E6",      # Soft Cyan for status bar text
    },
)

# 18. Theme: "Glitch Grid"
glitch_grid_theme = Theme(
    name="glitch_grid",
    primary="#007FFF",       # Electric Blue (default buttons)
    secondary="#9400D3",     # Glitchy Purple (system critical buttons)
    accent="#FF1493",        # Hot Pink (input focus, error/warning)
    foreground="#00FF00",    # Bright Green
    background="#000000",    # True Black
    surface="#1A1A1A",       # Dark Gray (default button background)
    panel="#0A0A0A",         # Almost Black (input backgrounds)
    success="#00DD00",       # Slightly different Bright Green
    warning="#FF1493",       # Hot Pink
    error="#FF1493",         # Hot Pink
    dark=True,
    variables={
        "input-border-default": "#9400D3",             # Glitchy Purple for input border
        "button-system-critical-foreground": "#00FF00",# Green text for critical buttons
        "button-system-critical-border": "#FF1493",   # Pink border for critical buttons
    },
)

# 19. Theme: "Paradise Virtua"
paradise_virtua_theme = Theme(
    name="paradise_virtua",
    primary="#FF007F",       # Magenta Rose (prominent interactive, borders)
    secondary="#8A8AFF",     # Lavender Blue (select buttons)
    accent="#FFA500",        # Orange Soda (input focus, button text)
    foreground="#AFEEEE",    # Light Aqua
    background="#005060",    # Dark Teal
    surface="#207080",       # Mid Teal (default button background)
    panel="#003040",         # Darker Teal (input backgrounds)
    success="#76D7C4",       # Aqua Green
    warning="#FFA500",       # Orange Soda
    error="#FF4D4D",         # Bright Red/Pink
    dark=True,
    variables={
        "button-default-foreground": "#FFA500", # Orange Soda for default button text
        "button-select-foreground": "#002030",  # Dark text for select buttons
        "titlebar-background": "#FF007F",       # Magenta Rose for title bar
        "titlebar-foreground": "#FFFFFF",       # White text for title bar
    },
)

# 20. Theme: "Lost Artifacts (Atari)"
lost_artifacts_atari_theme = Theme(
    name="lost_artifacts_atari",
    primary="#A08050",       # Muted Gold (prominent interactive, focus)
    secondary="#706080",     # Faded Purple (secondary interactive elements, borders)
    accent="#408080",        # Atari Teal (input border)
    foreground="#D0D0D0",    # Off-White
    background="#262A33",    # Very Dark Blue-Gray
    surface="#363A43",       # Darker Gray (default button background)
    panel="#1E222B",         # Even Darker Blue-Gray (input backgrounds)
    success="#609060",       # Muted Atari Green
    warning="#A08050",       # Muted Gold (can be warning too)
    error="#B07070",         # Dusty Rose
    dark=True,
    variables={
        "text-muted": "#70A0A0",                # Muted Cyan for descriptions
        "button-critical-background": "#B07070",# Dusty Rose for critical buttons
        "button-critical-foreground": "#FFFFFF",# White text for critical buttons
    },
)

# 21. Theme: "Green Phosphor Terminal"
green_phosphor_terminal_theme = Theme(
    name="green_phosphor_terminal",
    primary="#33FF33",       # Bright Green (interactive elements)
    secondary="#00AA00",     # Darker Green (borders, less important)
    accent="#66FF66",        # Lighter Bright Green (highlights, focus)
    foreground="#00FF00",    # Bright Green (main text)
    background="#000000",    # True Black
    surface="#001A00",       # Very Dark Green (button background)
    panel="#050505",         # Almost Black (input background)
    success="#33FF33",       # Bright Green
    warning="#FFFF00",       # Classic Terminal Yellow
    error="#FF3333",         # Classic Terminal Red
    dark=True,
    variables={
        "text-muted": "#00AA00",
        "text-highlight": "#66FF66",
        "log-output-background": "#020202",
        "log-output-foreground": "#00CF00",
    },
)

# 22. Theme: "Zero Day Stealth"
zero_day_stealth_theme = Theme(
    name="zero_day_stealth",
    primary="#3A6A8A",       # Muted Action Blue
    secondary="#4A5A7A",     # Desaturated Blue (hover, focus, borders)
    accent="#4A5A7A",        # Desaturated Blue (input focus, matches secondary)
    foreground="#A0B0C0",    # Light Gray-Blue
    background="#101015",    # Very Dark Charcoal
    surface="#1A1A20",       # Slightly Lighter Dark (default button background)
    panel="#0A0A0D",         # Even Darker (input background)
    success="#5A8A5A",       # Muted Green
    warning="#B0A070",       # Muted Yellow/Amber
    error="#8A4A4A",         # Muted Red
    dark=True,
    variables={
        "text-highlight": "#C0D0E0",
        "command-palette-background": "#15151A",
        "command-palette-border": "#4A5A7A",
    },
)

# 23. Theme: "Red Team Ops"
red_team_ops_theme = Theme(
    name="red_team_ops",
    primary="#FF3333",       # Bright Red
    secondary="#AA0000",     # Darker Red
    accent="#FF3333",        # Bright Red (input focus, matches primary)
    foreground="#CCCCCC",    # Light Gray
    background="#181818",    # Dark Gray, almost Black
    surface="#282828",       # Darker Gray (default button background)
    panel="#101010",         # Very Dark Gray (input background)
    success="#33CC33",       # Contrasting Bright Green
    warning="#FF3333",       # Bright Red
    error="#FF0000",         # Even brighter/pure Red for distinct error
    dark=True,
    variables={
        "input-border-default": "#882222", # Muted Red
        "progressbar-color": "#FF3333",
        "button-critical-action-foreground": "#FFFFFF",
    },
)

# 24. Theme: "Deep Dive Cyberspace"
deep_dive_cyberspace_theme = Theme(
    name="deep_dive_cyberspace",
    primary="#33CCFF",       # Lighter Electric Blue
    secondary="#0077CC",     # Medium Blue (hover, borders)
    accent="#33CCFF",        # Lighter Electric Blue (input focus, matches primary)
    foreground="#00FFFF",    # Bright Cyan
    background="#051025",    # Very Dark Blue
    surface="#002040",       # Dark Blue/Teal (default button background)
    panel="#000515",         # Even Darker Blue (input background)
    success="#33FF99",       # Contrasting Bright Mint Green
    warning="#FFFF33",       # Contrasting Bright Yellow
    error="#FF33AACC",       # Contrasting Bright Pink/Magenta
    dark=True,
    variables={
        "text-info": "#33CCFF",
        "input-border-default": "#0077CC",
        "button-execute-foreground": "#000020", # Dark text on light blue button
        "network-graph-background": "#0A1530",
        "network-graph-border": "#005577",
    },
)

# 25. Theme: "Ghost in the Shell" (Monochrome & Minimal)
ghost_in_the_shell_theme = Theme(
    name="ghost_in_the_shell",
    primary="#708090",       # SlateGray (focus accent, main interactive color)
    secondary="#555555",     # Medium Gray (borders, less active elements)
    accent="#708090",        # SlateGray (input focus, matches primary)
    foreground="#D0D0D0",    # Light Gray
    background="#222222",    # Dark Gray
    surface="#333333",       # Slightly Lighter Dark Gray (default button background)
    panel="#111111",         # Very Dark Gray (input background, status bar)
    success="#709080",       # Very subtle desaturated green
    warning="#A09070",       # Very subtle desaturated yellow/amber
    error="#907070",         # Very subtle desaturated red
    dark=True,
    variables={
        "text-muted": "#888888",
        "input-border-default": "#555555",
        "statusbar-background": "#111111",
        "statusbar-foreground": "#A0A0A0",
        "statusbar-border-top": "#555555", # Assuming this would map to a TCSS property
    },
)

ALL_THEMES = [
    classic_terminal_green_theme,
    modern_dark_dracula_theme,
    paper_light_theme,
    high_contrast_yellow_black_theme,
    ocean_depths_theme,
    solarized_dark_theme,
    solarized_light_theme,
    monokai_pro_theme,
    gruvbox_dark_theme,
    gruvbox_light_theme,
    cyberpunk_neon_theme,
    earthy_nature_theme,
    pastel_dreams_theme,
    sweet_sorbet_theme,
    cloudy_day_theme,
    kawaii_candy_theme,
    bunny_fluff_theme,
    neon_sunset_drive_theme,
    palm_mall_theme,
    glitch_grid_theme,
    paradise_virtua_theme,
    lost_artifacts_atari_theme,
    green_phosphor_terminal_theme,
    zero_day_stealth_theme,
    red_team_ops_theme,
    deep_dive_cyberspace_theme,
    ghost_in_the_shell_theme,
]

# Example of a theme with the 'variables' attribute as shown in Textual docs:
# MY_THEMES["arctic_example"] = Theme(
#     name="arctic_example",
#     primary=Color.parse("#88C0D0"),
#     secondary=Color.parse("#81A1C1"),
#     accent=Color.parse("#B48EAD"),
#     foreground=Color.parse("#D8DEE9"),
#     background=Color.parse("#2E3440"),
#     success=Color.parse("#A3BE8C"),
#     warning=Color.parse("#EBCB8B"),
#     error=Color.parse("#BF616A"),
#     surface=Color.parse("#3B4252"),
#     panel=Color.parse("#434C5E"),
#     dark=True,
#     variables={
#         "block-cursor-text-style": "none", # Needs to be a Style object if not a simple string
#         "footer-key-foreground": Color.parse("#88C0D0"),
#         "input-selection-background": Color.parse("#81a1c1").with_alpha(0.35),
#     },
# )

# Example of how you might collect them all for registration
# ALL_THEMES = [
#     classic_terminal_green_theme,
#     modern_dark_dracula_theme,
#     paper_light_theme,
#     high_contrast_yellow_black_theme,
#     ocean_depths_theme,
#     solarized_dark_theme,
#     solarized_light_theme,
#     monokai_pro_theme,
#     gruvbox_dark_theme,
#     gruvbox_light_theme,
#     cyberpunk_neon_theme,
#     earthy_nature_theme,
# ]