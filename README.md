# chatbook (tldw TUI) README

# chatbook (a Terminal User Interface)

Current status: Working ✓*
- Little flickery if you select different tabs, have to select logs first and then it will properly show the placeholders.
- Not all APIs work. Ollama is busted as an example.
- Chat requests/edits/deletion of messages is in and working.
- Placeholder buttons are in, and don't do anything (thumbs up/down, speaker)

- UI is very much WIP.
Launch: `python3 run.py` or `python3 -m app` 


Default `config.toml` stored at/read from `~/.config/tldw_cli/config.toml`

Default user DB stored at `~/.share/tldw_cli/` (Planned)


Inspiration
- https://github.com/darrenburns/elia/tree/main/elia_chat
- https://github.com/darrenburns/posting
