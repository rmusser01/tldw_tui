# tldw_chatbook README

A Textual TUI for interacting with various LLM APIs, managing conversation history, characters, notes, and more.

Current status: Working/In-Progress

## Features
- **Current Features:**
  - Connect to multiple LLM providers (Local: Llama.cpp, Ollama, Kobold.cpp, vLLM, Aphrodite, Custom-OpenAI API endpoint ; Commercial: OpenAI, Anthropic, Cohere, Deepseek, Google, Groq, Mistral, OpenRouter)
  - Character Card functionality (WIP)
  - Conversation history management
  - Notes and keyword management
  - Textual TUI interface
  - Configuration via `config.toml`
    - Default `config.toml` stored at `~/.config/tldw_cli/config.toml`
    - Default user DB stored at `~/.share/tldw_cli/`
  - Environment variable support for API keys
  - Inspiration from Elia Chat Widgets
  - Launch with `python3 -m tldw_chatbook.app` (working on pip packaging)
- **Planned Features:**
  - Conversation Forking + History Management thereof
  - Improved notes and keyword management (Support for syncing notes from a local folder/file)
  - Additional LLM provider support
  - More robust configuration options
  - Enhanced character chat functionality (ASCII art for pictures?)
  - Improved conversation history management (exporting)
  - Support for 'ranking' conversation replies (create your own conversational datasets)



Launch: `python3 -m tldw_chatbook.app`


Default `config.toml` stored at `~/.config/tldw_cli/config.toml`

Default user DB stored at `~/.share/tldw_cli/


Inspiration
https://github.com/darrenburns/elia/tree/main/elia_chat/widgets


## Installation

```bash
pip install tldw_chatbook
```

## Usage

After installation, run the application from your terminal:

```bash
tldw_chatbook
```

## Configuration

The application uses a `config.toml` file located at `~/.config/tldw_cli/config.toml`.
On first run, a default configuration file will be created if one doesn't exist. You'll need to edit this file to add your API keys for the services you want to use.

Environment variables can also be used for API keys, as specified in the `[api_settings.<provider>]` sections of the `config.toml` (e.g., `OPENAI_API_KEY`).

## Contributing

(Details on how to contribute, if applicable)

## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE](LICENSE) file for details.
```