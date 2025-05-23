# tldw_chatbook README

A Textual TUI for interacting with various LLM APIs, managing conversation history, characters, notes, and more.

Current status: Working. Weak DB support, lots of bugs there. Chat works great. ( I think )

## Features

*   Connect to multiple LLM providers (OpenAI, Anthropic, local models via Ollama, etc.).
*   Character chat functionality.
*   Conversation history management.
*   Notes and keyword management.


Launch: `python3 -m app`


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