# tldw_chatbook README

A Textual TUI for interacting with various LLM APIs, managing conversation history, characters, notes, and more.

Current status: Working/In-Progress

![Screenshot](https://github.com/rmusser01/tldw_chatbook/blob/main/static/PoC-Frontpage.PNG?raw=true)
### Quick Start
- **Via Manual Installation**
  - Clone the repository: `git clone https://github.com/rmusser01/tldw_chatbook`
  - Setup a virtual environment (optional but recommended): 
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  - Install the dependencies: `pip install -r requirements.txt`
  - Run the application:
    - If you are in the root directory of the repository: `python3 -m tldw_chatbook.app`

## Current Features
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

### Planned Features
- Conversation Forking + History Management thereof (Already implemented, but needs more testing/UI buildout)
- Improved notes and keyword management (Support for syncing notes from a local folder/file - think Obsidian)
- Additional LLM provider support (e.g., more local providers, more commercial providers)
- More robust configuration options (e.g., more environment variable support, more config.toml options)
- Enhanced character chat functionality (e.g., ASCII art for pictures, 'Generate Character' functionality)
- Improved conversation history management (e.g., exporting conversations, better search functionality)
- Support for 'ranking' conversation replies (create your own conversational datasets, e.g., for training or fine-tuning models)
- Workflows - e.g., Ability to create structured workflows, like a task list or a series of steps to follow, with the ability to execute them in order with checkpoints after each step. (Agents?)
- Agentic functionality (e.g., ability to create agents that can perform tasks based on conversation history or notes, think workflow automation with checkpoints)
  - First goal will be the groundwork/framework for building it out more, and then for coding, something like Aider?
  - Separate from the workflows, which are more like structured task lists or steps to follow. Agentic functionality will be more about creating workflows, but not-fully structured, that adapt based on the 'agents' decisions.
- Mindmap functionality (e.g., ability to create mindmaps from conversation history or notes)
- Support for more media types (e.g., images, audio, video - Ingestion thereof)
- Support for Server Syncing (e.g., ability to sync conversations, notes, characters, Media DB and prompts across devices)
- Support for RAG (Retrieval-Augmented Generation) functionality (e.g., ability to retrieve relevant information from conversations, notes, characters, Media DB and prompts)
- Support for Web Search (e.g., ability to search the web for relevant information based on conversation history or notes)
  - Already implemented, but needs more testing/UI buildout
- Support for audio playback + Generation (e.g., ability to play audio files, generate audio from text - longer term goal, has to run outside of the TUI)
  - Also Support for image generation and video playback + Generation (e.g., ability to play video files, generate video from text - longer term goal, has to run outside of the TUI)

### Getting Started
- **Via pip**
  - Install the package via pip: `pip install tldw_chatbook` (Not yet available, but will be soon)
  - Run the application from your terminal: `tldw_chatbook`
- **Via Manual Installation**
  - Clone the repository: `git clone https://github.com/rmusser01/tldw_chatbook`
  - Setup a virtual environment (optional but recommended) and run tldw_chatbook: 
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate or .venv\Scripts\activate
    ```
  - Install the dependencies: `pip install -r requirements.txt` 
    ```bash
    pip install -r requirements.txt
    ```
  - Run the application:
    - If you are in the root directory of the repository: `python3 -m tldw_chatbook.app`
    ```bash
    python3 -m tldw_chatbook.app
    ```
- **Configuration**
  - The application uses a `config.toml` file located at `~/.config/tldw_cli/config.toml`.
  - On first run, a default configuration file will be created if one doesn't exist. You'll need to edit this file to add your API keys for the services you want to use.
  - Environment variables can also be used for API keys, as specified in the `[api_settings.<provider>]` sections of the `config.toml` (e.g., `OPENAI_API_KEY`).
- **User Database**
  - The application uses several user databases stored at `~/.share/tldw_cli/`.
  - The databases are: 
    - `tldw_chatbook_ChaChaNotes.db` - This sqlite DB stores your conversations, characters, and notes.
    - `tldw_chatbook_media_v2.db` - This sqlite DB stores the user's ingested media files.
    - `tldw_chatbook_prompts.db` - This sqlite DB stores the user's prompts.
  - Each database is created on first run if it doesn't already exist.

### Project Structure

Here's a brief overview of the main directories in the project:

*   **`tldw_chatbook/`**: Contains the core source code of the application.
    *   **`app.py`**: Main application entry point.
    *   **`Screens/`**: Application screens (main views).
    *   **`Widgets/`**: Reusable TUI components.
    *   **`UI/`**: More complex UI structures and panels.
    *   **`Chat/`**: Logic for chat functionalities and LLM interactions.
    *   **`DB/`**: Database interaction modules.
    *   **`LLM_Calls/`**: Modules for calling LLM APIs.
    *   **`Notes/`**: Notes management logic.
    *   **`Event_Handlers/`**: Application event handling.
*   **`Docs/`**: Project documentation (WIP)
*   **`Tests/`**: Contains all automated tests.
*   **`css/`**: Stylesheets for the Textual TUI.
*   **`static/`**: Static assets like images.
*   **`Helper_Scripts/`**: Utility scripts for various tasks.

### Inspiration
https://github.com/darrenburns/elia

## Contributing
- Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute to this project.(WIP)
- (Realistically, this is a work in progress, so contributions are welcome, but please be aware that the codebase is still evolving and may change frequently.)
- Make a pull request against the `dev` branch, where development happens prior to being merged into `main`.

## License

This project is licensed under the GNU Affero General Public License - see the [LICENSE](LICENSE) file for details.

### Contact
For any questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/rmusser01/tldw) or contact me directly on the tldw_Project Discord or via the email in my profile.
