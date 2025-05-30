# Architecture & Design of tldw_chatbook



## Project Architecture Overview (Simplified)

`tldw_chatbook` is a Textual-based Terminal User Interface (TUI) application. Here's a very high-level overview of its main components:

- **`tldw_chatbook/`**: The main package containing the application code.
- **`tldw_chatbook/app.py`**: The main entry point of the application, setting up the Textual app.
- **`tldw_chatbook/Character_Chat/`**: Logic related to character-based chat.
- **`tldw_chatbook/Chat/`**: Core logic for handling chat interactions, LLM API calls, and conversation management.
- **`tldw_chatbook/DB/`**: Database interaction layer (SQLite).
  - **`ChaChaNotes_DB.py`**: SQLite Database library used for Chats, Characters, and Notes.
  - **`Client_Media_DB_v2.py`**: Manages Media database operations. This is where all ingested media is stored.
  - **`Prompts_DB.py`**: Manages the prompts database related operations.
  - **`Sync_Client.py`**: Handles synchronization of client DBs with the server. (WIP)
- **`tldw_chatbook/Event_Handlers/`**: Contains event handlers for various user interactions.
- **`tldw_chatbook/LLM_Calls/`**: Modules for making API calls to various LLM providers.
- **`tldw_chatbook/Metrics/`**: Contains modules for local application metrics. Local-Only.
- **`tldw_chatbook/Notes/`**: Interop Library for Notes functionality. (Eventually will hold local note syncing)
- **`tldw_chatbook/Prompt_Management/`**: Contains modules for managing prompts.
- **`tldw_chatbook/Screens/`**: Contains a single screen, this and UI should be merged into a single directory.
- **`tldw_chatbook/Third_Party/`**: Contains third-party libraries and utilities.
- **`tldw_chatbook/Tools/`**: Contains various interfaces to 3rd-party tools and services.
- **`tldw_chatbook/TTS/`**: Contains modules for Text-to-Speech functionality.
- **`tldw_chatbook/UI/`**: Contains different views of the application's various tabs (Was supposed to be screens but haven't bothered to rename it all).
- **`tldw_chatbook/Utils/`**: Contains utility functions and classes used across the application.
- **`tldw_chatbook/Web_Scraping/`**: Contains modules for web scraping functionality.
- **`tldw_chatbook/Widgets/`**: Reusable UI components used across different screens.
- **`tldw_chatbook/Config.py`**: Contains all configuration settings for the application, including API keys, database paths, and other settings.
- **`tldw_chatbook/Constants.py`**: Contains all constants used throughout the application, such as default values and error messages.
- **`tldw_chatbook/Logging_Config.py`**: Contains the logging configuration for the application, setting up loggers, handlers, and formatters.