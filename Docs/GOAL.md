# GOAL

### Goal of this Project
For now, the goal of this project is to create a terminal-based chat application that allows users to interact with large language models (LLMs) in a conversational manner. 
    - The application aims to provide a user-friendly interface for managing conversations, characters, and notes, while also allowing users to ingest media files and use custom prompts.
    - It should remain modular, lightweight, and minimal dependencies by default.
        - That said, it should allow for great extensibility and customization, allowing users to add their own features and functionality as needed.


### Key Features
- **Conversational Interface**: Users can chat with LLMs, manage conversations, and fork conversations.
- **Character Management**: Users can create, edit, import and export  characters, each with their own personality and traits.
- **Note Management**: Users can create, edit, import and export notes, which can be linked to conversations or characters.
- **Media Ingestion**: Users can ingest media files (images, audio, video) and then use them in conversation or perform analysis thereof.
- **Prompt Management**: Users can create, clone, edit, import and export custom prompts for LLMs.
- **Web Search**: Users can search the web for relevant information based on conversation history or notes.
- **RAG Search**: Users can retrieve relevant information from conversations, notes, characters, media DB and prompts.
- **Configuration**: The application is configurable via a `config.toml` file, allowing users to set API keys and other settings.
- **TUI Interface**: The application provides a text-based user interface (TUI) for easy navigation and interaction.
- **TTS and STT Support**: The application supports text-to-speech (TTS) and speech-to-text (STT) functionality, allowing users to interact with the application using voice commands.
- **Extensibility**: The application is designed to be modular and extensible, allowing users to add their own features and functionality as needed.
  - This includes:
    - Custom LLM providers
    - Workflows
    - Agents
    - Custom media types (e.g., Allow for a user to implement a binary analysis pipeline, e.g., for malware analysis, ship the file off and retrieve the results)
    - Chat pipeline plugins (Need to be able to add custom plugins to the chat pipeline, e.g., for custom processing of messages(ChatDictionaries++), or for custom actions based on messages(ChatDictionaries++))

