# Contributing to tldw_chatbook

First off, thank you for considering contributing to tldw_chatbook! This project is open source and we(I) welcome contributions from the community. Whether you're fixing bugs, adding new features, or improving documentation, your help is greatly appreciated.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/rmusser01/tldw_chatbook/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

If you have general questions, feel free to reach out on the [tldw_Project Discord](https://discord.gg/your-discord-link-here) or open an issue.

## Setting up the Development Environment

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/rmusser01/tldw_chatbook.git
    cd tldw_chatbook
    ```
3.  **Set up a virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    You might also want to install optional tools/features:
    ```bash
    pip install -r requirements-optional-example-placeholder.txt
    ```

5.  **Set up pre-commit hooks** (Optional, but recommended if the project adds them later):
    This project might use pre-commit hooks to ensure code style and quality. If `pre-commit` is used, install it and set up the hooks:
    ```bash
    pip install pre-commit
    pre-commit install
    ```

## Coding Style

*   Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. (I try)
*   Use clear and descriptive variable and function names.
*   Add comments to your code where necessary to explain complex logic. over explained versus mysterious and demure.
*   Ensure your code is well-formatted. Consider using a tool like Black or Ruff. (Is appreciated, even though I don't use it currently, I may in the future)

## Running Tests

Currently, the project has tests located in the `Tests/` directory (WIP). To run the tests:

1.  Make sure you have installed all dependencies, including any test-specific dependencies.
2.  Navigate to the root directory of the project.
3.  Run the tests using a test runner like `pytest` (you might need to install it: `pip install pytest`).
    ```bash
    pytest ./Tests
    ```

Please ensure all existing tests pass and, if you're adding a new feature or fixing a bug, add new tests to cover your changes.

## Submitting Pull Requests

1.  **Create a new branch** for your changes:
    ```bash
    git checkout -b feature/your-feature-name  # For new features
    # or
    git checkout -b fix/your-bug-fix-name    # For bug fixes
    ```
2.  **Make your changes** and commit them with clear and descriptive commit messages.
3.  **Push your changes** to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
4.  **Open a pull request** from your fork to the `dev` branch of the `rmusser01/tldw_chatbook` repository.
5.  In your pull request description, clearly explain the changes you've made and why. If it addresses an existing issue, link to it (e.g., "Fixes #123").
6.  Be prepared to discuss your changes and make adjustments if requested by the maintainers.

## Project Architecture Overview (Simplified)

`tldw_chatbook` is a Textual-based Terminal User Interface (TUI) application. Here's a very high-level overview of its main components:

*   **`tldw_chatbook/app.py`**: The main entry point of the application, setting up the Textual app.
*   **`tldw_chatbook/Screens/`**: Contains the different screens or views of the application.
*   **`tldw_chatbook/Widgets/`**: Reusable UI components used across different screens.
*   **`tldw_chatbook/Chat/`**: Core logic for handling chat interactions, LLM API calls, and conversation management.
*   **`tldw_chatbook/Character_Chat/`**: Logic related to character-based chat.
*   **`tldw_chatbook/Notes/`**: Functionality for managing notes.
*   **`tldw_chatbook/DB/`**: Database interaction layer (SQLite).
*   **`tldw_chatbook/LLM_Calls/`**: Modules for making API calls to various LLM providers.
*   **`config.toml` (user-specific)**: Manages user configurations, including API keys. Found at `~/.config/tldw_cli/config.toml`.
*   **`Tests/`**: Contains all the tests for the project.

This is a simplified view. For a deeper understanding, you'll need to explore the codebase.

## Other Resources

*   [Project README.md](README.md)
*   [Issue Tracker](https://github.com/rmusser01/tldw_chatbook/issues)

Thank you for contributing!
