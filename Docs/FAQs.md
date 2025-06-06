# Frequently Asked Questions (FAQs)

## Table of Contents
- [What is the purpose of this documentation?](#what-is-the-purpose-of-this-documentation)
- [How can I contribute to this project?](#how-can-i-contribute-to-this-project)
- [Where can I find the source code?](#where-can-i-find-the-source-code)
- [How do I report a bug or issue?](#how-do-i-report-a-bug-or-issue)
- [How do I request a feature?](#how-do-i-request-a-feature)
- [How do I get help or support?](#how-do-i-get-help-or-support)
- [What are the system requirements?](#what-are-the-system-requirements)
- [How do I install the application?](#how-do-i-install-the-application)
- [How do I run the application?](#how-do-i-run-the-application)
- [How do I update the application?](#how-do-i-update-the-application)
- [How do I uninstall the application?](#how-do-i-uninstall-the-application)
- [How do I configure the application?](#how-do-i-configure-the-application)
- [How do I use the application?](#how-do-i-use-the-application)
- [How do I customize the application?](#how-do-i-customize-the-application)
- [How do I troubleshoot common issues?](#how-do-i-troubleshoot-common-issues)
- [How do I reset the application?](#how-do-i-reset-the-application)
- [How do I back up my data?](#how-do-i-back-up-my-data)
- [How do I restore my data?](#how-do-i-restore-my-data)
- [How do I delete my data?](#how-do-i-delete-my-data)
- [How do I manage my data?](#how-do-i-manage-my-data)
- [How do I export my data?](#how-do-i-export-my-data)
- [How do I import my data?](#how-do-i-import-my-data)
- [How do I sync my data?](#how-do-i-sync-my-data)
- [How do I share my data?](#how-do-i-share-my-data)
- [How do I secure my data?](#how-do-i-secure-my-data)
- [How do I encrypt my data?](#how-do-i-encrypt-my-data)
- [How do I decrypt my data?](#how-do-i-decrypt-my-data)



### Windows Terminal Users

# Handling Large Pastes in Windows Terminal

When using this application (or any command-line application) within Windows Terminal, you might encounter a warning when attempting to paste a large amount of text (typically over 5 KiB). This is a built-in safety feature of Windows Terminal itself.

## Windows Terminal's Paste Warnings

Windows Terminal has specific settings that control how paste operations are handled:

1.  **`largePasteWarning`**:
    *   **Default**: `true`
    *   **Behavior**: If you try to paste text exceeding 5 KiB, Windows Terminal will display a confirmation dialog asking if you want to proceed. If you select "No" (or cancel), the text may not be pasted into the application.
    *   This is the most common warning users encounter when dealing with large text blocks.

2.  **`multiLinePasteWarning`**:
    *   **Default**: `true`
    *   **Behavior**: If you try to paste text that contains multiple lines, Windows Terminal will display a confirmation dialog. This is a security measure, as pasting multiple lines (each potentially a command) into a shell could have unintended consequences.

These settings are part of Windows Terminal's configuration and are independent of this application's behavior.

## Workaround / Configuration

If you frequently paste large amounts of text and find this warning disruptive, you can configure Windows Terminal to disable it.

**To change these settings:**

1.  Open Windows Terminal.
2.  Go to **Settings** (usually by clicking the dropdown arrow in the tab bar or pressing `Ctrl+,`).
3.  In the settings UI, navigate to the "Interaction" section (the names might vary slightly depending on your Terminal version).
4.  Look for options related to "Warn when pasting large amounts of text" (for `largePasteWarning`) and "Warn when pasting text with multiple lines" (for `multiLinePasteWarning`).
5.  Alternatively, you can directly edit the `settings.json` file:
    *   Click on "Open JSON file" in the Settings tab.
    *   In the root of the JSON structure, you can add or modify these properties:
        ```json
        "largePasteWarning": false, // Disables the warning for large pastes
        "multiLinePasteWarning": false // Disables the warning for multi-line pastes
        ```
    *   Set the desired value to `false` to disable the warning.

**Important Considerations:**

*   **Security**: Disabling `multiLinePasteWarning` can be risky, especially if you paste commands from untrusted sources, as it won't prompt you before potentially executing multiple commands.
*   **Application Behavior**: If these terminal warnings are disabled, large amounts of text will be sent directly to the application. This application does not currently implement its own separate warning for large pastes, relying on the user's terminal configuration.

If Windows Terminal's `largePasteWarning` is enabled and you click "Yes" to proceed with the paste, but the text still doesn't appear correctly in the application, this might indicate that the terminal itself is still having trouble sending the entirety of the large input to the application, or the application itself might have other limitations (though the 5KB issue described by the user seems directly tied to the terminal's warning).

Refer to the official [Windows Terminal Interaction Settings Documentation](https://docs.microsoft.com/en-us/windows/terminal/customize-settings/interaction) for the most up-to-date information.



### MLX on Mac
https://github.com/google/sentencepiece/issues/1083


### Samplers
https://rentry.org/samplers