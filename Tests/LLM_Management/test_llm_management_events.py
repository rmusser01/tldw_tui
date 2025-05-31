import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widgets import Input, RichLog

from tldw_chatbook.Event_Handlers.llm_management_events import handle_start_llamacpp_server_button_pressed
from tldw_chatbook.app import TldwCli  # Assuming TldwCli is the app class

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app():
    """Fixture to create a mock TldwCli app instance."""
    app = MagicMock(spec=TldwCli)
    app.loguru_logger = MagicMock()  # Mock the logger if it's used directly in the handler

    # Mock query_one to return specific widget mocks
    widget_mocks = {
        "#llamacpp-exec-path": MagicMock(spec=Input, value="/path/to/server"),
        "#llamacpp-model-path": MagicMock(spec=Input, value="/path/to/model.gguf"),
        "#llamacpp-host": MagicMock(spec=Input, value="127.0.0.1"),
        "#llamacpp-port": MagicMock(spec=Input, value="8001"),
        "#llamacpp-additional-args": MagicMock(spec=Input, value="--n-gpu-layers 33 --verbose"),
        "#llamacpp-log-output": MagicMock(spec=RichLog),
    }

    def query_one_side_effect(selector, widget_type):
        # Ensure the widget_type matches if provided (it is in the handler)
        mock_widget = widget_mocks.get(selector)
        if not mock_widget:
            raise Exception(f"Mock for selector {selector} not found")  # Should not happen in these tests
        if not isinstance(mock_widget, widget_type):
            raise Exception(f"Mock for {selector} is {type(mock_widget)} not {widget_type}")
        return mock_widget

    app.query_one = MagicMock(side_effect=query_one_side_effect)
    app.run_worker = AsyncMock()  # Use AsyncMock if run_worker is an async method or returns an awaitable
    app.notify = MagicMock()

    # Mock Path.is_file to return True for valid paths
    # This is crucial for path validation checks in the handler
    with patch("pathlib.Path.is_file", return_value=True) as _:
        yield app  # Yield the app to allow the patch to be active during the test


async def test_handle_start_llamacpp_server_button_pressed_basic_command(mock_app):
    """Test basic command construction with all fields provided."""
    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.run_worker.assert_called_once()
    call_args = mock_app.run_worker.call_args

    # The actual command is the second element in the 'args' tuple of the call_args
    # args=[app_instance, command_list]
    # Ensure 'args' key exists and has at least two elements
    assert 'args' in call_args.kwargs, "Worker 'args' not found in call_args.kwargs"
    assert len(call_args.kwargs['args']) == 2, "Worker 'args' does not have two elements"

    actual_command = call_args.kwargs['args'][1]

    expected_command = [
        "/path/to/server",
        "--model", "/path/to/model.gguf",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--n-gpu-layers", "33",
        "--verbose"
    ]
    assert actual_command == expected_command
    mock_app.notify.assert_called_with("Llama.cpp server starting…")


async def test_handle_start_llamacpp_server_no_additional_args(mock_app):
    """Test command construction when additional arguments are empty."""
    mock_app.query_one("#llamacpp-additional-args").value = ""  # Override default for this test

    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.run_worker.assert_called_once()
    call_args = mock_app.run_worker.call_args
    actual_command = call_args.kwargs['args'][1]

    expected_command = [
        "/path/to/server",
        "--model", "/path/to/model.gguf",
        "--host", "127.0.0.1",
        "--port", "8001",
    ]
    assert actual_command == expected_command
    mock_app.notify.assert_called_with("Llama.cpp server starting…")


async def test_handle_start_llamacpp_server_additional_args_with_spaces(mock_app):
    """Test command construction with additional arguments containing spaces (quoted)."""
    mock_app.query_one("#llamacpp-additional-args").value = '--custom-path "/mnt/my models/llama" --another-arg value'

    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.run_worker.assert_called_once()
    call_args = mock_app.run_worker.call_args
    actual_command = call_args.kwargs['args'][1]

    expected_command = [
        "/path/to/server",
        "--model", "/path/to/model.gguf",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--custom-path", "/mnt/my models/llama",  # shlex.split handles the quotes
        "--another-arg", "value"
    ]
    assert actual_command == expected_command


async def test_handle_start_llamacpp_server_default_host_port(mock_app):
    """Test that default host and port are used if inputs are empty."""
    mock_app.query_one("#llamacpp-host").value = ""
    mock_app.query_one("#llamacpp-port").value = ""

    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.run_worker.assert_called_once()
    call_args = mock_app.run_worker.call_args
    actual_command = call_args.kwargs['args'][1]

    # Default host is 127.0.0.1, default port is 8001 (as per handler logic)
    expected_command = [
        "/path/to/server",
        "--model", "/path/to/model.gguf",
        "--host", "127.0.0.1",
        "--port", "8001",
        "--n-gpu-layers", "33",
        "--verbose"
    ]
    assert actual_command == expected_command


async def test_handle_start_llamacpp_server_missing_exec_path(mock_app):
    """Test validation: executable path is missing."""
    mock_app.query_one("#llamacpp-exec-path").value = ""  # Missing exec path

    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.notify.assert_called_with("Executable path is required.", severity="error")
    mock_app.run_worker.assert_not_called()


async def test_handle_start_llamacpp_server_invalid_exec_path(mock_app):
    """Test validation: executable path is not a file."""
    # We need to make Path(exec_path).is_file() return False for this specific input
    # The fixture currently patches it globally to True.
    # We can re-patch it within this test for more specific behavior.
    with patch("pathlib.Path.is_file", side_effect=lambda p: str(p) != "/invalid/path/server") as mock_is_file:
        mock_app.query_one("#llamacpp-exec-path").value = "/invalid/path/server"

        await handle_start_llamacpp_server_button_pressed(mock_app)

        mock_app.notify.assert_called_with("Executable not found at: /invalid/path/server", severity="error")
        mock_app.run_worker.assert_not_called()
        # Check that is_file was called with the correct path
        mock_is_file.assert_any_call(mock_app.query_one("#llamacpp-exec-path").value)


async def test_handle_start_llamacpp_server_missing_model_path(mock_app):
    """Test validation: model path is missing."""
    mock_app.query_one("#llamacpp-model-path").value = ""  # Missing model path

    await handle_start_llamacpp_server_button_pressed(mock_app)

    mock_app.notify.assert_called_with("Model path is required.", severity="error")
    mock_app.run_worker.assert_not_called()


async def test_handle_start_llamacpp_server_invalid_model_path(mock_app):
    """Test validation: model path is not a file."""
    with patch("pathlib.Path.is_file", side_effect=lambda p: str(p) != "/invalid/path/model.gguf") as mock_is_file:
        # Ensure exec path is valid for this test
        mock_app.query_one("#llamacpp-exec-path").value = "/path/to/server"  # Valid mock path
        mock_is_file.side_effect = lambda p: True if str(p) == "/path/to/server" else (
                    str(p) != "/invalid/path/model.gguf")

        mock_app.query_one("#llamacpp-model-path").value = "/invalid/path/model.gguf"

        await handle_start_llamacpp_server_button_pressed(mock_app)

        mock_app.notify.assert_called_with("Model file not found at: /invalid/path/model.gguf", severity="error")
        mock_app.run_worker.assert_not_called()
        # is_file is called for exec_path first, then for model_path
        assert mock_is_file.call_count >= 2
        # The last call (or one of the calls) should be with the invalid model path
        mock_is_file.assert_any_call(mock_app.query_one("#llamacpp-model-path").value)

# To run these tests, you would typically use pytest:
# pytest Tests/Event_Handlers/test_llm_management_events.py
# (Assuming pytest and pytest-asyncio are installed)
# Also ensure that tldw_chatbook is in PYTHONPATH or installable
# and that the necessary dependencies for textual are available.
# The patch for Path.is_file in the fixture and some tests is important
# to bypass actual filesystem checks.
#
# Note on the patch in the fixture:
# The `with patch(...) as _:` in the fixture means Path.is_file will return True
# for ALL calls during the tests that use this fixture, unless a test re-patches it.
# The `_` is used because we don't need to access the MagicMock object for the patch itself
# in most test cases using the fixture.
# For tests like `test_handle_start_llamacpp_server_invalid_exec_path`, we re-patch
# `pathlib.Path.is_file` with a more specific side_effect to test the False condition.
#
# The side_effect for query_one in the fixture is a simple way to return different
# mocks based on the selector. A more robust approach for complex scenarios might

# involve a dictionary lookup as shown.
#
# The check `if not isinstance(mock_widget, widget_type):` in `query_one_side_effect`
# is to more closely mimic Textual's `query_one` behavior which also checks the type.
#
# The `pytestmark = pytest.mark.asyncio` line at the top is important for pytest-asyncio
# to correctly run the async test functions.
#
# The `mock_app.loguru_logger = MagicMock()` line is added to the fixture to prevent
# errors if the logger is accessed (e.g., `app.loguru_logger.info(...)`).
#
# The `test_handle_start_llamacpp_server_invalid_exec_path` and
# `test_handle_start_llamacpp_server_invalid_model_path` have been updated
# to show how to make `Path.is_file` return `False` for specific paths by
# re-patching within the test or using a more complex side_effect.
# The `side_effect=lambda p: str(p) != "/invalid/path/server"` means `is_file`
# will return `True` for any path *except* "/invalid/path/server".
# For the invalid model path test, the side_effect is a bit more complex to ensure
# the exec_path still appears valid while the model_path does not.
#
# The assertion `mock_is_file.assert_any_call(...)` is used because `is_file` might be
# called multiple times (once for exec_path, once for model_path). We just want to ensure
# it was called with the specific path we're testing.
#
# Final check on `call_args` for `run_worker`:
# The arguments to `run_worker` in the main code are passed as `args=[app, command]`.
# So, in the test, `mock_app.run_worker.call_args` will be a Call object.
# If using `args=...`, then `call_args.args` would be `([app, command],)`.
# If using `kwargs={'args': ...}` or `args=...` as a kwarg, then `call_args.kwargs['args']`
# would be `[app, command]`. The current code uses `args=[app, command]` directly in the
# `run_worker` call, so it's passed as a keyword argument named `args`.
# `call_args.kwargs['args']` is `[<MagicMock spec='TldwCli' id='...'>, ['/path/to/server', ...]]`
# So `call_args.kwargs['args'][1]` is the command list.
# The assertions have been updated to reflect this.
#
# Added a check in the fixture's `query_one_side_effect` to ensure the mock widget found
# is an instance of the `widget_type` argument, as the actual code uses this.
# Example: `app.query_one("#llamacpp-exec-path", Input)`
# This makes the mock more accurate.
#
# Added `app.loguru_logger = MagicMock()` to the fixture.
# The handler code uses `logger = getattr(app, "loguru_logger", logging.getLogger(__name__))`
# so the mock app needs this attribute.
#
# Updated the `test_handle_start_llamacpp_server_invalid_model_path` to correctly mock `is_file`
# such that the exec_path is considered valid, but the model_path is not. This requires
# the side_effect to be a bit more conditional.
#
# The test `test_handle_start_llamacpp_server_invalid_exec_path` was also refined for clarity on how `is_file` is mocked.
#
# Final review of the run_worker call in the handler:
# `app.run_worker(run_llamacpp_server_worker, args=[app, command], ...)`
# This means `call_args.args` will be `(run_llamacpp_server_worker,)`
# and `call_args.kwargs` will be `{'args': [app, command], 'group': ..., ...}`.
# The tests correctly access `call_args.kwargs['args'][1]`.
