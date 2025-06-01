"""llm_management_events_vllm.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **vLLM** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates vLLM-specific logic from the main llm_management_events.py.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from textual.containers import Container
from textual.css.query import QueryError
from textual.worker import Worker, WorkerState
from textual.widgets import Input, RichLog, TextArea, Button

if TYPE_CHECKING:
    from ..app import TldwCli
    # Assuming LLMManagementWindow might be needed for type hints if methods are complex
    # from ..UI.LLM_Management_Window import LLMManagementWindow

# Imports for shared functions from the original events file
from .llm_management_events import _make_path_update_callback, _stream_process, stream_worker_output_to_log
# Import for FileOpen and Filters
from ..Third_Party.textual_fspicker import FileOpen, Filters

__all__ = [
    "handle_vllm_browse_python_button_pressed",
    "handle_vllm_browse_model_button_pressed",
    "run_vllm_server_worker",
    "handle_start_vllm_server_button_pressed",
    "handle_stop_vllm_server_button_pressed",
]

###############################################################################
# ─── vLLM UI helpers ────────────────────────────────────────────────────────
###############################################################################


async def handle_vllm_browse_python_button_pressed(app: "TldwCli") -> None:
    """Let the user pick the Python interpreter used for vLLM (venv, etc.)."""

    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Python interpreter for vLLM",
            filters=Filters(("Python executable", lambda p: p.name.startswith("python"))),
        ),
        callback=_make_path_update_callback(app, "vllm-python-path"),
    )


async def handle_vllm_browse_model_button_pressed(app: "TldwCli") -> None:
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Model (checkpoint or GGUF) for vLLM",
            filters=Filters(("All files", lambda p: True)),
        ),
        callback=_make_path_update_callback(app, "vllm-model-path"),
    )

###############################################################################
# ─── Worker functions
###############################################################################

def run_vllm_server_worker(app_instance: "TldwCli", command: List[str]):
    """Worker that launches *vllm* via the given *command* list."""

    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    logger.info("vLLM worker begins: %s", " ".join(command))
    app_instance.vllm_server_process = None  # Clear any old process reference

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        app_instance.vllm_server_process = process  # Store the process

        app_instance.call_from_thread(
            app_instance._update_vllm_log, f"vLLM server started (PID: {process.pid})…\n"
        )
        _stream_process(app_instance, "_update_vllm_log", process)
        process.wait()
        yield f"vLLM server exited with code: {process.returncode}\n"
    except FileNotFoundError:
        app_instance.vllm_server_process = None # Clear process on error
        msg = f"ERROR: vLLM interpreter not found: {command[0]}\n"
        logger.error(msg.rstrip())
        app_instance.call_from_thread(app_instance._update_vllm_log, msg)
        yield msg
    except Exception as err:
        app_instance.vllm_server_process = None # Clear process on error
        msg = f"ERROR in vLLM worker: {err}\n"
        logger.error(msg.rstrip(), exc_info=True)
        app_instance.call_from_thread(app_instance._update_vllm_log, msg)
        yield msg
    finally:
        app_instance.vllm_server_process = None # Ensure process is cleared

###############################################################################
# ─── vLLM – start/stop handlers ──────────────────────────────────────────────
###############################################################################

async def handle_start_vllm_server_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start vLLM server.")

    try:
        python_path_input = app.query_one("#vllm-python-path", Input)
        model_path_input = app.query_one("#vllm-model-path", Input)
        host_input = app.query_one("#vllm-host", Input)
        port_input = app.query_one("#vllm-port", Input)
        additional_args_input = app.query_one("#vllm-additional-args", TextArea)
        log_output_widget = app.query_one("#vllm-log-output", RichLog)

        python_path = python_path_input.value.strip() or "python"
        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8002"
        additional_args_str = additional_args_input.text.strip()

        if model_path and not Path(model_path).exists():
            app.notify(f"Model path not found: {model_path}", severity="error")
            model_path_input.focus()
            return

        command = [
            python_path,
            "-m",
            "vllm.entrypoints.api_server",
            "--host",
            host,
            "--port",
            port,
        ]
        if model_path:
            command.extend(["--model", model_path])
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_vllm_server_worker,
            args=[app, command],
            group="vllm_server",
            description="Running vLLM API server",
            exclusive=True,
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#vllm-log-output"
            ),
        )
        app.notify("vLLM server starting…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing to start vLLM server: %s", err, exc_info=True)
        app.notify("Error setting up vLLM server start.", severity="error")


async def handle_stop_vllm_server_button_pressed(app: "TldwCli") -> None:
    """Stops the vLLM server process if it's running."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop vLLM server.")

    log_output_widget = app.query_one("#vllm-log-output", RichLog)

    if hasattr(app, "vllm_server_process") and app.vllm_server_process:
        process = app.vllm_server_process
        if process.poll() is None:  # Process is running
            logger.info(f"Stopping vLLM server process (PID: {process.pid}).")
            log_output_widget.write(f"Stopping vLLM server (PID: {process.pid})...\n")
            process.terminate()  # or process.kill()
            try:
                process.wait(timeout=10)  # Wait for up to 10 seconds
                logger.info("vLLM server process terminated.")
                log_output_widget.write("vLLM server stopped.\n")
                app.notify("vLLM server stopped.")
            except subprocess.TimeoutExpired:
                logger.warning("Timeout waiting for vLLM server to terminate. Killing.")
                log_output_widget.write("vLLM server did not stop gracefully, killing...\n")
                process.kill()
                process.wait() # Ensure it's killed
                log_output_widget.write("vLLM server killed.\n")
                app.notify("vLLM server killed after timeout.", severity="warning")
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Error during vLLM server termination: {e}", exc_info=True)
                log_output_widget.write(f"Error stopping vLLM server: {e}\n")
                app.notify(f"Error stopping vLLM server: {e}", severity="error")
            finally:
                app.vllm_server_process = None
        else:
            logger.info("vLLM server process was found but is not running.")
            log_output_widget.write("vLLM server is not currently running.\n")
            app.notify("vLLM server is not running.", severity="warning")
            app.vllm_server_process = None # Clear the stale process reference
    else:
        logger.info("No vLLM server process found to stop.")
        log_output_widget.write("vLLM server is not currently running.\n")
        app.notify("vLLM server is not running.", severity="warning")

#
# End of llm_management_events_vllm.py
########################################################################################################################
