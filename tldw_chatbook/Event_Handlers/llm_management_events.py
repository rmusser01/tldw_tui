"""llm_management_events.py

A collection of helper callbacks, worker functions and event‑handler
coroutines used by the **LLM Management** tab in *tldw‑cli*.

The file now supports **four** back‑ends:

* **Llamafile**
* **Llama.cpp**
* **vLLM**
* **Model Download** via Hugging Face

The design intentionally mirrors the style that was started by a previous
LLM so that each back‑end offers a familiar set of helpers:

* *browse‑…* handlers for file/directory pickers
* *_make_…_path_update_callback* helpers to populate `Input` widgets
* *run_…_worker* background workers streaming stdout → `RichLog`
* *handle_start_…_button_pressed* coroutines that validate UI fields,
  build command‑lines and spawn the workers with `app.run_worker`
* *stream_…_worker_output_to_log* (generic) that forward the *final*
  messages from a worker to its log widget.

The UI layer itself (layouts, widgets, tab switching, etc.) lives in
`llm_management.py` next to this module.  That file builds a left‑hand
vertical `VerticalTabs` widget whose four tabs bind to the handlers
below.
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
    from ..app import TldwCli  # pragma: no cover – runtime import only
    from ..UI.LLM_Management_Window import LLMManagementWindow # For type hints if used within methods

from ..Local_Inference.mlx_lm_inference_local import start_mlx_lm_server, stop_mlx_lm_server
# subprocess already imported

from ..Third_Party.textual_fspicker import FileOpen, FileSave, Filters

__all__ = [
    # ─── Llamafile ────────────────────────────────────────────────────────────
    "handle_llamafile_browse_exec_button_pressed",
    "handle_llamafile_browse_model_button_pressed",
    "handle_start_llamafile_server_button_pressed",
    # ─── Llama.cpp ────────────────────────────────────────────────────────────
    "handle_llamacpp_browse_exec_button_pressed",
    "handle_llamacpp_browse_model_button_pressed",
    "handle_start_llamacpp_server_button_pressed",
    "handle_stop_llamacpp_server_button_pressed",
    # ─── vLLM ─────────────────────────────────────────────────────────────────
    "handle_vllm_browse_python_button_pressed",
    "handle_vllm_browse_model_button_pressed",
    "handle_start_vllm_server_button_pressed",
    "handle_stop_vllm_server_button_pressed",
    # ─── Model download ───────────────────────────────────────────────────────
    "handle_browse_models_dir_button_pressed",
    "handle_start_model_download_button_pressed",
    # ─── Ollama ───────────────────────────────────────────────────────────────
    "handle_ollama_nav_button_pressed",
    # ─── MLX-LM ───────────────────────────────────────────────────────────────
    "handle_mlx_lm_nav_button_pressed",
]

###############################################################################
# Generic helpers
###############################################################################


def _make_path_update_callback(app: "TldwCli", input_widget_id: str):
    """Return a callback that sets *input_widget_id*'s value to a picked path."""

    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))

    async def _callback(selected_path: Optional[Path]) -> None:
        if selected_path:
            try:
                input_widget = app.query_one(f"#{input_widget_id}", Input)
                input_widget.value = str(selected_path)
                logger.info(
                    "Updated input #%s with path: %s", input_widget_id, selected_path
                )
            except Exception as err:  # pragma: no cover – UI querying
                logger.error(
                    "Error updating input #%s: %s", input_widget_id, err, exc_info=True
                )
                app.notify(
                    f"Error setting path for {input_widget_id}.", severity="error"
                )
        else:
            logger.info("File selection cancelled for #%s.", input_widget_id)

    return _callback


###############################################################################
# ─── Llamafile helpers ───────────────────────────────────────────────────────
###############################################################################


async def handle_llamafile_browse_exec_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llamafile browse executable button pressed.")

    exec_filters = Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Executable",
            filters=exec_filters,
        ),
        callback=_make_path_update_callback(app, "llamafile-exec-path"),
    )


async def handle_llamafile_browse_model_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llamafile browse model button pressed.")

    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Model (.gguf)",
            filters=gguf_filters,
        ),
        callback=_make_path_update_callback(app, "llamafile-model-path"),
    )


###############################################################################
# ─── Worker functions
###############################################################################


# Each run_…_worker uses the same streaming pattern – consider refactoring, but
# explicit duplication keeps each implementation easy to tweak individually.

def _stream_process(app_instance: "TldwCli", log_fn_name: str, process: subprocess.Popen):
    """Utility: pump *process* stdout into *app_instance.log_fn_name*."""

    log_fn = getattr(app_instance, log_fn_name)
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            app_instance.call_from_thread(log_fn, line)
        process.stdout.close()


def run_llamafile_server_worker(app_instance: "TldwCli", command: List[str]):
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    logger.info("Llamafile worker begins: %s", " ".join(command))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        app_instance.call_from_thread(
            app_instance._update_llamafile_log,
            f"Llamafile process started (PID: {process.pid})…\n",
        )
        _stream_process(app_instance, "_update_llamafile_log", process)
        process.wait()
        yield f"Llamafile server exited with code: {process.returncode}\n"
    except FileNotFoundError:
        msg = f"ERROR: Llamafile executable not found: {command[0]}\n"
        logger.error(msg.rstrip())
        app_instance.call_from_thread(app_instance._update_llamafile_log, msg)
        yield msg
    except Exception as err:  # pragma: no cover
        msg = f"ERROR in Llamafile worker: {err}\n"
        logger.error(msg.rstrip(), exc_info=True)
        app_instance.call_from_thread(app_instance._update_llamafile_log, msg)
        yield msg


def run_llamacpp_server_worker(app_instance: "TldwCli", command: List[str]):
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    logger.info("Llama.cpp worker begins: %s", " ".join(command))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        app_instance.call_from_thread(
            app_instance._update_llamacpp_log,
            f"Llama.cpp server process started (PID: {process.pid})…\n",
        )
        _stream_process(app_instance, "_update_llamacpp_log", process)
        process.wait()
        yield f"Llama.cpp server exited with code: {process.returncode}\n"
    except FileNotFoundError:
        msg = f"ERROR: Llama.cpp executable not found: {command[0]}\n"
        logger.error(msg.rstrip())
        app_instance.call_from_thread(app_instance._update_llamacpp_log, msg)
        yield msg
    except Exception as err:
        msg = f"ERROR in Llama.cpp worker: {err}\n"
        logger.error(msg.rstrip(), exc_info=True)
        app_instance.call_from_thread(app_instance._update_llamacpp_log, msg)
        yield msg


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


def run_model_download_worker(app_instance: "TldwCli", command: List[str]):
    """Background worker that executes *command* to download a model.

    The implementation simply shells‑out to **huggingface‑cli** so that we
    do **not** add an unconditional *huggingface‑hub* dependency to
    *tldw‑cli*.  Users that prefer the Python API can adapt this easily.
    """

    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    logger.info("Model‑download worker begins: %s", " ".join(command))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path.home()),  # run in a writable location
        )

        app_instance.call_from_thread(
            app_instance._update_model_download_log,
            f"Download started (PID: {process.pid})…\n",
        )
        _stream_process(app_instance, "_update_model_download_log", process)
        process.wait()
        yield f"Download command exited with code: {process.returncode}\n"
    except FileNotFoundError:
        msg = "ERROR: huggingface‑cli (or specified executable) not found.\n"
        logger.error(msg.rstrip())
        app_instance.call_from_thread(app_instance._update_model_download_log, msg)
        yield msg
    except Exception as err:  # pragma: no cover
        msg = f"ERROR in model‑download worker: {err}\n"
        logger.error(msg.rstrip(), exc_info=True)
        app_instance.call_from_thread(app_instance._update_model_download_log, msg)
        yield msg


###############################################################################
# ─── shared output streaming coroutine ───────────────────────────────────────
###############################################################################


async def stream_worker_output_to_log(app: "TldwCli", worker: Worker, log_widget_id: str):
    """Forward *worker*'s yielded final messages to *log_widget_id*."""

    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))

    try:
        log_widget = app.query_one(log_widget_id, RichLog)
        wrote_final = False
        if worker.state == WorkerState.SUCCESS:
            async for line in worker.output:
                log_widget.write(line.rstrip())
                wrote_final = True
            if not wrote_final:
                log_widget.write(
                    f"--- Worker {worker.name} finished successfully (no final message) ---"
                )
        elif worker.state == WorkerState.ERROR:
            async for line in worker.output:
                log_widget.write(line.rstrip())
                wrote_final = True
            if not wrote_final:
                log_widget.write(
                    f"--- Worker {worker.name} failed (no final message) ---"
                )
                if worker.result is not None:
                    log_widget.write(f"Error details: {worker.result}")
    except QueryError:
        logger.error("RichLog %s not found to attach worker %s", log_widget_id, worker.name)


###############################################################################
# ─── Llamafile – start handler ───────────────────────────────────────────────
###############################################################################


async def handle_start_llamafile_server_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start Llamafile server.")

    try:
        exec_path_input = app.query_one("#llamafile-exec-path", Input)
        model_path_input = app.query_one("#llamafile-model-path", Input)
        host_input = app.query_one("#llamafile-host", Input)
        port_input = app.query_one("#llamafile-port", Input)
        additional_args_input = app.query_one("#llamafile-additional-args", TextArea)
        log_output_widget = app.query_one("#llamafile-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8000"
        additional_args_str = additional_args_input.text.strip()

        if not exec_path:
            app.notify("Llamafile executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify(f"Llamafile executable not found at: {exec_path}", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify(f"Model file not found at: {model_path}", severity="error")
            model_path_input.focus()
            return

        command = [
            exec_path,
            "-m",
            model_path,
            "--host",
            host,
            "--port",
            port,
        ]
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_llamafile_server_worker,
            args=[app, command],
            group="llamafile_server",
            description="Running Llamafile server process",
            exclusive=True,
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#llamafile-log-output"
            ),
        )
        app.notify("Llamafile server starting…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing to start Llamafile server: %s", err, exc_info=True)
        app.notify("Error setting up Llamafile server start.", severity="error")


###############################################################################
# ─── Llama.cpp UI helpers ────────────────────────────────────────────────────
###############################################################################


async def handle_llamacpp_browse_exec_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llama.cpp browse executable button pressed.")

    exec_filters = Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp executable (e.g. main, server.py)",
            filters=exec_filters,
        ),
        callback=_make_path_update_callback(app, "llamacpp-exec-path"),
    )


async def handle_llamacpp_browse_model_button_pressed(app: "TldwCli") -> None:
    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp Model (.gguf)",
            filters=gguf_filters,
        ),
        callback=_make_path_update_callback(app, "llamacpp-model-path"),
    )


async def handle_start_llamacpp_server_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start Llama.cpp server.")

    try:
        exec_path_input = app.query_one("#llamacpp-exec-path", Input)
        model_path_input = app.query_one("#llamacpp-model-path", Input)
        host_input = app.query_one("#llamacpp-host", Input)
        port_input = app.query_one("#llamacpp-port", Input)
        additional_args_input = app.query_one("#llamacpp-additional-args", Input)
        log_output_widget = app.query_one("#llamacpp-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8001"
        additional_args_str = additional_args_input.value.strip()

        if not exec_path:
            app.notify("Executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify(f"Executable not found at: {exec_path}", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify(f"Model file not found at: {model_path}", severity="error")
            model_path_input.focus()
            return

        command = [
            exec_path,
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            port,
        ]
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_llamacpp_server_worker,
            args=[app, command],
            group="llamacpp_server",
            description="Running Llama.cpp server process",
            exclusive=True,
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#llamacpp-log-output"
            ),
        )
        app.notify("Llama.cpp server starting…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing to start Llama.cpp server: %s", err, exc_info=True)
        app.notify("Error setting up Llama.cpp server start.", severity="error")


async def handle_stop_llamacpp_server_button_pressed(app: "TldwCli") -> None:
    """Placeholder for stopping the Llama.cpp server."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("Stop Llama.cpp server button pressed - functionality to be implemented.")
    app.notify("Stop server functionality is not yet implemented.", severity="warning")


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


###############################################################################
# ─── Model download UI helpers ──────────────────────────────────────────────
###############################################################################


async def handle_browse_models_dir_button_pressed(app: "TldwCli") -> None:
    """Open a directory picker so the user can choose the *models* directory."""

    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            select_dirs=True,
            title="Select models directory",
            filters=Filters(("All", lambda p: True)),
        ),
        callback=_make_path_update_callback(app, "models-dir-path"),
    )


async def handle_start_model_download_button_pressed(app: "TldwCli") -> None:
    """Validate inputs and launch *run_model_download_worker*."""

    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to download a model from Hugging Face.")

    try:
        models_dir_input = app.query_one("#models-dir-path", Input)
        url_or_repo_input = app.query_one("#model-url-or-repo", Input)
        revision_input = app.query_one("#model-revision", Input)
        log_output_widget = app.query_one("#model-download-log-output", RichLog)

        models_dir = models_dir_input.value.strip()
        url_or_repo = url_or_repo_input.value.strip()
        revision = revision_input.value.strip()

        if not models_dir:
            app.notify("Models directory is required.", severity="error")
            models_dir_input.focus()
            return
        if not Path(models_dir).exists():
            app.notify(f"Models directory not found: {models_dir}", severity="error")
            models_dir_input.focus()
            return
        if not url_or_repo:
            app.notify("Model URL or repo name is required.", severity="error")
            url_or_repo_input.focus()
            return

        command = [
            "huggingface-cli",
            "download",
            url_or_repo,
            "--local-dir",
            models_dir,
            "--local-dir-use-symlinks",  # don't duplicate files if they exist
        ]
        if revision:
            command.extend(["--revision", revision])

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_model_download_worker,
            args=[app, command],
            group="model_download",
            description="Downloading model via huggingface‑cli",
            exclusive=False,
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#model-download-log-output"
            ),
        )
        app.notify("Model download started…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing model download: %s", err, exc_info=True)
        app.notify("Error setting up model download.", severity="error")


###############################################################################
# ─── Ollama UI helpers ──────────────────────────────────────────────────────
###############################################################################


async def handle_ollama_nav_button_pressed(app: "TldwCli") -> None:
    """Handle the Ollama navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        view_areas = content_pane.query(".llm-view-area")

        for view in view_areas:
            if view.id:  # Only hide if it has an ID
                logger.debug(f"Hiding view #{view.id}")
                view.styles.display = "none"
            else: # pragma: no cover
                logger.warning("Found a .llm-view-area without an ID, not hiding it.")

        ollama_view = app.query_one("#llm-view-ollama", Container)
        logger.debug(f"Showing view #{ollama_view.id}")
        ollama_view.styles.display = "block"
        #app.notify("Switched to Ollama view.")

    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to Ollama view: Could not find required UI elements.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while switching to Ollama view.", severity="error")


###############################################################################
# ─── MLX-LM UI helpers ──────────────────────────────────────────────────────
###############################################################################


async def handle_mlx_lm_nav_button_pressed(app: "TldwCli") -> None:
    """Handle the MLX-LM navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("MLX-LM nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        view_areas = content_pane.query(".llm-view-area")

        for view in view_areas:
            if view.id:  # Only hide if it has an ID
                logger.debug(f"Hiding view #{view.id}")
                view.styles.display = "none"
            else: # pragma: no cover
                logger.warning("Found a .llm-view-area without an ID, not hiding it.")

        mlx_lm_view = app.query_one("#llm-view-mlx-lm", Container)
        logger.debug(f"Showing view #{mlx_lm_view.id}")
        mlx_lm_view.styles.display = "block"
        #app.notify("Switched to MLX-LM view.") # Optional: uncomment if you want a notification

        # Set initial button states when MLX-LM view is shown
        # This assumes TldwCli app has an attribute 'mlx_server_process' initialized to None
        if not hasattr(app, 'mlx_server_process'):
            app.mlx_server_process = None # Initialize if not present

        start_button = mlx_lm_view.query_one("#mlx-start-server-button", Button)
        stop_button = mlx_lm_view.query_one("#mlx-stop-server-button", Button)

        if app.mlx_server_process and app.mlx_server_process.poll() is None:
            # Server is likely running
            start_button.disabled = True
            stop_button.disabled = False
        else:
            # Server is not running or process object is stale
            start_button.disabled = False
            stop_button.disabled = True

    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_mlx_lm_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to MLX-LM view: Could not find required UI elements.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_mlx_lm_nav_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while switching to MLX-LM view.", severity="error")


async def handle_start_mlx_server_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start MLX-LM server.")

    # Ensure 'app' has 'mlx_server_process' attribute, initialized to None somewhere
    # This should ideally be done in TldwCli's __init__ or on_mount.
    if not hasattr(app, 'mlx_server_process'):
        app.mlx_server_process = None # Initialize if not present, though this is a fallback

    log_output_widget: Optional[RichLog] = None # Initialize for broader scope in case of early QueryError
    start_button: Optional[Button] = None
    stop_button: Optional[Button] = None

    try:
        # Querying through app, assuming #llm-view-mlx-lm is unique and globally accessible under app.screen
        # More robustly, one might pass LLMManagementWindow instance or query via a more specific parent.
        llm_mlx_view_container = app.query_one("#llm-view-mlx-lm", Container)

        model_path_input = llm_mlx_view_container.query_one("#mlx-model-path", Input)
        host_input = llm_mlx_view_container.query_one("#mlx-host", Input)
        port_input = llm_mlx_view_container.query_one("#mlx-port", Input)
        additional_args_area = llm_mlx_view_container.query_one("#mlx-additional-args", TextArea)
        log_output_widget = llm_mlx_view_container.query_one("#mlx-log-output", RichLog)
        start_button = llm_mlx_view_container.query_one("#mlx-start-server-button", Button)
        stop_button = llm_mlx_view_container.query_one("#mlx-stop-server-button", Button)

        model_path = model_path_input.value.strip()
        host = host_input.value.strip()
        port_str = port_input.value.strip()
        additional_args = additional_args_area.text.strip()

        log_output_widget.clear()

        if not model_path:
            log_output_widget.write("Error: MLX Model Path is required.")
            app.notify("MLX Model Path is required.", severity="error")
            model_path_input.focus()
            return

        if not host:
            log_output_widget.write("Error: Host is required.")
            app.notify("Host is required.", severity="error")
            host_input.focus()
            return

        if not port_str:
            log_output_widget.write("Error: Port is required.")
            app.notify("Port is required.", severity="error")
            port_input.focus()
            return
        try:
            port_val = int(port_str)
        except ValueError:
            log_output_widget.write("Error: Port must be a valid number.")
            app.notify("Port must be a valid number.", severity="error")
            port_input.focus()
            return

        if app.mlx_server_process and app.mlx_server_process.poll() is None:
            log_output_widget.write(f"MLX-LM server is already running (PID: {app.mlx_server_process.pid}).")
            app.notify("MLX-LM server is already running.", severity="warning")
            return

        log_output_widget.write(f"Attempting to start MLX-LM server with model: {model_path} on {host}:{port_val}...")

        server_process_instance = start_mlx_lm_server(model_path, host, port_val, additional_args)
        app.mlx_server_process = server_process_instance

        if app.mlx_server_process and app.mlx_server_process.poll() is None:
            log_output_widget.write(f"MLX-LM server process started successfully (PID: {app.mlx_server_process.pid}).")
            log_output_widget.write("Note: Full log streaming from the server is not yet implemented in this view. Check console if needed.")
            app.notify("MLX-LM server started.")
            start_button.disabled = True
            stop_button.disabled = False
        else:
            log_output_widget.write("Error: Failed to start MLX-LM server. Check application logs for details.")
            app.notify("Failed to start MLX-LM server.", severity="error")
            start_button.disabled = False # Ensure start button is re-enabled on failure
            stop_button.disabled = True

    except QueryError as e:
        logger.error(f"QueryError in handle_start_mlx_server_button_pressed: {e}", exc_info=True)
        if log_output_widget: log_output_widget.write(f"UI Error: Could not find required elements for MLX-LM: {e}")
        app.notify("Error accessing MLX-LM UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error starting MLX-LM server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")
        if start_button: start_button.disabled = False
        if stop_button: stop_button.disabled = True


async def handle_stop_mlx_server_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop MLX-LM server.")

    log_output_widget: Optional[RichLog] = None
    start_button: Optional[Button] = None
    stop_button: Optional[Button] = None

    try:
        llm_mlx_view_container = app.query_one("#llm-view-mlx-lm", Container)
        log_output_widget = llm_mlx_view_container.query_one("#mlx-log-output", RichLog)
        start_button = llm_mlx_view_container.query_one("#mlx-start-server-button", Button)
        stop_button = llm_mlx_view_container.query_one("#mlx-stop-server-button", Button)

        if hasattr(app, 'mlx_server_process') and app.mlx_server_process:
            if app.mlx_server_process.poll() is None: # Process is running
                log_output_widget.write(f"Stopping MLX-LM server (PID: {app.mlx_server_process.pid})...")
                stop_mlx_lm_server(app.mlx_server_process) # stop_mlx_lm_server handles terminate/kill/wait
                app.mlx_server_process = None # Clear the stored process
                log_output_widget.write("MLX-LM server stop command issued.")
                app.notify("MLX-LM server stopped.")
            else: # Process already terminated
                log_output_widget.write(f"MLX-LM server (PID: {app.mlx_server_process.pid}) was already stopped.")
                app.notify("MLX-LM server was already stopped.", severity="info")
                app.mlx_server_process = None # Clear the stale process object
        else:
            log_output_widget.write("MLX-LM server is not currently running or no process tracked.")
            app.notify("MLX-LM server is not running.", severity="warning")
            if hasattr(app, 'mlx_server_process'): # If attribute exists but is None
                 app.mlx_server_process = None

        start_button.disabled = False
        stop_button.disabled = True

    except QueryError as e:
        logger.error(f"QueryError in handle_stop_mlx_server_button_pressed: {e}", exc_info=True)
        if log_output_widget: log_output_widget.write(f"UI Error: Could not find required elements for MLX-LM: {e}")
        app.notify("Error accessing MLX-LM UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error stopping MLX-LM server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred while stopping the server: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")
        # Attempt to set buttons to a safe state, though server state is uncertain
        if start_button: start_button.disabled = False
        if stop_button: stop_button.disabled = True


#
# End of llm_management_events.py
########################################################################################################################
