# llm_management_events.py
#
# Imports
from __future__ import annotations

import functools
#
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
#
# Third-Party Libraries
from textual.containers import Container
from textual.css.query import QueryError
from textual.worker import Worker, WorkerState
from textual.widgets import Input, RichLog, TextArea, Button
#
# Local Imports
if TYPE_CHECKING:
    from ..app import TldwCli  # pragma: no cover – runtime import only
from ..Local_Inference.mlx_lm_inference_local import start_mlx_lm_server, stop_mlx_lm_server
# subprocess already imported
from ..Third_Party.textual_fspicker import FileOpen, Filters
#
########################################################################################################################
#
#
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
#


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
    # ─── Model download ───────────────────────────────────────────────────────
    "handle_browse_models_dir_button_pressed",
    "handle_start_model_download_button_pressed",
    # ─── MLX-LM ───────────────────────────────────────────────────────────────
    "handle_mlx_lm_nav_button_pressed",
    "handle_start_mlx_server_button_pressed",
    "handle_stop_mlx_server_button_pressed",
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
                    f"Updated input {input_widget_id} with path:{selected_path}"
                )
            except Exception as err:  # pragma: no cover – UI querying
                logger.error(
                    f"Error updating input #{input_widget_id}: {err}", input_widget_id, err, exc_info=True
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


# FIXME
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


def run_llamacpp_server_worker(app_instance: "TldwCli", command: List[str]) -> str:
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"Llama.cpp WORKER (communicate test) starting with command: {quoted_command}")

    process = None
    final_status_message = f"Llama.cpp WORKER (communicate test): Default status for {quoted_command}"

    try:
        logger.debug("Llama.cpp WORKER (communicate test): Attempting to start subprocess...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,  # text=True implies this, but good to be explicit
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        pid = process.pid if process else 'Unknown'
        logger.info(f"Llama.cpp WORKER (communicate test): Subprocess launched, PID: {pid}")
        app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                      f"[PID:{pid}] Llama.cpp server starting (communicate test)...\n")

        # communicate() will wait for the process to terminate
        # Set a timeout; a server should run longer, but if it exits quickly, we'll catch it.
        # If it's a *real* server, this timeout will be hit, and that's an error for this test.
        # For a server that's meant to run indefinitely, communicate() isn't the right final tool,
        # but it's good for diagnosing if it even starts and produces initial output or errors.
        timeout_seconds = 30  # Increased timeout to see if it's just slow to start/error

        try:
            stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
            logger.info(
                f"Llama.cpp WORKER (communicate test): communicate() completed. PID {pid}, Exit Code: {process.returncode}")

            if stdout_data:
                logger.info(f"Llama.cpp WORKER (communicate test) STDOUT:\n{stdout_data.strip()}")
                app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                              f"--- STDOUT (PID:{pid}) ---\n{stdout_data.strip()}\n")
            if stderr_data:
                logger.error(f"Llama.cpp WORKER (communicate test) STDERR:\n{stderr_data.strip()}")
                app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                              f"--- STDERR (PID:{pid}) ---\n[bold red]{stderr_data.strip()}[/]\n")

            if process.returncode != 0:
                final_status_message = f"Llama.cpp server (PID:{pid}) exited with ERROR code: {process.returncode}."
                if stderr_data: final_status_message += f"\nSTDERR: {stderr_data.strip()}"
            else:
                # If it exits with 0 but quickly, it's not acting like a persistent server.
                final_status_message = f"Llama.cpp server (PID:{pid}) exited with code: {process.returncode} (SUCCESS)."
                if stdout_data: final_status_message += f"\nSTDOUT: {stdout_data.strip()}"

            app_instance.call_from_thread(app_instance._update_llamacpp_log, f"{final_status_message}\n")
            return final_status_message

        except subprocess.TimeoutExpired:
            logger.warning(
                f"Llama.cpp WORKER (communicate test): Timeout ({timeout_seconds}s) expired for PID {pid}. Server is likely running or hung.")
            app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                          f"[PID:{pid}] Server process timed out during 'communicate' test (expected if it's a long-running server).\n")
            # If it times out, it means the server is running (which is good for a server).
            # For this *test*, we'll consider it a special kind of "success" meaning "it didn't crash immediately".
            # The actual streaming logic would then take over in a non-test scenario.
            # However, for run_worker, we need to return something.
            # To keep the worker alive for a real server, we would not use communicate() here.
            # This version of the worker is for diagnostics.
            # We'll kill it to ensure the worker function terminates.
            process.kill()
            logger.info(f"Llama.cpp WORKER (communicate test): Killed PID {pid} after timeout.")
            return f"Llama.cpp server (PID:{pid}) was still running after {timeout_seconds}s (killed for test). This indicates it started."

    except FileNotFoundError:
        msg = f"ERROR: Llama.cpp executable not found: {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(app_instance._update_llamacpp_log, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in Llama.cpp worker (communicate test): {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(app_instance._update_llamacpp_log, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"Llama.cpp WORKER (communicate test): Worker function for command '{quoted_command}' finishing.")
        if process and process.poll() is None:  # Should be terminated by communicate or kill
            logger.warning(
                f"Llama.cpp WORKER (communicate test): Process PID {process.pid if process else 'N/A'} still running in finally. Killing.")
            process.kill()


# FIXME
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

    # FIXME
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

        # MODIFICATION: Added thread=True
        # MODIFICATION: Ensured 'name' is not set to 'command' (it wasn't in the snippet, but for safety)
        app.run_worker(
            run_llamafile_server_worker,
            args=[app, command],
            group="llamafile_server",
            description="Running Llamafile server process",
            exclusive=True,
            thread=True,  # <--- ADDED THIS
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

        command: List[str] = [
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
        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n")

        worker_callable = functools.partial(run_llamacpp_server_worker, app, command)

        logger.debug(f"Type of 'app' object: {type(app)}")
        logger.debug(f"Bound 'app.run_worker' method: {app.run_worker}")
        logger.debug(f"Preparing to call app.run_worker for Llama.cpp with partial: {worker_callable}")

        app.run_worker(
            worker_callable,
            group="llamacpp_server",
            description="Running Llama.cpp server process",
            exclusive=True,
            thread=True
        )

        app.notify("Llama.cpp server starting…")
    except Exception as err:
        logger.error(f"Error preparing to start Llama.cpp server: {err}", exc_info=True)
        app.notify("Error setting up Llama.cpp server start.", severity="error")


async def handle_stop_llamacpp_server_button_pressed(app: "TldwCli") -> None:
    """Stops the Llama.cpp server process if it is running."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop Llama.cpp server.")

    log_output_widget = None
    try:
        log_output_widget = app.query_one("#llamacpp-log-output", RichLog)

        if hasattr(app, 'llamacpp_server_process') and \
           app.llamacpp_server_process and \
           app.llamacpp_server_process.poll() is None:
            pid = app.llamacpp_server_process.pid
            logger.info(f"Stopping Llama.cpp server (PID: {pid}).")
            log_output_widget.write(f"Stopping Llama.cpp server (PID: {pid})...\n")

            app.llamacpp_server_process.terminate()
            try:
                app.llamacpp_server_process.wait(timeout=5) # Wait for graceful termination
                logger.info("Llama.cpp server terminated.")
                log_output_widget.write("Llama.cpp server stopped successfully.\n")
                app.notify("Llama.cpp server stopped.")
                # Update button states
                start_button = app.query_one("#llamacpp-start-server-button", Button)
                stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                start_button.disabled = False
                stop_button.disabled = True
            except subprocess.TimeoutExpired:
                logger.warning(f"Llama.cpp server (PID: {pid}) did not terminate gracefully. Killing.")
                log_output_widget.write(f"Llama.cpp server (PID: {pid}) did not stop in time, killing...\n")
                app.llamacpp_server_process.kill() # Force kill
                app.llamacpp_server_process.wait() # Wait for kill to complete
                logger.info("Llama.cpp server killed.")
                log_output_widget.write("Llama.cpp server killed.\n")
                app.notify("Llama.cpp server killed.", severity="warning")
                # Update button states even after kill
                start_button = app.query_one("#llamacpp-start-server-button", Button)
                stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                start_button.disabled = False
                stop_button.disabled = True
            except Exception as e: # Catch other errors during wait/terminate like process already exited
                logger.error(f"Error during Llama.cpp server termination (PID: {pid}): {e}", exc_info=True)
                log_output_widget.write(f"Error stopping Llama.cpp server (PID: {pid}): {e}\n")
                app.notify(f"Error stopping Llama.cpp server: {e}", severity="error")
                # Attempt to reset buttons on error
                try:
                    start_button = app.query_one("#llamacpp-start-server-button", Button)
                    stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                    start_button.disabled = False
                    stop_button.disabled = True
                except QueryError as q_err: # pragma: no cover
                    logger.error(f"Failed to query buttons to reset state after termination error: {q_err}", exc_info=True)
        else:
            logger.info("Llama.cpp server is not running or process attribute is missing.")
            log_output_widget.write("Llama.cpp server is not running or was already stopped.\n")
            app.notify("Llama.cpp server is not running.", severity="warning")
            # Update button states
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True

    except QueryError as e: # pragma: no cover
        logger.error(f"Could not find #llamacpp-log-output or a button: {e}", exc_info=True)
        app.notify("Error: UI widget not found during stop operation.", severity="error")
        # Attempt to reset buttons even if log widget is missing
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True
        except QueryError as q_err: # pragma: no cover
             logger.error(f"Failed to query buttons to reset state after QueryError: {q_err}", exc_info=True)
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Error stopping Llama.cpp server: {e}", exc_info=True)
        if log_output_widget: # Check if log_widget was found before error
            log_output_widget.write(f"An unexpected error occurred while stopping the server: {e}\n")
        app.notify(f"An unexpected error occurred: {e}", severity="error")
        # Attempt to reset buttons on generic error
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True
        except QueryError as q_err: # pragma: no cover
            logger.error(f"Failed to query buttons to reset state after generic error: {q_err}", exc_info=True)
    finally:
        if hasattr(app, 'llamacpp_server_process'):
            app.llamacpp_server_process = None
        logger.debug("Set app.llamacpp_server_process to None.")
        # Final attempt to ensure buttons are in a consistent state
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            # If server process is None here, it means it's stopped or was never started.
            is_running = hasattr(app, 'llamacpp_server_process') and \
                         app.llamacpp_server_process and \
                         app.llamacpp_server_process.poll() is None

            start_button.disabled = is_running
            stop_button.disabled = not is_running
        except QueryError as q_err: # pragma: no cover
            logger.error(f"Failed to query buttons in finally block: {q_err}", exc_info=True)


###############################################################################
# ─── Model download UI helpers ──────────────────────────────────────────────
###############################################################################


async def handle_browse_models_dir_button_pressed(app: "TldwCli") -> None:
    """Open a directory picker so the user can choose the *models* directory."""

    # FIXME
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
    logger.info("User requested to download a model from Hugging Face.")

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
        if not Path(models_dir).exists(): # Should be is_dir()
            app.notify(f"Models directory not found or is not a directory: {models_dir}", severity="error")
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
            str(Path(models_dir) / url_or_repo.split("/")[-1]), # Download into a subfolder
            "--local-dir-use-symlinks",
            "False", # Explicitly False for actual copy
        ]
        if revision:
            command.extend(["--revision", revision])

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_model_download_worker,
            args=[app, command],
            group="model_download",
            description="Downloading model via huggingface-cli",
            exclusive=False, # Can run multiple downloads
            thread=True,  # <--- ADDED THIS
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#model-download-log-output"
            ),
        )
        app.notify("Model download started…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing model download: %s", err, exc_info=True)
        app.notify("Error setting up model download.", severity="error")






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

        # FIXME
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

        # FIXME
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
                app.notify("MLX-LM server was already stopped.", severity="information")
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
