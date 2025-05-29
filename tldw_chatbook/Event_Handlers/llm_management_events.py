import logging
import subprocess  # For running external processes
from typing import TYPE_CHECKING, List, Optional
from pathlib import Path

from textual.css.query import QueryError
from textual.widgets import Input, TextArea, RichLog
from textual.worker import Worker, WorkerState

if TYPE_CHECKING:
    from ..app import TldwCli  # Assuming TldwCli is in app.py at .. level


# Worker function to run the Llamafile server
def run_llamafile_server_worker(command: List[str]):  # Not async
    logger = logging.getLogger(__name__)
    logger.info(f"Worker starting Llamafile with command: {' '.join(command)}")
    try:
        # Ensure command[0] (executable path) is absolute or resolvable
        # Path(command[0]).resolve(strict=True) # Optional: resolve path strictly before Popen

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True  # Ensures text=True behavior
        )

        # Yield a message that the process has started
        yield f"Llamafile process started (PID: {process.pid})...\n"

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                yield line  # Send line back to main thread
            process.stdout.close()

        process.wait()
        yield f"Llamafile server exited with code: {process.returncode}\n"

    except FileNotFoundError:
        error_msg = f"ERROR: Llamafile executable not found at: {command[0]}\n"
        logger.error(error_msg.strip())
        yield error_msg
    except Exception as e:
        error_msg = f"ERROR in Llamafile worker: {e}\n"
        logger.error(f"Error in Llamafile worker: {e}", exc_info=True)
        yield error_msg


async def stream_llamafile_worker_output_to_log(app: 'TldwCli', worker: Worker):
    logger = getattr(app, 'loguru_logger', logging)
    log_widget_id = "#llamafile-log-output"  # Specific to Llamafile

    try:
        log_widget = app.query_one(log_widget_id, RichLog)

        # Process lines already buffered in worker.output if any
        # This is a simplified way to get output. Textual might offer more robust ways
        # for continuous streaming for long-running workers.
        # output_lines = [] # Not strictly needed if writing line by line
        if worker.state == WorkerState.RUNNING or worker.state == WorkerState.SUCCESS:
            lines_processed_now = 0
            async for line in worker.output:  # Consume the AsyncIterator
                if isinstance(line, str):
                    log_widget.write(line.strip())  # Write line by line
                else:
                    log_widget.write(str(line).strip())  # Convert non-str to str
                lines_processed_now += 1
            if lines_processed_now > 0:
                logger.debug(f"Streamed {lines_processed_now} lines from worker {worker.name} to {log_widget_id}")

        if worker.state == WorkerState.SUCCESS:
            log_widget.write(f"--- Llamafile worker {worker.name} finished successfully ---")
        elif worker.state == WorkerState.ERROR:
            log_widget.write(f"--- Llamafile worker {worker.name} failed ---")
            # Error messages should have been yielded by the worker itself.

    except QueryError:
        logger.error(f"Failed to find RichLog widget {log_widget_id} for worker {worker.name}")
    except Exception as e:
        logger.error(f"Error streaming output for worker {worker.name} to {log_widget_id}: {e}", exc_info=True)


async def handle_start_llamafile_server_button_pressed(app: 'TldwCli') -> None:
    logger = getattr(app, 'loguru_logger', logging)
    logger.info("Attempting to start Llamafile server...")

    try:
        exec_path_input = app.query_one("#llamafile-exec-path", Input)
        model_path_input = app.query_one("#llamafile-model-path", Input)
        host_input = app.query_one("#llamafile-host", Input)
        port_input = app.query_one("#llamafile-port", Input)
        additional_args_input = app.query_one("#llamafile-additional-args", TextArea)
        log_output_widget = app.query_one("#llamafile-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        model_path = model_path_input.value.strip()
        host = host_input.value.strip()
        port = port_input.value.strip()
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

        log_output_widget.clear()
        log_output_widget.write(f"Starting Llamafile server with model: {model_path}...")
        log_output_widget.write(f"Executable: {exec_path}")

        command = [
            exec_path,
            "-m", model_path,
            "--host", host,
            "--port", port,
            # Add other necessary default llamafile arguments if any
        ]
        if additional_args_str:
            # Basic split for additional args, consider shlex for more robustness if needed
            command.extend(additional_args_str.split())

        logger.info(f"Llamafile command: {' '.join(command)}")

        log_output_widget.write(f"Executing: {' '.join(command)}\n")  # Log the command being run

        # Make sure to pass the command list to the worker
        app.run_worker(
            run_llamafile_server_worker,  # The new worker function
            args=[command.copy()],  # Pass a copy of the command list as args
            group="llamafile_server",  # Group for potential management
            description="Running Llamafile server process.",
            exclusive=True  # Only one Llamafile server at a time from this UI
        )
        app.notify("Llamafile server starting...")

    except Exception as e:
        logger.error(f"Error preparing to start Llamafile server: {e}", exc_info=True)
        app.notify("Error setting up Llamafile server start.", severity="error")

#
# End of llm_management_events.py
########################################################################################################################
