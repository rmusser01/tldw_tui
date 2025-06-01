# mlx_lm_inference_local.py
#
# Imports
import logging
import os
import subprocess
from typing import Optional
#
# Third-party Libraries
#
# Local Imports
#
#
########################################################################################################################
#
# Functions:

def start_mlx_lm_server(
    model_path: str,
    host: str,
    port: int,
    additional_args: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """
    Starts the MLX LM server using subprocess.Popen.

    Args:
        model_path: Path to the MLX model (HuggingFace ID or local path).
        host: Host address for the server.
        port: Port for the server.
        additional_args: Optional string of additional arguments for the server command.

    Returns:
        A subprocess.Popen object if successful, None otherwise.
    """
    command = [
        "python", "-m", "mlx_lm.server",
        "--model", model_path,
        "--host", host,
        "--port", str(port)
    ]
    if additional_args:
        command.extend(additional_args.split())

    logging.info(f"Starting MLX-LM server with command: {' '.join(command)}")
    try:
        # Set environment variable to disable output buffering for Python
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True, # Ensure text mode for stdout/stderr
            env=env
        )
        logging.info(f"MLX-LM server started with PID: {process.pid}")
        return process
    except FileNotFoundError:
        logging.error(
            "Error starting MLX-LM server: 'python' or 'mlx_lm.server' not found. "
            "Ensure Python is installed and mlx-lm is in your Python path (pip install mlx-lm)."
        )
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while starting MLX-LM server: {e}", exc_info=True)
        return None

def stop_mlx_lm_server(process: subprocess.Popen) -> None:
    """
    Stops the MLX LM server process.

    Args:
        process: The subprocess.Popen object representing the server process.
    """
    if not process:
        logging.warning("Stop MLX-LM server: No process provided.")
        return

    if process.poll() is not None:
        logging.info(f"MLX-LM server (PID: {process.pid}) already terminated with code {process.returncode}.")
        return

    logging.info(f"Stopping MLX-LM server with PID: {process.pid}...")
    try:
        process.terminate()
        try:
            process.wait(timeout=10) # Increased timeout for graceful shutdown
            logging.info(f"MLX-LM server (PID: {process.pid}) terminated gracefully with code {process.returncode}.")
        except subprocess.TimeoutExpired:
            logging.warning(
                f"MLX-LM server (PID: {process.pid}) did not terminate gracefully within timeout. Killing..."
            )
            process.kill()
            try:
                process.wait(timeout=5) # Wait for kill
                logging.info(f"MLX-LM server (PID: {process.pid}) killed, return code {process.returncode}.")
            except subprocess.TimeoutExpired:
                logging.error(f"MLX-LM server (PID: {process.pid}) did not die even after kill. Manual intervention may be needed.")
        except Exception as e_wait: # Catch other errors during wait (e.g. InterruptedError)
            logging.error(f"Error waiting for MLX-LM server (PID: {process.pid}) to terminate: {e_wait}", exc_info=True)


    except ProcessLookupError: # If the process was already gone
        logging.info(f"MLX-LM server (PID: {process.pid}) was already gone before explicit stop.")
    except Exception as e_term:
        logging.error(f"Error during initial termination of MLX-LM server (PID: {process.pid}): {e_term}", exc_info=True)
        # If terminate fails, try to kill as a fallback if it's still running
        if process.poll() is None:
            logging.info(f"Attempting to kill MLX-LM server (PID: {process.pid}) as terminate failed.")
            process.kill()
            try:
                process.wait(timeout=5)
                logging.info(f"MLX-LM server (PID: {process.pid}) killed after terminate failed, return code {process.returncode}.")
            except Exception as e_kill_wait:
                 logging.error(f"Error waiting for MLX-LM server (PID: {process.pid}) to die after kill: {e_kill_wait}", exc_info=True)

#
# End of mlx_lm_inference_local.py
########################################################################################################################
