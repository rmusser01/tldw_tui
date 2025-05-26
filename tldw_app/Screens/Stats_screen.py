# Metrics_Screen.py
#
# Description: Screen for displaying metrics.
#
# Imports
import logging
from pathlib import Path
from textual.widgets import Static, Label
from textual.containers import VerticalScroll
#
########################################################################################################################
#
# Functions:

METRICS_LOG_PATH = Path("/tmp/app_metrics.log")

class StatsScreen(Static):
    """
    A screen to display application metrics from a log file.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics: dict[str, str] = {}
        logging.info("MetricsScreen initialized. Log path: %s", METRICS_LOG_PATH)

    def on_mount(self):
        """Load metrics when the screen is mounted."""
        logging.info("MetricsScreen on_mount: Calling load_metrics.")
        self.load_metrics()
        # After loading, we need to recompose or update the display.
        # Textual handles reactive updates well, but for initial load like this,
        # we might need to explicitly tell it to refresh if compose relies on data
        # that wasn't ready at the initial compose call.
        # However, since on_mount happens before the first paint after adding to DOM,
        # the compose method called subsequently should have the data.
        # If not, we might need self.refresh() or re-query and update.

    def load_metrics(self):
        """
        Loads metrics from the predefined log file.
        Parses key-value pairs and updates self.metrics.
        Handles FileNotFoundError and parsing errors.
        """
        logging.info("Attempting to load metrics from: %s", METRICS_LOG_PATH)
        self.metrics = {}  # Reset metrics before loading

        try:
            with open(METRICS_LOG_PATH, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):  # Skip empty lines and comments
                        continue
                    try:
                        key, value = line.split("=", 1)
                        self.metrics[key.strip()] = value.strip()
                    except ValueError:
                        logging.warning(
                            "MetricsScreen: Malformed line #%d in %s: '%s'. Skipping.",
                            i + 1, METRICS_LOG_PATH, line
                        )
            logging.info("Successfully loaded %d metrics.", len(self.metrics))
            if not self.metrics:
                logging.info("Metrics file was empty or contained no valid data.")
                self.metrics = {"info": "Metrics file is empty or contains no valid data."}

        except FileNotFoundError:
            logging.error("MetricsScreen: Log file not found at %s.", METRICS_LOG_PATH)
            self.metrics = {"error": f"Metrics log file not found: {METRICS_LOG_PATH.name}"}
        except Exception as e:
            logging.exception("MetricsScreen: An unexpected error occurred while loading metrics.")
            self.metrics = {"error": f"An unexpected error occurred: {e}"}

    def compose(self):
        """Create child widgets for the screen based on loaded metrics."""
        logging.info("MetricsScreen composing. Current metrics: %s", self.metrics)
        with VerticalScroll(id="metrics-container"):
            if not self.metrics: # Should not happen if load_metrics sets a message for empty/error
                logging.warning("MetricsScreen compose: self.metrics is unexpectedly empty at compose time.")
                yield Label("No metrics loaded or an error occurred. Check logs.")
            elif "error" in self.metrics:
                yield Label(f"[bold red]Error loading metrics:[/]\n{self.metrics['error']}")
            elif "info" in self.metrics: # For specific info like "file empty"
                 yield Label(f"[italic]{self.metrics['info']}[/]")
            elif not self.metrics: # Truly empty after load_metrics somehow (fallback)
                 yield Label("No metrics available.")
            else:
                yield Label("[b u]Application Metrics[/b u]\n")
                for key, value in self.metrics.items():
                    # Simple formatting for now
                    display_key = key.replace("_", " ").capitalize()
                    yield Label(f"[b]{display_key}:[/b] {value}")
        logging.info("MetricsScreen compose finished.")

#
#
# End of Metrics_Screen.py
########################################################################################################################
