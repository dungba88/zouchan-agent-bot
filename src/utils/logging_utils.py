import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Set log level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[logging.StreamHandler()],  # Log to the console (stdout)
    )
