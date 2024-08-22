import os
import sys
import logging
from datetime import datetime

# Define the logging format
logging_str = "[%(asctime)s - %(levelname)s - %(module)s - %(message)s]"

# Generate the log file name with the current date and time
LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"

# Define the directory where logs will be stored
log_dir = os.path.join(os.getcwd(), "LOGS")

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Define the full log file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger with a specific name
logger = logging.getLogger("AUTOENCODERLOGGER")
