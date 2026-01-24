import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("local_nexus.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LocalNexus")

def log_info(message: str):
    logger.info(message)

def log_error(message: str):
    logger.error(message)
