# separate script to handle logging over the project  - - - > [ GLOBAL LOGGING ]

import logging
import os


# checks if the directory exists
os.makedirs("logs",exist_ok=True)

# global logger function
def get_logger(name, logfile):
  try:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

       # duplicate prevent in file logging
        if not logger.handlers:
           file_handler = logging.FileHandler(logfile) # saves logfile
           stream_handler = logging.StreamHandler() # show in terminal

           formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
           file_handler.setFormatter(formatter)
           stream_handler.setFormatter(formatter)

           logger.addHandler(file_handler)
           logger.addHandler(stream_handler)

        return logger

  except Exception as e:
      logging.error(f'Something went wrong: {e}')
      return None

