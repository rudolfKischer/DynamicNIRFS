import logging
import io
import sys
import os
# Logging configuration

# get parent file directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging_file_name = 'simulation.log'
logging_file_path = os.path.join(parent_dir, logging_file_name)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(filename)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    handlers=[
                        logging.FileHandler(logging_file_name),
                      logging.StreamHandler()
                    ]
)

logger = logging.getLogger(__name__)


class BlenderLogInterCeptor(io.StringIO):
    def __init__(self):
        super().__init__()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    
    def write(self, message):
        if message.strip():
            logger.info(f'BLENDER [{message.strip()}]')
    
    def flush(self):
        pass
    
    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().close()