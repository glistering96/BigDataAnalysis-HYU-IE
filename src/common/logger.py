"""_summary_

    logging module.
    
    Create an instance of logger and add handlers to it.
    The logger instance is named 'logger'.
    
    Any other modules that need to log should import this module and use the logger instance.
    
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path


BASEDIR = str(Path(__file__).parent.parent.parent)


def init_logger(config):
    # create logger
    level = config['level']
    
    logger = logging.getLogger('logger')
    logger.setLevel(level)
    
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()

    # add formatter to ch
    ch.setFormatter(formatter)
    
    # add ch to logger
    logger.addHandler(ch)
    
    # file handler
    file_nm = config['name']
    cur_time = time.strftime("%Y%m%d-%H%M")
    log_path = Path(f'{BASEDIR}/{file_nm}/logs')
    
    if '.txt' not in file_nm:
        file_nm = f'{file_nm}-{cur_time}.txt'
    
    
    if not log_path.exists():
        log_path.mkdir(parents=True)
        
    fh = logging.FileHandler(f"{log_path}/{file_nm}")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
    