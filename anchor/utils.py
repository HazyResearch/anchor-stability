"""
Utils file.
"""

import logging
import time
from datetime import timedelta


def load_vocab_list(vocab_file):
    vocab = []
    fin = open(vocab_file, "r", encoding="utf-8")
    for line in fin:
        try:
            w, _ = line.rstrip().split(' ')
            vocab.append(w)
        except:
            print(line)
    return vocab

def load_vocab(vocab_file):
    vocab = {}
    fin = open(vocab_file, "r", encoding="utf-8")
    for line in fin:
        w, count = line.rstrip().split(' ')
        vocab[w] = int(count)
    return vocab


# https://github.com/facebookresearch/DME
class LogFormatter(object):
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


# https://github.com/facebookresearch/DME
def create_logger(filepath):
    """
    Create a logger.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
