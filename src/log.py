from typing import Text, TextIO, Optional

import sys

from colorama import Fore

LOG_LEVEL = 1
PROC_NAME: Optional[Text] = None

def _output(level: int, fp: TextIO, header: Text, color: Text, message: object):
    if level < LOG_LEVEL:
        return

    proc_name = '' if PROC_NAME is None else f'/"{PROC_NAME}"'
    if fp.isatty():
        fp.write(f'{color}({header}{proc_name}){Fore.RESET} {message}\n')
    else:
        fp.write(f'({header}{proc_name}) {message}\n')
    fp.flush()

def debug(message: object):
    _output(0, sys.stdout, 'debug', Fore.GREEN, message)

def info(message: object):
    _output(1, sys.stdout, 'info', Fore.BLUE, message)

def warn(message: object):
    _output(2, sys.stderr, 'warn', Fore.YELLOW, message)

def error(message: object):
    _output(3, sys.stderr, 'ERROR', Fore.RED, message)

def fatal(message: object, returncode: Optional[int] = -1):
    _output(999, sys.stderr, 'FATAL', Fore.RED, message)
    exit(returncode)