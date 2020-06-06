import sys
from typing import Text, TextIO, Optional

from colorama import Fore

LOG_LEVEL = 1

def _output(level: int, fp: TextIO, header: Text, color: Text, message: object):
    if level < LOG_LEVEL:
        return
    if fp.isatty():
        fp.write(f'{color}({header}){Fore.RESET} {message}\n')
    else:
        fp.write(f'({header}) {message}\n')

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