import os
import socket
from contextlib import closing

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(BASEDIR, "database.db")


def get_available_port():
    """Returns a port that's not being used"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


HOST = "127.0.0.1"
PORT = get_available_port()


class Configuration:
    SECRET_KEY = "37ce401419a7b05582cd19ee14ad958c5b05be9dcd11dac020ed495f254f0c70"
