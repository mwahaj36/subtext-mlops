import os
import socket

import pytest

# Never initialize production resources, even on a developer machine with .env.
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = ""
os.environ["ASTRA_DB_API_ENDPOINT"] = ""
os.environ["DATABASE_URL"] = ""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original_connect = socket.socket.connect

    def connect(sock, address):
        # Windows asyncio builds its wakeup pipe with a loopback socket pair.
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return original_connect(sock, address)
        raise AssertionError("Unit tests must not use the network")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unit tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", forbidden)
