from typing import Optional
from pyckpt import rpc as rpc
from tests.test_task import find_free_port


class EchoServer:
    def echo(self, msg: str) -> str:
        return msg

class SumServer(object):
    def sum(self, x, y):
        return x + y

    def raise_exception(self):
        raise RuntimeError("raise_exception")


echo_server: Optional[rpc.Server] = None
echo_server_port: Optional[int] = None

sum_server: Optional[rpc.Server] = None
sum_server_port: Optional[int] = None


def setup_module():
    global echo_server
    global echo_server_port
    global sum_server
    global sum_server_port

    echo_server = rpc.Server(EchoServer())
    echo_server_port = find_free_port()
    echo_server.start("localhost", echo_server_port)

    sum_server = rpc.Server(SumServer())
    sum_server_port = find_free_port()
    sum_server.start("localhost", sum_server_port)

def teardown_module():
    echo_server.stop()
    sum_server.stop()


def test_rpc_echo():
    client = rpc.Client()
    client.connect("localhost", echo_server_port)

    msg =  "hello world"
    ret = client.call("echo", msg)
    assert isinstance(ret, str)
    client.close()

def test_rpc_sum():
    client = rpc.Client()
    client.connect("localhost", sum_server_port)
    assert client.call("sum", 1, 2) == 3
    client.close()
