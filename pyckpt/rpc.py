import asyncio
from asyncio import StreamReader, StreamWriter, Task
from concurrent.futures import Executor, ThreadPoolExecutor
import logging
from threading import Event, Thread
from typing import Generic, NamedTuple, Optional, Sequence, TypeVar

import msgpack

logger = logging.getLogger(__name__)

T = TypeVar("T")

DATA_BLOCK_SIZE = 1024**2


class Request(NamedTuple):
    method_name: str
    args: tuple


class ReturnValue(NamedTuple, Generic[T]):
    value: T


class Server(Generic[T]):
    def __init__(self, dispatcher: T, executor: Optional[Executor] = None):
        self._dispatcher = dispatcher
        self._event_loop = asyncio.new_event_loop()
        if executor is None:
            executor = ThreadPoolExecutor(1)
        self._event_loop.set_default_executor(executor)
        self._server: Optional[asyncio.Server] = None
        self._worker_thread: Optional[Thread] = None

        self._start_event = Event()
        self._stop_event: Optional[asyncio.Event] = None
        self._running_serves: set[Task] = set()

    async def _handle_request(self, request: Request):
        logger.debug("server handle request: %s", request)
        value = await asyncio.get_running_loop().run_in_executor(
            None,
            getattr(self._dispatcher, request.method_name),
            *request.args,
        )
        return value

    async def _serve(self, reader: StreamReader, writer: StreamWriter):
        self._running_serves.add(asyncio.current_task())
        packer = msgpack.Packer()
        unpacker = msgpack.Unpacker(use_list=False)
        peer_name = (writer.get_extra_info("peername"),)
        logger.debug("server receive incoming connection: %s", peer_name)
        try:
            while True:
                logger.debug("server wait read...")
                request_data = await reader.read(DATA_BLOCK_SIZE)
                if len(request_data) == 0:
                    writer.close()
                    return
                unpacker.feed(request_data)
                for req in unpacker:
                    return_value = await self._handle_request(Request(*req))
                    logger.debug("server send response: %s", return_value)
                    writer.write(packer.pack(ReturnValue(return_value)))
                    logger.debug("server wait write...")
                    await writer.drain()
        except Exception as exception:
            logger.error(
                "server connection %s broke with exception %s", peer_name, exception
            )
            writer.close()
            return

    async def _start_serve(self, host: str | Sequence[str], port: int):
        self._server = await asyncio.start_server(self._serve, host, port)
        logger.info("start listening on %s:%s", host, port)
        self._start_event.set()
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()
        self._server.close()
        self._stop_event = None

    def stop(self):
        async def _stop_serve():
            logger.debug("stopping server...")
            for serves in self._running_serves:
                try:
                    await serves
                except Exception as e:
                    logger.error("server serve error: %s", e)
            logger.debug("server stopped.")
            self._stop_event.set()

        future = asyncio.run_coroutine_threadsafe(_stop_serve(), self._event_loop)
        future.result()
        self._worker_thread.join()
        self._worker_thread = None

    def start(self, host: str | Sequence[str], port: int):
        self._worker_thread = Thread(
            target=self._event_loop.run_until_complete,
            args=(self._start_serve(host, port),),
        )
        self._worker_thread.start()
        self._start_event.wait()


class Client:
    def __init__(self):
        self._reader: Optional[StreamReader] = None
        self._writer: Optional[StreamWriter] = None
        self._packer = msgpack.Packer()
        self._unpacker = msgpack.Unpacker(use_list=False)

    async def connect_async(self, host: str, port: int):
        logger.debug("client start connecting to %s:%s", host, port)
        self._reader, self._writer = await asyncio.open_connection(host, port)
        logger.debug("client connected(%s:%s)", host, port)

    async def call_async(self, method_name: str, *args):
        request = Request(method_name, args)
        logger.debug("client send request: %s", request)
        data = self._packer.pack(request)
        self._writer.write(data)
        while True:
            response = await self._reader.read(DATA_BLOCK_SIZE)
            self._unpacker.feed(response)
            for response in self._unpacker:
                response = ReturnValue(*response).value
                logger.debug("client receive repose: %s", response)
                return response

    def close(self):
        logger.debug("client closing connection...")
        self._writer.write_eof()
        self._writer.close()
        logger.debug("client connection closed.")

    def connect(self, host: str, port: int):
        asyncio.get_event_loop().run_until_complete(
            self.connect_async(host, port),
        )

    def call(self, method_name: str, *args):
        if self._reader is None:
            raise ValueError("connect client first")
        return asyncio.get_event_loop().run_until_complete(
            self.call_async(method_name, *args)
        )
