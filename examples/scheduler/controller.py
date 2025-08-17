import time
import threading
from scheduler import worker
import util
import queue
import json
import socket


class Controller(object):
    def __init__(self, port: int, num_workers: int) -> None:
        super().__init__()

        self._logger = util.make_logger(__name__)

        self._num_workers = num_workers
        self._workers = []  # worker_addr: (ip, port)
        self.done_queue = queue.Queue()

        self._server = threading.Thread(
            target=self.controller_server, args=(port,), damen=True
        )
        self._server.start()

        self.wait_for_workers()
        self._jump_time = 0
        self._start_time = time.time()

    def set_start_time(self):
        self._start_time = time.time()

    def get_time(self):
        return time.time() + self._jump_time - self._start_time

    def controller_server(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", port))
        s.listen(10)

        while True:
            conn, _ = s.accept()
            threading.Thread(target=self.handle_worker, args=(conn,)).start()

    def handle_worker(self, conn):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                msg = json.loads(data.decode())
                cmd = msg.get("cmd")

                if cmd == "register_worker":
                    worker_ip = msg["worker_ip"]
                    worker_port = msg["worker_port"]
                    worker_id = len(self._workers)
                    self._workers.append((worker_ip, worker_port))
                    conn.sendall(
                        json.dumps({"status": "ok", "worker_id": worker_id}).encode()
                    )

                elif cmd == "done":
                    job_id = msg["job_id"]
                    job_counter = msg["job_counter"]
                    worker_id = msg["worker_id"]
                    gpus = msg["gpus"]
                    returncode = msg["returncode"]
                    self.done_queue.put(
                        (self.get_time(), job_id, worker_id, gpus, returncode)
                    )
                    self._logger.info(
                        f"controller, done, {worker_id}, {job_id} - {job_counter} @ {worker_id}, {gpus}, return code: {returncode}"
                    )
                    conn.sendall(json.dumps({"status": "ok"}).encode())

        except Exception as e:
            print(f"Worker connection error: {e}")
        finally:
            conn.close()

    def wait_for_workers(self):
        while len(self._workers) < self._num_workers:
            time.sleep(5)

    def kill_workers(self):
        for i in range(len(self._workers)):
            self.exit_command(i)

    def send_msg(self, worker_id, msg):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self._workers[worker_id])
            s.sendall(json.dumps(msg).encode())
            reply = s.recv(4096)
        return json.loads(reply.decode())

    def execute(self, job_info):
        node_id_list = [node[id] for node in job_info["placements"]["nodes"]]
        worker_id = min(list(node_id_list))
        reply = self.send_msg(worker_id, {"cmd": "execute", "job_info": job_info})
        return reply.get("status") == "ok"

    def kill(self, job_info):
        node_id_list = [node[id] for node in job_info["placements"]["nodes"]]
        worker_id = min(list(node_id_list))
        reply = self.send_msg(worker_id, {"cmd": "kill", "job_info": job_info})
        return reply.get("status") == "ok"

    def exit_command(self, worker_id):
        reply = self.send_msg(worker_id, {"cmd": "exit"})
        return reply.get("status") == "ok"
