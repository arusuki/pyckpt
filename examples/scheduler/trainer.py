import socket
import threading
import json
import time
import util


class Trainer(object):
    def __init__(self, scheduler_ip, scheduler_port, trainer_ip, trainer_port, job_id):
        self._trainer_ip = trainer_ip
        self._trainer_port = trainer_port
        self._scheduler_ip = scheduler_ip
        self._scheduler_port = scheduler_port
        self._job_id = job_id

        self._logger = util.make_logger(__name__)
        self._start_time = time.time()
        self._finished_iteraions = 0

        self._server = threading.Thread(
            target=self.trainer_server, args=(trainer_port,), daemon=True
        )
        self._server.start()
        self.register()

        self._logger.info(f"job {self._job_id}, trainer, start, {self._start_time}")

    def trainer_server(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", port))
        s.listen(10)
        self._logger.info(f"Trainer socket server listening on port {port}")
        while True:
            conn, _ = s.accept()
            threading.Thread(target=self.handle_trainer, args=(conn,)).start()

    def register(self):
        success = False
        while not success:
            try:
                self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._server.connect((self._scheduler_ip, self._scheduler_port))
                msg = {
                    "cmd": "register_trainer",
                    "trainer_ip": self._trainer_ip,
                    "trainer_port": self._trainer_port,
                    "job_id_list": [self._job_id],
                }
                self._server.sendall(json.dumps(msg).encode())
                reply = self._server.recv(4096)
                if json.loads(reply.decode()).get("status") == "ok":
                    self._logger.info(
                        f"trainer, register, {self._job_id}, {self._trainer_ip}:{self._trainer_port}"
                    )
                    success = True
            except Exception:
                time.sleep(5)

    def handle_trainer(self, conn):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                msg = json.loads(data.decode())
                cmd = msg.get("cmd")
                if cmd == "query_stats":
                    reply = {
                        "finished_iterations": self._finished_iteraions,
                    }
                    conn.sendall(json.dumps(reply).encode())
        finally:
            conn.close()

    def report_itertime(self, iter_time):
        msg = {
            "cmd": "report_itertime",
            "job_id": [self._job_id],
            "iter_time": iter_time,
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._scheduler_ip, self._scheduler_port))
            s.sendall(json.dumps(msg).encode())
            reply = s.recv(4096)
            if json.loads(reply.decode()).get("status") == "ok":
                self._logger.info(
                    f"job {self._job_id} reported iteration time {iter_time}"
                )
            else:
                self._logger.error(
                    f"Failed to report iteration time for job {self._job_id}"
                )
