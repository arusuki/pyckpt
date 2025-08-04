import util
import threading
import time
import os
import socket
import json

from task import Task


class Worker(object):
    def __init__(
        self,
        master_ip,
        master_port,
        worker_ip,
        worker_port,
        gpus: str,
        trace_name,
        this_dir,
    ) -> None:
        super().__init__()

        self._logger = util.make_logger(__name__)

        self._master_ip = master_ip
        self._master_port = master_port
        self._work_ip = worker_ip
        self._worker_port = worker_port
        self._server = threading.Thread(
            target=self.worker_server, args=(self._worker_port,), daemon=True
        )
        self._server.start()

        self._worker_id = None
        self._trace_name = trace_name
        self._this_dir = this_dir
        self._check_task_flag = True
        self._gpus = gpus.split(",")
        self._num_gpus = len(self._gpus)
        self.register()

        self._tasks = dict()

    def worker_server(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", port))
        s.listen(10)
        self._logger.info(f"Worker socket server listening on port {port}")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=self.handle_controller, args=(conn,)).start()

    def register(self):
        success = False
        while not success:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self._master_ip, self._master_port))
                    msg = {
                        "cmd": "register_worker",
                        "worker_ip": self._work_ip,
                        "worker_port": self._worker_port,
                        "num_gpus": self._num_gpus,
                    }
                    s.sendall(json.dumps(msg).encode())
                    reply = s.recv(4096)
                    reply = json.loads(reply.decode())
                    if reply.get("status") == "ok":
                        self._worker_id = reply.get("worker_id")
                        self._logger.info(
                            f"worker registered: {self._worker_id} {self._work_ip}:{self._worker_port}"
                        )
                        success = True
            except Exception as e:
                self._logger.info(f"register failed, retrying... {e}")
                time.sleep(5)

    def handle_controller(self, conn):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    return
                msg = json.loads(data.decode())
                cmd = msg.get("cmd")

                if cmd == "execute":
                    job_info = msg["job_info"]
                    task = Task(
                        job_info, self._master_ip, self._trace_name, self._this_dir
                    )
                    cmd = task.run()
                    job_id = max(task._job_id)
                    job_counter = max(task._job_counter)
                    self._tasks[(job_id, job_counter)] = (task, job_info)
                    self._logger.info(
                        f"{self._worker_id}, execute, {task._job_id} - {task._job_counter}, {task._gpus}, {' '.join(cmd)}"
                    )
                    conn.sendall(json.dumps({"status": "ok"}).encode())

                elif cmd == "kill":
                    job_info = msg["job_info"]
                    job_id = max(job_info["job_id"])
                    job_counter = max(job_info["job_counter"])

                    if (job_id, job_counter) in self._tasks:
                        task, _ = self._tasks.pop((job_id, job_counter))
                        task.terminate()
                        task.wait()
                        self._logger.info(
                            f"{self._worker_id}, kill, {job_id} - {job_counter}, {job_info['gpus']}"
                        )
                        conn.sendall(json.dumps({"status": "ok"}).encode())

                elif cmd == "exit":
                    self._logger.info(f"{self._worker_id} exit")
                    self._check_task_flag = False
                    conn.sendall(json.dumps({"status": "ok"}).encode())
                    os._exit(0)
                    
        finally:
            conn.close()

    # 检查任务完成并上报
    def check_tasks(self):
        while self._check_task_flag:
            finished_tasks = []
            for (job_id, job_counter), (task, job_info) in list(self._tasks.items()):
                if task.return_code is None:
                    continue
                self.report_done(job_id, job_counter, task._gpus, task.return_code)
                finished_tasks.append((job_id, job_counter))
            for key in finished_tasks:
                self._tasks.pop(key)
            time.sleep(2)

    # 上报任务完成
    def report_done(self, job_id, job_counter, gpus, returncode):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self._master_ip, self._master_port))
                msg = {
                    "cmd": "done",
                    "job_id": job_id,
                    "job_counter": job_counter,
                    "worker_id": self._worker_id,
                    "gpus": gpus,
                    "returncode": returncode,
                }
                s.sendall(json.dumps(msg).encode())
                reply = s.recv(4096)
                if(json.loads(reply.decode()).get("status") == "ok"):
                    self._logger.info(
                        f"report_done: {job_id} - {job_counter}, {gpus}, return code: {returncode}"
                    )
        except Exception as e:
            self._logger.info(f"report_done failed: {e}")

