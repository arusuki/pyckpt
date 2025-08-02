from controller import Controller
from cluster import CLUSTER

import socket
import json
import threading
import util
import copy
from jobs import JOBS


class Scheduler(object):
    def __init__(self, scheduler_port: int, controller_port: int) -> None:
        super().__init__()

        self._logger = util.make_logger(__name__)
        self._scheduler_port = scheduler_port

        self._trainers = {}
        self._server = threading.Thread(
            target=self.scheduler_server, args=(scheduler_port,), daemon=True
        )
        self._server.start()

        self._num_workers = CLUSTER.num_node_p_switch
        self._controller = Controller(controller_port, self._num_workers)

        # self._start_time = self._controller.get_time()

    def get_time(self):
        return self._controller.get_time()

    def scheduler_server(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", port))
        s.listen(10)

        while True:
            conn, _ = s.accept()
            threading.Thread(target=self.handle_trainer, args=(conn,)).start()

    def handle_trainer(self, conn):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                msg = json.loads(data.decode())
                cmd = msg.get("cmd")

                if cmd == "register_trainer":
                    job_id_list = msg["job_id_list"]
                    job_id = max(job_id_list)
                    self._trainers[job_id] = (msg["trainer_ip"], msg["trainer_port"])

                    conn.sendall(json.dumps({"status": "ok"}).encode())
                    self._logger.info(
                        f"scheduler, register, {job_id}-{job_id_list}, {msg['trainer_ip']}:{msg['trainer_port']}"
                    )

                elif cmd == "report_itertime":
                    job_id = msg["job_id"]
                    iter_time = msg["iter_time"]
                    for rjob_id in job_id:
                        if rjob_id >= 0:
                            rjob = JOBS.find_runnable_job(rjob_id)
                            rjob["real_itertime"] = copy.deepcopy(list(iter_time))
                    self._logger.info(
                        f"scheduler, update job {job_id} iter_time {list(iter_time)}"
                    )
                    conn.sendall(json.dumps({"status": "ok"}).encode())

        except Exception as e:
            print(f"Trainer connection error: {e}")
        finally:
            conn.close()

    def query_stats(self, job_id_list):
        job_id = max(job_id_list)
        assert job_id in self._trainers
        addr = self._trainers[job_id]
        msg = {"cmd": "query_stats"}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(addr)
                s.sendall(json.dumps(msg).encode())
                reply = s.recv(4096)
                status = json.loads(reply.decode()).get("status")
                if status == "ok":
                    return json.loads(reply.decode()).get("finished_iterations")
            except Exception:
                self._logger.error(f"query stats to trainer {job_id} failed")
                return None

    def has_ready_jobs(self, tmp_time):
        if len(JOBS.job_events) > 0 and JOBS.job_events[0]["time"] <= tmp_time:
            return True
        else:
            return False

    def has_running_trainers(self, running_jobs):
        if running_jobs > self._controller.done_queue.qsize():
            return True
        else:
            return False
