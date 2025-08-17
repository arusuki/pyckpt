import argparse
import os
from queue import Queue
from threading import Thread
from time import sleep
import pyckpt.rpc as rpc
import torch
from pyckpt.task import load_checkpoint, main, resume_checkpoint

LISTEN_PORT = 9387
HOST = "localhost"
LISTEN_ADDRESS = f"{HOST}:{LISTEN_PORT}"


@main(LISTEN_ADDRESS)
def counter():
    i = torch.tensor(0)
    while True:
        sleep(1)
        print(i)
        i += 1


@main(LISTEN_ADDRESS)
def ping_pong():
    def ping(input: Queue[int], output: Queue[int]):
        while True:
            value = output.get()
            print(f"ping: {value}")
            sleep(1)
            input.put(value + 1)

    def pong(input: Queue[int], output: Queue[int]):
        while True:
            value = input.get()
            print(f"pong: {value}")
            output.put(value + 1)

    input, output = Queue(), Queue()
    ponger = Thread(target=pong, args=(input, output))
    ponger.start()
    output.put(0)
    ping(input, output)


def checkpoint():
    client = rpc.Client()
    client.connect(HOST, LISTEN_PORT)
    client.call("checkpoint", os.getcwd(), "counter", None)


def resume():
    loaded = load_checkpoint(os.getcwd(), "counter")
    resume_checkpoint(loaded, LISTEN_ADDRESS)


def main():
    parser = argparse.ArgumentParser(
        description="A script with run, checkpoint, and resume operations."
    )
    parser.add_argument(
        "operation",
        choices=["run", "checkpoint", "resume", "ping-pong"],
        nargs="?",  # Make the argument optional
        default="run",
    )
    args = parser.parse_args()
    # Map operations to functions
    operations = {
        "run": counter,
        "checkpoint": checkpoint,
        "resume": resume,
        "ping-pong": ping_pong,
    }
    operations[args.operation]()


if __name__ == "__main__":
    main()
