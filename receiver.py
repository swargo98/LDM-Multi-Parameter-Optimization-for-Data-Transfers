import os
import mmap
import time
import socket
import pprint
import argparse
import logging as log
import numpy as np
import psutil
import multiprocessing as mp
from config_receiver import configurations
from threading import Lock

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _preallocate_fd(fd, size):
    try:
        os.posix_fallocate(fd, 0, size)
    except AttributeError:
        os.ftruncate(fd, size)
    except OSError:
        os.ftruncate(fd, size)


chunk_size = mp.Value("i", 1024*1024)
cpus = mp.Manager().list()

log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
if configurations["loglevel"] == "debug":
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.DEBUG,
    )

    mp.log_to_stderr(log.DEBUG)
else:
    log.basicConfig(
        format=log_FORMAT,
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=log.INFO
    )

    mp.log_to_stderr(log.INFO)


def worker(sock, process_num):
    while True:
        try:
            client, address = sock.accept()
            log.info("{u} connected".format(u=address))
            process_status[process_num] = 1
            total = 0
            d = client.recv(1).decode()
            if not d:
                time.sleep(0.5)
            while d:
                header = ""
                while d != '\n':
                    header += str(d)
                    d = client.recv(1).decode()

                if file_transfer:
                    parts = header.split(",")
                    if len(parts) != 4:
                        log.error("Bad header: %r", header)
                        break


                    filename_rel = str(parts[0])
                    offset       = int(parts[1])   # where this chunk *should* start
                    to_rcv_goal  = int(parts[2])   # how many bytes the sender intended for THIS chunk
                    total_size   = int(parts[3])

                    abs_path = os.path.join(root, filename_rel)
                    _ensure_dir(abs_path)
                    flags = os.O_CREAT | os.O_RDWR
                    if direct_io:
                        flags |= os.O_DIRECT | os.O_SYNC  # beware alignment!
                    fd = os.open(abs_path, flags, 0o644)
                    if offset == 0:
                        _preallocate_fd(fd, int(total_size))

                    remaining = to_rcv_goal
                    cs = chunk_size.value if hasattr(chunk_size, "value") else chunk_size
                    max_block = int(cs) if cs else (1 << 20)

                    current = offset

                    while remaining > 0:
                        n = min(remaining, max_block)
                        buf = client.recv(n)
                        if not buf:
                            # sender paused/closed mid-chunk; keep what we wrote so far
                            log.debug("incomplete chunk: %s off=%d recv=%d of %d",
                                    filename_rel, offset, to_rcv_goal - remaining, to_rcv_goal)
                            break

                        written = os.pwrite(fd, buf, current)
                        log.debug(written)
                        if written != len(buf):
                            log.error("short pwrite %d != %d for %s", written, len(buf), filename_rel)
                            break

                        current += written
                        remaining -= written
                        total += written


                else:
                    chunk = client.recv(chunk_size.value)
                    while chunk:
                        chunk = client.recv(chunk_size.value)

                d = client.recv(1).decode()

            total_mb = np.round(total / (1024 * 1024))
            log.info("%s exited. total received %s MB", address, total_mb)
            client.close()
            # process_status[process_num] = 0
        except Exception as e:
            log.error(str(e))


def report_throughput():
    global cpus
    time.sleep(1)

    while sum(process_status) > 0:
        t1 = time.time()
        cpus.append(psutil.cpu_percent())
        log.info(f"cpu: curr - {np.round(cpus[-1], 4)}, avg - {np.round(np.mean(cpus), 4)}")
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    parser=argparse.ArgumentParser()
    parser.add_argument("--host", help="Receiver Host Address")
    parser.add_argument("--port", help="Receiver Port Number")
    parser.add_argument("--data_dir", help="Receiver Data Directory")
    args = vars(parser.parse_args())
    # pp.pprint(f"Command line arguments: {args}")

    if args["host"]:
        configurations["receiver"]["host"] = args["host"]

    if args["port"]:
        configurations["receiver"]["port"] = int(args["port"])

    if args["data_dir"]:
        configurations["data_dir"] = args["data_dir"]

    configurations["cpu_count"] = mp.cpu_count()
    pp.pprint(configurations)

    direct_io = False
    root = configurations["data_dir"]
    HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]

    file_transfer = True
    if "file_transfer" in configurations and configurations["file_transfer"] is not None:
        file_transfer = configurations["file_transfer"]

    num_workers = configurations["max_cc"]

    sock = socket.socket()
    sock.bind((HOST, PORT))
    sock.listen(num_workers)

    iter = 0
    while iter<1:
        iter += 1
        log.info(">>>>>> Iterations: {0} >>>>>>".format(iter))

        process_status = mp.Array("i", [0 for _ in range(num_workers)])
        workers = [mp.Process(target=worker, args=(sock, i,)) for i in range(num_workers)]
        for p in workers:
            p.daemon = True
            p.start()

        process_status[0] = 1

        while sum(process_status) > 0:
            time.sleep(0.1)

        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.1)