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
# === RECEIVER: file table + range tracking ===
from threading import Lock

# class RangeSet:
#     """Track [start,end) intervals; supports add+merge and covered length."""
#     __slots__ = ("iv",)  # list of (s,e), non-overlapping, sorted by s
#     def __init__(self):
#         self.iv = []

#     def add(self, s, e):
#         if e <= s: return
#         out = []
#         placed = False
#         for a,b in self.iv:
#             if b < s:            # entirely before new
#                 out.append((a,b))
#             elif e < a:          # entirely after new
#                 if not placed:
#                     out.append((s,e)); placed = True
#                 out.append((a,b))
#             else:                # overlap
#                 s = min(s, a)
#                 e = max(e, b)
#         if not placed:
#             out.append((s,e))
#         # normalize
#         out.sort(key=lambda x: x[0])
#         # print(f"Previous intervals: {self.iv}, added [{s},{e}), merged to {out}")
#         self.iv = out

#     def covered(self):
#         return sum(b - a for a,b in self.iv)

# # filename_rel -> state {fd,total,ranges,lock}
# # file_state = {}
# # file_state_lock = Lock()

# from multiprocessing import Manager
# mgr = Manager()
# file_state = mgr.dict()
# file_state_lock = mgr.RLock()

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

# def get_file_entry(filename_rel, total_size, direct_io=False):
#     """Open+preallocate once, store state for concurrent writers."""
#     with file_state_lock:
#         print("getting file entry:", filename_rel, total_size)
#         print("current file_state: ", file_state)
#         st = file_state.get(filename_rel)
#         if st is not None:
#             return st

#         abs_path = os.path.join(root, filename_rel)
#         _ensure_dir(abs_path)
#         flags = os.O_CREAT | os.O_RDWR
#         if direct_io:
#             flags |= os.O_DIRECT | os.O_SYNC  # beware alignment!
#         fd = os.open(abs_path, flags, 0o644)
#         _preallocate_fd(fd, int(total_size))
#         st = {"fd": fd, "total": int(total_size), "ranges": RangeSet(), "lock": Lock()}
#         file_state[filename_rel] = st
#         return st

# def note_block_received(filename_rel, start, length):
#     """Record a received [start, start+length) block; return True if file complete."""
#     st = file_state[filename_rel]
#     end = start + length
#     with st["lock"]:
#         # print("noting block:", filename_rel, start, length)
#         st["ranges"].add(start, end)
#         return st["ranges"].covered() >= st["total"]
    
# def close_file_if_complete(filename_rel):
#     st = file_state.get(filename_rel)
#     if not st:
#         return False
#     print("filename_rel:", filename_rel)
#     print("total:", st["total"])
#     print("covered:", st["ranges"].covered())
#     with st["lock"]:
#         # idempotency guard
#         if st.get("closed"):
#             return False
#         if st["ranges"].covered() >= st["total"]:
#             try:
#                 # Optional but recommended for durability
#                 try:
#                     os.fsync(st["fd"])
#                 except Exception:
#                     pass
#                 os.close(st["fd"])
#             finally:
#                 st["closed"] = True
#             return True
#     return False



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

                # print(header)
                if file_transfer:
                    parts = header.split(",")
                    if len(parts) != 4:
                        log.error("Bad header: %r", header)
                        break

                    # print(parts)

                    filename_rel = str(parts[0])
                    offset       = int(parts[1])   # where this chunk *should* start
                    to_rcv_goal  = int(parts[2])   # how many bytes the sender intended for THIS chunk
                    total_size   = int(parts[3])

                    # st = get_file_entry(filename_rel, total_size, direct_io=False)
                    abs_path = os.path.join(root, filename_rel)
                    _ensure_dir(abs_path)
                    flags = os.O_CREAT | os.O_RDWR
                    if direct_io:
                        flags |= os.O_DIRECT | os.O_SYNC  # beware alignment!
                    fd = os.open(abs_path, flags, 0o644)
                    if offset == 0:
                        _preallocate_fd(fd, int(total_size))
                    # fd = st["fd"]

                    remaining = to_rcv_goal
                    cs = chunk_size.value if hasattr(chunk_size, "value") else chunk_size
                    max_block = int(cs) if cs else (1 << 20)

                    # receive loop for this chunk; tolerate mid-chunk stop
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

                        # note_block_received(filename_rel, current, written)
                        current += written
                        remaining -= written
                        total += written

                    # log.info(f"Chunk done: {filename_rel}, offset: {offset}, to_rcv_goal: {to_rcv_goal}, total_size: {total_size}, received: {to_rcv_goal - remaining}")

                    # if close_file_if_complete(filename_rel):
                    #     log.info("File complete: %s (%d bytes)", filename_rel, st["total"])


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
            # raise e


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

        # while True:
        #     try:
        #         time.sleep(1)
        #     except:
        #         break

        process_status[0] = 1
        # reporting_process = mp.Process(target=report_throughput)
        # reporting_process.daemon = True
        # reporting_process.start()

        while sum(process_status) > 0:
            time.sleep(0.1)

        # reporting_process.terminate()
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.1)