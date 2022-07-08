#!/usr/bin/env python3
import argparse
import ctypes
import logging
import os
import sys
import time

import climetlab as cml
import dask
import pandas as pd
from asyncssh import set_debug_level, set_log_level

# import climetlab_s2s_ai_challenge
# from climetlab.mirrors.directory_mirror import DirectoryMirror
# from dask.distributed import get_worker
from dask.distributed import Worker, WorkerPlugin, performance_report
from distributed import Client, LocalCluster, SSHCluster

from ecmwf_ml_utils.parse_slurm import parse_slurm_nodes

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--num-workers-per-node",
        required=True,
        type=int,
        help="Number of worker processes per node",
    )
    required_args.add_argument(
        "--worker-memory-limit",
        required=True,
        type=str,
        help="Memory that each worker can use, e.g. '5GB' or '5000M', or 'auto'.",
    )
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "--num-threads-per-worker",
        required=False,
        type=int,
        default=1,
        help="Number of threads per worker [default == 1]",
    )
    optional_args.add_argument(
        "--scheduler-port",
        required=False,
        type=int,
        default=8786,
        help="Port the scheduler is listening on",
    )
    optional_args.add_argument(
        "--dashboard-port",
        required=False,
        type=int,
        default=8787,
        help="Dask dashboard port",
    )
    optional_args.add_argument(
        "--temp-dir",
        required=False,
        type=str,
        default=os.environ["TMPDIR"],
        help="Path where to store temporary files",
    )
    return parser.parse_args()


def get_logger(
    name,
    filename=None,
    mode="a+",
    logger_debug_level=logging.DEBUG,
    handler_debug_level=logging.DEBUG,
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    msgfmt="[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName).30s] [%(levelname)s] %(message)s",
) -> logging.Logger:

    logger = logging.getLogger(name=name)
    if logger.hasHandlers():
        return logger

    def build_handler():
        handler = logging.StreamHandler(sys.stdout)
        if filename:
            handler = logging.FileHandler(filename, mode=mode)
        handler.setLevel(handler_debug_level)
        formatter = logging.Formatter(msgfmt, datefmt=datefmt)
        setattr(formatter, "converter", time.gmtime)
        handler.setFormatter(formatter)
        return handler

    logger.setLevel(logger_debug_level)
    logger.addHandler(build_handler())
    return logger


def sanitize_string(s):
    return "".join([x for x in s if x in "_" or x.isalnum()])


class LoggerWorkerPlugin(WorkerPlugin):
    # This name is used in the dask list of plugins
    name = "custom_logger"

    def __init__(self, directory):
        self.directory = directory
        self.logger = None

    def setup(self, worker: Worker) -> None:
        name = sanitize_string(f"log_{worker.worker_address}")

        # set logger to custom filename
        filename = os.path.join(self.directory, name)
        self.logger = get_logger(name, filename=filename)

        # Use the same file also for the logger of "distributed.worker"
        log = logging.getLogger("distributed.worker")
        for h in log.handlers:
            log.removeHandler(h)
        log.addHandler(self.logger.handlers[0])

        self.logger.info(f"Worker setup() completed for worker {worker.worker_address}")

    def teardown(self, worker: Worker) -> None:
        self.logger.info(f"Worker teardown() for worker {worker.worker_address}")
        del worker  # TODO: remove this?


def trim_worker_memory() -> int:
    """
    https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
    https://distributed.dask.org/en/stable/worker-memory.html"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except OSError as e:
        print(f"WARNING: trim_worker_memory won't work here ({e})")


class NonDaskClient:
    def __init__(self, *args, **kwargs):
        print(f"Non dask client. Ignoring {args} {kwargs}")

    def submit(self, func, *args, **kwargs):
        func(*args, **kwargs)

    def gather(self, *args, **kwargs):
        pass

    def shutdown(self, *args, **kwargs):
        pass


class CustomSshClient(Client):
    def __init__(
        self,
        num_workers_per_node: int,
        worker_memory_limit: str,
        local_temp_dir: str,
        num_threads_per_worker: int = 1,
        scheduler_port: int = 8786,
        dashboard_port: int = 8787,
        nodes=None,
        dask_log_dir=None,
    ):
        """
        A client created with an SSH cluster
        """
        set_log_level(logging.ERROR)
        set_debug_level(level=1)  # asyncssh_debug_level = 1, # 1 minimal,2,3

        # Get the values needed for cluster creation, as supplied by the batch-
        # system-specific batch runscript.
        nodes = parse_slurm_nodes(nodes)

        # assert "-" in nodes, "Must run this on 2 or more nodes!"
        # basename, node_nos = nodes.split("-", maxsplit=1)
        # sep = "-" if "-" in node_nos else ","
        # node_nos = node_nos[1:-1].split(sep)
        # nodes = [basename + "-" + node_no for node_no in node_nos]
        # print(f"These are the nodes we're running on: {nodes}")
        kwargs = dict(
            # scheduler runs on the "root" (1st) node
            # num_workers_per_node on each of the nodes, including the 1st
            hosts=[nodes[0], *nodes],
            connect_options={
                "client_host_keysign": True,
                "known_hosts": None,
            },
            worker_options={
                "memory_limit": worker_memory_limit,
                "local_directory": local_temp_dir,
                "n_workers": num_workers_per_node,
                "nthreads": num_threads_per_worker,
            },
            scheduler_options={
                "port": scheduler_port,
                "dashboard_address": f":{dashboard_port}",
            },
        )
        LOGGER.debug(kwargs)
        try:
            cluster = SSHCluster(**kwargs)
        except RuntimeError as e:
            LOGGER.error("--------------------------------------------------")
            LOGGER.error("--------------------------------------------------")
            LOGGER.error("ERROR not connect with SSH.")
            LOGGER.error("Could not connect with SSH. Using a local cluster.")
            LOGGER.error(str(e))
            LOGGER.error("--------------------------------------------------")
            LOGGER.error("--------------------------------------------------")
            cluster = LocalCluster("127.0.0.1:8786", n_workers=2, threads_per_worker=2)

        """
    distributed:
    worker:
    # Fractions of worker process memory at which we take action to avoid memory
    # blowup. Set any of the values to False to turn off the behavior entirely.
        memory:
        target: fraction of managed memory where we start spilling to disk
        spill: fraction of process memory where we start spilling to disk
        pause: fraction of process memory at which we pause worker threads ("False" means do not pause)
        terminate: fraction of process memory at which we terminate the worker ("False" means do not terminate)
    """

        dask.config.set(
            {
                # this high initial guess tells the scheduler to spread tasks
                "distributed.scheduler.unknown-task-duration": "10s",
                # worker memory management
                "distributed.worker.memory.target": 0.92,
                "distributed.worker.memory.spill": 0.92,
                "distributed.worker.memory.pause": False,
                "distributed.worker.memory.terminate": False,
            }
        )

        super().__init__(cluster)
        if dask_log_dir:
            self.register_worker_plugin(LoggerWorkerPlugin(dask_log_dir))

        self.run(trim_worker_memory)


class CustomLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def __call__(self) -> logging.Logger:
        try:
            return dask.distributed.get_worker().plugins[LoggerWorkerPlugin.name].logger
        except ValueError:
            return self._logger
        except KeyError as e:
            raise Exception(
                "The worker logger plugin was not initialized correctly!"
            ) from e

    def __getattr__(self, name):
        return getattr(self(), name)


def print_(msg):
    print(f"******* {msg} **********")


def main() -> None:

    # TODO: check / remove this main() func

    args = parse_args()

    # TODO: in climetlab, read .climetlab/settings.yaml to make this work everywhere
    # cml.mirror('s2s').activate() ?
    # cml.mirror('s2s', source = ...).activate() ?
    # cml.mirror('s2s', dataset = ...).activate() ?

    # mirror = DirectoryMirror("/ec/res4/scratch/mafp/mirror")
    # mirror.activate()

    # TODO: remove this env var?
    worker_logdir = os.environ.get("DASK_LOG_DIR", ".tmp")
    os.makedirs(worker_logdir, exist_ok=True)
    print(args.temp_dir)
    client = create_ssh_client(
        args.num_workers_per_node,
        args.worker_memory_limit,
        args.num_threads_per_worker,
        args.scheduler_port,
        args.dashboard_port,
        args.temp_dir,
    )
    client.register_worker_plugin(LoggerWorkerPlugin(worker_logdir))
    client.run(trim_worker_memory)

    s = cml.load_source(
        "dummy-source",
        kind="netcdf",
        dims=["forecast_time", "realization", "lead_time", "latitude", "longitude"],
        variables=["t2m", "tp"],
    )
    ds = s.to_xarray()
    t2m_avg = client.compute(
        ds.mean(("forecast_time", "realization", "lead_time")).std(
            ("latitude", "longitude")
        )
    ).result()

    print(ds)

    with performance_report(filename="./dask-report.html"):

        ds = cml.load_dataset(
            "s2s-ai-challenge-training-input",
            origin="ecmwf",
            date=pd.date_range(start="2020-01-02", end="2020-07-31", freq="7D"),
            parameter=["t2m"],
            format="netcdf",
        ).to_xarray()

        print_("DATASET:")
        print(ds)

        t2m_avg = client.compute(
            ds.mean(("forecast_time", "realization", "lead_time")).std(
                ("latitude", "longitude")
            )
        ).result()

        print_("CLIENT:")
        print(client)

        print_("T2M AVERAGE:")
        print(t2m_avg)


if __name__ == "__main__":
    main()
