"""Launching tool for DGL distributed training"""
import argparse
import logging
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import time
from functools import partial
from threading import Thread
from typing import Optional


def cleanup_proc(get_all_remote_pids, conn):
    """This process tries to clean up the remote training tasks."""
    print("cleanup process runs")
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == "exit":
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for (ip, port), pids in remote_pids.items():
            kill_process(ip, port, pids)
    print("cleanup process exits")


def kill_process(ip, port, pids):
    """ssh to a remote machine and kill the specified processes."""
    curr_pid = os.getpid()
    killed_pids = []
    # If we kill child processes first, the parent process may create more again. This happens
    # to Python's process pool. After sorting, we always kill parent processes first.
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print("kill process {} on {}:{}".format(pid, ip, port), flush=True)
        kill_cmd = (
            "ssh -o StrictHostKeyChecking=no -p "
            + str(port)
            + " "
            + ip
            + " 'kill {}'".format(pid)
        )
        subprocess.run(kill_cmd, shell=True)
        killed_pids.append(pid)
    # It's possible that some of the processes are not killed. Let's try again.
    for i in range(3):
        killed_pids = get_killed_pids(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break
        else:
            killed_pids.sort()
            for pid in killed_pids:
                print(
                    "kill process {} on {}:{}".format(pid, ip, port), flush=True
                )
                kill_cmd = (
                    "ssh -o StrictHostKeyChecking=no -p "
                    + str(port)
                    + " "
                    + ip
                    + " 'kill -9 {}'".format(pid)
                )
                subprocess.run(kill_cmd, shell=True)


def get_killed_pids(ip, port, killed_pids):
    """Get the process IDs that we want to kill but are still alive."""
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ",".join(killed_pids)
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -p {} -h'".format(killed_pids)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode("utf-8").split("\n"):
        l = p.split()
        if len(l) > 0:
            pids.append(int(l[0]))
    return pids


def execute_remote(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
    port: int,
    username: Optional[str] = "",
) -> Thread:
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        state_q: A queue collecting Thread exit states.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        thread_list:
        username: Optional. If given, this will specify a username to use when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a username to avoid permission issues.

    Returns:
        thread: The Thread whose run() is to run the `cmd` on the remote host. Returns when the cmd completes
            on the remote host.
    """
    ip_prefix = ""
    if username:
        ip_prefix += "{username}@".format(username=username)

    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} '{cmd}'".format(
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )

    # thread func to run the job
    def run(ssh_cmd, state_q):
        try:
            subprocess.check_call(ssh_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception:
            state_q.put(-1)

    thread = Thread(
        target=run,
        args=(
            ssh_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread


def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine."""
    pids = []
    curr_pid = os.getpid()
    # Here we want to get the python processes. We may get some ssh processes, so we should filter them out.
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -aux | grep python | grep -v StrictHostKeyChecking'"
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode("utf-8").split("\n"):
        l = p.split()
        if len(l) < 2:
            continue
        # We only get the processes that run the specified command.
        res = re.search(cmd_regex, p)
        if res is not None and int(l[1]) != curr_pid:
            pids.append(l[1])

    pid_str = ",".join([str(pid) for pid in pids])
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'pgrep -P {}'".format(pid_str)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode("utf-8").split("\n")
    all_pids = []
    for pid in set(pids + pids1):
        if pid == "" or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids


def get_all_remote_pids(hosts, ssh_port, udf_command):
    """Get all remote processes."""
    remote_pids = {}
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # When creating training processes in remote machines, we may insert some arguments
        # in the commands. We need to use regular expressions to match the modified command.
        cmds = udf_command.split()
        new_udf_command = " .*".join(cmds)
        pids = get_remote_pids(ip, ssh_port, new_udf_command)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids


def construct_torch_dist_launcher_cmd(
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Constructs the torch distributed launcher command.
    Helper function.

    Args:
        num_trainers:
        num_nodes:
        node_rank:
        master_addr:
        master_port:

    Returns:
        cmd_str.
    """
    torch_cmd_template = (
        # "python3 "
        #"/miniconda3/envs/capsule/bin/python3.8 -m torch.distributed.launch "
        "/miniconda3/envs/capsule/bin/torchrun "
        "--nproc_per_node={nproc_per_node} "
        "--nnodes={nnodes} "
        "--node_rank={node_rank} "
        "--master_addr={master_addr} "
        "--master_port={master_port}"
    )
    return torch_cmd_template.format(
        nproc_per_node=num_trainers,
        nnodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


def wrap_udf_in_torch_dist_launcher(
    udf_command: str,
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Wraps the user-defined function (udf_command) with the torch.distributed.run module.

     Example: if udf_command is "python3 run/some/trainer.py arg1 arg2", then new_df_command becomes:
         "python3 -m torch.distributed.run <TORCH DIST ARGS> run/some/trainer.py arg1 arg2

    udf_command is assumed to consist of pre-commands (optional) followed by the python launcher script (required):
    Examples:
        # simple
        python3.7 path/to/some/trainer.py arg1 arg2

        # multi-commands
        (cd some/dir && python3.7 path/to/some/trainer.py arg1 arg2)

    IMPORTANT: If udf_command consists of multiple python commands, then this will result in undefined behavior.

    Args:
        udf_command:
        num_trainers:
        num_nodes:
        node_rank:
        master_addr:
        master_port:

    Returns:

    """
    torch_dist_cmd = construct_torch_dist_launcher_cmd(
        num_trainers=num_trainers,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    # Auto-detect the python binary that kicks off the distributed trainer code.
    # Note: This allowlist order matters, this will match with the FIRST matching entry. Thus, please add names to this
    #       from most-specific to least-specific order eg:
    #           (python3.7, python3.8) -> (python3)
    # The allowed python versions are from this: https://www.dgl.ai/pages/start.html
    python_bin_allowlist = (
        "python3.6",
        "python3.7",
        "python3.8",
        "python3.9",
        "python3",
        # for backwards compatibility, accept python2 but technically DGL is a py3 library, so this is not recommended
        "python2.7",
        "python2",
    )
    # If none of the candidate python bins match, then we go with the default `python`
    python_bin = "python"
    for candidate_python_bin in python_bin_allowlist:
        if candidate_python_bin in udf_command:
            python_bin = candidate_python_bin
            break

    # transforms the udf_command from:
    #     python path/to/dist_trainer.py arg0 arg1
    # to:
    #     python -m torch.distributed.run [DIST TORCH ARGS] path/to/dist_trainer.py arg0 arg1
    # Note: if there are multiple python commands in `udf_command`, this may do the Wrong Thing, eg launch each
    #       python command within the torch distributed launcher.
    new_udf_command = udf_command.replace(
        python_bin, f"{torch_dist_cmd}"
    )
    return new_udf_command



def wrap_cmd_with_local_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired env vars with the following properties:
    (1) env vars persist for the entire `cmd`, even if it consists of multiple "chained" commands like:
        cmd = "ls && pwd && python run/something.py"
    (2) env vars don't pollute the environment after `cmd` completes.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_with_local_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A string containing env vars, eg "VAR1=val1 VAR2=val2"

    Returns:
        cmd_with_env_vars:

    """
    # use `export` to persist env vars for entire cmd block. required if udf_command is a chain of commands
    # also: wrap in parens to not pollute env:
    #     https://stackoverflow.com/a/45993803
    return f"(export {env_vars}; {cmd})"


def wrap_cmd_with_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra env vars

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_with_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Args:
        cmd:
        env_vars: A list of strings containing env vars, e.g., ["VAR1=value1", "VAR2=value2"]

    Returns:
        cmd_with_env_vars:
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_with_local_envvars(cmd, env_vars)


def get_available_port(ip):
    """Get available port with specified ip."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(2233, 65535):
        try:
            sock.connect((ip, port))
        except:
            return port
    raise RuntimeError("Failed to get available port for ip~{}".format(ip))


def submit_jobs(args, udf_command, dry_run=False):
    """Submit distributed jobs (server and client processes) via ssh"""
    if dry_run:
        print("Currently it's in dry run mode which means no jobs will be launched.")
    dry_run_cmds = []
    hosts = []
    thread_list = []

    # Get the IP addresses of the cluster.
    ip_config = os.path.join(args.workspace, args.ip_config)
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of ip_config.")


    state_q = queue.Queue()
   
    master_addr = hosts[0][0]
    master_port = get_available_port(master_addr)
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # [1] Transform udf_command to follow torch's dist launcher format: `torchrun ... UDF`
        torch_dist_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=args.num_trainers,
            num_nodes=len(hosts),
            node_rank=node_id,
            master_addr=master_addr,
            master_port=master_port,
        )
        # [2] add ENV vars
        client_env_vars = "OMP_NUM_THREADS=4"
        cmd = wrap_cmd_with_local_envvars(
            torch_dist_udf_command, client_env_vars
        )

        # [3] add extra env vars(by lists)
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        
        # [4] auto update bash and go right path of run script
        cmd = "source ~/.bashrc;"+ cmd
        cmd = "cd " + str(args.workspace) + "; " + cmd
        dry_run_cmds.append(cmd)
        
        # [5] execute remote
        if not dry_run:
            thread_list.append(
                execute_remote(
                    cmd, state_q, ip, args.ssh_port, username=args.ssh_username
                )
            )

    # return commands if in dry run mode
    if dry_run:
        return dry_run_cmds

    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1, conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, udf_command)
    process = multiprocessing.Process(target=cleanup_proc, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info("Stop launcher")
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send("cleanup")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            # Record err_code We record one of the error if there are multiple
            err = err_code

    # The training processes complete. We should tell the cleanup process to exit.
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)


def main():
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument("--ssh_port", type=int, default=22, help="SSH Port.")
    parser.add_argument(
        "--ssh_username",
        default="",
        help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
        "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
        "instead of 'ssh 1.2.3.4 CMD'",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Path of user directory of distributed tasks. \
                This is used to specify a destination location where \
                the contents of current directory will be rsyncd",
    )
    parser.add_argument(
        "--num_trainers",
        type=int,
        help="The number of trainer processes per machine",
    )
    parser.add_argument(
        "--num_omp_threads",
        type=int,
        help="The number of OMP threads per trainer",
    )
    parser.add_argument(
        "--ip_config",
        type=str,
        help="The file (in workspace) of IP configuration for server processes",
    )
    parser.add_argument(
        "--num_server_threads",
        type=int,
        default=1,
        help="The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on \
                        the same machine. By default, it is 1.",
    )
    parser.add_argument(
        "--graph_format",
        type=str,
        default="csc",
        help='The format of the graph structure of each partition. \
                        The allowed formats are csr, csc and coo. A user can specify multiple \
                        formats, separated by ",". For example, the graph format is "csr,csc".',
    )
    parser.add_argument(
        "--extra_envs",
        nargs="+",
        type=str,
        default=[],
        help="Extra environment parameters need to be set. For example, \
                        you can set the LD_LIBRARY_PATH and NCCL_DEBUG by adding: \
                        --extra_envs LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH NCCL_DEBUG=INFO ",
    )
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, "Please provide user command line."
    assert (
        args.num_trainers is not None and args.num_trainers > 0
    ), "--num_trainers must be a positive number."
    assert (
        args.num_server_threads > 0
    ), "--num_server_threads must be a positive number."
    assert (
        args.workspace is not None
    ), "A user has to specify a workspace with --workspace."
    assert (
        args.ip_config is not None
    ), "A user has to specify an IP configuration file with --ip_config."
    if args.num_omp_threads is None:
        args.num_omp_threads = max(
            multiprocessing.cpu_count() // 2 // args.num_trainers, 1
        )
        print(
            "The number of OMP threads per trainer is set to",
            args.num_omp_threads,
        )

    udf_command = str(udf_command[0])
    if "python" not in udf_command:
        raise RuntimeError(
            "Capsule launching script can only support Python executable file."
        )

    submit_jobs(args, udf_command)

if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()


## run command
"""
python3 ./launch.py \
    --workspace /Capsule/src/dist \
    --num_trainers 1 \
    --ip_config /Capsule/ip_config.txt \
    "python3 dist_sage.py"
"""

"""
python3 ./launch.py \
    --workspace /Capsule/src/datapart \
    --num_trainers 1 \
    --ip_config /Capsule/ip_config.txt \
    "python3 dist2subG.py"
"""