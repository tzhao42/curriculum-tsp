"""Script to generate job scripts."""

import os
import pathlib


class Job:
    def __init__(
        self, name, time, num_nodes, cpus, mem, gpus, gpu_size, partition
    ):
        """Create a job."""
        self._name = name
        self._time = time  # number of hours
        self._num_nodes = num_nodes
        self._cpus = cpus
        self._mem = mem
        self._gpus = gpus
        self._gpu_size = gpu_size
        self._partition = partition

        self._filename = f"{self._name}.sh"
        open(self._filename, "w").close()

    def _generate_header(self):
        """Generate header."""
        self._writeline("#!/bin/bash")
        self._writeline(f"#SBATCH -t {self._time}:00:00")
        self._writeline(f"#SBATCH -N {self._num_nodes}")
        self._writeline(f"#SBATCH -n {self._cpus}")
        self._writeline(f"#SBATCH --mem={self._mem}GB")
        self._writeline(f"#SBATCH --gres=gpu:{self._gpus}")
        self._writeline(f"#SBATCH --constraint={self._gpu_size}GB")
        self._writeline(f"#SBATCH -p {self._partition}")
        self._writeline("")
        self._writeline("# This script was generate automatically")
        self._writeline("")
        self._writeline(
            'SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/'
            'ortools-pytorch-gpu.simg"'
        )
        self._writeline(
            'STARTING_DIRECTORY="/om2/user/tzhao/6883/curriculum-tsp"'
        )
        self._writeline("")
        self._writeline(r"cd ${STARTING_DIRECTORY}")
        self._writeline("module load openmind/singularity")
        self._writeline("hostname")
        self._writeline("nvidia-smi")
        self._writeline("")

    def _generate_execution(
        self,
        mode,
        run_name,
        num_nodes,
        epochs,
        curriculum,
        regen=None,
        val_set=None,
        load=None,
    ):
        """Create execution string.

        Might not be set: load, val_set, regen
        """
        run_string = r"singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py "
        run_string += f"--mode {mode} "
        run_string += f"--run-name {run_name} "

        # training
        run_string += f"--num-nodes {num_nodes} "
        run_string += f"--epochs {epochs} "
        run_string += f"--curriculum {curriculum} "

        if regen:
            run_string += "--regen "

        run_string += f"--val-set {val_set} "

        # testing
        if mode == "test":
            run_string += f"--load {load} "

        run_string += "&"
        return run_string

    def add_executions(
        self,
        mode_l,
        run_name_l,
        num_nodes_l,
        epochs_l,
        curriculum_l,
        regen_l,
        val_set_l,
        load_l,
    ):
        """Add executions to a job."""
        assert len(mode_l) == len(run_name_l)
        assert len(mode_l) == len(num_nodes_l)
        assert len(mode_l) == len(epochs_l)
        assert len(mode_l) == len(curriculum_l)
        assert len(mode_l) == len(regen_l)
        assert len(mode_l) == len(val_set_l)
        assert len(mode_l) == len(load_l)

        self._num_cases = len(mode_l)
        self._mode_l = mode_l
        self._run_name_l = run_name_l
        self._num_nodes_l = num_nodes_l
        self._epochs_l = epochs_l
        self._curriculum_l = curriculum_l
        self._regen_l = regen_l
        self._val_set_l = val_set_l
        self._load_l = load_l

    def _generate_footer(self):
        """Generate footer."""
        for i in range(self._num_cases):
            curr_execution = self._generate_execution(
                self._mode_l[i],
                self._run_name_l[i],
                self._num_nodes_l[i],
                self._epochs_l[i],
                self._curriculum_l[i],
                self._regen_l[i],
                self._val_set_l[i],
                self._load_l[i],
            )
            self._writeline(curr_execution)

        self._writeline("")
        self._writeline("")
        self._writeline("sleep 2h")
        self._writeline("nvidia-smi")
        self._writeline("sleep 72h")
        self._writeline("")
        self._writeline("")

    def _writeline(self, line):
        """Write a line into the script."""
        with open(self._filename, "a+") as f:
            f.write(line + "\n")

    def generate(self):
        """Generate job file."""
        self._generate_header()
        self._generate_footer()
        with open("run-all.sh", "a+") as f:
            f.write(f"sbatch {self._filename}\n")



def generate_jobs(
    om_name_l,
    om_time_l,
    om_num_nodes_l,
    om_cpus_l,
    om_mem_l,
    om_gpus_l,
    om_gpu_size_l,
    om_partition_l,
    execution_dict,
):
    """Generate jobs."""
    assert len(om_name_l) == len(om_time_l)
    assert len(om_name_l) == len(om_num_nodes_l)
    assert len(om_name_l) == len(om_cpus_l)
    assert len(om_name_l) == len(om_mem_l)
    assert len(om_name_l) == len(om_gpus_l)
    assert len(om_name_l) == len(om_gpu_size_l)
    assert len(om_name_l) == len(om_partition_l)

    mode_l = execution_dict["mode_l"]
    run_name_l = execution_dict["run_name_l"]
    num_nodes_l = execution_dict["num_nodes_l"]
    epochs_l = execution_dict["epochs_l"]
    curriculum_l = execution_dict["curriculum_l"]
    regen_l = execution_dict["regen_l"]
    val_set_l = execution_dict["val_set_l"]
    load_l = execution_dict["load_l"]

    assert len(mode_l) == len(run_name_l)
    assert len(mode_l) == len(num_nodes_l)
    assert len(mode_l) == len(epochs_l)
    assert len(mode_l) == len(curriculum_l)
    assert len(mode_l) == len(regen_l)
    assert len(mode_l) == len(val_set_l)
    assert len(mode_l) == len(load_l)

    assert len(mode_l) // len(om_name_l) - len(mode_l) / len(om_name_l) == 0

    executions_per_machine = len(mode_l) // len(om_name_l)

    all_mode_l = [
        mode_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_run_name_l = [
        run_name_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_num_nodes_l = [
        num_nodes_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_epochs_l = [
        epochs_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_curriculum_l = [
        curriculum_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_regen_l = [
        regen_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_val_set_l = [
        val_set_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]
    all_load_l = [
        load_l[
            executions_per_machine * i : executions_per_machine * (i + 1)
        ]
        for i in range(len(mode_l) // executions_per_machine)
    ]

    for i in range(len(om_name_l)):
        job = Job(
            om_name_l[i],
            om_time_l[i],
            om_num_nodes_l[i],
            om_cpus_l[i],
            om_mem_l[i],
            om_gpus_l[i],
            om_gpu_size_l[i],
            om_partition_l[i],
        )
        job.add_executions(
            all_mode_l[i],
            all_run_name_l[i],
            all_num_nodes_l[i],
            all_epochs_l[i],
            all_curriculum_l[i],
            all_regen_l[i],
            all_val_set_l[i],
            all_load_l[i],
        )
        job.generate()


def validate_load_l(load_l):
    """Validate load_l."""
    base_dir = pathlib.Path(__file__).parent.absolute().parents[0]
    log_dir = os.path.join(base_dir, "logs")

    for load in load_l:
        task, nodes = load.split("-")[0], load.split("-")[1]
        parent_dir = f"{task}-{nodes}"
        target_dir = os.path.join(log_dir, parent_dir, load)
        assert os.path.isdir(target_dir)


def validate_num_nodes(num_nodes_l, val_set_l, load_l):
    """Validate number of nodes."""
    for i in range(len(num_nodes_l)):
        num_nodes = str(num_nodes_l[i])
        val_nodes = val_set_l[i].split("-")[1]
        load_nodes = load_l[i].split("-")[1]
        assert num_nodes == val_nodes
        assert num_nodes == load_nodes


if __name__ == "__main__":

    # resetting all-run.sh
    open("run-all.sh", 'w').close()
    with open("run-all.sh", "a+") as f:
        f.write("#!/bin/bash\n\n")


    # generating execution parameters
    mode_l = ["all" for i in range(40)]

    run_name_l = ["static-exp-{:0>2d}-epochs-30".format(i) for i in range(10)]
    run_name_l += ["regen-exp-{:0>2d}-epochs-30".format(i) for i in range(10)]
    run_name_l += ["static-exp-{:0>2d}-epochs-30".format(i) for i in range(10)]
    run_name_l += ["regen-exp-{:0>2d}-epochs-30".format(i) for i in range(10)]

    num_nodes_l = [20 for i in range(20)]
    num_nodes_l += [100 for i in range(20)]

    epochs_l = [30 for i in range(40)]

    curriculum_l = [i for i in range(10)]
    curriculum_l += [i for i in range(10)]
    curriculum_l += [i for i in range(10)]
    curriculum_l += [i for i in range(10)]

    regen_l = [False for i in range(10)]
    regen_l += [True for i in range(10)]
    regen_l += [False for i in range(10)]
    regen_l += [True for i in range(10)]

    val_set_l = ["tsp-20-val-1000.npy" for i in range(20)]
    val_set_l += ["tsp-100-val-1000.npy" for i in range(20)]

    load_l = [None for i in range(40)]

    # validating
    # validate_load_l(load_l)
    validate_num_nodes(num_nodes_l, val_set_l, load_l)

    # execution dict
    execution_dict_small = {
        "mode_l": mode_l[:20],
        "run_name_l": run_name_l[:20],
        "num_nodes_l": num_nodes_l[:20],
        "epochs_l": epochs_l[:20],
        "curriculum_l": curriculum_l[:20],
        "regen_l": regen_l[:20],
        "val_set_l": val_set_l[:20],
        "load_l": load_l[:20],
    }
    execution_dict_large = {
        "mode_l": mode_l[20:],
        "run_name_l": run_name_l[20:],
        "num_nodes_l": num_nodes_l[20:],
        "epochs_l": epochs_l[20:],
        "curriculum_l": curriculum_l[20:],
        "regen_l": regen_l[20:],
        "val_set_l": val_set_l[20:],
        "load_l": load_l[20:],
    }

    # generating scripts
    om_name_l = ["e20-static-batch-{:0>2d}".format(i) for i in range(5)]
    om_name_l += ["e20-regen-batch-{:0>2d}".format(i) for i in range(5)]
    om_time_l = [72 for i in range(10)]
    om_num_nodes_l = [1 for i in range(10)]
    om_cpus_l = [4 for i in range(5)]
    om_cpus_l += [8 for i in range(5)]
    om_mem_l = [12 for i in range(10)]
    om_gpus_l = [1 for i in range(10)]
    om_gpu_size_l = [6 for i in range(10)]
    om_partition_l = ["normal" for i in range(10)]
    execution_dict = execution_dict_small

    generate_jobs(
        om_name_l,
        om_time_l,
        om_num_nodes_l,
        om_cpus_l,
        om_mem_l,
        om_gpus_l,
        om_gpu_size_l,
        om_partition_l,
        execution_dict,
    )

    om_name_l = ["e100-static-exp-{:0>2d}".format(i) for i in range(10)]
    om_name_l += ["e100-regen-exp-{:0>2d}".format(i) for i in range(10)]
    om_time_l = [72 for i in range(20)]
    om_num_nodes_l = [1 for i in range(20)]
    om_cpus_l = [16 for i in range(20)]
    om_mem_l = [16 for i in range(20)]
    om_gpus_l = [1 for i in range(20)]
    om_gpu_size_l = [12 for i in range(20)]
    om_partition_l = ["normal" for i in range(20)]
    execution_dict = execution_dict_large

    generate_jobs(
        om_name_l,
        om_time_l,
        om_num_nodes_l,
        om_cpus_l,
        om_mem_l,
        om_gpus_l,
        om_gpu_size_l,
        om_partition_l,
        execution_dict,
    )
