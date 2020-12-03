"""Code for generating loss plots from log.log files."""

import os
import glob
import shutil
import pathlib

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = pathlib.Path(__file__).parent.absolute().parents[1]
LOG_DIR = os.path.join(BASE_DIR, "logs")

def get_run_dirs():
    """Get directories of each run."""
    return sorted([p for p in glob.glob(os.path.join(LOG_DIR, "*", "*")) if os.path.isdir(p)])

def analyze_run_dir(run_dir):
    """Extract relevant lines from run directory."""
    log_path = os.path.join(run_dir, "log.log")
    with open(log_path) as f:
        content = f.readlines()
    relevant_lines = [line for line in content if "loss/reward/val reward" in line]
    return relevant_lines

def process_line(rel_line):
    """Process a relevant line."""
    col_idx_1 = rel_line.find(":")
    col_idx_2 = rel_line.find(":", col_idx_1 + 1)
    sub = rel_line[col_idx_1+1:col_idx_2]

    # remove whitespace
    sub_stripped = "".join(sub.split())
    vals = sub_stripped.split(",")[:3]
    float_vals = [float(val) for val in vals]

    # loss, training path lengths, validation path lengths
    return float_vals

def process_run_dir(run_dir):
    """Process a run directory."""
    rel_lines = analyze_run_dir(run_dir)
    processed_lines = [process_line(rel_line)[1:] for rel_line in rel_lines]

    # should be [training path lengths, validation path lengths]

    xs = [i for i in range(len(processed_lines))]
    train_ys = [l[0] for l in processed_lines]
    val_ys = [l[1] for l in processed_lines]

    # getting run name
    dirname = run_dir.split("/")[-1]
    subname = dirname.split("-")[:-1]
    title = "-".join(subname)

    # getting save path
    savepath = os.path.join(run_dir, "training_plot.png")

    # save current plot to directory
    create_training_plot(xs, train_ys, val_ys, title, savepath)

def create_training_plot(xs, train_ys, val_ys, title, savepath):
    """Makes a training plot."""
    plt.plot(xs, train_ys, label = "Train")
    plt.plot(xs, val_ys, label = "Val")
    plt.xlabel("Epochs")
    plt.ylabel("Avg Tour Length")
    plt.title(title)
    plt.legend()
    plt.savefig(savepath)
    plt.close()

def coagulate_training_plots():
    """Copies all training plots into a single directory."""
    plots_dir = os.path.join(LOG_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    all_plot_paths = [p for p in glob.glob(os.path.join(LOG_DIR, "*", "*", "training_plot.png"))]

    for plot_path in all_plot_paths:
        dirname = plot_path.split("/")[-2]
        subname = dirname.split("-")[:-1]
        title = "-".join(subname)
        dst = os.path.join(plots_dir, title + ".png")

        # copying
        shutil.copyfile(plot_path, dst)

def produce_loss_plots():
    """Produces loss plots for everything run in the log directory."""
    run_dirs = get_run_dirs()
    for curr_dir in run_dirs:
        process_run_dir(curr_dir)
    coagulate_training_plots()


if __name__ == "__main__":
    produce_loss_plots()
    
