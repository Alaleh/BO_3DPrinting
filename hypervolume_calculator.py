import copy
import torch
import argparse
import numpy as np
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated


def read_file_as_list(f_name, delimiter):
    with open(f_name, 'r') as f:
        input_file = f.readlines()
    f.close()
    file_vals = [[float(x) for x in input_file[i].split(delimiter)] for i in range(len(input_file))]
    for i in range(len(file_vals)):
        file_vals[i][0] = file_vals[i][0]
    return np.asarray(file_vals)


def calc_hypervolume(objectives, rp):
    hv = Hypervolume(rp)
    objectives = torch.tensor((objectives), dtype=torch.float64)
    pareto_mask = is_non_dominated(objectives)
    pareto_y = objectives[pareto_mask]
    volume = hv.compute(pareto_y)
    return volume, [i+1 for i in range(len(pareto_mask)) if pareto_mask[i]]


def main():
    parser = argparse.ArgumentParser(description='3D-printing')
    parser.add_argument('--experiment', default=2, type=int, help='Experiment number (1 or 2)')
    args = parser.parse_args()
    experiment_number = args.experiment

    data = read_file_as_list('results/experiment_' + str(experiment_number) + '/output.txt', ',')

    if experiment_number == 1:
        rp = torch.tensor([-270.0, -20.0, -20.0, -20.0], dtype=torch.float64)
    elif experiment_number == 2:
        rp = torch.tensor([-230, -4.0, -4.0, -4.0], dtype=torch.float64)

    init_count = 4
    print(len(data))
    for i in range(init_count, len(data) + 1):
        hv_c, par = calc_hypervolume(data[:i], rp)
        # print("At iteration number ", i-4, " the value of hypervolume is ", hv_c)
        # print("At iteration number ", i-4, " the points in the pareto front are on row number: ", par)
        # # print(par)
        print(hv_c)


if __name__ == '__main__':
    main()
