import copy
import torch
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
    return volume


def main():
    data = read_file_as_list('results/output.txt', ',')
    print(data)
    rp = torch.tensor([-270.0, -20.0, -20.0, -20.0], dtype=torch.float64)
    init_count = 4
    print(len(data))
    for i in range(init_count, len(data) + 1):
        print(i-4, calc_hypervolume(data[:i], rp))


if __name__ == '__main__':
    main()
