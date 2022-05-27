import copy
import random

import torch
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list, optimize_acqf_discrete
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement


# code works for maximization of the objectives
# multiply the objective that needs to be minimized with a negative sign

def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    '''
        function to initialize GP model
        train_x: inputs, train_obj: outputs
        covar_module: special custom-designed kernel if required
    '''
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def read_input_output():
    with open('input.txt', 'r') as f:
        input_file = f.readlines()
    f.close()

    with open('output.txt', 'r') as f:
        output_file = f.readlines()
    f.close()

    inputs = []
    outputs = []
    for i in range(len(output_file)):
        inputs.append([float(x) for x in input_file[i][1:-2].split(', ')])
        outputs.append([float(x) for x in output_file[i].split(',')])
    train_x = torch.tensor(inputs)
    train_y = torch.tensor(outputs)

    return train_x, train_y, inputs


def generate_feasible_points(already_evaluated):
    data_points = []
    n_ranges = [[4.0, 15.0], [115.0, 250.0]]
    value_ranges = [[4, 155, 115], [4.1, 156.5, 116.5], [4.2, 158, 118], [4.3, 159.5, 119.5], [4.4, 161, 121],
                    [4.5, 162.5, 122.5], [4.6, 164, 124], [4.7, 165.5, 125.5], [4.8, 167, 127], [4.9, 168.5, 128.5],
                    [5, 170, 130], [5.1, 171, 131], [5.2, 172, 132], [5.3, 173, 133], [5.4, 174, 134], [5.5, 175, 135],
                    [5.6, 176, 136], [5.7, 177, 137], [5.8, 178, 138], [5.9, 179, 139], [6, 180, 140],
                    [6.1, 181.5, 141], [6.2, 183, 142], [6.3, 184.5, 143], [6.4, 186, 144], [6.5, 187.5, 145],
                    [6.6, 189, 146], [6.7, 190.5, 147], [6.8, 192, 148], [6.9, 193.5, 149], [7, 195, 150],
                    [7.1, 196, 151], [7.2, 197, 152], [7.3, 198, 153], [7.4, 199, 154], [7.5, 200, 155],
                    [7.6, 201, 156], [7.7, 202, 157], [7.8, 203, 158], [7.9, 204, 159], [8, 205, 160],
                    [8.1, 207, 161.5], [8.2, 209, 163], [8.3, 211, 164.5], [8.4, 213, 166], [8.5, 215, 167.5],
                    [8.6, 217, 169], [8.7, 219, 170.5], [8.8, 221, 172], [8.9, 223, 173.5], [9, 225, 175],
                    [9.1, 226, 177], [9.2, 227, 179], [9.3, 228, 181], [9.4, 229, 183], [9.5, 230, 185],
                    [9.6, 231, 187], [9.7, 232, 189], [9.8, 233, 191], [9.9, 234, 193], [10, 235, 195],
                    [10.1, 235.5, 195.5], [10.2, 236, 196], [10.3, 236.5, 196.5], [10.4, 237, 197],
                    [10.5, 237.5, 197.5], [10.6, 238, 198], [10.7, 238.5, 198.5], [10.8, 239, 199],
                    [10.9, 239.5, 199.5], [11, 240, 200], [11.1, 240, 200.5], [11.2, 240, 201], [11.3, 240, 201.5],
                    [11.4, 240, 202], [11.5, 240, 202.5], [11.6, 240, 203], [11.7, 240, 203.5], [11.8, 240, 204],
                    [11.9, 240, 204.5], [12, 240, 205], [12.1, 240, 205], [12.2, 240, 205], [12.3, 240, 205],
                    [12.4, 240, 205], [12.5, 240, 205], [12.6, 240, 205], [12.7, 240, 205], [12.8, 240, 205],
                    [12.9, 240, 205], [13, 240, 205], [13.1, 240.5, 205.5], [13.2, 241, 206], [13.3, 241.5, 206.5],
                    [13.4, 242, 207], [13.5, 242.5, 207.5], [13.6, 243, 208], [13.7, 243.5, 208.5], [13.8, 244, 209],
                    [13.9, 244.5, 209.5], [14, 245, 210], [14.1, 245.5, 211], [14.2, 246, 212], [14.3, 246.5, 213],
                    [14.4, 247, 214], [14.5, 247.5, 215], [14.6, 248, 216], [14.7, 248.5, 217], [14.8, 249, 218],
                    [14.9, 249.5, 219], [15, 250, 220]]
    for i in value_ranges:
        for j in np.arange(i[2], i[1] + 0.1, 0.1):
            if [i[0], np.round(j, 1)] not in already_evaluated:
                data_points.append([i[0], np.round(j, 1)])

    res = torch.tensor(data_points, dtype=torch.float32)

    for i in range(len(n_ranges)):
        res[:, i] = (res[:, i] - n_ranges[i][0]) / (n_ranges[i][1] - n_ranges[i][0])

    # print(len(res))

    return res


def main():
    # number of outputs for each evaluation
    NUM_OBJECTIVES = 5
    NUM_FEATURES = 2
    # speed, pressure range:
    input_ranges = [[4.0, 15.0], [115.0, 250.0]]

    train_x, train_y, used_points = read_input_output()

    for i in range(NUM_FEATURES):
        train_x[:, i] = (train_x[:, i] - input_ranges[i][0]) / (input_ranges[i][1] - input_ranges[i][0])

    # reference point which is a lower bound on all objectives
    ref_point = torch.tensor([0.] * NUM_OBJECTIVES)

    # print("y means: ", [torch.mean(train_y[:, i]) for i in range(NUM_OBJECTIVES)])

    if len(train_x) > 1:
        for i in range(NUM_OBJECTIVES):
            ref_point[i] = (ref_point[i] - torch.mean(train_y[:, i])) / (torch.std(train_y[:, i]))
            train_y[:, i] = (train_y[:, i] - torch.mean(train_y[:, i])) / (torch.std(train_y[:, i]))

    # print(f'reference point {ref_point}')
    mll, model = initialize_model(train_x, train_y)  # initialize the model
    fit_gpytorch_model(mll)  # fit the GP model

    # defining the acquisition function
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
    acq_func = ExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning, )

    # acquisition function optimization - continuous
    # new_input, acq_vals = optimize_acqf(acq_function=acq_func,
    #                 bounds=torch.tensor([[0, 0], [1, 1]], dtype=torch.float32), q=200, Sequential=False,
    #                 num_restarts=10, raw_samples=1024)
    #

    # acquisition function optimization - discrete
    feasible_points = generate_feasible_points(used_points)
    new_input, acq_vals = optimize_acqf_discrete(acq_function=acq_func, q=1,
                                                 choices=feasible_points, dtype=torch.float32)

    train_x_to_evaluate = []

    for i in range(NUM_FEATURES):
        train_x_to_evaluate.append(
            np.round(input_ranges[i][0] + (input_ranges[i][1] - input_ranges[i][0]) * new_input[0][i].item(), 1))

    print('input to evaluate : speed = ', train_x_to_evaluate[0], ' , and pressure = ', train_x_to_evaluate[1])
    with open('input.txt', 'a+') as f:
        print(train_x_to_evaluate, file=f)
    # torch.save({'inputs': train_x, 'outputs': train_y, 'gp_state_dict': model.state_dict()},
    #            'multiobj_bo_num_iters_' + str(len(train_x)) + '.pkl')


if __name__ == '__main__':
    main()
