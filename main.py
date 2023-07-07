import os
import gc
import copy
import time
import torch
import random
import numpy as np
from botorch import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list, optimize_acqf_discrete, _split_batch_eval_acqf


# code works for maximization of the objectives
# multiply the objective that needs to be minimized with a negative sign

def initialize_model(train_x, train_obj):
    '''
        function to initialize GP model
        train_x: inputs, train_obj: outputs
        covar_module: special custom-designed kernel if required
    '''
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def read_file_as_list(f_name, delimiter):
    with open(f_name, 'r') as f:
        input_file = f.readlines()
    f.close()
    file_vals = [[float(x) for x in input_file[i].split(delimiter)] for i in range(len(input_file))]
    return file_vals


def read_input_output():
    inputs = read_file_as_list('results/input.txt', ',')
    outputs = read_file_as_list('results/output.txt', ',')
    train_x = torch.tensor(inputs)
    train_y = torch.tensor(outputs)

    return train_x, train_y, inputs


def generate_feasible_points(already_evaluated, normalize=True):
    """
    returns a list of lists. Each sub-list contains [nozzle_feature, speed, pressure]
    """
    data_points = []
    n_ranges = [[0.26, 0.61], [4.0, 15.0], [98.0, 449.0]]

    filenames_input_space = os.listdir('input_space/')
    filenames_input_space.sort()

    for f in range(len(filenames_input_space)):
        f_name = filenames_input_space[f]
        t = f_name.split('_')
        r1, r2 = float(t[0][-4:]), float(t[1][:4])
        # Ran into rounding error when calculated this in the code
        if f < 2:
            vals_size = 8
        else:
            vals_size = 10
        nozzle_values = [np.round(r1 + 0.01 * r, 2) for r in range(vals_size)]
        other_features = read_file_as_list('input_space/' + f_name, ',')
        for i in other_features:
            p1, p2 = i[2], i[1]
            vals_size = int((p2 - p1) / 0.1) + 1
            line_features = [np.round(p1 + 0.1 * p, 1) for p in range(vals_size)]
            for j in line_features:
                for k in nozzle_values:
                    new_v = [k, i[0], j]
                    if new_v not in already_evaluated:
                        data_points.append(new_v)

    res = torch.tensor(data_points, dtype=torch.double)

    if normalize:
        for i in range(len(n_ranges)):
            res[:, i] = (res[:, i] - n_ranges[i][0]) / (n_ranges[i][1] - n_ranges[i][0])

    return res


def edited_optimize_acqf_discrete(acq_function, choices, max_batch_size=1, device="cpu"):
    choices_batched = choices.unsqueeze(-2)
    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(acq_function=acq_function, X=choices_batched,
                                            max_batch_size=max_batch_size).to(device)
    best_idx = torch.argmax(acq_values)
    print(acq_values)
    return choices_batched[best_idx]


def clear_gpu_memory(variables=None):
    torch.cuda.empty_cache()
    gc.collect()
    del variables


def main():
    t0 = time.time()

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # print("Using ", device)

    NUM_FEATURES = 3
    NUM_OBJECTIVES = 4
    tkwargs = {"dtype": torch.double, "device": device}
    # thickness, speed, pressure range:
    input_ranges = [[0.26, 0.61], [4.0, 15.0], [98.0, 449.0]]
    output_ranges = [[-360, -10], [-20.0, 0.0], [-20.0, 0.0], [-20.0, 0.0]]

    train_x, train_y, used_points = read_input_output()
    train_x = train_x.to(**tkwargs)
    train_y = train_y.to(**tkwargs)

    if not len(used_points):
        feasible_points = generate_feasible_points(used_points, normalize=False)
        random_initials = np.random.randint(0, len(feasible_points), size=4)
        for i in random_initials:
            print(feasible_points[i].tolist())
        return

    for i in range(NUM_FEATURES):
        train_x[:, i] = (train_x[:, i] - input_ranges[i][0]) / (input_ranges[i][1] - input_ranges[i][0])

    for i in range(NUM_OBJECTIVES):
        train_y[:, i] = (train_y[:, i] - output_ranges[i][0]) / (output_ranges[i][1] - output_ranges[i][0])

    # real reference point which is a lower bound on all objectives
    # ref_point = torch.tensor([-101.01,-20.01,-3.01,-1.01], **tkwargs)
    ref_point = torch.tensor([0.0, 0.0, 0.0, 0.0], **tkwargs)

    mll, model = initialize_model(train_x, train_y)  # initialize the model
    # print("Finished initializing the model")

    fit_gpytorch_model(mll)  # fit the GP model
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
    acq_func = ExpectedHypervolumeImprovement(model=model, ref_point=ref_point.tolist(), partitioning=partitioning, )
    # print("Finished defining the acquisition function")

    # acquisition function optimization - discrete
    feasible_points = generate_feasible_points(used_points).to(**tkwargs)
    # print("Finished generating the set of valid possible inputs")
    # print("There were ", len(feasible_points), " Candidates")
    clear_gpu_memory()
    # feasible_points = koila.lazy(feasible_points)
    new_input = edited_optimize_acqf_discrete(acq_function=acq_func, choices=feasible_points,
                                              max_batch_size=1024, device=device)
    # print("Finished optimizing the acquisition function")

    train_x_to_evaluate = []

    for i in range(NUM_FEATURES):
        train_x_to_evaluate.append(
            np.round(input_ranges[i][0] + (input_ranges[i][1] - input_ranges[i][0]) * new_input[0][i].item(), 2))

    print('input to evaluate : layer thickness = ', train_x_to_evaluate[0], ' speed = ', train_x_to_evaluate[1],
          ' , and pressure = ', train_x_to_evaluate[2])
    with open('results/input.txt', 'a+') as f:
        print(','.join([str(i) for i in train_x_to_evaluate]), file=f)

    # print("This iteration took ", time.time() - t0, "Seconds")


if __name__ == '__main__':
    main()
