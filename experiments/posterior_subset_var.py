from logging import warning
import torch

from sbi.inference.snle import SNLE_A

from sbi_feature_importance.utils import random_subsets_of_dims, skip_dims, record_scaler, now
from sbi_feature_importance.snle import ReducablePosterior, MarginalLikelihoodEstimator, CalibratedPrior

from sbi_feature_importance.experiment_helper import (
    SimpleDB,
    Task,
    TaskManager,
    str2int,
)

import argparse
from functools import partial
from torch import manual_seed as tseed
from numpy.random import seed as npseed


#-------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", help="select marginal policy", default="default")
parser.add_argument("-k", "--policy_kwargs", help="kwargs for marginal policy", default="{}")
parser.add_argument("-s", "--seed", help="set random seed", default=0, type=int)
parser.add_argument("-n", "--name", help="set name of the experiment.", default="default_experiment")
parser.add_argument("-w", "--sample_with", help="whether to sample with mcmc or rejection.", default="rejection")
parser.add_argument("-m", "--max_workers", help="how many workers to use.", default=-1, type=int)
parser.add_argument("-d", "--data_dir", help="from which directory to import presimulated the data")

main_args = parser.parse_args()

# Set random seed
npseed(main_args.seed)
tseed(main_args.seed)

# init database and logger
storage_location = "./"
name = main_args.name
db = SimpleDB(storage_location + name)

data_storage_location = "./"
data_name = main_args.data_dir
data = SimpleDB(data_storage_location + data_name)

# MISSING init logger

#-------------------------------------------------------------------------------
# import prior and observations

features = [0,1,2,3,8,13,18,19,21,22]
x_o = data.query("x_o")
prior = data.query("prior")
x = data.query("x")
theta = data.query("theta")

#-------------------------------------------------------------------------------
def training_loop(theta, x, dims, prior, marginal_policy, method_tag, policy_kwargs):    
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posterior_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    inference = SNLE_A(prior, show_progress_bars=False, density_estimator="mdn")
    inference_ = MarginalLikelihoodEstimator(inference)
    inference_.set_marginal_loss_policy(marginal_policy, **policy_kwargs)
    
    t4 = now()
    estimator = inference_.append_simulations(theta, x[:,dims]).train(training_batch_size=100)
    calibrated_prior = CalibratedPrior(inference._prior).train(theta,x[:, dims])
    posterior = inference_.build_posterior(estimator, prior=calibrated_prior, sample_with=main_args.sample_with)
    record_scaler(now()-t4, f"train_{method_tag}_{dims}", db.location + "/timings.txt")
    
    db.write(f"posterior_{method_tag}_{dims}", posterior)
    return posterior

def direct_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"direct_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    posterior = db.query(f"posterior_{method_tag}_{dims}")
    
    t5 = now()
    thetas = posterior.sample((n_samples,), context[:,dims].view(1,-1), **kwargs)
    record_scaler(now()-t5, f"direct_{method_tag}_{dims}", db.location + "/timings.txt")

    db.write(f"direct_{method_tag}_{dims}", thetas)
    return thetas

def approx_sampling_loop(n_samples, context, dims, method_tag, dim_idxs, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posthoc_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    full_posterior = db.query(f"posterior_{method_tag}_{features}")
    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dim_idxs)
    
    t6 = now()
    thetas = posterior.sample((n_samples,), context[:,features], **kwargs)
    record_scaler(now()-t6, f"posthoc_{method_tag}_{dims}", db.location + "/timings.txt")

    db.write(f"posthoc_{method_tag}_{dims}", thetas)
    return thetas


method_tag = f"{main_args.policy}_{main_args.sample_with}_{main_args.seed}"
if "num_subsets" in main_args.policy_kwargs:
    method_tag += f"_{eval(main_args.policy_kwargs)['num_subsets']}"
if "subset_len" in main_args.policy_kwargs:
    method_tag += f"_{eval(main_args.policy_kwargs)['subset_len']}"

task_queue = []
if main_args.policy == "default":
    for dims in [[1,2,3,8,13,18,19,21,22],[0,1,3,8,13,18,19,21,22]]:
        args = (theta, x, dims, prior, main_args.policy, method_tag, eval(main_args.policy_kwargs))
        kwargs = {}
        task = Task(task=training_loop, args=args, kwargs=kwargs, name=f"train_{method_tag}_{dims}", priority=4)
        task_queue.append(task)
        
        args = (3000, x_o, dims, method_tag)
        kwargs = {"warmup_steps":100}
        task = Task(task=direct_sampling_loop, args=args, kwargs=kwargs, name=f"direct_{method_tag}_{dims}", priority=2)
        task_queue.append(task)
else:
    args = (theta, x, features, prior, main_args.policy, method_tag, eval(main_args.policy_kwargs))
    kwargs = {}
    task = Task(task=training_loop, args=args, kwargs=kwargs, name=f"train_{method_tag}_{features}", priority=4)
    task_queue.append(task)
    all_dim_idxs = list(range(len(features)))
    for dims, dim_idxs in zip([[1,2,3,8,13,18,19,21,22],[0,1,3,8,13,18,19,21,22]], [[1,2,3,4,5,6,7,8,9],[0,1,3,4,5,6,7,8,9]]):
        args = (3000, x_o, dims, method_tag, dim_idxs)
        kwargs = {"warmup_steps":100}
        task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{dims}", priority=2)
        task_queue.append(task)


mp_queue = TaskManager(task_queue, main_args.max_workers)
mp_queue.execute_tasks()