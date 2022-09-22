import torch

from sbi.inference.snle import SNLE_A

from sbi_feature_importance.utils import record_scaler
from sbi_feature_importance.snle import ReducablePosterior, MarginalLikelihoodEstimator, CalibratedPrior
from sbi_feature_importance.metrics import sample_kl

from sbi_feature_importance.experiment_helper import (
    SimpleDB,
    Task,
    TaskManager,
    str2int,
)

import argparse
from torch import manual_seed as tseed
from numpy.random import seed as npseed


#-------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", help="select marginal policy", default="experimental")
parser.add_argument("-s", "--seed", help="set random seed", default=0, type=int)
parser.add_argument("-f", "--featuresets", help="set method to select feature subsets. 'best' or 'random' ", default="best")
parser.add_argument("-n", "--name", help="set name of the experiment.", default="default_experiment")
parser.add_argument("-w", "--sample_with", help="whether to sample with mcmc or rejection.", default="mcmc")
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
x_o = data.query("x_o")[:,features]
prior = data.query("prior")
x = data.query("x")[:,features]
theta = data.query("theta")

all_dims = features

#-------------------------------------------------------------------------------
# define training and sampling routines
def training_loop(theta, x, dims, prior, marginal_policy, method_tag):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posterior_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)

    inference = SNLE_A(prior, show_progress_bars=False, density_estimator="mdn")
    inference_ = MarginalLikelihoodEstimator(inference)
    assert "experimental" in marginal_policy, "tree search only works for experimental loss"
    inference_.set_marginal_loss_policy(marginal_policy, num_subsets=1, subset_len=1, gamma=0)

    estimator = inference_.append_simulations(theta, x[:,dims]).train(training_batch_size=100)
    calibrated_prior = CalibratedPrior(inference._prior).train(theta,x[:, dims])
    posterior = inference_.build_posterior(estimator, prior=calibrated_prior, sample_with=main_args.sample_with)
    
    db.write(f"posterior_{method_tag}_{dims}", posterior)
    return posterior

def approx_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posthoc_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    full_posterior = db.query(f"posterior_{method_tag}_{all_dims}")
    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dims)
    
    thetas = posterior.sample((n_samples,), context, **kwargs)

    db.write(f"posthoc_{method_tag}_{dims}", thetas, mode="replace, disc")
    return thetas

#-------------------------------------------------------------------------------
# create tasks and fill task queue
method_tag = f"{main_args.featuresets}_{main_args.sample_with}_{main_args.seed}"

task_queue = []
args = (theta, x, all_dims, prior, main_args.policy, method_tag)
kwargs = {}
task = Task(task=training_loop, args=args, kwargs=kwargs, name=f"train_{method_tag}_{all_dims}", priority=4)
task.execute()

args = (1000, x_o, all_dims, method_tag)
kwargs = {}
task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{all_dims}", priority=2)
task.execute()

good_fts = []
X = db.query(f"posthoc_{method_tag}_{all_dims}")
for i in range(len(all_dims)):
    task_queue = []
    remaining_dims = [d for d in all_dims if d not in good_fts]
    kls = torch.zeros((len(remaining_dims),))
    for ft in remaining_dims:
        dims = sorted(good_fts.copy() + [ft])
        
        args = (1000, x_o, dims, method_tag)
        kwargs = {"warmup_steps":100}
        task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{dims}", priority=2)
        task_queue.append(task)
    
    mp_queue = TaskManager(task_queue, main_args.max_workers)
    mp_queue.execute_tasks()

    for i, ft in enumerate(remaining_dims):
        dims = sorted(good_fts.copy() + [ft])
        Y = db.query(f"posthoc_{method_tag}_{dims}")
        kls[i] = sample_kl(X, Y)
    assert not any(abs(kls).isnan()) and not any(abs(kls).isinf()), "KLs contain NaNs or Infs."
    
    # choose best feature
    if main_args.featuresets == "best":
        selected_ft_idx = int(torch.argmin(kls))
        next_best_ft = remaining_dims[selected_ft_idx]
        selected_kl = kls[selected_ft_idx]

    # choose random feature
    elif main_args.featuresets == "random":
        selected_ft_idx = torch.randint(0,len(remaining_dims),(1,))
        next_best_ft = remaining_dims[selected_ft_idx]
        selected_kl = kls[selected_ft_idx]
    else:
        raise ValueError("Provide featuresets kwarg. Either 'best' or 'random' ")
    print(next_best_ft, selected_kl, kls)

    good_fts.append(next_best_ft)
    record_scaler(selected_kl, next_best_ft, db.location + f"/{method_tag}_best_fts.txt")
print(good_fts)