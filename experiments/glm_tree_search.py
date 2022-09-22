from numpy import argmin
import torch

from sbi_feature_importance.toymodels import GLM
from sbi.inference.snle import SNLE_A

from sbi_feature_importance.utils import random_subsets_of_dims, skip_dims, record_scaler, now
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
parser.add_argument("-p", "--policy", help="select marginal policy", default="default")
parser.add_argument("-s", "--seed", help="set random seed", default=0, type=int)
parser.add_argument("-n", "--name", help="set name of the experiment.", default="default_experiment")
parser.add_argument("-w", "--sample_with", help="whether to sample with mcmc or rejection.", default="rejection")
parser.add_argument("-m", "--max_workers", help="how many workers to use.", default=1, type=int)

main_args = parser.parse_args()

# Set random seed
npseed(main_args.seed)
tseed(main_args.seed)

# init database and logger
storage_location = "../results/"
# storage_location = "./"
name = main_args.name
db = SimpleDB(storage_location + name)

# MISSING init logger

#-------------------------------------------------------------------------------
# define HH model / simulator
glm = GLM(seed=main_args.seed)
prior = glm.Prior

theta_o, x_o = glm.sample_joint((1,))

db.write("prior", prior)
db.write(f"theta_o_{main_args.seed}", theta_o)
db.write(f"x_o_{main_args.seed}", x_o)

all_dims = list(range(4))


theta, x =  glm.sample_joint((1000,))
# db.write("x", x)
# db.write("theta", theta)

#-------------------------------------------------------------------------------
# define training and sampling routines
def training_loop(theta, x, dims, prior, marginal_policy, method_tag):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posterior_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)

    inference = SNLE_A(prior, show_progress_bars=False, density_estimator="mdn")
    inference_ = MarginalLikelihoodEstimator(inference)
    if "random" in marginal_policy:
        inference_.set_marginal_loss_policy(marginal_policy, num_subsets=3, max_len=3)
    elif "experimental" in marginal_policy:
        inference_.set_marginal_loss_policy(marginal_policy, num_subsets=3, subset_len=2)
    elif "default" in marginal_policy:
        inference_.set_marginal_loss_policy(marginal_policy)
    else:
        raise ValueError("no valid loss policy selected")

    estimator = inference_.append_simulations(theta, x[:,dims]).train(training_batch_size=100)
    posterior = inference_.build_posterior(estimator, prior=prior, sample_with=main_args.sample_with)
    
    db.write(f"posterior_{method_tag}_{dims}", posterior)
    return posterior

def direct_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"direct_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    posterior = db.query(f"posterior_{method_tag}_{dims}")
    
    thetas = posterior.sample((n_samples,), context[:,dims].view(1,-1), **kwargs)

    db.write(f"direct_{method_tag}_{dims}", thetas)
    return thetas

def approx_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posthoc_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    full_posterior = db.query(f"posterior_{method_tag}_{all_dims}")
    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dims)
    
    thetas = posterior.sample((n_samples,), context, **kwargs)

    db.write(f"posthoc_{method_tag}_{dims}", thetas)
    return thetas

#-------------------------------------------------------------------------------
# create tasks and fill task queue
method_tag = f"{main_args.policy}_{main_args.sample_with}_{main_args.seed}"

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
        kwargs = {}
        task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{dims}", priority=2)
        task_queue.append(task)
    
    mp_queue = TaskManager(task_queue, main_args.max_workers)
    mp_queue.execute_tasks()

    for i, ft in enumerate(remaining_dims):
        dims = sorted(good_fts.copy() + [ft])
        Y = db.query(f"posthoc_{method_tag}_{dims}")
        kls[i] = sample_kl(X, Y)
    print(next_best_ft, kls.min())
    assert not any(abs(kls).isnan()) and not any(abs(kls).isinf()), "KLs contain NaNs or Infs."
    # choose best feature
    next_best_ft = remaining_dims[int(torch.argmin(kls))]
    
    # # choose random feature
    # next_best_ft = remaining_dims[torch.randint(0,len(remaining_dims),(1,))]

    good_fts.append(next_best_ft)
    record_scaler(kls.min(), next_best_ft, db.location + "/best_fts.txt")
print(good_fts)