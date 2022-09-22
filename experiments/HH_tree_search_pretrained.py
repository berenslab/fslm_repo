import torch

from sbi.inference.snle import SNLE_A

from sbi_feature_importance.utils import record_scaler
from sbi_feature_importance.snle import ReducablePosterior, MarginalLikelihoodEstimator, CalibratedPrior
from sbi_feature_importance.metrics import sample_kl, mmd

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
parser.add_argument("-q", "--posterior_dir", help="from which directory to import pretrained posterior")
parser.add_argument("-t", "--tag_of_posterior", help="specify which posterior by tag")

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

posterior_storage_location = "./"
posterior_name = main_args.posterior_dir 
posteriors = SimpleDB(posterior_storage_location + posterior_name)

# MISSING init logger

#-------------------------------------------------------------------------------
# import prior and observations
features = [0,1,2,3,8,13,18,19,21,22]
x_o = data.query("x_o")[:,features]
prior = data.query("prior")

all_dims = list(range(len(features)))

#-------------------------------------------------------------------------------
def approx_sampling_loop(full_posterior, n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posthoc_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dims)
    
    thetas = posterior.sample((n_samples,), context, **kwargs)

    db.write(f"posthoc_{method_tag}_{dims}", thetas, mode="replace, disc")
    return thetas

#-------------------------------------------------------------------------------
# create tasks and fill task queue
method_tag = f"{main_args.featuresets}_{main_args.sample_with}_{main_args.seed}_{main_args.tag_of_posterior}"
full_posterior_dict = posteriors.find(main_args.tag_of_posterior)
full_posterior_dict = {key:val for key, val in full_posterior_dict.items() if "posterior" in key}
if len(full_posterior_dict) ==1:
    key, full_posterior = full_posterior_dict.popitem()
    db.write(f"posterior_loc_{method_tag}", f"the location of the posterior used: {posteriors.location}{key}", mode="replace, disc")
else:
    raise ValueError("The tag identifies none, two or more posteriors. be more specific")

args = (full_posterior, 1000, x_o, all_dims, method_tag)
kwargs = {"warmup_steps":100}
task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{all_dims}", priority=2)
task.execute()

good_fts = []
X = db.query(f"posthoc_{method_tag}_{all_dims}")
for i in range(len(all_dims)):
    task_queue = []
    remaining_dims = [d for d in all_dims if d not in good_fts]
    discrepancies = torch.zeros((len(remaining_dims),))
    for ft in remaining_dims:
        dims = sorted(good_fts.copy() + [ft])
        
        args = (full_posterior, 1000, x_o, dims, method_tag)
        kwargs = {"warmup_steps":100}
        task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{dims}", priority=2)
        task_queue.append(task)
    
    mp_queue = TaskManager(task_queue, main_args.max_workers)
    mp_queue.execute_tasks()

    for i, ft in enumerate(remaining_dims):
        dims = sorted(good_fts.copy() + [ft])
        Y = db.query(f"posthoc_{method_tag}_{dims}")
        # discrepancy[i] = sample_kl(X, Y)
        discrepancies[i] = mmd(X, Y)
    assert not any(abs(discrepancies).isnan()) and not any(abs(discrepancies).isinf()), "Metric contains NaNs or Infs."
    
    # choose best feature
    if main_args.featuresets == "best":
        selected_ft_idx = int(torch.argmin(discrepancies))
        next_best_ft = remaining_dims[selected_ft_idx]
        selected_discrepancy = discrepancies[selected_ft_idx]

    # choose random feature
    elif main_args.featuresets == "random":
        selected_ft_idx = int(torch.randint(0,len(remaining_dims),(1,)))
        next_best_ft = remaining_dims[selected_ft_idx]
        selected_discrepancy = discrepancies[selected_ft_idx]
    else:
        raise ValueError("Provide featuresets kwarg. Either 'best' or 'random' ")
    print(next_best_ft, selected_discrepancy, discrepancies)

    good_fts.append(next_best_ft)
    record_scaler(selected_discrepancy, next_best_ft, db.location + f"/{method_tag}_best_fts.txt")
print(good_fts)