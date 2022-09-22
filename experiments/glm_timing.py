import torch

from sbi_feature_importance.toymodels import GLM
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
from torch import manual_seed as tseed
from numpy.random import seed as npseed


#-------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--mode", help="select posthoc or direct", default="posthoc")
parser.add_argument("-s", "--seed", help="set random seed", default=0, type=int)
parser.add_argument("-f", "--featuresets", help="set method to obtain feature subsets. 'random' or 'skip' ", default="skip")
parser.add_argument("-n", "--name", help="set name of the experiment.", default="default_experiment")
parser.add_argument("-w", "--sample_with", help="whether to sample with mcmc or rejection.", default="rejection")
parser.add_argument("-m", "--max_workers", help="how many workers to use.", default=1, type=int)

main_args = parser.parse_args()

if main_args.featuresets == "random":
    get_subsets = random_subsets_of_dims
elif main_args.featuresets == "skip":
    get_subsets = skip_dims

# Set random seed
npseed(main_args.seed)
tseed(main_args.seed)

# init database and logger
storage_location = "./"
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


theta, x =  glm.sample_joint((10000,))
# db.write("x", x)
# db.write("theta", theta)

#-------------------------------------------------------------------------------
# define training and sampling routines
def training_loop(theta, x, dims, prior, method_tag):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"posterior_{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    
    t4 = now()
    inference = SNLE_A(prior, show_progress_bars=False, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x[:,dims]).train(training_batch_size=100)
    posterior = inference.build_posterior(estimator, sample_with=main_args.sample_with)
    record_scaler(now()-t4, f"train_{method_tag}_{dims}", db.location + "/timings.txt")
    
    db.write(f"posterior_{method_tag}_{dims}", posterior)
    return posterior

def direct_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"{method_tag}_{dims}") + main_args.seed
    npseed(seed)
    tseed(seed)
    
    posterior = db.query(f"posterior_{method_tag}_{dims}")
    
    t5 = now()
    thetas = posterior.sample((n_samples,), context[:,dims].view(1,-1), **kwargs)
    record_scaler(now()-t5, f"sample_{method_tag}_{dims}", db.location + "/timings.txt")

    db.write(f"{method_tag}_{dims}", thetas)
    return thetas

def approx_sampling_loop(n_samples, context, dims, method_tag, **kwargs):
    # ensures different subprocesses have a different but repeatable random state
    seed = str2int(f"{method_tag}_{dims}") + main_args.seed 
    npseed(seed)
    tseed(seed)
    
    full_posterior = db.query(f"posterior_{method_tag}_{all_dims}")
    posterior = ReducablePosterior(full_posterior)
    posterior.marginalise(dims)
    
    t6 = now()
    thetas = posterior.sample((n_samples,), context, **kwargs)
    record_scaler(now()-t6, f"sample_{method_tag}_{dims}", db.location + "/timings.txt")

    db.write(f"{method_tag}_{dims}", thetas)
    return thetas

#-------------------------------------------------------------------------------
# create tasks and fill task queue
method_tag = f"{main_args.mode}_{main_args.sample_with}_{main_args.seed}"

task_queue = []
if main_args.mode == "direct":
    for dims in [all_dims] + get_subsets(all_dims):
        args = (theta, x, dims, prior, method_tag)
        kwargs = {}
        task = Task(task=training_loop, args=args, kwargs=kwargs, name=f"train_{method_tag}_{dims}", priority=4)
        task_queue.append(task)
        
        args = (500, x_o, dims, method_tag)
        kwargs = {}
        task = Task(task=direct_sampling_loop, args=args, kwargs=kwargs, name=f"direct_{method_tag}_{dims}", priority=2)
        task_queue.append(task)
if main_args.mode == "posthoc":
    args = (theta, x, all_dims, prior, method_tag)
    kwargs = {}
    task = Task(task=training_loop, args=args, kwargs=kwargs, name=f"train_{method_tag}_{all_dims}", priority=4)
    task_queue.append(task)
    for dims in [all_dims] + get_subsets(all_dims):
        args = (500, x_o, dims, method_tag)
        kwargs = {}
        task = Task(task=approx_sampling_loop, args=args, kwargs=kwargs, name=f"posthoc_{method_tag}_{dims}", priority=2)
        task_queue.append(task)

#-------------------------------------------------------------------------------
# execute tasks in parallel and in order of task priority
mp_queue = TaskManager(task_queue, main_args.max_workers)
mp_queue.execute_tasks()