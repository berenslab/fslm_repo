import torch

from ephys_helper.hh_simulator import HHSimulator
from sbi.utils.torchutils import BoxUniform
from brian2 import clear_cache

from sbi_feature_importance.utils import OptimisedPrior, record_scaler, now

from sbi_feature_importance.experiment_helper import (
    SimpleDB
)

import argparse
from torch import manual_seed as tseed
from numpy.random import seed as npseed


#-------------------------------------------------------------------------------
# init experiment

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="set random seed", default=0, type=int)
parser.add_argument("-n", "--name", help="set name of the experiment.", default="HHsim_default")
parser.add_argument("-m", "--max_workers", help="how many workers to use.", default=1, type=int)
parser.add_argument("-o", "--optimise_on", help="how many sapmles to optimise the prior on.", default=10000, type=int)
parser.add_argument("-t", "--train_on", help="how many samples to train on.", default=100000, type=int)

main_args = parser.parse_args()

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
model_parameters = torch.tensor(
    [
        [
            71.1,  # ENa [mV] : Reversal potential of sodium.
            -101.3,  # EK [mV] : Reversal potential of potassium.
            131.1,  # ECa [mV] : Reversal potential of calcium.
            36.0,  # T_1 [°C] :  36 °C from paper MartinPospischil et al.
            34,  # T_2 [°C] :  Experimental temperature.
            3.0,  # Q10 : temperature coeff.
            11.97,  # tau [ms] : Membrane time constant.
            126.2,  # input_res [Mohm] : Membrane input resistance.
            float("nan"),  # C [uF/cm^2] : Membrane capcitance per area.
            float("nan"),  # gNa [mS] : Channel conductance of sodium.
            float("nan"),  # gK [mS] : Channel conductance of potassium.
            float("nan"),  # gM [mS] : Channel conductance for adpt. K currents.
            float("nan"),  # gLeak [mS] : Leak conductance.
            float("nan"),  # gL [mS] : Channel conductance for Calcium current.
            float("nan"),  # tau_max [s] :
            float("nan"),  # VT [mV] : Threshold voltage.
            float("nan"),  # Eleak  [mV] : Reversal potential of leak currents.
            float("nan"),  # rate_to_SS_factor : Correction factor.
        ]
    ]
)

# instantiate simulator and set a stimulus
hh_model = HHSimulator(cythonise=True)  # Whether to use cython or numpy backend
hh_model.set_static_parameters(new_model_params=model_parameters)
stimulus_protocol = {
    "dt": 0.04,
    "duration": 800,
    "stim_end": 700,
    "stim_onset": 100,
    "I": 300,
}
V0 = -70

hh_model.set_stimulus(protocol_params=stimulus_protocol)

def simulator(theta):
    return hh_model.simulate_and_summarise(theta, V0=V0, n_workers=main_args.max_workers, batch_size=50, rnd_seed=main_args.seed)

#-------------------------------------------------------------------------------
# define prior and observations
prior_min = [0.4, 0.5, 1e-4, -3e-5, 1e-4, -3e-5, 50, -90, -110, 0.1]
prior_max = [3, 80.0, 30, 0.6, 0.8, 0.6, 3000, -40, -50, 3]
prior = BoxUniform(prior_min, prior_max)

theta_o = torch.tensor([[ 2.8248e+00,  6.1847e+01,  7.8308e+00,  1.8434e-01,  
                        1.5106e-01, 2.2927e-01,  2.3197e+03, -6.6100e+01, 
                        -9.3987e+01,  1.2552e+00]])
x_o = simulator(theta_o)

db.write("model_parameters", model_parameters)
db.write("stimulus_protocol", stimulus_protocol)
db.write("theta_o", theta_o)
db.write("x_o", x_o)

all_dims = list(range(23))

#-------------------------------------------------------------------------------
# optimise prior and simulate
t0 = now()
theta_train = prior.sample((main_args.optimise_on,))
x_train = simulator(theta_train)
record_scaler(now()-t0, "simulate training data", db.location + "/timings.txt")
clear_cache('cython')

t2 = now()
opt_prior = OptimisedPrior(prior)
opt_prior.train(theta_train, x_train)
record_scaler(now()-t2, "optimise prior", db.location + "/timings.txt")

t3 = now()
theta_opt = opt_prior.sample((main_args.train_on,))
x_opt = simulator(theta_opt)
record_scaler(now()-t3, "Simulate data", db.location + "/timings.txt")
clear_cache('cython')

db.write("prior", opt_prior)
db.write("x", x_opt)
db.write("theta", theta_opt)
