#!/bin/bash

python3 simulateHH.py -o 100000 -t 1000000 -n HH1M -m 23 simulate.sh

###############################################################################################

# FIG 2
for i in {0..9}
do
python3 glm_timing_fixed_xo.py -s $i -p posthoc -f skip -n fig2_glm -w mcmc -m 5
python3 glm_timing_fixed_xo.py -s $i -p posthoc -f skip -n fig2_glm -w rejection -m 5
done

for i in {1..10}
do
python3 glm_timing_fixed_xo.py -s $i -p direct -f skip -n fig2_glm -w mcmc -m 5
python3 glm_timing_fixed_xo.py -s $i -p direct -f skip -n fig2_glm -w rejection -m 5

python3 glm_timing_fixed_xo_maf.py -s $i -p direct -f skip -n fig2_glm -w mcmc -m 5
python3 glm_timing_fixed_xo_maf.py -s $i -p direct -f skip -n fig2_glm -w rejection -m 5
done

# FIG 3
for i in {0..9}
do
python3 reducable_posterior_less_fts.py -d HH1M -s $i -p default -f skip -n fig3 -w mcmc -m 11
python3 reducable_posterior_less_fts.py -k {'num_subsets':7,'subset_len':1,'gamma':0.5} -d HH1M -s $i -p experimental -f skip -n fig3 -w mcmc -m 11
python3 reducable_posterior_less_fts.py -k {'num_subsets':1,'subset_len':1,'gamma':0.0} -d HH1M -s $i -p experimental -f skip -n fig3 -w mcmc -m 11
python3 reducable_posterior_less_fts.py -k {'num_subsets':5,'subset_len':2,'gamma':0.5} -d HH1M -s $i -p experimental -f skip -n fig3 -w mcmc -m 11

python3 reducable_posterior_less_fts_maf.py -d HH1M -s $i -p default -f skip -n fig3 -w mcmc -m 11
done

for i in {0..9}
do
python3 train_full_posteriors.py -k {'num_subsets':1,'subset_len':1,'gamma':0} -d HH1M -s $i -p experimental -f 1 -n posterior_var_10ft -w mcmc -m 11
python3 train_full_posteriors.py -k {'num_subsets':5,'subset_len':2,'gamma':0.5} -d HH1M -s $i -p experimental -f 1 -n posterior_var_10ft -w mcmc -m 11
python3 train_full_posteriors.py -k {'num_subsets':7,'subset_len':1,'gamma':0.5} -d HH1M -s $i -p experimental -f 1 -n posterior_var_10ft -w mcmc -m 11
done

for i in {0..9}
do
python3 posterior_subset_var.py -d HH1M -s $i -p default -n posterior_var_10ft_subsets -w mcmc -m 2
python3 posterior_subset_var.py -k {'num_subsets':5,'subset_len':2,'gamma':0.5} -d HH1M -s $i -p experimental -n posterior_var_10ft_subsets -w mcmc -m 2
python3 posterior_subset_var.py -k {'num_subsets':7,'subset_len':1,'gamma':0.5} -d HH1M -s $i -p experimental -n posterior_var_10ft_subsets -w mcmc -m 2
python3 posterior_subset_var.py -k {'num_subsets':1,'subset_len':1,'gamma':0.0} -d HH1M -s $i -p experimental -n posterior_var_10ft_subsets -w mcmc -m 2
done

# FIG 4
for i in {0..9}
do
python3 HH_tree_search_pretrained.py -d HH1M -q posterior_var_10ft -t 2_1_1_0 -s $i -f random -p experimental -n fig4_HHtree_rev_kl -w mcmc -m 10
python3 HH_tree_search_pretrained.py -d HH1M -q posterior_var_10ft -t 2_1_1_0 -s $i -f best -p experimental -n fig4_HHtree_rev_kl -w mcmc -m 10
done