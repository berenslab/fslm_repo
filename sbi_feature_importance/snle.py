from __future__ import annotations

import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from nflows.flows import Flow
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.snle.snle_base import LikelihoodEstimator
# types
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.utils.data import DataLoader, TensorDataset

import sbi_feature_importance.utils as fi_utils


def build_reducable_posterior(inference_obj, **kwargs):
    posterior = inference_obj.build_posterior(**kwargs)
    return ReducablePosterior(posterior)


def ReducablePosterior(posterior: RejectionPosterior or MCMCPosterior):
    r"""Factory function to wrap `MCMCPosterior` or `RejectionPosterior`.

    Provides passthrough to wrap the posterior instance in its respective
    reducable class wrapper `ReducableRejectionPosterior` or
    `ReducableMCMCPosterior` depending on the instance's class.

    Example:
        ```
        posterior = infer(simulator, prior, "SNLE_A", num_simulations)
        reducable_posterior = ReducablePosterior(posterior)
        reducable_posterior.marginalise(list_of_dims_to_keep)
        reducable_posterior.sample()
        ```
    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Returns:
        ReducableRejectionPosterior` or `ReducableMCMCPosterior`
    """
    if isinstance(posterior, RejectionPosterior):
        return ReducableRejectionPosterior(deepcopy(posterior))
    elif isinstance(posterior, MCMCPosterior):
        return ReducableMCMCPosterior(deepcopy(posterior))


class ReducableBasePosterior:
    r"""Provides marginalisation functionality to for `MCMCPosterior` and
    `RejectionPosterior`.


    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior or RejectionPosterior) -> None:
        self._wrapped_posterior = posterior

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_posterior, attr)

    def marginalise(
        self, dims: List[int], inplace: bool = True
    ) -> Optional[ReducablePosterior]:
        r"""Marginalise likelihood distribution of the likelihood-based posterior.

        Marginalises the MDN-based likelihood $p(x_1, ..., x_N|\theta)$ such that
        $$
        p(\theta|x_{subset}) \propto p(x_{subset}|\theta) p(\theta)
        $$
        , where $x_{susbet} \susbet (x_1, ..., x_N)$

        Args:
            dims: Feature dimensions to keep.
            inplace: Whether to return a marginalised copy of self, or to
                marginalise self directly.

        Returns:
            red_posterior: If inplace=False, returns a marginalised copy of self.
        """
        likelihood_estimator = self.potential_fn.likelihood_estimator
        marginal_likelihood = ReducableLikelihoodEstimator(likelihood_estimator, dims)
        if inplace:
            self.potential_fn.likelihood_estimator = marginal_likelihood
        else:
            red_posterior = deepcopy(self)
            red_posterior.potential_fn.likelihood_estimator = marginal_likelihood
            return red_posterior


class ReducableRejectionPosterior(ReducableBasePosterior, RejectionPosterior):
    r"""Wrapper for `RejectionPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features p(\theta|x1)
    of x_o = (x1, x2).

    Args:
        posterior: `RejectionPosterior` instance trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: RejectionPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableMCMCPosterior(ReducableBasePosterior, MCMCPosterior):
    r"""Wrapper for `MCMCPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features $p(\theta|x1)$
    of $x_o = (x1, x2)$.

    Args:
        posterior: `MCMCPosterior` instance that trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableLikelihoodEstimator:
    r"""Adds marginalisation functionality to mdn based likelihood estimators.

    Supports `.log_prob` of the MoG likelihood.

    Its main purpose is to emulate the likelihood estimator employed in
    `LikelihoodbasedPotential`.

    Args:
        likelihood_estimator: Flow instance that was trained using a MDN.
        marginal_dims: List of x dimensions to consider. Dimensions not
            in `marginal_dims` are marginalised out.

    Attributes:
        likelihood_net: Conditional density estimator for the likelihood.
        dims: List of x dimensions to consider in the evaluation.
    """

    def __init__(
        self, likelihood_estimator: Flow, marginal_dims: Optional[List[int]] = None
    ) -> None:
        self.likelihood_net = likelihood_estimator
        self.dims = marginal_dims

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self.likelihood_net.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self.likelihood_net.eval()

    def marginalise(
        self, context: Tensor, dims: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Marginalise MoG and return new mixture parameters.

        Args:
            context: Condition in $p(x|\theta)$.
            dims: List of dimensions to keep.

        Returns:
            logits: log-weights for each component of the marginal distributions.
            mu_x: means of the marginal distributution for each component.
            precfs_xx: precision factors of the marginal distribution for
                each component.
            sumlogdiag: Sum of the logarithms of the diagonal elements of the
                precision factors of the marginal distributions for each component.
        """

        # reset to unmarginalised params
        logits, means, precfs, _ = fi_utils.extract_and_transform_mog(
            self.likelihood_net, context
        )

        mask = torch.zeros(means.shape[-1], dtype=bool)
        mask[dims] = True

        # Make a new precisions with correct ordering
        mu_x = means[:, :, mask]
        precfs_xx = precfs[:, :, mask]
        precfs_xx = precfs_xx[:, :, :, mask]

        # set new GMM parameters
        sumlogdiag = torch.sum(
            torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2
        )
        return logits, mu_x, precfs_xx, sumlogdiag

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Evaluate the Mixture of Gaussian (MoG)
        probability density function at a value x.

        Args:
            inputs: Values at which to evaluate the MoG pdf.
            context: Conditiones the likelihood distribution.

        Returns:
            Log probabilities at values specified by theta.
        """
        logits, means, precfs, sumlogdiag = self.marginalise(context, self.dims)
        prec = precfs.transpose(3, 2) @ precfs

        return mdn.log_prob_mog(inputs[:, self.dims], logits, means, prec, sumlogdiag)


class MarginalLikelihoodEstimator(LikelihoodEstimator):
    r"""Adds additional terms to the loss to obtain more robust marginal estimates.

    Example:
    ```
    inference_ = SNLE_A(better_prior, show_progress_bars=True, density_estimator="mdn")
    inference = MarginalLikelihoodEstimator(inference_)
    inference.set_marginal_loss_policy("skip")
    ```

    Args:
        estimator: SNLE `LikelihoodEstimator` to wrap. Its lossfunction will
            be replaced by a custom loss function, that can be specified
            via `set_marginal_loss_policy()`.
    """

    def __init__(
        self, estimator: LikelihoodEstimator, marginal_loss_policy: Optional[str] = None
    ) -> None:
        self._wrapped_estimator = estimator
        self.modded_loss = False
        self.set_marginal_loss_policy(marginal_loss_policy)
        self.train = self.add_policy_warn(self.train)

    def add_policy_warn(self, train: Callable) -> Callable:
        """Warns if training is called without marginal loss policy.

        Args:
            train: self.train

        Returns:
            Wrapper for train that warns if the loss function does not include
            marginal terms.
        """

        def train_wrapper(*args, **kwargs):
            if not self.modded_loss:
                warnings.warn(
                    (
                        "No loss policy was set. The default NLE loss will be used. "
                        "You can specify a policy with `set_marginal_loss_policy`."
                    )
                )
            return train(*args, **kwargs)

        return train_wrapper

    def __getattr__(self, attr):
        """Forward attrs to wrapped object if not existant in self."""
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_estimator, attr)

    def mog_from_mdn(self, context: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract MoG params from MDN.

        Args:
            context: theta context for likelihood.

        Returns:
            logits, means, precision factors and sum of the log diagonals
                of mog.
        """
        return fi_utils.extract_and_transform_mog(self._neural_net, context)
        # return fi_utils.extract_mog_from_flow(self._neural_net, context)

    @staticmethod
    def marginalise_mog(
        logits: Tensor, means: Tensor, precfs: Tensor, dims: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Marginalise MoG and return new mixture parameters.

        Args:
            context: Condition in $p(x|\theta)$.
            dims: List of dimensions to keep.

        Returns:
            logits: log-weights for each component of the marginal distributions.
            mu_x: means of the marginal distributution for each component.
            precfs_xx: precision factors of the marginal distribution for
                each component.
            sumlogdiag: Sum of the logarithms of the diagonal elements of the
                precision factors of the marginal distributions for each component.
        """

        # reset to unmarginalised params
        mask = torch.zeros(means.shape[-1], dtype=bool)
        mask[dims] = True

        # Make a new precisions with correct ordering
        mu_x = means[:, :, mask]
        precfs_xx = precfs[:, :, mask]
        precfs_xx = precfs_xx[:, :, :, mask]

        # set new GMM parameters
        sumlogdiag = torch.sum(
            torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2
        )

        return logits, mu_x, precfs_xx, sumlogdiag

    def set_marginal_loss_policy(
        self, policy: Optional[str] = "default", **kwargs
    ) -> Callable:
        """Sets the loss marginal loss policy that will be followed.

        Along the maximising the negative log _probs of the likelihood, marginal
        likelihoods of choice can also be incorporated into the loss function.
        Several options are available to create marginal distributions from a
        superset of dimensions:
        - skip: skip 1 or more dimensions, i.e. [0,1,2] -> [[1,2], [0,2], [0,1]]
        - permute: Sorted permutations from a min to a max number of dimensions,
            i.e. (min,max)=(1,2); [0,1,2] -> [[1,2], [0,2], [0,1], [0], [1], [2]]
        - random_subset: Random subset of possible permutations, i.e.
            [0,1,2] -> [[1,2], [0,1], [1], [2]]
        - one_d_marginals: [0,1,2] -> [[0], [1], [2]]

        Args:
            policy: Specify policy. If none is explicitely stated or the stated
                policy does not exist, then no marginal probabilities are
                calculated in the loss function.

        Returns:
            Callable: Policy, that if provided with list of dims will return a
                generator over which to itterate, i.e. `list(policy(dims))`
                might return [[0], [1], [2]] for policy="skip".
        """

        def skip(dims: List[int], skip_n: int = 1):
            """Returns subets with n skipped dimensions.

            [0,1,2] -> [[1,2], [0,2], [0,1]]

            Args:
                dims: list/superset of available dimensions.
                skip_n: How many dims to skip with in each subset.

            Yields:
                subset of dimensions.
            """
            for subset in fi_utils.skip_dims(dims, skip_n):
                yield subset

        def permute(
            dims: List[int], min_ndims: int = 1, max_ndims: int = 2
        ) -> Generator:
            """Permute: Sorted permutations from a min to a max number of dimensions,
            i.e. (min,max)=(1,2); [0,1,2] -> [[1,2], [0,2], [0,1], [0], [1], [2]].

            Args:
                dims: list/superset of available dimensions.
                min_ndims: min num of dims in perumtation of subset.
                max_ndims: min num of dims in perumtation of subset.

            Yields:
                subset of dimensions.
            """
            for subset in fi_utils.permute_dims(dims[-1] + 1, min_ndims, max_ndims):
                yield subset

        def one_d_marginals(dims: List[int]) -> Generator:
            """Returns one_d_marginals.

            [0,1,2] -> [[0], [1], [2]]

            Args:
                dims: list/superset of available dimensions.

            Yields:
                subset of dimensions.
            """
            for dim in dims:
                yield [dim]

        def random_subset(
            dims: List[int], num_subsets: int = 10, min_len: int = 1, max_len: int = 9
        ) -> Generator:
            """Random_subset: Random subset of possible permutations, i.e.
            [0,1,2] -> [[1,2], [0,1], [1], [2]].

            Args:
                dims: list/superset of available dimensions.
                num_subsets: Number of subsets to return.
                min_len: Min number of dims per subset.
                max_len: Max number of dims per subset.

            Yields:
                subset of dimensions.
            """
            for subset in fi_utils.random_subsets_of_dims(
                dims, num_subsets, min_len, max_len
            ):
                yield subset

        def experimental_loss(num_subsets=10, subset_len=10, gamma=1):
            return partial(
                self._loss_experimental,
                num_subsets=num_subsets,
                subset_len=subset_len,
                gamma=gamma,
            )

        def no_policy(dims):
            """yields nothing, ergo marginal loss term is skipped."""
            return
            yield

        if hasattr(self, "_loss_backup"):  # untested
            self._loss = self._loss_backup

        if policy is None:
            dim_subset_policy = no_policy
        elif "skip" in policy.lower():
            dim_subset_policy = lambda dims: skip(dims, **kwargs)
        elif "permute" in policy.lower():
            dim_subset_policy = lambda dims: permute(dims, **kwargs)
        elif "random" in policy.lower():
            dim_subset_policy = lambda dims: random_subset(dims, **kwargs)
        elif "1d" in policy.lower():
            dim_subset_policy = lambda dims: one_d_marginals(dims, **kwargs)
        elif "experimental" in policy.lower():
            self._loss_backup = self._loss
            self._loss = experimental_loss(**kwargs)
            dim_subset_policy = None
        else:
            dim_subset_policy = no_policy

        if dim_subset_policy != no_policy:
            self.modded_loss = True
        else:
            self.modded_loss = False
        self.marginal_loss_policy = dim_subset_policy

    def marginal_losses(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Calculates the marginal log_probs for a given policy.

        Args:
            inputs: x.
            context: theta.

        Returns:
            Tensor: with average of the log_probs.
        """
        # transform, mog = self.mog_from_mdn(context)
        # logits, means, precs, precfs, sumlogdiag = mog
        # noise, logabsdet = transform(inputs)
        # scale = transform._transforms[0]._scale

        logits, means, precfs, _ = self.mog_from_mdn(context)
        dims = list(range(inputs.shape[1]))

        dims = list(range(inputs.shape[1]))

        N = 0
        total_log_probs = 0
        for subset in self.marginal_loss_policy(dims):
            logits_xx, means_xx, precfs_xx, sumlogdiag_xx = self.marginalise_mog(
                logits, means, precfs, subset
            )
            prec_xx = precfs_xx.transpose(3, 2) @ precfs_xx
            # logabsdet_x = torch.log(scale[subset]).sum()

            total_log_probs += (
                mdn.log_prob_mog(
                    # noise[:, subset], logits_xx, means_xx, prec_xx, sumlogdiag_xx
                    inputs[:, subset],
                    logits_xx,
                    means_xx,
                    prec_xx,
                    sumlogdiag_xx,
                )
                # + logabsdet_x
            )
            N += 1
        if N == 0:
            N = 1
        return 1 / N * total_log_probs

    def _loss(self, theta: Tensor, x: Tensor, gamma: float = 1.0) -> Tensor:
        """Replaces the original loss function of `LikelihoodEstimator`.

        Args:
            theta: parameter tensor
            x: observation tensor
            gamma: weights the importance of marginal losses.

        Returns:
            loss
        """
        loss = 0
        loss += self.marginal_losses(x, context=theta)
        loss *= gamma
        loss += self._neural_net.log_prob(x, context=theta)
        return -loss

    def _loss_experimental(
        self,
        theta: Tensor,
        x: Tensor,
        num_subsets: int = 5,
        subset_len: int = 2,
        gamma: float = 1.0,
    ) -> Tensor:
        """Replaces the original loss function of `LikelihoodEstimator`.

        Args:
            theta: parameter tensor
            x: observation tensor
            num_subsets: Specifies how many marginal terms are added to the loss.
            subset_len: Specifies how many features the subsets should contain.
            gamma: weights the importance of marginal losses.

        Returns:
            loss
        """

        # transform, mog = self.mog_from_mdn(theta)
        # logits, means, precs, precfs, sumlogdiag = mog
        logits, means, precfs, sumlogdiag = self.mog_from_mdn(theta)

        batchsize, n_mixtures, n_dims = means.shape

        n_logits = logits.repeat(num_subsets, 1)
        n_means = means.repeat(num_subsets, 1, 1)
        n_precfs = precfs.repeat(num_subsets, 1, 1, 1)
        # n_scale = transform._transforms[0]._scale.repeat(num_subsets * batchsize, 1)

        dims = (
            torch.multinomial(torch.ones(num_subsets, n_dims) / num_subsets, subset_len)
            # ensures repeats are grouped by subsets
            .T.repeat(batchsize, 1)
            .T.reshape(batchsize * num_subsets, 1, subset_len)
            .repeat(1, n_mixtures, 1)
        )
        mask = torch.sum(
            torch.nn.functional.one_hot(dims, n_dims), 2, keepdim=True
        ).bool()

        # noise, logabsdet = transform(x)
        # n_noise = noise.repeat(num_subsets, 1)[mask[:, 0].squeeze()].view(
        #     num_subsets * batchsize, -1
        # )
        # n_logabsdet = (
        #     torch.log(n_scale[mask[:, 0, 0]])
        #     .view(batchsize * num_subsets, subset_len)
        #     .sum(dim=1)
        # )
        inpts = x.repeat(num_subsets, 1)[mask[:, 0].squeeze()].view(
            num_subsets * batchsize, -1
        )

        # Make a new precisions with correct ordering
        mu_x = n_means[mask.squeeze(2)].reshape(
            num_subsets * batchsize, n_mixtures, subset_len
        )

        precfs_xx = n_precfs.transpose(3, 2)[mask.repeat(1, 1, n_dims, 1)].reshape(
            num_subsets * batchsize, n_mixtures, n_dims, subset_len
        )
        precfs_xx = precfs_xx.transpose(3, 2)[mask.repeat(1, 1, subset_len, 1)].reshape(
            num_subsets * batchsize, n_mixtures, subset_len, subset_len
        )

        # set new GMM parameters
        n_sumlogdiag = torch.sum(
            torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2
        )

        logp_full = (
            # mdn.log_prob_mog(noise, logits, means, precs, sumlogdiag) + logabsdet
            mdn.log_prob_mog(
                x, logits, means, precfs.transpose(3, 2) @ precfs, sumlogdiag
            )
        )

        logp_marginals = (
            mdn.log_prob_mog(
                # n_noise,
                inpts,
                n_logits,
                mu_x,
                precfs_xx.transpose(3, 2) @ precfs_xx,
                n_sumlogdiag,
            )
            # + n_logabsdet
        )
        logp_marginals = logp_marginals.reshape(num_subsets, batchsize)

        return -logp_full - gamma * logp_marginals.mean(0)


class NaNCalibration(nn.Module):
    r"""Learns calibration bias to compensate for NaN observations.

    Logistic regressor predicts which parameters cause NaN observations.
    The model is optimised with a binary cross entropy loss.

    The forward pass computes $p(valid|\theta)$.

    For reference (https://openreview.net/pdf?id=kZ0UYdhqkNY)

    Args:
        input_dim: Number of parameters in $\theta$.
        device: On which device to train.

    Attributes:
        linear: Linear layer.
        device: Which device is used.
    """

    def __init__(self, input_dim: int, device: str = "cpu"):
        super(NaNCalibration, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.device = device

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Implements logistic regression

        Args:
            inputs: theta.

        Returns:
            outputs: outputs of the forward pass. $p(valid|\theta)$
        """
        outputs = torch.sigmoid(self.linear(inputs))
        return outputs

    def log_prob(self, theta: Tensor) -> Tensor:
        r"""log of the forward pass, i.e. $p(valid|\theta)$.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            $log(p(valid|\theta))$.
        """
        probs = self.forward(theta.view(-1, self.n_params))
        return torch.log(probs)

    def train(
        self,
        theta_train: Tensor,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio = 0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> NaNCalibration:
        r"""Trains classifier to predict the probability of valid observations.

        Returns self for calls such as:
        `nan_likelihood = NaNCallibration(n_params).train(theta, y)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Observation labels. 0 if x_i contains no NaNs, 1 if does.
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation 
                improvement.
            val_ratio: How any training_examples to split of for validation.
            
        Returns:
            self
        """
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_split = int(len(theta_train) * val_ratio)

        train_data = TensorDataset(theta_train[n_split:], y_train[n_split:])
        val_data = TensorDataset(theta_train[:n_split], y_train[:n_split])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        best_val_log_prob = 0
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            train_log_probs_sum = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, y_batch = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )
                train_losses = criterion(self.forward(theta_batch), y_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                optimizer.step()

                train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size
                )

            val_log_probs_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, y_batch = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                    )
                    val_losses = criterion(self.forward(theta_batch), y_batch)
                    val_log_probs_sum -= val_losses.sum().item()

            val_log_prob_average = val_log_probs_sum / (
                len(val_loader) * val_loader.batch_size
            )

            if epoch == 0 or val_log_prob_average > best_val_log_prob:
                best_val_log_prob = val_log_prob_average
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > stop_after_epoch:
                print(f"converged after {epoch} epochs.")
                break

            if verbose:
                print(
                    "\r[{}] train loss: {:.5f}, val_loss: {:.5f}".format(
                        epoch, train_log_prob_average, val_log_prob_average
                    ),
                    end="",
                )
        return self


class CalibratedPrior(Distribution):
    r"""Prior distribution that can be calibrated to compensate for the
    likelihood bias, that is caused by ignoring non-valid observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given prior distribution $p(\theta)$, we can hence obtain a calibratedd
    prior $\tilde{p}(\theta)$ according to:
    $$
    \tilde{p}(\theta) \propto p(\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Its has a modified `.log_prob()` method compared to the base_prior. While,
    sampling is passed on to the base prior.
    The support is inherited from the base prior distribution.


    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_prior = CallibratedPrior(inference._prior).train(theta,x)
    posterior = inference.build_posterior(prior=calibrated_prior, sample_with='rejection')
    ```

    Args:
        prior: A base_prior distribution.
        device: Which device to use.

    Attributes:
        base_prior: The prior distribution that is optimised.
        dim: The dimensionality of $\theta$.
        nan_likelihood: Classifier to predict if $\theta$ will produce NaNs in
            observation.
    """

    def __init__(self, prior: Any, device: str = "cpu"):
        r"""
        Args:
            prior: A prior distribution that supports `.log_prob()` and
                `.sample()`.
            device: Which device to use. Should be same as for `prior`.
        """
        self.base_prior = prior
        self.dim = prior.sample((1,)).shape[1]
        self.nan_calibration = None
        self.device = device
        self._mean = prior.mean
        self._variance = prior.variance

    @property
    def mean(self):
        if self.nan_calibration is None:
            return self.base_prior.mean
        else:
            return self._mean

    @property
    def variance(self):
        if self.nan_calibration is None:
            return self.base_prior.variance
        else:
            return self._variance

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.base_prior.arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.base_prior.support

    def log_prob(self, theta: Tensor) -> Tensor:
        """Prob of calibrated prior.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            p_nan: log_probs for $p(valid|\theta)$
        """
        if self.nan_calibration is None:
            warnings.warn(
                "Evaluating non calibrated prior! To calibrate, call .train() first!"
            )
            return self.base_prior.log_prob(theta)
        else:
            p_no_nan = self.nan_calibration(theta.view(-1, self.dim)).view(-1)
            p = self.base_prior.log_prob(theta)
            return p + p_no_nan

    def sample(self, *args, **kwargs):
        """Pass through to `self.base_prior.sample()`

        Returns:
            Samples from `base_prior`
        """
        return self.base_prior.sample(*args, **kwargs)

    def train(
        self,
        theta_train: Tensor,
        x_train: Tensor = None,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio = 0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> CalibratedPrior:
        r"""Trains classifier to predict which parameters produce valid observations.

        Callibration factor is then added to log_prob of `base_prior`.

        The model is a logistic regressor optimised with a binary cross entropy
        loss.

        Returns self for calls such as:
        `trained_prior = CalibratedPrior(prior).train(theta, x)`

        Args:
            theta_train: Sets of parameters for training.
            x_train: Set of training observations that some of which include
                NaN features. Will be used to create labels y_train, depending
                on presence of NaNs (True if x_i contains NaN, else False).
            y_train: Labels whether corresponding theta_train produced an
                observation x_train that included NaN features.
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation 
                improvement.
            val_ratio: How any training_examples to split of for validation.

        Returns:
            self trained calibrated prior.
        """
        if x_train is not None:
            has_nan = fi_utils.includes_nan(x_train).view(-1, 1)
        elif y_train is not None:
            has_nan = y_train.view(-1, 1)
        else:
            raise ValueError("Please provide y_train or x_train.")

        self.nan_calibration = NaNCalibration(self.dim, device=self.device).train(
            theta_train,
            has_nan.float(),
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            stop_after_epoch=stop_after_epoch,
            verbose=verbose,
        )

        # estimating mean and variance from samples
        samples = self.sample((10000,))
        self._mean = samples.mean()
        self._variance = samples.var()

        return self


class CalibratedLikelihoodEstimator:
    r"""Modifies the likelihood by a calibration factor.

    Wraps the trained likelihood estimator and applies a calibration term to
    compensate for discarded training data due to NaN observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given likelihood distribution $p(x|\theta)$, we can hence obtain a
    calibrated likelihood $\tilde{p}(x|\theta)$ according to:
    $$
    \tilde{p}(x|\theta) \propto p(x|\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_likelihood = calibrate_likelihood_estimator(estimator, theta, x)
    posterior = inference.build_posterior(density_estimator=calibrated_likelihood, sample_with='rejection')
    ```

    Args:
        likelihood_estimator: A likelihood estimator (a Flow).
        calibration_f: A trained callibration network that has learned
            $p(valid|\theta)$.

    Attributes:
        calibration_f: calibration factor for likelihood that has been trained
        partially on NaNs.
    """

    def __init__(self, likelihood_estimator: Flow, calibration_f: NaNCalibration):
        self._wrapped_estimator = likelihood_estimator
        self.calibration_f = calibration_f

    def __getattr__(self, attr):
        """Forward attrs to wrapped object if not existant in self."""
        if attr == "log_prob":
            return getattr(self._wrapped_estimator, attr)
        elif attr in self.__dict__:
            return getattr(self, attr)

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self._wrapped_estimator.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self._wrapped_estimator.eval()

    def log_prob(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        r"""calibrated likelihood.

        Adds the calibration log_prob $p(valid|\theta)$ on top of the likelihood
        log_prob.

        Args:
            inputs: where to evaluate the likelihood.
            context: context of the likelihood.

        Returns:
            calibrated likelihoods.
        """
        return self._wrapped_estimator.log_prob(inputs, context) + torch.log(
            self.calibration_f(context)
        )


def calibrate_likelihood_estimator(
    estimator: Flow, theta: Tensor, x: Tensor, **train_kwargs
) -> CalibratedLikelihoodEstimator:
    r"""Calibrates the likelihood estimator if partially trained on NaNs.

    It learns a calibration function $c_ζ(θ)=p(valid|θ)$ on the training data an applies it to the
    likelihood.

    Args:
        estimator: likelihood estimator with log_prob method.
        theta: parameters.
        x: observations.

    Returns:
        a calibrated likelihood function.
    """
    calibration_f = NaNCalibration(int(theta.shape[1]))
    y = fi_utils.includes_nan(x).float()
    calibration_f.train(theta, y, **train_kwargs)
    return CalibratedLikelihoodEstimator(deepcopy(estimator), calibration_f)
