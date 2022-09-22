# for transfer learning feature subsets
import time
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors.direct_posterior import DirectPosterior
# this is temporary until pytorch implementation
from sklearn.mixture import GaussianMixture
# types
from torch import Tensor, optim

import sbi_feature_importance.utils as fi_utils
from sbi_feature_importance.utils import GMM

# TODO: Integrate evidence distr. into ReducablePosterior self.evidence with sample & log_prob
# If sample_reduced_features or log_prob_reduced_features is called evidence dist
# can be used!


class ReducableDirectPosterior(DirectPosterior):
    r"""Temporary Wrapper for `DirectPosterior` instaces based on a trained MDN.
    Structered such that evetything can be integrated into `DirectPosterior`
    directly. Implements functionality to evaluate and sample the posterior
    based on a reduced subset of features p(\theta|x1) of x_o = (x1, x2).

    Example:
        ```
        # trained posterior instance
        estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(estimator)

        # wrapping instance and marginalising observations
        mdn_wrapper = ReducableDirectPosterior(posterior)

        mdn_wrapper.sample_reduced_features(n_samples, evidence, xdim_subset, x_o)
        ```
    Args:
        mdn_posterior: sbi `DirectPosterior` instance that has been
            trained using an MDN.
    Attributes:
        Inherits all its attributes from `DirectPosterior`.
    """

    def __init__(self, mdn_posterior: DirectPosterior):
        if type(mdn_posterior.net._distribution) is mdn:
            # wrap copy of input object into self
            self.__class__ = type(
                "ReducedDirectPosterior",
                (self.__class__, deepcopy(mdn_posterior).__class__),
                {},
            )
            self.__dict__ = deepcopy(mdn_posterior).__dict__
        else:
            raise AttributeError("Posterior does not contain a MDN.")

    def sample_reduced_features(
        self,
        sample_shape: Tuple[int, int],
        evidence: Tensor,
        xdims: List,
        context: Tensor = None,
        rnd_seed: int = 0,
        **gmm_kwargs,
    ) -> Tensor:
        r"""Draw samples from the posterior distributution conditoned only on
        feature subset.

        Given $x_o = (x_1,x_2)$ the posterior samples will be drawn from
        $p(\theta|x_1)$. `xdims` decides which dimensions are going to be part
        of $x_1$.

        Args:
            sample_shape: Shape of the samples to be drawn. (n_samples,n_batches).
            evidence: Samples from the marginal likelihood / evidence
                $x \sim p(x)$. Needed for MC approximation of $p(\theta|x_1)$.
            xdims: The indices of the dimensions to keep for the context,
                i.e. those of $x_1$.
            context: Observation $x_o = (x_1,x_2)$ with different features.
            rnd_seed: Seeds the random number generator in pytorch.
            gmm_kwargs: keywords like n_mixtures and max_iter for the GMM fit of
                $p(x)$.

        Returns:
            samples from the posterior distribution conditioned only on a
                specific subset of features of $x_o$.
        """
        torch.manual_seed(rnd_seed)
        n_samples = torch.Size(sample_shape).numel()
        samples = 0

        # ensure context is set
        self.default_x = context
        assert self.default_x != None, "No context has been set."

        if len(xdims) != evidence.shape[1]:
            # MC approximation of p(\theta|x1) = \sum p(\theta|x1,x2i), where x2i ~ p(x2|x1)
            x1_x2_o = sample_conditonal_evidence(
                n_samples, evidence, xdims, context, **gmm_kwargs
            )
            logits_mog, means_mog, precfs_mog, _ = fi_utils.extract_and_transform_mog(
                self.net, x1_x2_o
            )

            samples = (
                mdn.sample_mog(1, logits_mog, means_mog, precfs_mog).detach().squeeze()
            )
        else:
            samples = self.sample(sample_shape, context)

        return samples.reshape((*sample_shape, -1))

    def log_prob_reduced_features(
        self,
        inputs: Tensor,
        evidence: Tensor,
        xdims: List[int],
        context: Tensor = None,
        num_mc_terms: int = 10000,
        rnd_seed: int = 0,
        **gmm_kwargs,
    ) -> Tensor:
        r"""Computes the log of the probability of the posterior based on a subset
        of features.

        The probability is approximated using an MC approximation of log p(\theta|x1)
        with $p(\theta|x1) = \sum_i p(\theta|x1,x_{2_i})$, where
        $x_{2_i} \sim p(x2|x1)$. The number of terms in the approximation
        is determined by `num_mc_terms`.

        Args:
            inputs: Points $\theta_i$ on the posterior that are to be evaluated.
            evidence: Samples from the marginal likelihood / evidence
                $x \sim p(x)$. Needed for MC approximation of $p(\theta|x_1)$.
            xdims: The indices of the dimensions to keep for the context,
                i.e. those of $x_1$.
            context: Observation $x_o = (x1,x2)$ with different features.
            rnd_seed: Seeds the random number generator in pytorch.
            gmm_kwargs: keywords like n_mixtures and max_iter for the GMM fit of
            $p(x)$.

        Returns:
            log_probs: Log of the posterior probabilities evaluated on a subset
                of features and approximated with MC.
        """
        # ensure context is set
        self.default_x = context
        assert self.default_x != None, "No context has been set."

        n_evals = inputs.shape[0]
        log_probs = torch.zeros(n_evals)

        if len(xdims) != evidence.shape[1]:
            # MC approximation of p(\theta|x1) = \sum p(\theta|x1,x2i), w. x2i ~ p(x2|x1)
            x1_x2_o = sample_conditonal_evidence(
                num_mc_terms, evidence, xdims, context, **gmm_kwargs
            )
            (
                logits_mog,
                means_mog,
                precfs_mog,
                sumlogdiag,
            ) = fi_utils.extract_and_transform_mog(self.net, x1_x2_o)
            precs_mog = precfs_mog.transpose(3, 2) @ precfs_mog

            for n in range(n_evals):
                mc_terms = mdn.log_prob_mog(
                    inputs[n].view(1, -1), logits_mog, means_mog, precs_mog, sumlogdiag
                )
                log_probs[n] = 1 / num_mc_terms * torch.sum(mc_terms)
        else:
            log_probs = self.log_prob(inputs, context)

        return log_probs


def sample_conditonal_evidence(
    n_samples: int,
    evidence: Tensor,
    xdims: List[int],
    condition: Tensor,
    rnd_seed: Optional[int] = None,
    **gmm_kwargs,
) -> Tensor:
    r"""Draw samples from the conditional evidence, approximated using a MoG.

    A MoG is fitted to the evidence $p(x)$ and then condicioned according to
    $p(x2|x1)$. Samples $x_{2_i}$ are then drawn from the conditioned distribution.
    The samples get appended to the original context such that ${x1,x2i}_1:N$.
    These samples can then be used in the MC approximation of $p(\theta|x1)$.

    Args:
        n_samples: Number of samples to be drawn from the MoG.
        evidence: Samples from the marginal likelihood / evidence $x \sim p(x)$.
            Needed for MC approximation of $p(\theta|x1)$.
        xdims: The indices of the dimensions to keep for the context,
            i.e. those of x1.
        context: Observation $x_o = (x1,x2)$ with different features.
        gmm_kwargs: keywords like n_mixtures and max_iter for the GMM fit of
            $p(x)$.

    Returns:
        x1_x2: Samples drawn from the conditional evidence
            $(x1_o,x2i) \sim p(x2|x1)$.
    """
    N, ndims = evidence.shape
    dims = list(range(ndims))
    if rnd_seed != None:
        torch.manual_seed(rnd_seed)

    x1dims = xdims
    x2dims = [i for i, y in enumerate([x not in x1dims for x in dims]) if y]

    # fit, condition MoG and sample x2 from conditional eviedence
    logits_px, means_px, precfs_px = fit_gmm(evidence, **gmm_kwargs)
    logits_cpx, means_cpx, precfs_cpx, _ = fi_utils.condition_mog(
        condition, x2dims, logits_px, means_px, precfs_px
    )  # x1 = x_o
    x2_c = mdn.sample_mog(n_samples, logits_cpx, means_cpx, precfs_cpx)

    x1_x2 = condition.repeat(n_samples, 1)
    x1_x2[:, x2dims] = x2_c
    return x1_x2


# USING SKLEARN IS A TEMPORARY MEASURE UNTIL THE PYTORCH IMPLEMENTATION IS FIXED
from sklearn.mixture import GaussianMixture


def fit_gmm(
    x: Tensor, n_mixtures: int = 10, max_iter: int = 1000
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fit a MoG to samples.
    Args:
        x: Samples from some distribution.
        n_mixtures: Number of Gaussian mixture components to use in MoG.
        max_iter: maximum number of iterations of EM optimisation.
    Returns:
        logits: Log-weights of each component.
        means: Means of each component.
        precfs: Precision factors of each component.
    """
    N, ndims = x.shape
    # gmm = GMM(n_mixtures)
    # gmm.fit(x)

    gmm = GaussianMixture(n_mixtures, max_iter=max_iter, covariance_type="full")
    gmm.fit(x)

    means = torch.from_numpy(gmm.means_).unsqueeze(0).float()
    logits = torch.log(torch.from_numpy(gmm.weights_)).unsqueeze(0).float()
    precs = torch.inverse(torch.from_numpy(gmm.covariances_))
    precfs = torch.cholesky(precs, upper=True).unsqueeze(0).float()

    # logits, means, precs = gmm.get_parameters()
    # precfs = torch.cholesky(precs, upper=True).unsqueeze(0)
    # logits = logits.unsqueeze(0)
    # means = means.unsqueeze(0)

    return logits, means, precfs


class MissingNet(nn.Module):
    r"""Linear layer that allows to zero out weights corresponding to feature dims.
    Useful in transferlearning on feature subsets.

    Used as an input to `embedding_net` in `posterior_nn`.

    Example:
    ```
    embedding_net = MissingNet(num_dim, num_dim)
    estimator = posterior_nn("mdn", embedding_net=embedding_net)
    inference = SNPE_C(prior, density_estimator=estimator)
    pretrained_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(pretrained_estimator)

    insertable_estimator = insert_pretrained(pretrained_estimator, dims)

    inference = SNPE_C(prior, density_estimator=insertable_estimator)
    transfer_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(transfer_estimator)
    ```

    Args:
        y_numel: Number of features.
        hidden_features: Number of hidden features.
        force_diagonal: Forces the input weights of a square matrix to be diagonal.
            I.e. The input features are only scaled independently.

    Attributes:
        force_diagonal: Whether to enforce a diagonal mask.
        linear: y = Ax^T + b.
        relu: Activation function of the layer.
        mask: Binary mask to null out weights during forward pass.
    """

    def __init__(
        self,
        y_numel: int,
        hidden_features: int,
        dims: List[int] = "all",
        force_diagonal: bool = False,
    ):
        super(MissingNet, self).__init__()
        self.force_diagonal = force_diagonal
        self.linear = nn.Linear(y_numel, hidden_features)
        self.relu = nn.ReLU()
        self.dropout_except4dims(dims)

        self.mask = self.enforce_diagonality(self.mask)

    def enforce_diagonality(self, mask):
        """Enforces diagonality constraints on input mask,
        if `force_diagonal` is true.

        Args:
            mask: Binary mask to diagonalise.

        Returns:
            mask: Potentially diagonalised binary mask.
        """
        if self.force_diagonal:
            assert (
                mask.shape[0] == mask.shape[1]
            ), "diagonality can only be inforced \
                if the number of hidden features machtes the number of input features!"
            mask *= torch.eye(mask.shape[0], dtype=bool)
            return mask
        else:
            return mask

    def dropout_except4dims(self, dims="all"):
        """Only keeps specified input features.

        This is realised by multiplying the weight matrix
        with a binary mask, that zeroes out all remaining dimensions.

        Args:
            dims: Specifies which dimensions / features to pass through
            the following network.
        """
        self.mask = torch.zeros_like(self.linear.weight.data)
        num_dims = self.linear.weight.shape[1]
        all_dims = list(range(num_dims))
        if dims == "all":
            dims = all_dims
        with torch.no_grad():
            self.mask[:, dims] = 1

        self.mask = self.enforce_diagonality(self.mask)
        self.linear.weight.data *= self.mask

    def forward(self, x: Tensor):
        """Linear forward pass through the embedding network.

        Implements y = BWx^T + b, where B is a binary mask
        and W are the weights.
        Args:
            x: Inputs.

        Returns:
            x: Transformed inputs.
        """
        self.linear.weight.data *= self.mask
        x = self.linear(x)
        x = self.relu(x)
        return x


def insert_pretrained(trained_net, dims: List[int] = "all"):
    """Helper function, that enables the insertion of a pretrained
    density estimator into SNPE.

    If `trained_net` contains an embedding net of type `MissingNet`,
    input features can be selectively dropped via dims.

    Args:
        trained_net: Pretrained posterior estimator instance.
        dims: Which dims / features to keep around in the embedding layer.

    Returns:
        lambda function that returns a copy of the input NN on any input x,y.
    """
    transfer_net = deepcopy(trained_net)
    if type(transfer_net._embedding_net[1]) == MissingNet:
        transfer_net._embedding_net[1].dropout_except4dims(dims)
    return lambda x, y: transfer_net


class BadFeatureEmbeddingNet(nn.Module):
    r"""Emebdding layer to learn feature imputations of "bad features".

    Implementation of:
    (https://papers.nips.cc/paper/2017/file/addfa9b7e234254d26e9c7f2af1005cb-Paper.pdf)

    $h(x_i) = x*(1-m(x_i)) + c_i*m(x_i)$, where $m(x_i)$ produces a binary output
    if a feature is good or bad. c_i are the feature imputations that
    are learned.

    Example:
    ```
    net = BadFeatureEmbeddingNet(num_dims, num_dims)
    posterior_nn = posterior_nn("mdn", embedding_net=net)
    inference = SNPE_C(prior=prior, density_estimator=posterior_nn)
    estimator = inference.append_simulations(theta, x).train(exclude_invalid_x=False)
    posterior = inference.build_posterior(estimator)
    ```

    Args:
        n_features: How many inputs the layer accepts.
        force_diagonal: Whether to enforce a diagonal mask.

    Attributes:
        force_diagonal: Whether to enforce a diagonal mask.
        linear: y = Ax^T + b.
        relu: Activation function of the layer.
        mask: Binary mask to null out off diagonal weights during forward pass.
    """

    def __init__(self, n_features, force_diagonal=True):
        super(BadFeatureEmbeddingNet, self).__init__()
        self.linear = nn.Linear(n_features, n_features, bias=False)
        self.mask = torch.eye(n_features, dtype=bool)
        self.force_diagonal = force_diagonal
        # self.relu = nn.ReLU()

    def ft_is_missing(self, x: Tensor):
        """Returns 1 for invalid inputs and 0 for intact features.

        Args:
            x: Input vector.

        Returns:
            Binary output if input is ("nan" or "inf") -> 1 or not -> 0
        """
        is_inf = x.isinf()
        is_nan = x.isnan()
        return torch.logical_or(is_nan, is_inf).float()

    def forward(self, x: Tensor):
        """Linear forward pass through the network.

        Implements y = BWx^T + b, where B is a binary mask
        and W are the weights.
        Args:
            x: Inputs.

        Returns:
            x: Transformed inputs.
        """
        if self.force_diagonal:
            self.linear.weight.data *= self.mask
        m = self.ft_is_missing(x)

        y = x.clone()
        y[m.bool()] = 0
        x = y + self.linear(m)
        # x = self.relu(x)
        return x
