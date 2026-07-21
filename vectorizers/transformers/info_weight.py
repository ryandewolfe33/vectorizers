import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse

from warnings import warn

MOCK_TARGET = np.ones(1, dtype=np.int64)
MOCK_BOOL = np.ones(1, dtype=np.bool)


@numba.njit(nogil=True)
def column_kl_divergence(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength,
    target=MOCK_TARGET,
):
    count_norm = count_data.sum()
    # Empty column
    if count_norm == 0:
        return 0
    result = 0.0
    current_idx = 0
    for i in range(baseline_probabilities.shape[0]):
        observed_probability = prior_strength * baseline_probabilities[i]
        if count_indices[current_idx] == i:
            observed_probability += (1-prior_strength) * count_data[current_idx] / count_norm
            current_idx += 1
        if observed_probability > 0:
            result += observed_probability * np.log(observed_probability / baseline_probabilities[i])
    return result


@numba.njit(nogil=True)
def supervised_column_kl_divergence(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength,
    target,
):
    observed = np.zeros_like(baseline_probabilities)
    for i in range(count_indices.shape[0]):
        idx = count_indices[i]
        label = target[idx]
        if label >= 0:
            observed[label] += count_data[i]
    observed_norm = observed.sum()
    result = 0.0
    for i in range(baseline_probabilities.shape[0]):
        observed_probability = (1-prior_strength) * observed[i] / observed_norm + prior_strength * baseline_probabilities[i]
        if observed_probability > 0:
            result += observed_probability * np.log(observed_probability / baseline_probabilities[i])
    return result


@numba.njit(nogil=True, parallel=True)
def column_weights(
    indptr,
    indices,
    data,
    baseline_probabilities,
    column_kl_divergence_func,
    prior_strength=0.1,
    target=MOCK_TARGET,
    column_groups=None,
):
    n_cols = indptr.shape[0] - 1
    weights = np.ones(n_cols)
    for i in numba.prange(n_cols):
        group = 0
        if column_groups is not None:
            group = column_groups[i]
        weights[i] = column_kl_divergence_func(
            indices[indptr[i] : indptr[i + 1]],
            data[indptr[i] : indptr[i + 1]],
            baseline_probabilities[group, :],
            prior_strength=prior_strength,
            target=target,
        )
    return weights


@numba.njit(nogil=True)
def compute_baseline_probabilities(
    indptr,
    indices,
    data,
    target=None,
    column_groups=None,
):
    """
    Compute the marginals to compare each column to. Returns
    an (n column groups) x (n samples) matrix (unsupervised) or an
    (n column groups) x (n targets) matrix (supervised) where each
    row is the marginal of the column group.

    indptr, indices, and data arrays are from csr format.
    """
    n_groups = 1
    if column_groups is not None:
        n_groups = column_groups.max() + 1
    n_targets = indptr.shape[0] - 1
    if target is not None:
        n_targets = target.max() + 1
    counts = np.zeros((n_groups, n_targets), dtype=np.int64)
    for row in range(indptr.shape[0] - 1):
        this_target = row
        if target is not None:
            if target[row] >= 0:
                this_target = target[row]
            else:
                continue
        for i in range(indptr[row], indptr[row + 1]):
            group = 0
            if column_groups is not None:
                group = column_groups[indices[i]]
            counts[group, this_target] += data[i]
    probabilities = counts / np.sum(counts, axis=1).reshape(-1, 1)
    return probabilities


def information_weight(
    data,
    prior_strength=0.1,
    target=None,
    column_groups=None,
):
    """Compute information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution.

    Parameters
    ----------
    data: scipy sparse matrix (n_samples, n_features)
        A matrix of count data where rows represent observations and
        columns represent features. Column weightings will be learned
        from this data.

    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    target: ndarray or None (optional, default=None)
        If supervised target labels are available, these can be used to define distributions
        over the target classes rather than over rows, allowing weights to be
        supervised and target based. If None then unsupervised weighting is used.

    column_groups: ndarray or None (optional, default=None)
        If columns have a natural grouping, i.e. cols 10-15 are a one-hot-encoding of a single
        categorical variable, we should compare the column distribution to the within group
        marginal. If passed None then all columns have the same group.

    Returns
    -------
    weights: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """
    if target is not None:
        column_kl_divergence_func = supervised_column_kl_divergence
    else:
        column_kl_divergence_func = column_kl_divergence

    csr_data = data.tocsr()
    baseline_probabilities = compute_baseline_probabilities(
        csr_data.indptr,
        csr_data.indices,
        csr_data.data,
        target,
        column_groups,
    )

    csc_data = data.tocsc()
    csc_data.sort_indices()
    weights = column_weights(
        csc_data.indptr,
        csc_data.indices,
        csc_data.data,
        baseline_probabilities,
        column_kl_divergence_func,
        prior_strength=prior_strength,
        target=target,
        column_groups=column_groups,
    )

    return weights


class InformationWeightTransformer(BaseEstimator, TransformerMixin):
    """A data transformer that re-weights columns of count data. Column weights
    are computed as information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior of pseudo counts equal to ``prior_strength`` times the
    baseline prior distribution.

    Parameters
    ----------
    prior_strength: float (optional, default=0.1)
        How strongly to weight the prior when doing a Bayesian update to
        derive a model based on observed counts of a column.

    Attributes
    ----------

    information_weights_: ndarray of shape (n_features,)
        The learned weights to be applied to columns based on the amount
        of information provided by the column.
    """

    def __init__(
        self,
        prior_strength=1e-4,
        approx_prior=None,
        weight_power=2.0,
        supervision_weight=0.95,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight

        if approx_prior is not None:
            warn(
                "Approx prior parameter is no longer used, and is only accepted"
                "for backwards compatibility."
            )

    def fit(self, X, y=None, column_groups=None, **fit_kwds):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        self:
            The trained model.
        """
        if not scipy.sparse.isspmatrix(X):
            X = scipy.sparse.csc_matrix(X)

        self.information_weights_ = information_weight(
            X,
            self.prior_strength,
            column_groups=column_groups,
        )

        mean_weight = np.mean(self.information_weights_)
        if mean_weight > 0:
            self.information_weights_ /= mean_weight
            # This should never happen
        self.information_weights_ = np.maximum(self.information_weights_, 0.0)
        self.information_weights_ = np.power(
            self.information_weights_, self.weight_power
        )

        if y is not None:
            unsupervised_power = (1.0 - self.supervision_weight) * self.weight_power
            supervised_power = self.supervision_weight * self.weight_power

            target_classes = np.unique(y)
            target_dict = dict(
                np.vstack((target_classes, np.arange(target_classes.shape[0]))).T
            )
            target = np.array(
                [np.int64(target_dict[label]) for label in y], dtype=np.int64
            )
            self.supervised_weights_ = information_weight(
                X,
                self.prior_strength,
                target=target,
                column_groups=column_groups,
            )
            mean_supervised_weight = np.mean(self.information_weights_)
            if mean_supervised_weight > 0:
                self.supervised_weights_ /= mean_supervised_weight
            # This should never happen
            self.supervised_weights_ = np.maximum(self.supervised_weights_, 0.0)
            self.supervised_weights_ = np.power(
                self.supervised_weights_, supervised_power
            )

            self.information_weights_ = (
                self.information_weights_ * self.supervised_weights_
            )

        return self

    def transform(self, X):
        """Reweight data ``X`` based on learned information weights of columns.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be transformed. Note that, as count data all
            entries should be positive or zero.

        Returns
        -------
        result: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The reweighted data.
        """
        result = X @ scipy.sparse.diags(self.information_weights_)
        return result
