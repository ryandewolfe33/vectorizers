import numba
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data
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
    zero_count_contribution,
    target=MOCK_TARGET,
):
    count_norm = count_data.sum()
    # Empty column
    if count_norm == 0:
        return 0
    result = 0.0
    current_idx = 0
    for i in range(baseline_probabilities.shape[0]):
        if count_indices[current_idx] == i:
            observed_probability = (
                prior_strength * baseline_probabilities[i]
                + (1 - prior_strength) * count_data[current_idx] / count_norm
            )
            result += observed_probability * np.log2(
                observed_probability / baseline_probabilities[i]
            )
            current_idx += 1
        else:
            result += (
                prior_strength * baseline_probabilities[i] * zero_count_contribution
            )
    return result


@numba.njit(nogil=True)
def column_kl_divergence_zero_prior(
    count_indices,
    count_data,
    baseline_probabilities,
    mock_prior,
    mock_zero_count_contribution,
    target=MOCK_TARGET,
):
    count_norm = count_data.sum()
    # Empty column
    if count_norm == 0:
        return 0
    result = 0.0
    current_idx = 0
    for idx, count in zip(count_indices, count_data):
        observed_probability = count / count_norm
        result += observed_probability * np.log2(
            observed_probability / baseline_probabilities[idx]
        )
    return result


@numba.njit(nogil=True)
def supervised_column_kl_divergence(
    count_indices,
    count_data,
    baseline_probabilities,
    prior_strength,
    zero_count_contribution,
    target,
):
    observed = np.zeros_like(baseline_probabilities)
    for i in range(count_indices.shape[0]):
        idx = count_indices[i]
        label = target[idx]
        if label >= 0:
            observed[label] += count_data[i]
    observed_norm = observed.sum()
    if observed_norm == 0:
        return 0
    result = 0.0
    for i in range(baseline_probabilities.shape[0]):
        if observed[i] == 0:
            result += (
                prior_strength * baseline_probabilities[i] * zero_count_contribution
            )
        else:
            observed_probability = (1 - prior_strength) * observed[
                i
            ] / observed_norm + prior_strength * baseline_probabilities[i]
            result += observed_probability * np.log2(
                observed_probability / baseline_probabilities[i]
            )
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
    zero_count_contribution = np.log2(prior_strength) if prior_strength > 0 else 0
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
            prior_strength,
            zero_count_contribution,
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
    prior_strength=1e-4,
    target=None,
    column_groups=None,
):
    """Compute information based weights for columns. The information weight
    is estimated as the amount of information gained by moving from a baseline
    model to a model derived from the observed counts. In practice this can be
    computed as the KL-divergence between distributions. For the baseline model
    we assume data will be distributed according to the row sums -- i.e.
    proportional to the frequency of the row. For the observed counts we use
    a background prior equal to ``prior_strength`` times the
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
        over the target classes rather than over rows, allowing weights to be supervised and
        target based. Allows negative values to denote unknown labels for semi-supervised
        weights. If None then unsupervised weighting is used (equivalent to each sample having)
        its own label).

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
    elif prior_strength == 0:
        column_kl_divergence_func = column_kl_divergence_zero_prior
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


class InformationWeightTransformer(TransformerMixin, BaseEstimator):
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
        prior_strength: float = 1e-4,
        approx_prior: None = None,
        weight_power: float = 1.0,
        supervision_weight: float = 0.95,
    ):
        self.prior_strength = prior_strength
        self.weight_power = weight_power
        self.supervision_weight = supervision_weight
        self.approx_prior = approx_prior
        if approx_prior is not None:
            warn(
                "Approx prior parameter is no longer used, and is only accepted"
                "for backwards compatibility.",
                DeprecationWarning,
            )

    def _validate_parameters(self):
        if self.prior_strength < 0 or self.prior_strength >= 1:
            raise ValueError("prior_strength must be at least 0 and less than 1.")
        if self.weight_power <= 0:
            raise ValueError("weight_power must be positive.")
        if self.supervision_weight <= 0 or self.supervision_weight > 1:
            raise ValueError("supervision_weight must be greater than 0 and at most 1.")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.input_tags.sparse = True
        tags.input_tags.positive_only = True
        tags.target_tags.one_d_labels = True
        return tags

    def _validation_kwargs(self, include_y=False):
        x_validation = {"accept_sparse": True, "ensure_non_negative": True}
        if not include_y:
            return x_validation
        y_validation = {"ensure_2d": False, "dtype": None}
        return (x_validation, y_validation)

    def fit(self, X, y=None, column_groups=None):
        """Learn the appropriate column weighting as information weights
        from the observed count data ``X``.

        Parameters
        ----------
        X: ndarray of scipy sparse matrix of shape (n_samples, n_features)
            The count data to be trained on. Note that, as count data all
            entries should be positive or zero.

        y: ndarray or list of shape (n_samples, )
            Class of each sample for supervised or semi-supervised transform.
            For semi-supervised, the element types must be convertable to ints
            and negative numbers represent no class. If the element type is not
            cannot be converted to int, each unique value is a class.

        Returns
        -------
        self:
            The trained model.
        """
        self._validate_parameters()
        # Not nice but validate_data does not return y if it is None
        if y is not None:
            X, y = validate_data(
                self, X, y, validate_separately=self._validation_kwargs(include_y=True)
            )
        else:
            X = validate_data(self, X, **self._validation_kwargs())
        X = scipy.sparse.csr_array(X)

        # Validate column groups
        if column_groups is not None:
            if len(column_groups) != X.shape[1]:
                raise ValueError(
                    "The number of columns must match the length of column groups."
                )
            # Make column_groups indexed with ints 0-n
            _, column_groups = np.unique(column_groups, return_inverse=True)

        self.information_weights_ = information_weight(
            X,
            self.prior_strength,
            column_groups=column_groups,
        )

        if y is not None:
            self.unsupervised_weights_ = self.information_weights_
            # Format y as array of ints if it is not
            if np.issubdtype(y.dtype, np.number) and not np.issubdtype(
                y.dtype, np.integer
            ):
                cast_y = y.astype(int)
                if np.all(y == cast_y):
                    warn(
                        f"Input y was cast from {y.dtype} to {cast_y.dtype} and will be treated"
                        "as array of ints. Consider passing y as an array of ints.",
                        UserWarning,
                    )
                    y = cast_y
                else:
                    warn(
                        f"Input y could not be cast from {y.dtype} to {cast_y.dtype} and will be"
                        "treated as array of objects (identical values have the same class).",
                        UserWarning,
                    )
            if not np.issubdtype(y.dtype, np.integer):
                target_classes = np.unique(y)
                target_dict = {
                    target_classes[i]: i for i in range(target_classes.shape[0])
                }
                y = np.array([target_dict[label] for label in y], dtype=np.int64)

            self.supervised_weights_ = information_weight(
                X,
                self.prior_strength,
                target=y,
                column_groups=column_groups,
            )

            unsupervised_power = (1.0 - self.supervision_weight)
            supervised_power = self.supervision_weight

            self.information_weights_ = (
                np.power(self.unsupervised_weights_, unsupervised_power) *
                np.power(self.supervised_weights_, supervised_power)
            )
        
        self.information_weights_ = np.power(
            self.information_weights_, self.weight_power
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
        X = validate_data(self, X, **self._validation_kwargs(), reset=False)
        result = X @ scipy.sparse.diags(self.information_weights_)
        return result
