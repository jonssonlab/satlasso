from abc import ABC
from functools import partial
import sys
import numbers
import warnings
from joblib import Parallel, delayed, effective_n_jobs

import numpy as np
from scipy import sparse

from sklearn.linear_model._base import RegressorMixin, LinearModel
from sklearn.linear_model._coordinate_descent import ElasticNet, LinearModelCV

from sklearn.linear_model._base import _pre_fit, _preprocess_data, _deprecate_normalize
from sklearn.linear_model._coordinate_descent import _alpha_grid, _set_order
from sklearn.model_selection import check_cv
from sklearn.utils import check_array, check_consistent_length, check_scalar, column_or_1d
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_non_negative, check_random_state

from sklearn.exceptions import NotFittedError

from sklearn.metrics import r2_score

import _cd_fast as cd_fast

## UTILS FROM SKLEARN
def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, only_non_negative=False
):
    """Validate sample weights.
    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)
    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
        Input sample weights.
    X : {ndarray, list, sparse matrix}
        Input data.
    only_non_negative : bool, default=False,
        Whether or not the weights are expected to be non-negative.
        .. versionadded:: 1.0
    dtype : dtype, default=None
        dtype of the validated `sample_weight`.
        If None, and the input `sample_weight` is an array, the dtype of the
        input is preserved; otherwise an array with the default numpy dtype
        is be allocated.  If `dtype` is not one of `float32`, `float64`,
        `None`, the output will be of dtype `float64`.
    copy : bool, default=False
        If True, a copy of sample_weight will be created.
    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            order="C",
            copy=copy,
            input_name="sample_weight",
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if only_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight
## 

def sat_path(
    X,
    y,
    s,
    *,
    eps=1e-3,
    n_l1_multipliers=100,
    l1_multipliers=None,
    precompute="auto",
    Xy=None,
    copy_X=True,
    coef_init=None,
    verbose=False,
    return_n_iter=False,
    positive=False,
    check_input=True,
    **kwargs
):
    X_offset_param = kwargs.pop("X_offset", None)
    X_scale_param = kwargs.pop("X_scale", None)
    sample_weight = kwargs.pop("sample_weight", None)
    tol = kwargs.pop("tol", 1e-4)
    max_iter = kwargs.pop("max_iter", 1000)
    random_state = kwargs.pop("random_state", None)
    selection = kwargs.pop("selection", "cyclic")

    if len(kwargs) > 0:
        raise ValueError("Unexpected parameters in kwargs", kwargs.keys())

    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(
            X,
            accept_sparse="csc",
            dtype=[np.float64, np.float32],
            order="F",
            copy=copy_X,
        )
        y = check_array(
            y,
            accept_sparse="csc",
            dtype=X.dtype.type,
            order="F",
            copy=False,
            ensure_2d=False,
        )
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(
                Xy, dtype=X.dtype.type, order="C", copy=False, ensure_2d=False
            )

    n_samples, n_features = X.shape

    ## TODO: incorporate in later data
    ## Separate unsaturated and saturated samples
    Xu, yu = _set_order(X[np.logical_not(s)], y[np.logical_not(s)], order="F")
    Xs, ys = _set_order(X[s.astype(bool)], y[s.astype(bool)], order="F")
    if sparse.issparse(s):
        s = s.asformat("csc")
    else:
        s = np.array(s, order="F")

    if sparse.isspmatrix(X):
        ## TODO: account for saturated
        if X_offset_param is not None:
            # As sparse matrices are not actually centered we need this to be passed to
            # the CD solver.
            X_sparse_scaling = X_offset_param / X_scale_param
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should have been passed through _pre_fit already if function is called
    # from ElasticNet.fit
    if check_input:
        ## TODO: account for saturated
        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X,
            y,
            Xy,
            precompute,
            normalize=False,
            fit_intercept=False,
            copy=False,
            check_input=check_input,
        ) 
    if l1_multipliers is None:
        # No need to normalize fit_intercept: it has been done above
        ## TODO: account for saturated
        l1_multipliers = _alpha_grid(
            X,
            y,
            Xy=Xy,
            l1_ratio=1.0,
            fit_intercept=False,
            eps=eps,
            n_alphas=n_l1_multipliers,
            normalize=False,
            copy_X=False,
        )
    elif len(l1_multipliers) > 1:
        l1_multipliers = np.sort(l1_multipliers)[::-1]  # make sure l1_multipliers are properly ordered

    n_l1_multipliers = len(l1_multipliers)
    dual_gaps = np.empty(n_l1_multipliers)
    n_iters = []

    rng = check_random_state(random_state)
    if selection not in ["random", "cyclic"]:
        raise ValueError("selection should be either random or cyclic.")
    random = selection == "random"

    coefs = np.empty((n_features, n_l1_multipliers), dtype=X.dtype)

    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, l1_multiplier in enumerate(l1_multipliers):
        # account for n_samples scaling in objectives between here and cd_fast
        l1_multiplier *= n_samples
        if sparse.isspmatrix(X):
            ## sparse sat coordinate descent 
            raise NotImplementedError
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=X.dtype.type, order="C")
            ## coordinate descent gram
            raise NotImplementedError
        elif precompute is False:
            model = cd_fast.sat_coordinate_descent(
                coef_, l1_multiplier, 0.0, X, Xu, Xs, y, yu, ys, s, max_iter, tol, rng, random, positive
            )
        else:
            raise ValueError(
                "Precompute should be one of True, False, 'auto' or array-like. Got %r"
                % precompute
            )
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        # we correct the scale of the returned dual gap, as the objective
        # in cd_fast is n_samples * the objective in this docstring.
        dual_gaps[i] = dual_gap_ / n_samples
        n_iters.append(n_iter_)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print("Path: %03i out of %03i" % (i, n_l1_multipliers))
            else:
                sys.stderr.write(".")

    if return_n_iter:
        return l1_multipliers, coefs, dual_gaps, n_iters
    return l1_multipliers, coefs, dual_gaps

def _path_residuals(
    X,
    y,
    s,
    saturation_value,
    sample_weight,
    train,
    test,
    normalize,
    fit_intercept,
    path,
    path_params,
    l1_multipliers=None,
    X_order=None,
    dtype=None,
):
    """Returns the MSE for the models computed by 'path'.
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    sample_weight : None or array-like of shape (n_samples,)
        Sample weights.
    train : list of indices
        The indices of the train set.
    test : list of indices
        The indices of the test set.
    path : callable
        Function returning a list of models on the path. See
        enet_path for an example of signature.
    path_params : dictionary
        Parameters passed to the path function.
    l1_multipliers : array-like, default=None
        Array of float that is used for cross-validation. If not
        provided, computed using 'path'.
    X_order : {'F', 'C'}, default=None
        The order of the arrays expected by the path function to
        avoid memory copies.
    dtype : a numpy dtype, default=None
        The dtype of the arrays expected by the path function to
        avoid memory copies.
    """
    X_train = X[train]
    y_train = y[train]
    s_train = s[train]
    X_test = X[test]
    y_test = y[test]
    s_test = s[test]
    if sample_weight is None:
        sw_train, sw_test = None, None
    else:
        sw_train = sample_weight[train]
        sw_test = sample_weight[test]
        n_samples = X_train.shape[0]
        # TLDR: Rescale sw_train to sum up to n_samples on the training set.
        # See TLDR and long comment inside ElasticNet.fit.
        sw_train *= n_samples / np.sum(sw_train)
        # Note: Alternatively, we could also have rescaled l1_multiplier instead
        # of sample_weight:
        #
        #     l1_multiplier *= np.sum(sample_weight) / n_samples

    if not sparse.issparse(X):
        for array, array_input in (
            (X_train, X),
            (y_train, y),
            (s_train, s),
            (X_test, X),
            (y_test, y),
            (s_test, s)
        ):
            if array.base is not array_input and not array.flags["WRITEABLE"]:
                # fancy indexing should create a writable copy but it doesn't
                # for read-only memmaps (cf. numpy#14132).
                array.setflags(write=True)

    if y.ndim == 1:
        precompute = path_params["precompute"]
    else:
        # No Gram variant of multi-task exists right now.
        # Fall back to default enet_multitask
        precompute = False

    ## TODO: accommodate saturated
    X_train, y_train, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
        X_train,
        y_train,
        None,
        precompute,
        normalize,
        fit_intercept,
        copy=False,
        sample_weight=sw_train,
    )

    path_params = path_params.copy()
    path_params["Xy"] = Xy
    path_params["X_offset"] = X_offset
    path_params["X_scale"] = X_scale
    path_params["precompute"] = precompute
    path_params["copy_X"] = False
    path_params["l1_multipliers"] = l1_multipliers
    # needed for sparse cd solver
    path_params["sample_weight"] = sw_train

    # Do the ordering and type casting here, as if it is done in the path,
    # X is copied and a reference is kept here
    X_train = check_array(X_train, accept_sparse="csc", dtype=dtype, order=X_order)
    l1_multipliers, coefs, _ = path(X_train, y_train, s_train, **path_params)
    del X_train, y_train, s_train

    if y.ndim == 1:
        # Doing this so that it becomes coherent with multioutput.
        coefs = coefs[np.newaxis, :, :]
        y_offset = np.atleast_1d(y_offset)
        y_test = y_test[:, np.newaxis]

    if normalize:
        nonzeros = np.flatnonzero(X_scale)
        coefs[:, nonzeros] /= X_scale[nonzeros][:, np.newaxis]

    intercepts = y_offset[:, np.newaxis] - np.dot(X_offset, coefs)
    X_test_coefs = safe_sparse_dot(X_test, coefs)

    s_test_clip = np.full_like(y_test[:, :, np.newaxis], np.inf)
    s_test_clip[s_test, ...] = saturation_value
    s_test_clip[s_test, ...] -= intercepts[s_test, ...]

    residues = np.clip(X_test_coefs, a_max=s_test_clip) - y_test[:, :, np.newaxis]
    residues += intercepts
    ## TODO: check above
    if sample_weight is None:
        this_mse = (residues**2).mean(axis=0)
    else:
        this_mse = np.average(residues**2, weights=sw_test, axis=0)

    return this_mse.mean(axis=0)

class SatLasso(RegressorMixin, LinearModel):
    """Linear regression with combined L1 priors as regularizer that permits saturated data.
    Minimizes the objective function::
            1 / (2 * n_samples) * ||yu - Xuw||^2_2
            + l1_multiplier * ||w||_1
            + max(0, ys-Xsw)
    where Xu, yu and Xs, ys denote the unsaturated and saturated data, respectively.

    Parameters
    ----------
    l1_multiplier : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization
        strength. `l1_multplier` must be a non-negative float i.e. in `[0, inf)`.
        When `l1_multiplier = 0`, the objective is equivalent to ordinary least
        squares accommodating saturated data.
    saturation : float or str, default="max"
        The determination of saturated data. If `saturation` is a string, it must 
        be "max", in which case the maximum value in the passed targets will be 
        considered saturated samples. If `saturation` is a float, samples with the
        given float value as target will be considered saturated samples.
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0 and will be removed in
            1.2.
    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.
    max_iter : int, default=1000
        The maximum number of iterations.
    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Parameter vector (w in the cost function formula).
    sparse_coef_ : sparse matrix of shape (n_features,)
        Sparse representation of the `coef_`.
    intercept_ : float
        Independent term in decision function.
    n_iter_ : int or list of int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    path = staticmethod(sat_path)

    def __init__(
        self,
        l1_multiplier=1.0,
        saturation="max",
        *,
        fit_intercept=True,
        normalize="deprecated",
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic"
    ):
        self.l1_multiplier = l1_multiplier
        self.saturation = saturation
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
    
    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data.
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target. Will be cast to X's dtype if necessary.
        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.
            .. versionadded:: 0.23
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        self : object
            Fitted estimator.
        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.
        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        _normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )

        check_scalar(
            self.l1_multiplier,
            "l1_multiplier",
            target_type=numbers.Real,
            min_val=0.0,
        )

        if isinstance(self.precompute, str):
            raise ValueError(
                "precompute should be one of True, False or array-like. Got %r"
                % self.precompute
            )

        if self.max_iter is not None:
            check_scalar(
                self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
            )

        check_scalar(self.tol, "tol", target_type=numbers.Real, min_val=0.0)

        # Remember if X is copied
        X_copied = False
        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X_copied = self.copy_X and self.fit_intercept
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csc",
                order="F",
                dtype=[np.float64, np.float32],
                copy=X_copied,
                multi_output=False,
                y_numeric=True,
            )
            y = check_array(
                y, order="F", copy=False, dtype=X.dtype.type, ensure_2d=False
            )
        
        if isinstance(self.saturation, float):
            if self.saturation not in y:
                raise ValueError("saturation float value does not exist in y.")
            self.saturation_value = self.saturation
        elif isinstance(self.saturation, str):
            if self.saturation not in ["max"]:
                raise ValueError("saturation should either be a float value or 'max'.")
            self.saturation_value = max(y)
        else:
            raise ValueError("saturation should either be a float value or 'max'.")

        ## Boolean array indicator for saturated samples
        s = (y == self.saturation_value).astype(float)

        n_samples, n_features = X.shape
        l1_multiplier = self.l1_multiplier

        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sample_weight is not None:
            if check_input:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            sample_weight = sample_weight * (n_samples / np.sum(sample_weight))

        # Ensure copying happens only once, don't do it again if done above.
        # X and y will be rescaled if sample_weight is not None, order='F'
        # ensures that the returned X and y are still F-contiguous.
        should_copy = self.copy_X and not X_copied
        ## TODO: account for saturated
        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X,
            y,
            None,
            self.precompute,
            _normalize,
            self.fit_intercept,
            copy=should_copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # coordinate descent needs F-ordered arrays and _pre_fit might have
        # called _rescale_data
        if check_input or sample_weight is not None:
            X, y = _set_order(X, y, order="F")

        if self.selection not in ["cyclic", "random"]:
            raise ValueError("selection should be either random or cyclic.")
        
        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((1, n_features), dtype=X.dtype, order="F")
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]


        _, this_coef, this_dual_gap, this_iter = self.path(
                    X,
                    y,
                    s,
                    eps=None,
                    n_l1_multipliers=None,
                    l1_multipliers=[l1_multiplier],
                    precompute=precompute,
                    Xy=Xy,
                    copy_X=True,
                    coef_init=coef_[0],
                    verbose=False,
                    return_n_iter=True,
                    positive=self.positive,
                    check_input=False,
                    ## from here on **kwargs
                    tol=self.tol,
                    X_offset=X_offset,
                    X_scale=X_scale,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    selection=self.selection,
                    sample_weight=sample_weight,
                )

        self.n_iter_ = this_iter[0]
        self.coef_ = this_coef[:, 0]
        self.dual_gap_ = this_dual_gap[0]

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # check for finiteness of coefficients
        if not all(np.isfinite(w).all() for w in [self.coef_, self.intercept_]):
            raise ValueError(
                "Coordinate descent iterations resulted in non-finite parameter"
                " values. The input data may contain large values and need to"
                " be preprocessed."
            )

        # return self for chaining fit and predict calls
        return self

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination of the prediction.
        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y` accommodating saturated samples.
        """

        try:
            check_is_fitted(self)
            saturation_value = self.saturation_value
        except NotFittedError:
            saturation_value = max(y) if self.saturation == "max" else self.saturation

        s_clip = np.full_like(y, np.inf)
        s_clip[y == saturation_value] = saturation_value

        y_pred = self.predict(X)
        y_pred = np.clip(y_pred, a_max=s_clip)

        return r2_score(y, y_pred, sample_weight)

    @property
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return sparse.csr_matrix(self.coef_)
    
    def _decision_function(self, X):
        """Decision function of the linear model.
        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)
        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
        check_is_fitted(self)
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        else:
            return super()._decision_function(X)

class SatLassoCV(RegressorMixin, LinearModel, ABC):
    """Linear regression with combined L1 priors as regularizer that permits saturated data.
    Minimizes the objective function::
            1 / (2 * n_samples) * ||yu - Xuw||^2_2
            + l1_multiplier * ||w||_1
            + max(0, ys-Xsw)
    where Xu, yu and Xs, ys denote the unsaturated and saturated data, respectively.

    Parameters
    ----------
    saturation : float or str, default="max"
        The determination of saturated data. If `saturation` is a string, it must 
        be "max", in which case the maximum value in the passed targets will be 
        considered saturated samples. If `saturation` is a float, samples with the
        given float value as target will be considered saturated samples.
    n_l1_multipliers : int, default=100
        Number of l1_multipliers along regularization path.
    l1_multipliers : ndarray, default=None
        List of l1_multipliers where to compute the models.
        If ``None`` l1_multipliers are set automatically.
    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.
    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        .. deprecated:: 1.0
            ``normalize`` was deprecated in version 1.0 and will be removed in
            1.2.
    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.
    max_iter : int, default=1000
        The maximum number of iterations.
    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    verbose : bool or int, default=0
        Amount of verbosity.
    n_jobs : int, default=None
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.
    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    l1_multiplier_ : float
        The amount of L1 penalization chosen by cross validation.
    coef_ : ndarray of shape (n_features,)
        Parameter vector (w in the cost function formula).
    sparse_coef_ : sparse matrix of shape (n_features,)
        Sparse representation of the `coef_`.
    intercept_ : float
        Independent term in decision function.
    mse_path_ : ndarray of shape (n_l1_multiplier, n_folds)
        Mean square error for the test set on each fold, varying l1_multiplier.
    l1_multipliers_ : ndarray of shape (n_l1_multipliers,)
        The grid of l1_multipliers used for fitting.
    dual_gap_ : float 
        The dual gap at the end of the optimization for the optimal l1_multiplier
        (``l1_multiplier_``).
    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal l1_multiplier.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    """

    path = staticmethod(sat_path)

    def __init__(
        self,
        saturation="max",
        *,
        n_l1_multipliers=100,
        l1_multipliers=None,
        fit_intercept=True,
        normalize="deprecated",
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        cv=None,
        verbose=0,
        n_jobs=None,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic"
    ):
        self.n_l1_multipliers = n_l1_multipliers
        self.l1_multipliers = l1_multipliers
        self.saturation = saturation
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def _get_estimator(self):
        return SatLasso()
    
    def _is_multitask(self):
        return False

    def _more_tags(self):
        return {
            "multioutput": False, 
                "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
    
    def fit(self, X, y, sample_weight=None):
        ## TODO: fix below
        """Fit linear model with coordinate descent.
        Fit is on grid of l1_multipliers and best l1_multiplier estimated by cross-validation.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : float or array-like of shape (n_samples,), \
                default=None
            Sample weights used for fitting and evaluation of the weighted
            mean squared error of each cv-fold. Note that the cross validated
            MSE that is finally used to find the best model is the unweighted
            mean over the (weighted) MSEs of each test fold.
        Returns
        -------
        self : object
            Returns an instance of fitted model.
        """

        # Do as _deprecate_normalize but without warning as it's raised
        # below during the refitting on the best l1_multiplier.
        _normalize = self.normalize
        if _normalize == "deprecated":
            _normalize = False

        # This makes sure that there is no duplication in memory.
        # Dealing right with copy_X is important in the following:
        # Multiple functions touch X and subsamples of X and can induce a
        # lot of duplication of memory
        copy_X = self.copy_X and self.fit_intercept

        check_y_params = dict(
            copy=False, dtype=[np.float64, np.float32], ensure_2d=False
        )
        if isinstance(X, np.ndarray) or sparse.isspmatrix(X):
            # Keep a reference to X
            reference_to_old_X = X
            # Let us not impose fortran ordering so far: it is
            # not useful for the cross-validation loop and will be done
            # by the model fitting itself

            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse="csc", dtype=[np.float64, np.float32], copy=False
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if sparse.isspmatrix(X):
                if hasattr(reference_to_old_X, "data") and not np.may_share_memory(
                    reference_to_old_X.data, X.data
                ):
                    # X is a sparse matrix and has been copied
                    copy_X = False
            elif not np.may_share_memory(reference_to_old_X, X):
                # X has been copied
                copy_X = False
            del reference_to_old_X
        else:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr. We also want to allow y to be 64 or 32 but check_X_y only
            # allows to convert for 64.
            check_X_params = dict(
                accept_sparse="csc",
                dtype=[np.float64, np.float32],
                order="F",
                copy=copy_X,
            )
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params),
                multi_output=False
            )
            copy_X = False

        check_consistent_length(X, y)

        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                "Multi-task not supported."
            )
        y = column_or_1d(y, warn=True)

        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        model = self._get_estimator()

        if self.selection not in ["random", "cyclic"]:
            raise ValueError("selection should be either random or cyclic.")

        if isinstance(self.saturation, float):
            if self.saturation not in y:
                raise ValueError("saturation float value does not exist in y.")
            self.saturation_value = self.saturation
        elif isinstance(self.saturation, str):
            if self.saturation not in ["max"]:
                raise ValueError("saturation should either be a float value or 'max'.")
            self.saturation_value = max(y)
        else:
            raise ValueError("saturation should either be a float value or 'max'.")

        ## Boolean array indicator for saturated samples
        s = (y == self.saturation_value).astype(float)

        # All LinearModelCV parameters except 'cv' are acceptable
        path_params = self.get_params()

        # FIXME: 'normalize' to be removed in 1.2
        # path_params["normalize"] = _normalize
        # Pop `intercept` and `normalize` that are not parameter of the path
        # function
        path_params.pop("normalize", None)
        path_params.pop("fit_intercept", None)

        path_params.pop("cv", None)
        path_params.pop("n_jobs", None)

        l1_multipliers = self.l1_multipliers

        check_scalar_l1_multiplier = partial(
            check_scalar,
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="left",
        )

        if l1_multipliers is None:
            ## TODO: check works with saturated
            l1_multipliers = [
                _alpha_grid(
                    X,
                    y,
                    l1_ratio=1.,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_l1_multipliers,
                    normalize=_normalize,
                    copy_X=self.copy_X,
                )
            ]
        else:
            # Making sure s entries are scalars.
            if np.isscalar(l1_multipliers):
                check_scalar_l1_multiplier(l1_multipliers, "l1_multipliers")
            else:
                # l1_multipliers is an iterable item in this case.
                for index, l1_multiplier in enumerate(l1_multipliers):
                    check_scalar_l1_multiplier(l1_multiplier, f"l1_multipliers[{index}]")
            # Making sure l1_multipliers is properly ordered.
            l1_multipliers = np.tile(np.sort(l1_multipliers)[::-1], (1, 1))

        # We want n_l1_multipliers to be the number of l1_multipliers.
        n_l1_multipliers = len(l1_multipliers[0])
        path_params.update({"n_l1_multipliers": n_l1_multipliers})

        path_params["copy_X"] = copy_X
        # We are not computing in parallel, we can modify X
        # inplace in the folds
        if effective_n_jobs(self.n_jobs) > 1:
            path_params["copy_X"] = False

        # init cross-validation generator
        cv = check_cv(self.cv)

        # Compute path for all folds and compute MSE to get the best l1_multiplier
        folds = list(cv.split(X, y))
        best_mse = np.inf

        # We do a double for loop folded in one, in order to be able to
        # iterate in parallel on l1_ratio and folds
        jobs = (
            delayed(_path_residuals)(
                X,
                y,
                s,
                sample_weight,
                train,
                test,
                _normalize,
                self.fit_intercept,
                self.path,
                path_params,
                l1_multipliers=this_l1_multipliers,
                X_order="F",
                dtype=X.dtype.type,
            )
            for this_l1_multipliers in l1_multipliers
            for train, test in folds
        )
        mse_paths = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(jobs)
        mse_paths = np.reshape(mse_paths, (1, len(folds), -1))
        # The mean is computed over folds.
        mean_mse = np.mean(mse_paths, axis=1)
        self.mse_path_ = np.squeeze(np.moveaxis(mse_paths, 2, 1))
        for each_l1_multipliers, mse_l1_multipliers in zip(l1_multipliers, mean_mse):
            i_best_l1_multiplier = np.argmin(mse_l1_multipliers)
            this_best_mse = mse_l1_multipliers[i_best_l1_multiplier]
            if this_best_mse < best_mse:
                best_l1_multiplier = each_l1_multipliers[i_best_l1_multiplier]
                best_mse = this_best_mse

        self.l1_multiplier_ = best_l1_multiplier
        if self.l1_multipliers is None:
            self.l1_multipliers_ = np.asarray(l1_multipliers)
            self.l1_multipliers_ = self.l1_multipliers_[0]
        # Remove duplicate l1_multipliers in case l1_multipliers is provided.
        else:
            self.l1_multipliers_ = np.asarray(l1_multipliers[0])

        # Refit the model with the parameters selected
        common_params = {
            name: value
            for name, value in self.get_params().items()
            if name in model.get_params()
        }
        model.set_params(**common_params)
        model.l1_multiplier = best_l1_multiplier
        model.copy_X = copy_X
        precompute = getattr(self, "precompute", None)
        if isinstance(precompute, str) and precompute == "auto":
            model.precompute = False

        if sample_weight is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.dual_gap_ = model.dual_gap_
        self.n_iter_ = model.n_iter_
        return self