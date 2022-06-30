from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings

from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._cython_blas cimport (_axpy, _dot, _asum, _ger, _gemv, _nrm2,
                                   _copy, _scal)
from sklearn.utils._cython_blas cimport RowMajor, ColMajor, Trans, NoTrans


from sklearn.utils._random cimport our_rand_r

from scipy.linalg.cython_blas cimport ssbmv, dsbmv

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef floating diff_abs_max(int n, floating* a, floating* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m

cdef void fispos(int n, floating* x, floating* y) nogil:
    """ y:= (x > 0).astype(float) """
    cdef int i
    cdef floating v
    for i in range(1, n):
        v = x[i]
        if v > 0:
            y[i] = 1.0
        else:
            y[i] = 0.0

cdef void fisneg(int n, floating* x, floating* y) nogil:
    """ y:= (x < 0).astype(float) """
    cdef int i
    cdef floating v
    for i in range(1, n):
        v = x[i]
        if v >= 0:
            y[i] = 0.0
        else:
            y[i] = 1.0

cdef void fnot(int n, floating* x) nogil:
    """ x := np.logical_not(x).astype(float) """
    cdef int i
    cdef floating v
    for i in range(1, n):
        v = x[i]
        if v > 0:
            x[i] = 0.0
        else:
            x[i] = 1.0

cdef void _sbmvl(int n, int k, floating alpha, floating* A, int lda, floating* x, int incx, floating beta, floating* y, int incy) nogil:
    """y := alpha*A*x + beta*y,
        where alpha and beta are scalars, x and y are n element vectors and
        A is an n by n symmetric band matrix provided as lower dimension, with k super-diagonals. 
    """
    if floating is float:
        ssbmv("L", &n, &k, &alpha, A, &lda, x, &incx, &beta, y, &incy)
    else:
        dsbmv("L", &n, &k, &alpha, A, &lda, x, &incx, &beta, y, &incy)

def sat_coordinate_descent(floating[::1] w,
                        floating alpha, floating beta,
                        floating[::1, :] X,
                        floating[::1, :] Xu,
                        floating[::1, :] Xs,
                        floating[::1] y,
                        floating[::1] yu,
                        floating[::1] ys,
                        floating[::1] s,
                        int max_iter, floating tol,
                        object rng, bint random=0, bint positive=0):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression.
        We minimize 1 / 2 * norm(y_u - X_u w, 2)^2 + alpha norm(w, 1) + max(0, y_s - X_s w) + (beta/2) * norm(w, 2)^2.
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_unsaturated_samples = Xu.shape[0]
    cdef unsigned int n_saturated_samples = Xs.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # compute norms of the columns of X
    cdef floating[::1] norm_cols_X = np.square(X).sum(axis=0)

    # initial value of the residuals
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    cdef floating[::1] Ru = np.empty(n_unsaturated_samples, dtype=dtype)
    cdef floating[::1] Rs = np.empty(n_saturated_samples, dtype=dtype)
    cdef floating[::1] I = np.empty(n_samples, dtype=dtype)
    cdef floating[::1] Is = np.empty(n_saturated_samples, dtype=dtype)
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)

    cdef floating tmp
    cdef floating w_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating w_norm2
    cdef floating l1_norm
    cdef floating const
    cdef floating A_norm2
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged.")

    with nogil:
        # R = y - np.dot(X, w)
        _copy(n_samples, &y[0], 1, &R[0], 1)
        _gemv(ColMajor, NoTrans, n_samples, n_features, -1.0, &X[0, 0],
              n_samples, &w[0], 1, 1.0, &R[0], 1)
        # Ru = yu - np.dot(Xu, w)
        _copy(n_unsaturated_samples, &yu[0], 1, &Ru[0], 1)
        _gemv(ColMajor, NoTrans, n_unsaturated_samples, n_features, -1.0, &Xu[0, 0],
              n_unsaturated_samples, &w[0], 1, 1.0, &Ru[0], 1)
        # Rs = ys - np.dot(Xs, w)
        _copy(n_saturated_samples, &ys[0], 1, &Rs[0], 1)
        _gemv(ColMajor, NoTrans, n_saturated_samples, n_features, -1.0, &Xs[0, 0],
              n_saturated_samples, &w[0], 1, 1.0, &Rs[0], 1)

        # tol *= np.dot(y, y)
        tol *= _dot(n_samples, &y[0], 1, &y[0], 1)

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    _axpy(n_samples, w_ii, &X[0,ii], 1, &R[0], 1)
                    
                    # Ru += w_ii * Xu[:,ii]
                    _axpy(n_unsaturated_samples, w_ii, &Xu[0, ii], 1, &Ru[0], 1)
                    
                # tmp = (Xu[:,ii]*Ru).sum()
                tmp = _dot(n_unsaturated_samples, &Xu[0, ii], 1, &Ru[0], 1)
                
                # tmp += (Xs[:,ii] * 1(Rs > 0)).sum()
                fispos(n_saturated_samples, &Rs[0], &Is[0])
                tmp = (tmp + _dot(n_saturated_samples, &Xs[0, ii], 1, &Is[0], 1))          

                if positive and tmp < 0:
                    w[ii] = 0.0
                else:
                    w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0)
                             / (norm_cols_X[ii] + beta))
                
                if w_ii != 0.0:
                    # Rs += w_ii * Xs[:,ii]
                    _axpy(n_saturated_samples, w_ii, &Xs[0, ii], 1, &Rs[0], 1)
                
                if w[ii] != 0.0:
                    # R -= w[ii] * X[:,ii] # Update residual
                    _axpy(n_samples, -w[ii], &X[0, ii], 1, &R[0], 1)
                    
                    # Ru -=  w[ii] * Xu[:,ii] # Update residual
                    _axpy(n_unsaturated_samples, -w[ii], &Xu[0, ii], 1, &Ru[0], 1)
                    
                    # Rs -=  w[ii] * Xs[:,ii] # Update residual
                    _axpy(n_saturated_samples, -w[ii], &Xs[0, ii], 1, &Rs[0], 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                d_w_max = fmax(d_w_max, d_w_ii)

                w_max = fmax(w_max, fabs(w[ii]))

            if (w_max == 0.0 or
                d_w_max / w_max < d_w_tol or
                n_iter == max_iter - 1):
                break

                ## TODO: duality gap BELOW
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion
                
                ## R = R * np.logical_not(np.logical_and(R < 0, s)) where s indicator for saturated samples
                fisneg(n_samples, &R[0], &I[0])
                _sbmvl(n_samples, 0, 1.0, &I[0], 1, &s[0], 1, 0.0, &I[0], 1)
                fnot(n_samples, &I[0])
                _sbmvl(n_samples, 0, 1.0, &I[0], 1, &R[0], 1, 0.0, &R[0], 1)
                
                # XtA = np.dot(X.T, R) - beta * w
                _copy(n_features, &w[0], 1, &XtA[0], 1)
                _gemv(ColMajor, Trans,
                      n_samples, n_features, 1.0, &X[0, 0], n_samples,
                      &R[0], 1,
                      -beta, &XtA[0], 1)

                if positive:
                    dual_norm_XtA = max(n_features, &XtA[0])
                else:
                    dual_norm_XtA = abs_max(n_features, &XtA[0])

                # R_norm2 = np.dot(R, R)
                R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)

                # w_norm2 = np.dot(w, w)
                w_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = _asum(n_features, &w[0], 1)

                # np.dot(R.T, y)
                gap += (alpha * l1_norm
                        - const * _dot(n_samples, &R[0], 1, &y[0], 1)
                        + 0.5 * beta * (1 + const ** 2) * (w_norm2))

                if gap < tol:
                    # return if we reached desired tolerance
                    break

        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                message = (
                    "Objective did not converge. You might want to increase "
                    "the number of iterations, check the scale of the "
                    "features or consider increasing regularisation. "
                    f"Duality gap: {gap:.3e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += (
                        " Linear regression models with null weight for the "
                        "l1 regularization term are more efficiently fitted "
                        "using one of the solvers implemented in "
                        "sklearn.linear_model.Ridge/RidgeCV instead."
                    )
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(w), gap, tol, n_iter + 1