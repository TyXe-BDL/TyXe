import math

import torch
from torch.distributions import constraints, constraint_registry, transforms
from torch.distributions.utils import lazy_property, _standard_normal

from pyro.distributions.torch_distribution import TorchDistribution


# move to torch.distributions.constraints?
class _RealMatrix(constraints.Constraint):
    def check(self, value):
        # using .all() twice here in order to implicitly check that the tensor has at least two dimensions.
        # This is similar to the _RealVector constraint -- it might make sense to explicitly check for .dim() >= 2?
        return (value == value).all(-1).all()


real_matrix = _RealMatrix()


@constraint_registry.biject_to.register(real_matrix)
@constraint_registry.transform_to.register(real_matrix)
def _transform_to_real(constraint):
    return transforms.identity_transform


def _broadcast_matrix_normal_params(row_mat, col_mat, loc_mat):
    r"""
    :param row_mat: (optional batch shape x) m x m tensor
    :param col_mat: (optional batch shape x) n x n tensor
    :param loc_mat: (optional batch shape x) m x n tensor
    :return: broadcasted tensors with same leading batch shape
    """
    m, n = loc_mat.shape[-2:]
    batch_shapes = row_mat.shape[:-2], col_mat.shape[:-2], loc_mat.shape[:-2]
    if any(batch_shapes):
        max_len = max(map(len, batch_shapes))

        def as_padded_tensor(s):
            t = torch.tensor(s).long()
            return torch.cat([t.new_ones(max_len - t.size(0)), t])

        batch_shape = torch.Size(torch.stack(tuple(map(
            as_padded_tensor, batch_shapes))).max(0)[0])
        return (row_mat.expand(batch_shape + (m, m)),
                col_mat.expand(batch_shape + (n, n)),
                loc_mat.expand(batch_shape + (m, n)))
    else:
        return row_mat, col_mat, loc_mat


def _batch_diag(bmat):
    return torch.diagonal(bmat, dim1=-2, dim2=-1)


def _batch_outer_product(bv1, bv2):
    r"""
    Calculates the outer product on a batch of vector with compatible batch shapes. Batch shapes must be
    broadcastable to be compatible. Takes in two tensors of shapes ... x m and ... x n and returns a tensor of
    shape ... x m x n.
    """
    return bv1.unsqueeze(-1) * bv2.unsqueeze(-2)


def _batch_trtrs_lower(flat_bB, flat_bA):
    return torch.stack([torch.trtrs(B, A, upper=False)[0] for B, A in zip(flat_bB, flat_bA)])


def _batch_kf_mahalanobis(bLU, bLV, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`vec(\mathbf{X})^\top\mathbf{M}^{-1}vec(\mathbf{X})` for a Kronecker
    factored M :math:`\mathbf{M} = \mathbf{U} \otimes \mathbf{V}`. Since M is Kronecker factored, the distance can
    be computed efficiently as :math:`tr(\mathbf{V}^{-1} \mathbf{X}^\top \mathbf{U}^{-1} \mathbf{X})`
    The function takes the lower Cholesky factors of U and V as arguments, i.e.
    :math:`\mathbf{U} = \mathbf{LU}\mathbf{LU}^\top` and :math:`\mathbf{V} = \mathbf{LV}\mathbf{LV}^\top`.
    :param bLU: tensor of shape m x m with possible leading batch dimensions
    :param bLV: tensor of shape n x n with possible leading batch dimensions
    :param bx: tensor of shape m x n with possible leading batch dimensions
    :return:
    """
    m, n = bx.shape[-2:]
    batch_shape = bx.shape[:-2]
    bLU = bLU.expand(batch_shape + (m, m))
    bLV = bLV.expand(batch_shape + (n, n))

    # the implementation is based on rotating the leading cholesky factor to the right, such that the trace of an
    # inner product of a matrixhas to be calculated, which can be done efficiently by squaring and summing all elements
    # TODO implement using torch.triangular_solve for torch >= 1.1 to avoid for-loop for batches
    mm1 = _batch_trtrs_lower(
        bx.view(-1, m, n).transpose(-1, -2), bLV.reshape(-1, n, n))
    mm2 = _batch_trtrs_lower(
        mm1.transpose(-1, -2), bLU.reshape(-1, m, m))
    return mm2.pow(2).sum((-2, -1)).reshape(batch_shape)


class MatrixNormal(TorchDistribution):
    r"""
    Creates a matrix normal distribution parameterized by a mean matrix and the covariance of the rows and the
    columns. This is equivalent to a multivariate normal distribution with the columns of the mean matrix stacked
    into a vector and the Kronecker product of the the row and column covariances as the covariance. However,
    all operations like sampling, evaluating the pdf or calculating the entropy are computationally much more
    efficient thanks to the Kronecker factorization of the covariance.

    The matrix normal distribution can be parameterized either in terms of a pair of positive definite covariance
    matrices :math:`\mathbf{\Sigma}_r` and :math:`\mathbf{\Sigma}_c`, a pair of positive definite precision matrices
    :math:`\mathbf{\Sigma}^{-1}_r` and :math:`\mathbf{\Sigma}^{-1}_c` or a pair of lower-triangular matrices
    :math:`\mathbf{L}_r` and :math:`\mathbf{L}_c` such that :math:`\mathbf{L}_r\mathbf{L}_r^\top = \mathbf{\Sigma}_r`
    and :math:`\mathbf{L}_c\mathbf{L}_c^\top = \mathbf{\Sigma}_c`. The latter two matrices can be obtained by the
    Cholesky decomposition of the covariance.

    Example:

        >>> m = MatrixNormal(torch.zeros(3, 2), torch.eye(3), torch.eye(2))
        >>> m.sample()  # matrix normally distributed with mean=`[[0,0],[0,0],[0,0]]` and row and column covariance=`I`.
        tensor([[ 0.7476, -0.5581],
                [-0.0741,  0.0298],
                [ 1.0855,  0.2974]])

    Args:
        loc (Tensor): mean of the distribution
        row_covariance_matrix (Tensor): positive-definite covariance matrix of the rows
        col_covariance_matrix (Tensor): positive-definite covariance matrix of the columns
        row_precision_matrix (Tensor): positive-definite precision matrix of the rows
        col_precision_matrix (Tensor): positive-definite precision matrix of the columns
        row_scale_tril (Tensor): lower-triangular factor of the row covariance, with positive-valued diagonal
        col_scale_tril (Tensor): lower-triangular factor of the column covariance, with positive-valued diagonal

    Note:
        Only one pair of :attr:`row_covariance_matrix`/:attr`col_covariance_matrix`,
        :attr:`row_precision_matrix`/:attr`col_precision_matrix` or :attr:`row_scale_tril`/:attr`col_scale_tril` can
        be specified. Using the scale_tril pair will be more efficient, since the other arguments are used internally
        only to calculate the Cholesky decomposition of the covariance matrices.
    """

    arg_constraints = {"loc": real_matrix,
                       "row_covariance_matrix": constraints.positive_definite,
                       "col_covariance_matrix": constraints.positive_definite,
                       "row_precision_matrix": constraints.positive_definite,
                       "col_precision_matrix": constraints.positive_definite,
                       "row_scale_tril": constraints.lower_cholesky,
                       "col_scale_tril": constraints.lower_cholesky}
    support = real_matrix
    has_rsample = True

    def __init__(self, loc,
                 row_covariance_matrix=None, col_covariance_matrix=None,
                 row_precision_matrix=None, col_precision_matrix=None,
                 row_scale_tril=None, col_scale_tril=None,
                 validate_args=None):
        # TODO it might be useful (similarly for the regular multivariate normal distribution) to also accept the
        #  Cholesky of the precision matrix and perform all operations either with the square root of the precision
        #  or with that of the covariance, depending on which one was used to parameterize the distribution. This
        #  would be to avoid explicitly invertion the precision matrix, which is numerically unstable.
        if loc.dim() < 2:
            raise ValueError("loc must be at least two-dimensional.")
        if ((row_covariance_matrix is not None and
                col_covariance_matrix is not None) +
                (row_precision_matrix is not None and
                 col_precision_matrix is not None) +
                (row_scale_tril is not None and
                 col_scale_tril is not None)) != 1:
            raise ValueError("Exactly one row/column pair out of covariance_matrix, "
                             "precision_matrix or scale_tril may be specified.")
        if row_scale_tril is not None:
            if row_scale_tril.dim() < 2:
                raise ValueError()
            if col_scale_tril.dim() < 2:
                raise ValueError()
            self.row_scale_tril, self.col_scale_tril_, self.loc = _broadcast_matrix_normal_params(
                row_scale_tril, col_scale_tril, loc)
        elif row_covariance_matrix is not None:
            if row_covariance_matrix.dim() < 2:
                raise ValueError()
            if col_covariance_matrix.dim() < 2:
                raise ValueError()
            self.row_covariance_matrix, self.col_covariance_matrix, self.loc = _broadcast_matrix_normal_params(
                row_covariance_matrix, col_covariance_matrix, loc)
        else:
            if row_precision_matrix.dim() < 2:
                raise ValueError
            if col_precision_matrix.dim() < 2:
                raise ValueError
            self.row_precision_matrix, self.col_precision_matrix, self.loc = _broadcast_matrix_normal_params(
                row_precision_matrix, col_precision_matrix, loc)

        batch_shape, event_shape = self.loc.shape[:-2], self.loc.shape[-2:]
        super(MatrixNormal, self).__init__(batch_shape, event_shape, validate_args)
        self._row_shape = (self._event_shape[-2], )
        self._col_shape = (self._event_shape[-1], )

        if row_scale_tril is not None:
            self._unbroadcasted_row_scale_tril = row_scale_tril
            self._unbroadcasted_col_scale_tril = col_scale_tril
        else:
            if row_precision_matrix is not None:
                self.row_covariance_matrix = torch.inverse(
                    self.row_precision_matrix)
                self.col_covariance_matrix = torch.inverse(
                    self.col_precision_matrix)
            self._unbroadcasted_row_scale_tril = torch.cholesky(
                self.row_covariance_matrix)
            self._unbroadcasted_col_scale_tril = torch.cholesky(
                self.col_covariance_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MatrixNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self._event_shape
        row_cov_shape = batch_shape + self._row_shape + self._row_shape
        col_cov_shape = batch_shape + self._col_shape + self._col_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_row_scale_tril = self._unbroadcasted_row_scale_tril
        new._unbroadcasted_col_scale_tril = self._unbroadcasted_col_scale_tril
        new._row_shape = self._row_shape
        new._col_shape = self._col_shape
        if 'row_covariance_matrix' in self.__dict__:
            new.row_covariance_matrix = self.row_covariance_matrix.expand(
                row_cov_shape)
            new.col_covariance_matrix = self.col_covariance_matrix.expand(
                col_cov_shape)
        if 'row_scale_tril' in self.__dict__:
            new.row_scale_tril = self.row_scale_tril.expand(row_cov_shape)
            new.col_scale_tril = self.col_scale_tril.expand(col_cov_shape)
        if 'row_precision_matrix' in self.__dict__:
            new.row_precision_matrix = self.row_precision_matrix.expand(
                row_cov_shape)
            new.col_precision_matrix = self.row_precision_matrix.expand(
                col_cov_shape)
        super(MatrixNormal, new).__init__(batch_shape,
                                          self.event_shape,
                                          validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def row_scale_tril(self):
        return self._unbroadcasted_row_scale_tril.expand(
            self._batch_shape + self._row_shape + self._row_shape)

    @lazy_property
    def col_scale_tril(self):
        return self._unbroadcasted_col_scale_tril.expand(
            self._batch_shape + self._col_shape + self._col_shape)

    @lazy_property
    def row_covariance_matrix(self):
        return (torch.matmul(
                self._unbroadcasted_row_scale_tril,
                self._unbroadcasted_row_scale_tril.transpose(-1, -2))
                .expand(self._batch_shape + self._row_shape + self._row_shape))

    @lazy_property
    def col_covariance_matrix(self):
        return (torch.matmul(
                self._unbroadcasted_col_scale_tril,
                self._unbroadcasted_col_scale_tril.transpose(-1, -2))
                .expand(self._batch_shape + self._col_shape + self._col_shape))

    @lazy_property
    def row_precision_matrix(self):
        row_scale_tril_inv = torch.inverse(self._unbroadcasted_row_scale_tril)
        return (torch.matmul(
                row_scale_tril_inv.transpose(-1, -2), row_scale_tril_inv)
                .expand(self._batch_shape + self._row_shape + self._row_shape))

    @lazy_property 
    def col_precision_matrix(self):
        col_scale_tril_inv = torch.inverse(self._unbroadcasted_col_scale_tril)
        return (torch.matmul(
                col_scale_tril_inv.transpose(-1, -2), col_scale_tril_inv)
                .expand(self._batch_shape + self._col_shape + self._col_shape))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        row_var = _batch_diag(self._unbroadcasted_row_scale_tril).pow(2).expand(
            self._batch_shape + self._row_shape)
        col_var = _batch_diag(self._unbroadcasted_col_scale_tril).pow(2).expand(
            self._batch_shape + self._col_shape)
        return _batch_outer_product(row_var, col_var)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self._unbroadcasted_row_scale_tril.matmul(eps).matmul(
            self._unbroadcasted_col_scale_tril.transpose(-1, -2))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        m = _batch_kf_mahalanobis(self._unbroadcasted_row_scale_tril,
                                  self._unbroadcasted_col_scale_tril,
                                  diff)
        half_log_det = self._cov_half_log_det()
        return -0.5 * (self._row_shape[0] * self._col_shape[0] *
                       math.log(2 * math.pi) + m) - half_log_det

    def entropy(self):
        half_log_det = self._cov_half_log_det()
        H = (half_log_det + 0.5 * self._row_shape[0] * self._col_shape[0] *
             (1.0 + math.log(2 * math.pi)))
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)

    def _cov_half_log_det(self):
        r"""
        Calculates half the log determinant of the full covariance, which for a Kronecker factored matrix
        :math:`\mathbf{\Sigma} = \mathbf{U} \otimes \mathbf{V}` with factors of shape U: m x m and V: n x n
        is :math:`\log det(\mathbf{M) = n \log det(\mathbf{U}) + m \log det(\mathbf{V})`
        """
        row_half_log_det = _batch_diag(
            self._unbroadcasted_row_scale_tril).log().sum(-1)
        col_half_log_det = _batch_diag(
            self._unbroadcasted_col_scale_tril).log().sum(-1)
        return (self._col_shape[0] * row_half_log_det +
                self._row_shape[0] * col_half_log_det)
