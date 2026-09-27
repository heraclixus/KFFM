import torch

from optimal_transport import RBFKernel, SobolevRBFKernel, create_kernel
from util.hilbert import sobolev_norm_squared_1d


def test_create_sobolev_rbf_kernel():
    kernel = create_kernel("sobolev_rbf", sigma=2.0, s=1.0)

    assert isinstance(kernel, SobolevRBFKernel)
    assert kernel.sigma == 2.0
    assert kernel.s == 1.0


def test_sobolev_rbf_cost_matches_hilbert_norm_1d():
    torch.manual_seed(0)
    sigma = 1.7
    kernel = SobolevRBFKernel(sigma=sigma, s=1.0, domain_length=1.0)
    x = torch.randn(3, 1, 16)
    y = torch.randn(4, 1, 16)

    gram = kernel(x, y)
    expected_norm_sq = sobolev_norm_squared_1d(x[0] - y[2], s=1.0, L=1.0)
    expected_kernel = torch.exp(-expected_norm_sq / (2 * sigma**2))

    assert torch.allclose(gram[0, 2], expected_kernel.squeeze(), atol=1e-5)


def test_sobolev_rbf_kernel_is_symmetric_with_unit_diagonal_2d():
    torch.manual_seed(0)
    kernel = SobolevRBFKernel(sigma=1.0, s=1.0, domain_length=1.0)
    x = torch.randn(5, 1, 8, 8)

    gram = kernel(x, x)

    assert torch.allclose(gram, gram.T, atol=1e-5)
    assert torch.allclose(torch.diag(gram), torch.ones(5), atol=1e-5)
    assert torch.isfinite(gram).all()


def test_flat_rbf_still_available():
    kernel = create_kernel("rbf", sigma=0.5)

    assert isinstance(kernel, RBFKernel)
