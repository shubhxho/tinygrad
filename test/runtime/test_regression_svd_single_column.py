import numpy as np
import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("shape", [(1, 1), (3, 1), (1, 3), (2, 3, 1), (2, 1, 3)])
@pytest.mark.parametrize("full_matrices", [False, True])
def test_svd_single_singular_value(shape, full_matrices):
  a = (np.arange(np.prod(shape), dtype=np.float32) + 1).reshape(shape)
  u, s, vh = [t.numpy() for t in Tensor(a).svd(full_matrices=full_matrices)]
  assert np.isfinite(u).all() and np.isfinite(s).all() and np.isfinite(vh).all()
  np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose((u[..., :1] * s[..., None, :]) @ vh[..., :1, :], a, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(u.swapaxes(-2, -1) @ u, np.broadcast_to(np.eye(u.shape[-1]), shape[:-2] + (u.shape[-1],)*2), atol=1e-5)
  np.testing.assert_allclose(vh @ vh.swapaxes(-2, -1), np.broadcast_to(np.eye(vh.shape[-2]), shape[:-2] + (vh.shape[-2],)*2), atol=1e-5)
