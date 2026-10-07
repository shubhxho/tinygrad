import numpy as np
import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid_single_scalar(indexing):
  out, = Tensor(3).meshgrid(indexing=indexing)
  assert out.shape == (1,)
  np.testing.assert_array_equal(out.numpy(), [3])

@pytest.mark.parametrize("shape", [(1, 2), (1, 1, 1)])
def test_meshgrid_rejects_matrices(shape):
  with pytest.raises(RuntimeError): Tensor.ones(shape).meshgrid()
  with pytest.raises(RuntimeError): Tensor.arange(3).meshgrid(Tensor.ones(shape))
