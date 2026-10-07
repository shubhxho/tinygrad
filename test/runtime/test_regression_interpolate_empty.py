import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("mode", ["linear", "nearest", "nearest-exact"])
@pytest.mark.parametrize("shape,size", [((0,), (2,)), ((2, 0), (3,)), ((2,), (0,)), ((2, 3), (0, 4)), ((2,), (-1,))])
def test_interpolate_empty_spatial_dimensions(mode, shape, size):
  with pytest.raises(AssertionError, match="spatial sizes must be positive"): Tensor.empty(shape).interpolate(size, mode=mode)

@pytest.mark.parametrize("mode", ["linear", "nearest", "nearest-exact"])
def test_interpolate_empty_batch(mode):
  out = Tensor.empty(0, 3).interpolate((5,), mode=mode)
  assert out.shape == (0, 5)
  assert out.numpy().size == 0
