import numpy as np
import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("shape,target", [((2,), (-1, -1)), ((), (-1,)), ((2,), (None, 2)), ((1, 2), (2,)), ((1,), ())])
def test_expand_invalid_dimensions(shape, target):
  with pytest.raises(ValueError): Tensor.ones(shape).expand(target)


def test_expand_existing_inferred_dimensions():
  a = np.arange(3).reshape(1, 3)
  np.testing.assert_array_equal(Tensor(a).expand(2, -1).numpy(), np.broadcast_to(a, (2, 3)))
  np.testing.assert_array_equal(Tensor(a).expand(2, None).numpy(), np.broadcast_to(a, (2, 3)))
  assert Tensor(a).expand(4, 2, -1).shape == (4, 2, 3)
