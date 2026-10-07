import numpy as np
import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("shape,repeats", [((1,), (-1,)), ((2, 1), (1, -1)), ((0, 2), (3,)), ((1, 2), (2,))])
def test_repeat_invalid_counts(shape, repeats):
  with pytest.raises(ValueError): Tensor.ones(shape).repeat(repeats)

@pytest.mark.parametrize("repeats", [(0, 2), (2, 3), (2, 1, 3)])
def test_repeat_valid_counts(repeats):
  a = np.arange(6).reshape(2, 3)
  np.testing.assert_array_equal(Tensor(a).repeat(repeats).numpy(), np.tile(a, repeats))
