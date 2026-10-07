import subprocess
import sys
import pytest
from tinygrad.codegen.opt.search import get_test_global_size
from tinygrad.helpers import prod

@pytest.mark.parametrize("shape,limit", [([16, 16, 16], 64), ([2, 3], 1), ([32, 32, 32], 4096)])
def test_beam_size_reduction_terminates(shape, limit):
  # A subprocess bounds the regression: the old loop never returns for small axes.
  code = f"from tinygrad.codegen.opt.search import get_test_global_size; print(get_test_global_size({shape!r}, {limit}, {{}}))"
  subprocess.run([sys.executable, "-c", code], check=True, timeout=10, capture_output=True)
  sizes, factor = get_test_global_size(shape, limit, {})
  assert 0 < prod(sizes) <= limit
  assert prod(sizes) * factor == prod(shape)

@pytest.mark.parametrize("limit", [0, -1])
def test_beam_size_rejects_invalid_limit(limit):
  code = f"from tinygrad.codegen.opt.search import get_test_global_size; get_test_global_size([1], {limit}, {{}})"
  result = subprocess.run([sys.executable, "-c", code], timeout=10, capture_output=True)
  assert result.returncode != 0 and b"ValueError" in result.stderr


def test_beam_empty_size():
  assert get_test_global_size([0, 4], 1, {}) == ([0, 4], 1.0)
