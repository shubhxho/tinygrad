import hashlib
import io
from unittest.mock import patch
import pytest
from tinygrad.helpers import fetch

@pytest.mark.parametrize("failure", ["short", "sha", "read"])
@pytest.mark.parametrize("cached", [False, True])
def test_failed_download_does_not_replace_cache(tmp_path, failure, cached):
  target = tmp_path / "cached.bin"
  if cached: target.write_bytes(b"old contents")
  response = io.BytesIO(b"new")
  response.status, response.headers = 200, {"content-length": "10" if failure == "short" else "3"}
  if failure == "read": response.read = lambda *args: (_ for _ in ()).throw(OSError("connection lost"))
  with patch("urllib.request.urlopen", return_value=response), pytest.raises((RuntimeError, OSError)):
    fetch("https://example.test/file", name=target, allow_caching=False, sha256="bad" if failure == "sha" else None)
  assert sorted(p.name for p in tmp_path.iterdir()) == (["cached.bin"] if cached else [])
  if cached: assert target.read_bytes() == b"old contents"


def test_successful_download_replaces_cache(tmp_path):
  target = tmp_path / "cached.bin"
  target.write_bytes(b"old")
  response = io.BytesIO(b"new")
  response.status, response.headers = 206, {"content-length": "3"}
  with patch("urllib.request.urlopen", return_value=response):
    assert fetch("https://example.test/file", name=target, allow_caching=False, sha256=hashlib.sha256(b"new").hexdigest()) == target
  assert target.read_bytes() == b"new"
  assert list(tmp_path.iterdir()) == [target]
