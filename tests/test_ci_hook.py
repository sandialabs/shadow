import subprocess


def test_ci_unittest():
    """No-op test to trigger CI pipeline."""
    assert True


def test_cuda_available():
    """No dependency test if CUDA is available on system."""
    assert subprocess.Popen('nvidia-smi')
