# tests/test_environment.py — Pytest suite to verify that the development environment
# is correctly configured and all required dependencies are importable.

import pytest
import sys


class TestPythonEnvironment:
    """Verify the Python runtime is functional."""

    def test_python_version(self):
        """Python 3.x is running."""
        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 10, "Python 3.10+ is recommended"


class TestCoreImports:
    """Verify every required package can be imported."""

    def test_import_torch(self):
        import torch
        assert torch.__version__

    def test_import_torchvision(self):
        import torchvision
        assert torchvision.__version__

    def test_import_torchaudio(self):
        import torchaudio
        assert torchaudio.__version__

    def test_import_librosa(self):
        import librosa
        assert librosa.__version__

    def test_import_numpy(self):
        import numpy
        assert numpy.__version__

    def test_import_pandas(self):
        import pandas
        assert pandas.__version__

    def test_import_sklearn(self):
        import sklearn
        assert sklearn.__version__

    def test_import_matplotlib(self):
        import matplotlib
        assert matplotlib.__version__

    def test_import_soundfile(self):
        import soundfile
        assert soundfile.__version__

    def test_import_streamlit(self):
        import streamlit
        assert streamlit.__version__

    def test_import_tqdm(self):
        import tqdm
        assert tqdm.__version__

    def test_import_pytest(self):
        assert pytest.__version__


class TestPyTorchDetails:
    """Verify PyTorch version and CUDA detection."""

    def test_pytorch_version_available(self):
        import torch
        version = torch.__version__
        assert isinstance(version, str)
        assert len(version) > 0

    def test_cuda_availability_detected(self):
        """CUDA availability can be queried (pass regardless of result)."""
        import torch
        cuda_available = torch.cuda.is_available()
        assert isinstance(cuda_available, bool)

    def test_torch_tensor_creation(self):
        """Basic tensor operations work."""
        import torch
        t = torch.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.sum().item() == 6.0
