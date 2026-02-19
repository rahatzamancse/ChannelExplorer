"""
Package Name: ChannelExplorer

This package provides the ChannelExplorer module for TensorFlow and PyTorch.

Main Classes:
- channelexplorer_tf: channelexplorer module for TensorFlow.
- channelexplorer_torch: channelexplorer module for PyTorch.
"""

try:
    from .channelexplorer_tf import ChannelExplorer_TF
except ImportError:
    pass

try:
    from .channelexplorer_torch import APAnalysisTorchModel
except ImportError:
    pass
