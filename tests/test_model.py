import pytest
import torch
import sys
import inspect
sys.path.append('src')
from model import Net

@pytest.fixture
def model():
    return Net()

def test_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, which exceeds the limit of 20,000"

def test_dropout_layers(model):
    dropout_layers = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert len(dropout_layers) > 0, "Model should contain dropout layers"

def test_batch_normalization(model):
    bn_layers = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert len(bn_layers) > 0, "Model should contain batch normalization layers"

def test_gap_no_linear(model):
    # Check for absence of fully connected (Linear) layers
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert len(linear_layers) == 0, "Model should not contain fully connected layers"
    
    # Check for presence of adaptive average pooling
    def contains_adaptive_avg_pool(model):
        source = inspect.getsource(model.forward)
        return 'adaptive_avg_pool' in source
    
    assert contains_adaptive_avg_pool(model), "Model should use Global Average Pooling (GAP)"

if __name__ == "__main__":
    pytest.main([__file__]) 