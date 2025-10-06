#!/usr/bin/env python3
"""
Model and Dependency Checker for KD Pruning Simulator
This script checks if all required models and dependencies are available.
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_package(package_name: str) -> Tuple[bool, str]:
    """Check if a package is available and return version info."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown version')
        return True, version
    except ImportError:
        return False, "Not installed"

def check_torch_models() -> Dict[str, bool]:
    """Check if PyTorch models can be imported."""
    models_status = {}
    
    # Check torchvision models
    try:
        import torchvision.models
        models_status['torchvision'] = True
        print("✓ torchvision is available")
        
        # Test specific model imports
        try:
            from torchvision.models import resnet18
            models_status['resnet18'] = True
            print("✓ resnet18 model can be imported")
        except ImportError as e:
            models_status['resnet18'] = False
            print(f"✗ resnet18 model import failed: {e}")
            
        try:
            from torchvision.models import mobilenet_v2
            models_status['mobilenet_v2'] = True
            print("✓ mobilenet_v2 model can be imported")
        except ImportError as e:
            models_status['mobilenet_v2'] = False
            print(f"✗ mobilenet_v2 model import failed: {e}")
            
    except ImportError:
        models_status['torchvision'] = False
        print("✗ torchvision is not available")
        models_status['resnet18'] = False
        models_status['mobilenet_v2'] = False
    
    return models_status

def check_transformers_models() -> Dict[str, bool]:
    """Check if transformers models can be imported."""
    models_status = {}
    
    try:
        import transformers
        models_status['transformers'] = True
        print("✓ transformers is available")
        
        # Test specific model imports
        try:
            from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer
            models_status['distilbert'] = True
            print("✓ DistilBert models can be imported")
        except ImportError as e:
            models_status['distilbert'] = False
            print(f"✗ DistilBert models import failed: {e}")
            
        try:
            from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
            models_status['t5'] = True
            print("✓ T5 models can be imported")
        except ImportError as e:
            models_status['t5'] = False
            print(f"✗ T5 models import failed: {e}")
            
    except ImportError:
        models_status['transformers'] = False
        print("✗ transformers is not available")
        models_status['distilbert'] = False
        models_status['t5'] = False
    
    return models_status

def test_model_loading() -> Dict[str, bool]:
    """Test if models can actually be loaded."""
    loading_status = {}
    
    # Test ResNet18
    try:
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)  # Use pretrained=False to avoid downloading
        loading_status['resnet18'] = True
        print("✓ ResNet18 model can be instantiated")
    except Exception as e:
        loading_status['resnet18'] = False
        print(f"✗ ResNet18 model instantiation failed: {e}")
    
    # Test MobileNetV2
    try:
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(width_mult=0.5)
        loading_status['mobilenet_v2'] = True
        print("✓ MobileNetV2 model can be instantiated")
    except Exception as e:
        loading_status['mobilenet_v2'] = False
        print(f"✗ MobileNetV2 model instantiation failed: {e}")
    
    # Test DistilBert
    try:
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        config = DistilBertConfig(dim=256, n_layers=3)
        model = DistilBertForSequenceClassification(config)
        loading_status['distilbert'] = True
        print("✓ DistilBert model can be instantiated")
    except Exception as e:
        loading_status['distilbert'] = False
        print(f"✗ DistilBert model instantiation failed: {e}")
    
    # Test T5
    try:
        from transformers import T5Config, T5ForConditionalGeneration
        config = T5Config(d_model=256, num_layers=6, d_ff=1024, num_heads=4, vocab_size=32128)
        model = T5ForConditionalGeneration(config)
        loading_status['t5'] = True
        print("✓ T5 model can be instantiated")
    except Exception as e:
        loading_status['t5'] = False
        print(f"✗ T5 model instantiation failed: {e}")
    
    return loading_status

def main():
    """Main function to run all checks."""
    print("=" * 60)
    print("KD PRUNING SIMULATOR - MODEL AND DEPENDENCY CHECKER")
    print("=" * 60)
    
    # Check core packages
    print("\n1. CORE PACKAGES:")
    print("-" * 30)
    
    packages = ['torch', 'torchvision', 'transformers', 'flask', 'flask_cors']
    package_status = {}
    
    for package in packages:
        available, version = check_package(package)
        package_status[package] = available
        status = "✓" if available else "✗"
        print(f"{status} {package}: {version}")
    
    # Check model imports
    print("\n2. MODEL IMPORTS:")
    print("-" * 30)
    torch_models = check_torch_models()
    transformer_models = check_transformers_models()
    
    # Test model loading
    print("\n3. MODEL INSTANTIATION:")
    print("-" * 30)
    loading_status = test_model_loading()
    
    # Summary
    print("\n4. SUMMARY:")
    print("-" * 30)
    
    all_packages_ok = all(package_status.values())
    all_models_ok = all(torch_models.values()) and all(transformer_models.values())
    all_loading_ok = all(loading_status.values())
    
    print(f"Core packages: {'✓ All available' if all_packages_ok else '✗ Some missing'}")
    print(f"Model imports: {'✓ All available' if all_models_ok else '✗ Some missing'}")
    print(f"Model loading: {'✓ All working' if all_loading_ok else '✗ Some failed'}")
    
    if not all_packages_ok or not all_models_ok or not all_loading_ok:
        print("\n5. INSTALLATION COMMANDS:")
        print("-" * 30)
        if not package_status.get('torchvision', False):
            print("pip install torchvision")
        if not package_status.get('transformers', False):
            print("pip install transformers")
        if not package_status.get('flask', False):
            print("pip install flask")
        if not package_status.get('flask_cors', False):
            print("pip install flask-cors")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 