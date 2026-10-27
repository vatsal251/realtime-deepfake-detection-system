import os
import argparse
import torch
import torch.nn as nn
from torchvision import models

from train_classifier import get_model

def create_mockup_checkpoint(checkpoint_path, arch="dual_fft"):
    """Creates a dummy checkpoint with random weights so exporting can be verified without training first."""
    print(f"Checkpoint not found. Creating a mockup checkpoint at {checkpoint_path}...")
    model = get_model(num_classes=2, arch=arch, pretrained=False)
    torch.save({
        'epoch': 0,
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'classes': ['fake', 'real'],
        'class_to_idx': {'fake': 0, 'real': 1},
        'accuracy': 0.5
    }, checkpoint_path)
    print("Mockup checkpoint created successfully.")

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Deepfake model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to PyTorch model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="mobilenetv3_deepfake.onnx", help="Path to save the exported ONNX model")
    
    args = parser.parse_args()
    
    # If no checkpoint exists, create a mockup weight file
    if not os.path.exists(args.checkpoint):
        create_mockup_checkpoint(args.checkpoint)
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # Extract structural details
    classes = checkpoint.get("classes", ["fake", "real"])
    num_classes = len(classes)
    
    # Detect architecture
    state_dict = checkpoint.get('model_state_dict', {})
    if checkpoint.get("arch") == "temporal_cnn_lstm" or "lstm.weight_ih_l0" in state_dict:
        from temporal_model import get_temporal_model
        arch = "temporal_cnn_lstm"
        print("Architecture detected: TEMPORAL_CNN_LSTM (Sequence of frames)")
        model = get_temporal_model(num_classes=num_classes, pretrained_spatial=False)
        model.load_state_dict(state_dict)
        model.eval()
        dummy_input = torch.randn(1, 8, 3, 224, 224, requires_grad=False)
        dynamic_axes = {"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}}
    else:
        saved_arch = checkpoint.get("arch", None)
        has_fft = "freq_branch.conv.0.weight" in state_dict
        has_eff = "spatial_features.7.2.block.3.0.weight" in state_dict or "features.7.2.block.3.0.weight" in state_dict

        if saved_arch == "efficientnet_dual" or (has_eff and has_fft):
            arch = "efficientnet_dual"
            in_channels = 4
        elif saved_arch == "efficientnet" or (has_eff and not has_fft):
            arch = "efficientnet"
            in_channels = 3
        elif saved_arch == "dual_fft" or has_fft:
            arch = "dual_fft"
            in_channels = 4
        else:
            arch = "mobilenet"
            in_channels = 3
        print(f"Architecture detected: {arch.upper()} (input channels: {in_channels})")

        model = get_model(num_classes=num_classes, arch=arch, pretrained=False)
        model.load_state_dict(state_dict)
        model.eval()
        dummy_input = torch.randn(1, in_channels, 224, 224, requires_grad=False)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    print(f"Model trained for classes: {classes} (Total classes: {num_classes})")
    input_names = ["input"]
    output_names = ["output"]


    
    # Optional: Dynamic batching configuration
    # This allows classifying variable batch sizes (e.g. 1 face, 2 faces, or N faces in standard forward pass)
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
    
    print(f"Exporting model to ONNX format at {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=14, # Standard stable opset version
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Export complete. Model saved in: {args.output}")
    print("Verification: Let's checks the ONNX model structure...")
    
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX model checker passed. Model structure is valid!")
    except ImportError:
        print("Install the 'onnx' library to verify the generated export format.")
    except Exception as e:
        print(f"Validation failed with error: {e}")

if __name__ == "__main__":
    main()
