import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    to highlight the exact facial regions responsible for FAKE or REAL predictions.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Auto-detect target convolutional layer
        if target_layer is None:
            if hasattr(model, 'spatial_features'):
                # DualStream model
                target_layer = model.spatial_features[-1]
            elif hasattr(model, 'features'):
                # Standard MobileNet or EfficientNet
                target_layer = model.features[-1]
            else:
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                        break
        self.target_layer = target_layer
        
        # Hook forward and backward passes
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=0):
        """
        input_tensor: (1, C, H, W) normalized tensor.
        target_class: 0 for FAKE, 1 for REAL.
        Returns: 2D numpy heatmap normalized to 0..1.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass on target score
        score = output[0, target_class]
        score.backward()

        # Compute channel weights via global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H, W)
        cam = F.relu(cam) # only positive features contributing to the class
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_heatmap(self, face_bgr, heatmap, alpha=0.5):
        """Overlays the Grad-CAM heatmap as a colored jet overlay onto face_bgr."""
        h, w = face_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(face_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

if __name__ == '__main__':
    from train_classifier import get_model
    m = get_model(2, arch='dual_fft', pretrained=False)
    cam = GradCAM(m)
    dummy = torch.randn(1, 4, 224, 224, requires_grad=True)
    heat = cam.generate_heatmap(dummy, target_class=0)
    print('Grad-CAM heatmap generated successfully, shape:', heat.shape)
