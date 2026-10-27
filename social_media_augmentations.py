import random
import numpy as np
import cv2
from PIL import Image, ImageFilter

class MotionBlur:
    """Simulates camera shake or rapid facial motion blur."""
    def __init__(self, p=0.4, max_kernel_size=9):
        self.p = p
        self.max_kernel_size = max_kernel_size

    def __call__(self, img_pil):
        if random.random() < self.p:
            k_size = random.choice(list(range(3, self.max_kernel_size + 1, 2)))
            angle = random.uniform(0, 180)
            
            # Create directional motion blur kernel
            M = cv2.getRotationMatrix2D((k_size / 2, k_size / 2), angle, 1)
            kernel = np.zeros((k_size, k_size), dtype=np.float32)
            kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
            kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
            kernel /= np.sum(kernel) + 1e-8

            img_np = np.array(img_pil)
            blurred = cv2.filter2D(img_np, -1, kernel)
            return Image.fromarray(blurred)
        return img_pil

class VideoCodecCompression:
    """Simulates blocky H.264 macroblock compression artifacts common on WhatsApp/Telegram."""
    def __init__(self, p=0.4, block_size_range=(4, 12)):
        self.p = p
        self.block_size_range = block_size_range

    def __call__(self, img_pil):
        if random.random() < self.p:
            img_np = np.array(img_pil)
            h, w = img_np.shape[:2]
            block = random.randint(*self.block_size_range)
            
            # Downsample to simulate macroblocks
            small_w, small_h = max(4, w // block), max(4, h // block)
            small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Blend 50% blocky with original to simulate partial P/B-frame artifacting
            blended = cv2.addWeighted(img_np, 0.5, blocky, 0.5, 0)
            return Image.fromarray(blended)
        return img_pil

if __name__ == '__main__':
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    mb = MotionBlur(p=1.0)
    vc = VideoCodecCompression(p=1.0)
    out1 = mb(dummy)
    out2 = vc(out1)
    print('Social Media Augmentations verified successfully!')
