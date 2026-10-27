import numpy as np
import cv2
import random
from PIL import Image

class SelfBlendedImageGenerator:
    """
    Self-Blended Image (SBI) Synthesizer (inspired by CVPR 2022 SBI paper).
    Takes a clean real face image and generates a realistic synthetic deepfake face
    with authentic blending boundary artifacts, color shifts, and resolution inconsistencies.
    """
    def __init__(self, p=0.5):
        self.p = p

    def generate_random_mask(self, h, w):
        """Generates a random face-like convex/elliptical polygon mask."""
        center_x = int(w * random.uniform(0.42, 0.58))
        center_y = int(h * random.uniform(0.45, 0.60))
        radius_x = int(w * random.uniform(0.25, 0.42))
        radius_y = int(h * random.uniform(0.28, 0.45))
        angle = random.uniform(-15, 15)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), angle, 0, 360, 255, -1)

        pts = []
        num_pts = random.randint(8, 14)
        for i in range(num_pts):
            theta = 2 * np.pi * i / num_pts
            r_scale = random.uniform(0.85, 1.15)
            px = int(center_x + radius_x * r_scale * np.cos(theta))
            py = int(center_y + radius_y * r_scale * np.sin(theta))
            pts.append([px, py])
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.array(pts, dtype=np.int32)], 255)
        
        combined_mask = cv2.bitwise_and(mask, poly_mask)
        blur_k = random.choice([3, 5, 7, 9, 11])
        feathered_mask = cv2.GaussianBlur(combined_mask, (blur_k, blur_k), 0)
        return feathered_mask.astype(np.float32) / 255.0

    def transform_source_patch(self, img):
        """Applies subtle affine transform and color mismatch to the source face."""
        h, w = img.shape[:2]
        
        # 1. Subtle affine warp
        scale = random.uniform(0.94, 1.06)
        rot = random.uniform(-4, 4)
        M = cv2.getRotationMatrix2D((w/2, h/2), rot, scale)
        dx = random.uniform(-w*0.03, w*0.03)
        dy = random.uniform(-h*0.03, h*0.03)
        M[0, 2] += dx
        M[1, 2] += dy
        warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 2. Color / contrast / gamma mismatch
        warped = warped.astype(np.float32)
        gamma = random.uniform(0.88, 1.14)
        warped = np.clip(np.power(warped / 255.0, gamma) * 255.0, 0, 255)

        alpha = random.uniform(0.90, 1.12)
        beta = random.uniform(-10, 10)
        warped = np.clip(alpha * warped + beta, 0, 255).astype(np.uint8)

        # 3. Resolution mismatch
        if random.random() < 0.6:
            down_scale = random.uniform(0.5, 0.85)
            small = cv2.resize(warped, (int(w * down_scale), int(h * down_scale)), interpolation=cv2.INTER_LINEAR)
            warped = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Subtle blur
        if random.random() < 0.5:
            k = random.choice([3, 5])
            warped = cv2.GaussianBlur(warped, (k, k), 0)

        return warped

    def blend(self, target_img):
        """Blends a transformed source patch onto the target face."""
        h, w = target_img.shape[:2]
        source_img = self.transform_source_patch(target_img)
        mask = self.generate_random_mask(h, w)[:, :, np.newaxis]
        fake_img = (mask * source_img + (1.0 - mask) * target_img).astype(np.uint8)
        return fake_img

    def __call__(self, img_pil):
        """Callable transform: takes PIL Image and returns (synthesized_PIL, is_fake)."""
        if random.random() < self.p:
            np_img = np.array(img_pil)
            blended = self.blend(np_img)
            return Image.fromarray(blended), True
        return img_pil, False

if __name__ == "__main__":
    gen = SelfBlendedImageGenerator(p=1.0)
    dummy = np.random.randint(50, 220, (224, 224, 3), dtype=np.uint8)
    fake = gen.blend(dummy)
    print("SBI Generator test successful! Output shape:", fake.shape)
