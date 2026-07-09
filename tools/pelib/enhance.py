"""pelib.enhance - shadow-lift + CLAHE + unsharp, for the DETECTOR only (mask maps to the original)."""
import numpy as np
import cv2


def enhance_for_detection(bgr, clahe=3.0, gamma=0.55, sharp=0.6, sat=1.25):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    out = np.power(np.clip(out, 0, 1), gamma)
    if sat != 1.0:
        hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    if sharp > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = np.clip(out * (1 + sharp) - blur * sharp, 0, 1)
    return (out * 255).astype(np.uint8)
