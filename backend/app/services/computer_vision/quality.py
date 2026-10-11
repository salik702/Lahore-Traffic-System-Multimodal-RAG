"""Pixel measurements, not probabilities of compliance."""
import base64
import cv2


def assess_quality(image, min_side=128, min_sharpness=25):
    if image is None or not image.size:
        return {"sufficient": False, "sharpness": 0, "reasons": ["Empty image"]}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    reasons = []
    if min(image.shape[:2]) < min_side:
        reasons.append("Image region is too small")
    if sharpness < min_sharpness:
        reasons.append("Image region is blurred or lacks visible detail")
    if brightness < 15 or brightness > 245:
        reasons.append("Image region is severely underexposed or overexposed")
    return {"sufficient": not reasons, "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2), "reasons": reasons}


def crop_region(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    return image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]


def clipped(bbox, shape, margin=2):
    h, w = shape[:2]
    return bbox[0] <= margin or bbox[1] <= margin or bbox[2] >= w-margin or bbox[3] >= h-margin


def image_url(image, max_side=None):
    if max_side and max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Image encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
