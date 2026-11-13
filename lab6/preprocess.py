import cv2


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    corrected_l = clahe.apply(l)

    corrected_lab = cv2.merge((corrected_l, a, b))
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

def prepare_frame(frame):
    prepared = frame.copy()
    prepared = apply_clahe(prepared)
    prepared = cv2.medianBlur(prepared, 5)
    return prepared
