import cv2


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = int(w * r), height
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
