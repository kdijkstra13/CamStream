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

ALIGNMENT_LEFT = 0
ALIGNMENT_CENTER = 1
ALIGNMENT_RIGHT = 2
ALIGNMENT_TOP = 3
ALIGNMENT_BOTTOM = 4

def fwh(font, text):
    l, t, r, b = font.getbbox(text)
    return r - l, b - t

def text_box(text, image_draw, font, box, horizontal_alignment=ALIGNMENT_LEFT, vertical_alignment=ALIGNMENT_TOP, **kwargs):
    x = box[0]
    y = box[1]
    width = box[2]
    height = box[3]
    lines = text.split('\n')
    true_lines = []
    for line in lines:
        if fwh(font, line)[0] <= width:
            true_lines.append(line)
        else:
            current_line = ''
            for word in line.split(' '):
                if fwh(font, current_line + word)[0] <= width:
                    current_line += ' ' + word
                else:
                    true_lines.append(current_line)
                    current_line = word
            true_lines.append(current_line)

    x_offset = y_offset = 0
    lineheight = fwh(font, true_lines[0])[1] * 1.2  # Give a margin of 0.2x the font height
    if vertical_alignment == ALIGNMENT_CENTER:
        y = int(y + height / 2)
        y_offset = - (len(true_lines) * lineheight) / 2
    elif vertical_alignment == ALIGNMENT_BOTTOM:
        y = int(y + height)
        y_offset = - (len(true_lines) * lineheight)

    for line in true_lines:
        linewidth = fwh(font, line)[0]
        if horizontal_alignment == ALIGNMENT_CENTER:
            x_offset = (width - linewidth) / 2
        elif horizontal_alignment == ALIGNMENT_RIGHT:
            x_offset = width - linewidth
        image_draw.text((int(x + x_offset), int(y + y_offset)), line, font=font, **kwargs)
        y_offset += lineheight
