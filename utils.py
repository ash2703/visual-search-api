import numpy as np
import cv2
import pathlib
import os
from typing import List
from PIL import Image


def concat_tile_resize(list_2d,
                       interpolation=cv2.INTER_CUBIC):
      # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h,
                                 interpolation=cv2.INTER_CUBIC)
                  for list_h in list_2d]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)


def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
      # take minimum width
    w_min = min(img.shape[1]
                for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)


def hconcat_resize(img_list,
                   interpolation=cv2.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation=interpolation)
                      for img in img_list]

    # return final image
    return cv2.hconcat(im_list_resize)


def stack_images_side_by_side(input_img: pathlib.Path ,img_paths: List[pathlib.Path], out_path: pathlib.Path = pathlib.Path("./static/stacked")):
    query_image = cv2.imread(str(input_img), cv2.IMREAD_UNCHANGED)
    img_stack = [cv2.cvtColor(query_image, cv2.COLOR_RGB2RGBA)]
    for image_path in img_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:  # If RGB, add alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            
        img_stack.append(img)
    tile = concat_tile_resize(
        [img_stack]
    )
    stacked_image_path = out_path / input_img.name
    os.makedirs(stacked_image_path.parent, exist_ok = True)
    cv2.imwrite(str(stacked_image_path), tile)
    return tile

def img_text_write(in_file, text):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw

    img = Image.open(in_file)
    draw = ImageDraw.Draw(img)
    draw.text((1, 1), text, (0, 0, 0))
    draw.text((0, 0), text, (255, 255, 255))
