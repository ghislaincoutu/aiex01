# TensorFlow -- Détection d'objets à partir d'une image

# Références :
# Tensorflow -- Object Detection
# https://www.tensorflow.org/hub/tutorials/object_detection
# Open Images Dataset V7 and Extensions
# https://storage.googleapis.com/openimages/web/index.html

import os
import time
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageColor
from PIL import Image
from six import BytesIO
from six.moves.urllib.request import urlopen
os.system("clear")

def pause():
    programPause = input("Appuyez sur la touche Retour pour continuer...")

print("TensorFlow -- Détection d’objets à partir d’une image")
print(f"Version TensorFlow : {tf.__version__}")
pause()

print("\nDéfinition des fonctions pour importer et traiter les images")
pause()

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.show()

def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename

def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()
):
    # Adds a bounding box to an image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        bbox = font.getbbox(display_str)
        text_width, text_height = bbox[2], bbox[3]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    # Overlay labeled boxes on an image with formatted scores and label names.
    colors = list(ImageColor.colormap.values())
    try:
        font = ImageFont.truetype("/var/aiex01/fonts/LiberationSans-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()
    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"), int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
            np.copyto(image, np.array(image_pil))
    return image

print("\nTéléchargemet de l’image à traiter")
pause()

def download_image(image33):
    match image33:
        case "01":
            url33 = "https://farm6.staticflickr.com/5314/5887463535_a88f862a81_o.jpg"
            return url33
        case "02":
            url33 = "https://farm6.staticflickr.com/3694/19007218479_03f8493049_o.jpg"
            return url33
        case "03":
            url33 = "http://localhost/dev01/medias/image001.jpg"
            return url33
        case _:
            print(f"Sélection incorrecte: {image33}")

image_url = download_image("02")
downloaded_image_path = download_and_resize_image(image_url, 1280, 800, True)

print("\nActivation des modules de détection d’objets")
pause()

module_handle = ("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
detector = hub.load(module_handle).signatures["default"]

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, path):
    img = load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[
        tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}
    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)
    image_with_boxes = draw_boxes(
        img.numpy(),
        result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"],
    )
    display_image(image_with_boxes)

print("\nExécution de la détection d’objets")
pause()
run_detector(detector, downloaded_image_path)
