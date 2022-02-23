import cv2
import numpy as np
import PIL.Image as Image
import albumentations as A


def visualize_bboxes(images, bboxes_list, color=(255, 0, 0), thickness=2):
    vis_images = []
    for img, bboxes in zip(images, bboxes_list):
        img = np.asarray(img)
        for bbox in bboxes:
            x1, y1, x2, y2 = [round(coord) for coord in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        vis_images.append(Image.fromarray(img))

    return vis_images


def records_to_arrays(records):
    bboxes_list = []
    images = []
    for record in records:
        bboxes = []
        images.append(record.img)
        for bbox in record.detection.bboxes:
            bboxes.append([round(i) for i in bbox.xyxy])
        bboxes_list.append(bboxes)

    return images, bboxes_list


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def transform_imgs(images, bboxes_list, t):
    transforms = A.Compose(
        t,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.7, label_fields=["class_labels"]
        ),
    )
    images_t = []
    bboxes_list_t = []
    for image, bboxes in zip(images, bboxes_list):
        class_labels = [1 for i in range(len(bboxes))]
        transformed = transforms(
            image=np.array(image), bboxes=bboxes, class_labels=class_labels, cropping_bbox=bboxes
        )
        images_t.append(Image.fromarray(transformed["image"]))
        bboxes_list_t.append(transformed["bboxes"])

    return {"images": images_t, "bboxes": bboxes_list_t}


def generate_transform_samples(records, transforms, name):
    images, bboxes = records_to_arrays(records)

    transformed = transform_imgs(images, bboxes, transforms)
    transformed_vis = visualize_bboxes(transformed["images"], transformed["bboxes"])

    grid = image_grid(transformed_vis, 3, 3)
    grid.save(name + ".png")

    return grid
