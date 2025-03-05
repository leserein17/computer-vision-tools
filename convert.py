import os
import json
import base64
import numpy as np
import zlib
from pycocotools import mask as mask_utils
import cv2

def numpy_serializer(obj):
    if isinstance(obj, (np.integer, np.int32, np.uint32, np.int64, np.uint64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable：{type(obj)}")

def mask2rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
    area = mask_utils.area(rle)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle, area

def create_category_mapper(categories):
    return {item['name']: item['id'] for item in categories}

def base64_2_mask(s, h, w, col, row, name):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    lrow, lcol = mask.shape
    mask_full = np.zeros((h, w), dtype=bool)
    mask_full[row: row+lrow, col: col+lcol] = mask
    return mask_full, lcol, lrow

def count_categories(ann_dir: str):
    categories = []
    seen = set()
    cid = 1

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'):
            continue

        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            data = json.load(f)
            cname = data["objects"][0]["classTitle"].replace("branch_of_", "")
            if cname not in seen:
                categories.append({
                    "id": cid,
                    "name": cname
                })
                seen.add(cname)
                cid += 1
    return categories


def supervisely2coco(ann_dir: str, output_path: str):
    coco = {
        "info": {
            "date_created": "2025-03-05",
            "description": ""
        },
        "licenses": None,
        "images": [],
        "annotations": [],
        "categories": []
    }

    images = []
    annotations = []

    categories = count_categories(ann_dir)
    coco["categories"] = categories
    category_mapper = create_category_mapper(categories)

    oid = 0
    for i, ann_file in enumerate(os.listdir(ann_dir)):
        if not ann_file.endswith('.json'):
            continue

        image = {
            "id": None,
            "width": None,
            "height": None,
            "file_name": None
        }

        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            data = json.load(f)
            # images
            id = i+1
            size = data["size"]

            image["id"] = id
            image["width"] = size["width"]
            image["height"] = size["height"]
            image["file_name"] = os.path.splitext(ann_file)[0]
            images.append(image)

            # annotations
            for ann in data["objects"]:
                annotation = {
                    "id": None,
                    "image_id": None,
                    "bbox": None,
                    "area": None,
                    "iscrowd": None,
                    "category_id": None,
                    "segmentation": None
                }

                oid += 1
                annotation["id"] = oid
                annotation["image_id"] = id
                bitmap = ann["bitmap"]["data"]
                origin = ann["bitmap"]["origin"]
                mask, w, h = base64_2_mask(bitmap, size["height"], size["width"], origin[0], origin[1], os.path.splitext(ann_file)[0])
                rle, area = mask2rle(mask)
                annotation["bbox"] = [origin[0], origin[1], w, h]
                annotation["area"] = area
                annotation["iscrowd"] = 0
                category = ann["classTitle"].replace("branch_of_", "")
                annotation["category_id"] = category_mapper[category]
                annotation["segmentation"] = rle
                annotations.append(annotation)
        print(f"{ann_file} has converted.")

    coco["images"] = images
    coco["annotations"] = annotations

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                coco,
                f,
                indent=4,
                ensure_ascii=False,
                default=numpy_serializer
            )


if __name__ == "__main__":
    ann_dir = "./urban-street/val/ann/"
    output_path = "./urban-street/annotations/instance_val.json"
    supervisely2coco(ann_dir, output_path)
