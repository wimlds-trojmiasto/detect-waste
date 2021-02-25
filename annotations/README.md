# Train and test with waste datasets

The basic steps are as below:

1. Prepare the customized mixed dataset: download all images of waste,
2. Prepare the customized annotations by runing:
    - `annotations_preprocessing_multi.sh` for mixed dataset,
    - or take provided here annotations.
3. Train, test, inference models on the chosen dataset.

## Annotations used in the Detect Waste in Pomerania project

Our annotations consist of:
- `annotations_train.json` and `annotations_test.json` for TACO dataset with 7 detect-waste categories object detection training,
- `annotations_binary_train.json` and `annotations_binary_test.json` for TACO dataset with 1 class object detection training,
- `binary_mixed_train.json` and `binary_mixed_train.json` for mixed dataset object detection training,
- `annotations_binary_mask_all_train.json` and `annotations_binary_mask_all_test.json` for mixed dataset instance segmentation training,
- and pseudoannotations (predictions from efficientdet) `openlittermap.json` for openlittermap dataset.

**Note**: Detect-waste only supports evaluating mAP of dataset in COCO format for now.
Users should convert the data into coco format.

### COCO annotation format

The necessary keys of COCO format is as below, for the complete details, please refer [here](https://cocodataset.org/#format-data).

```json
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```