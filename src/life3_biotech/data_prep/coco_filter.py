import json
from pathlib import Path
from pconst import const


class CocoFilter:
    """
    This class filters the COCO dataset

    Attributes:
    logger: Logger object used to log events to.
    """

    def __init__(self, logger):
        self.logger = logger

    def _process_info(self):
        self.info = self.coco["info"]

    def _process_licenses(self):
        self.licenses = self.coco["licenses"]

    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.coco["categories"]:
            cat_id = category["id"]
            super_category = category["supercategory"]

            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category["name"])
            else:
                self.logger.error(f"ERROR: Skipping duplicate category id: {category}")

            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {
                    cat_id
                }  # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):

        self.images = dict()
        self.remove_images_id = []
        for image in self.coco["images"]:
            image_id = image["id"]
            if image_id not in self.images:
                self.images[image_id] = image
                if self.images[image_id]["file_name"] in const.EXCLUDED_IMAGES:
                    self.remove_images_id.append(image_id)
                    self.logger.info(
                        f"Image id to be removed: {image_id} - Filename: {self.images[image_id]['file_name']}"
                    )
            else:
                self.logger.error(f"ERROR: Skipping duplicate image id: {image}")

    def _process_segmentations(self):
        """
        Read and create dictionary in following format {image_id: list of corresponding annotations dict}
        Images without annotations are not recorded.
        """
        self.segmentations = dict()

        for segmentation in self.coco["annotations"]:
            image_id = segmentation["image_id"]
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _filter_categories(self):
        """Find category ids matching args
        Create mapping from original category id to new category id
        Create new collection of categories
        """
        missing_categories = set(const.TILE_COCO_FILTER_CATEGORIES) - self.category_set
        if len(missing_categories) > 0:
            self.logger.error(f"Did not find categories: {missing_categories}")

        self.new_category_map = dict()
        new_id = 1
        for key, item in self.categories.items():
            if item["name"] in const.TILE_COCO_FILTER_CATEGORIES:
                self.new_category_map[key] = new_id
                new_id += 1

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category["id"] = new_id
            self.new_categories.append(new_category)

    def _filter_annotations(self):
        """Create new collection of annotations matching category ids
        Keep track of image ids matching annotations
        """
        self.new_segmentations = []
        self.new_image_ids = set()
        for image_id, segmentation_list in self.segmentations.items():
            if image_id not in self.remove_images_id:
                for segmentation in segmentation_list:
                    original_seg_cat = segmentation["category_id"]
                    if original_seg_cat in self.new_category_map.keys():
                        new_segmentation = dict(segmentation)
                        new_segmentation["category_id"] = self.new_category_map[
                            original_seg_cat
                        ]
                        self.new_segmentations.append(new_segmentation)
                        self.new_image_ids.add(image_id)

    def _filter_images(self):
        """Create new collection of images"""
        self.new_images = []
        for image_id in self.new_image_ids:
            if image_id not in self.remove_images_id:
                self.new_images.append(self.images[image_id])

    def filter_coco(self, input_json_path, output_json_path):
        """
        Based on https://github.com/immersive-limit/coco-manager
        Remove any extra categories
        Give the categories new ids (counting up from 1)
        Find any annotations that reference the desired categories (class_map) - configurable in yml file (pipelines.yml)
        Filter out extra annotations
        Filter out images not referenced by any annotations
        Filter out images by excluding setting (excluded_images) in configurable in yml file (pipelines.yml)
        Save a new json file
        """

        # Verify input path exists
        if not input_json_path.exists():
            self.logger.error("Coco annotation json path not found.")

        # Load the json
        with open(input_json_path) as json_file:
            self.coco = json.load(json_file)

        # Process the json
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        # Filter to specific categories
        self._filter_categories()
        self._filter_annotations()
        self._filter_images()

        # Build new JSON
        new_master_json = {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.new_images,
            "annotations": self.new_segmentations,
            "categories": self.new_categories,
        }

        # Write the JSON to a file
        with open(output_json_path, "w+") as output_file:
            json.dump(new_master_json, output_file)

        self.logger.info("Filtered coco annotations json saved.")
