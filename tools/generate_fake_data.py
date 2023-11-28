#!/usr/bin/env python3
# GENERATE FAKE DATA.py
#   by Lut99
#
# Created:
#   28 Nov 2023, 13:30:54
# Last edited:
#   28 Nov 2023, 14:18:10
# Auto updated?
#   Yes
#
# Description:
#   Attempt to generate some fake data that is compatible with ImageNet but
#   not so insanely large (for testing purposes).
#

import argparse
import os
import random
import string
import sys
import typing

from torchvision import datasets, transforms


##### CONSTANTS #####
# The format of images we save.
FORMAT = "JPEG"
# The extension of images to save.
EXTENSION = "jpeg"





##### HELPER FUNCTIONS #####
def assert_dir(path: str, fix_dirs: bool) -> int:
    """
        Asserts the given directory exists.

        If `fix_dirs` is True, then it will attempt to create it if missing instead of errorring.

        Returns an exit code (0 is good, anything else is bad).
    """

    # Check if it exists
    if not os.path.exists(path):
        # It doesn't; fix it _or_ complain
        if fix_dirs:
            print(f" - Creating missing directory '{path}'...")
            try:
                os.makedirs(path)
            except IOError as e:
                print(f"ERROR: Failed to create directory '{path}': {e}", file=sys.stderr)
                return e.errno
        else:
            print(f"ERROR: Directory '{path}' does not exist", file=sys.stderr)
            return 1

    # Folder exists
    return 0





##### ENTRYPOINT #####
def main(output_dir: str, fix_dirs: bool, n_train_images: int, n_val_images: int, size: typing.Tuple[int, int], num_classes: int) -> int:
    """
        Main function of the script.

        # Arguments
        - `output_dir`: The directory to write to.
        - `fix_dirs`: Wether to create missing directories (True) or error if any occur (False).
        - `n_train_images`: The number of images to generate for training.
        - `n_val_images`: The number of images to generate for validation.
        - `size`: The size of the generated images (as a WxH tuple).
        - `num_classes`: The number of classes to generate for.

        # Returns
        An exit code for the script. `0` means OK, anything else means bad.
    """

    # Assert the output directory exists and open it as an ImageFolder
    print(f"Preparing output environment in '{output_dir}'...")
    assert_dir(output_dir, fix_dirs=fix_dirs)

    # Create test and training dirs and their classes
    for mode in ["train", "val"]:
        # Assert the mode exists
        mode_dir = os.path.join(output_dir, mode)
        assert_dir(mode_dir, fix_dirs=fix_dirs)

        # Generate the classes
        for clss in range(num_classes):
            clss_dir = os.path.join(mode_dir, str(clss))
            assert_dir(clss_dir, fix_dirs=fix_dirs)



    # Attempt to write a fake dataset there
    print(f"Writing {n_train_images} (train) + {n_val_images} (validation) fake images of {size[0]}x{size[1]} divided in {num_classes} classes to '{output_dir}'...")
    class_counters = { mode: { clss: 0 for clss in range(num_classes) } for mode in ["train", "val"] }
    for (mode, n_images) in [("train", n_train_images), ("val", n_val_images)]:
        mode_dir = os.path.join(output_dir, mode)
        for clss in range(num_classes):
            clss_dir = os.path.join(mode_dir, str(clss))

            # Create a generator for this mode and save all of its images to disc
            data = datasets.FakeData(size=n_images, image_size=size, num_classes=1)

            # Write the datasets
            for img, _ in data:
                # Determine the image name
                if mode == "train":
                    image_name = f"{clss}_{class_counters[mode][clss]}.{EXTENSION}"
                else:
                    image_name = f"ILSVRC2012_val_{class_counters[mode][clss]:0>8}.{EXTENSION}"

                # Compute the output name
                img_path = os.path.join(clss_dir, image_name)
                print(f" - Writing image of class {clss} to '{img_path}'")

                # Write the file
                try:
                    img.save(img_path, FORMAT)
                except Exception as e:
                    print(f"ERROR: Failed to write image '{img_path}': {e}", file=sys.stderr)
                    return 1

                # Increment the counter
                class_counters[mode][clss] += 1



    # Done!
    return 0



# Actual entrypoint
if __name__ == "__main__":
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("OUTPUT", help="The directory to generate the fake data in.")
    parser.add_argument("-f", "--fix-dirs", action="store_true", help="If given, creates the output directory if it does not exist.")

    parser.add_argument("-t", "--train-n", type=int, default=50, help="The number of samples to generate for the training phase PER CLASS.")
    parser.add_argument("-v", "--val-n", type=int, default=10, help="The number of samples to generate for the validation phase PER CLASS.")
    parser.add_argument("-s", "--size", default="256x256", help="The size of the images to generate. Given as a `<WIDTH>x<HEIGHT>` pair (e.g., `1024x768`)")
    parser.add_argument("-c", "--classes", type=int, default=1000, help="The number of classes that the images can belong to.")

    # Parse the arguments
    args = parser.parse_args()
    x_pos = args.size.find("x")
    size = (int(args.size[:x_pos]), int(args.size[x_pos+1:]))

    # Run main
    exit(main(args.OUTPUT, args.fix_dirs, args.train_n, args.val_n, size, args.classes))
