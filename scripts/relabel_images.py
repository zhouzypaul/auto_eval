"""
Program to relabel images from pickle files, allowing for back-labeling and
saving positive/negative classifications to separate files.
"""
import argparse
import os
import pickle
import signal
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scripts.ft_paligemma import process_image


def save_progress(output_dir, positive_images, negative_images, current_idx):
    """Save current progress to a temporary file."""
    progress_file = os.path.join(output_dir, ".relabel_progress.pkl")
    with open(progress_file, "wb") as f:
        pickle.dump(
            {
                "positive_images": positive_images,
                "negative_images": negative_images,
                "current_idx": current_idx,
            },
            f,
        )
    print(f"\nProgress saved at image {current_idx}")


def load_progress(output_dir):
    """Load progress from temporary file if it exists."""
    progress_file = os.path.join(output_dir, ".relabel_progress.pkl")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "rb") as f:
                progress = pickle.load(f)
            print(f"Restored progress from image {progress['current_idx']}")
            return progress
        except Exception as e:
            print(f"Error loading progress file: {e}")
    return None


def cleanup_progress(output_dir):
    """Remove temporary progress file."""
    progress_file = os.path.join(output_dir, ".relabel_progress.pkl")
    if os.path.exists(progress_file):
        os.remove(progress_file)


def signal_handler(sig, frame):
    """Handle interrupt signal by saving progress before exit."""
    print("\nInterrupt received. Saving progress...")
    if hasattr(signal_handler, "save_args"):
        save_progress(*signal_handler.save_args)
    sys.exit(0)


def main(input_dir, output_dir, input_format):
    positive_images = []
    negative_images = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Try to load previous progress
    progress = load_progress(output_dir)
    if progress:
        positive_images = progress["positive_images"]
        negative_images = progress["negative_images"]
        start_idx = progress["current_idx"]
    else:
        start_idx = 0

    # Find all pickle files in input directory
    pickle_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
    if not pickle_files:
        print(f"No pickle files found in {input_dir}")
        return

    # Load all images from pickle files
    all_images = []
    for pkl_file in pickle_files:
        with open(os.path.join(input_dir, pkl_file), "rb") as f:
            try:
                data = pickle.load(f)
                if isinstance(data, list):
                    all_images.extend(data)
                else:
                    all_images.append(data)
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                continue

    if not all_images:
        print("No valid images found in pickle files")
        return

    # Set up signal handler with necessary arguments for saving progress
    signal_handler.save_args = (output_dir, positive_images, negative_images, start_idx)
    signal.signal(signal.SIGINT, signal_handler)

    current_idx = start_idx
    last_save_time = time.time()

    # convert format to rgb
    if input_format.lower() == "bgr":
        all_images = [
            {
                "observation": {
                    "image_primary": cv2.cvtColor(
                        img_dict["observation"]["image_primary"], cv2.COLOR_BGR2RGB
                    )
                }
            }
            for img_dict in all_images
        ]

    while current_idx < len(all_images):
        # Get current image data
        img_data = all_images[current_idx]
        img = img_data["observation"]["image_primary"]

        img = process_image(img)
        img_data["observation"][
            "image_primary"
        ] = img  # This updates the image in all_images since it's a reference

        # Display image
        cv2.imshow("Image", img)
        print(f"\nImage {current_idx + 1}/{len(all_images)}")
        print(
            "Press: 'p' for positive, 'n' for negative, 's' to skip, 'b' to go back, 'q' to quit"
        )

        key = cv2.waitKey(0) & 0xFF
        if key == ord("p"):
            positive_images.append(img_data)
            current_idx += 1
        elif key == ord("n"):
            negative_images.append(img_data)
            current_idx += 1
        elif key == ord("s"):
            current_idx += 1
        elif key == ord("b") and current_idx > 0:
            # Go back to previous image
            current_idx -= 1
            # Remove the last added image from either positive or negative list
            prev_img = all_images[current_idx]["observation"]["image_primary"]
            if (
                positive_images
                and positive_images[-1]["observation"]["image_primary"] is prev_img
            ):
                positive_images.pop()
            elif (
                negative_images
                and negative_images[-1]["observation"]["image_primary"] is prev_img
            ):
                negative_images.pop()
        elif key == ord("q"):
            break
        else:
            print("Invalid input!")

        cv2.destroyAllWindows()

        # Save progress every 5 minutes
        current_time = time.time()
        if current_time - last_save_time > 300:  # 300 seconds = 5 minutes
            save_progress(output_dir, positive_images, negative_images, current_idx)
            last_save_time = current_time

    # Save final results
    if positive_images:
        with open(os.path.join(output_dir, "positives.pkl"), "wb") as f:
            pickle.dump(positive_images, f)
            print(f"Saved {len(positive_images)} positive images")

    if negative_images:
        with open(os.path.join(output_dir, "negatives.pkl"), "wb") as f:
            pickle.dump(negative_images, f)
            print(f"Saved {len(negative_images)} negative images")

    # Clean up progress file after successful completion
    cleanup_progress(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel images from pickle files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing pickle files with images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save positive and negative pickle files",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["rgb", "bgr"],
        default="rgb",
        help="Input image format (rgb or bgr). OpenCV displays in RGB format.",
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.input_format)
