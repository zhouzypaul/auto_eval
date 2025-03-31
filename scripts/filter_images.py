"""
goes through the saved images by the classifier, and uses p/n keyboards to separate
the images input positive/negatives, and use them to re-train the classifier
"""
import argparse
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main(input_folder, output_folder):
    positive_images = []
    negative_images = []

    # Keep track of all image files and current index
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    current_idx = 0

    while current_idx < len(png_files):
        filename = png_files[current_idx]
        # Visualize the image
        img_path = os.path.join(input_folder, filename)
        img = mpimg.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        cv2.imshow("Image", img)

        # Ask user for input using cv2.waitKey
        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
        if key == ord("p"):
            positive_images.append({"observation": {"image_primary": img}})
            current_idx += 1
        elif key == ord("n"):
            negative_images.append({"observation": {"image_primary": img}})
            current_idx += 1
        elif key == ord("s"):
            # skip this file
            current_idx += 1
        elif key == ord("b") and current_idx > 0:
            # Go back to previous image
            current_idx -= 1
            # Remove the last added image from either positive or negative list
            if (
                positive_images
                and positive_images[-1]["observation"]["image_primary"] is img
            ):
                positive_images.pop()
            elif (
                negative_images
                and negative_images[-1]["observation"]["image_primary"] is img
            ):
                negative_images.pop()
        elif key == ord("q"):
            # Quit the program
            break
        else:
            print(
                "Invalid input. Use: 'p' for positive, 'n' for negative, 's' to skip, 'b' to go back, 'q' to quit"
            )

        cv2.destroyAllWindows()  # Close the image window after input

    # Create pickle files
    with open(os.path.join(output_folder, "positives.pkl"), "wb") as pos_file:
        pickle.dump(positive_images, pos_file)
        print("Positive images saved to positives.pkl")

    with open(os.path.join(output_folder, "negatives.pkl"), "wb") as neg_file:
        pickle.dump(negative_images, neg_file)
        print("Negative images saved to negatives.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument(
        "--input_folder", type=str, help="The folder containing .png images"
    )
    parser.add_argument(
        "--output_folder", type=str, help="The folder to save the pickle files"
    )
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
