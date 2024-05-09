import os
import shutil

def copy_images_with_annotations(src_img_dir, src_label_dir, dest_img_dir, dest_label_dir):
    # Create destination directories if they don't exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    # Iterate through image files in the source directory
    for img_filename in os.listdir(src_img_dir):
        img_path = os.path.join(src_img_dir, img_filename)

        # Check if the corresponding annotation file exists
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(src_label_dir, label_filename)

        if os.path.exists(label_path):
            # Copy the image to the destination directory
            shutil.copy(img_path, dest_img_dir)

            # Copy the corresponding annotation file to the destination directory
            shutil.copy(label_path, dest_label_dir)

if __name__ == "__main__":
    # Replace these paths with your actual directory paths
    source_img_directory = "results"
    source_label_directory = "aligned-labels"
    destination_img_directory = "align-images-v2"
    destination_label_directory = "align-labels-v2"

    # Copy images with annotations to the destination directories
    copy_images_with_annotations(
        source_img_directory, source_label_directory,
        destination_img_directory, destination_label_directory
    )
