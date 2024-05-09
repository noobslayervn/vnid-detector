import os
from PIL import Image

# Path to the folder containing images
folder_path = './cropped_images_v1'

# Get list of images in the folder
images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Open each image one by one
for image_file in images:
    image_path = os.path.join(folder_path, image_file)
    
    # Open the image
    image = Image.open(image_path)
    image.show()
    
    # Get user input
    content = input("Enter content for this image (press Enter to skip): ")
    
    # Close the image
    image.close()
    
    # Write content to a text file
    with open('content.txt', 'a') as f:
        f.write(f"{image_file}\t{content}\n")
