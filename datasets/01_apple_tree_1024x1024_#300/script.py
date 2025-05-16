import cv2
import os

# Paths to the folders
images_folder = "images"
semantics_folder = "semantics"
output_folder = "output"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get sorted lists of image and mask files
image_files = sorted(os.listdir(images_folder))
mask_files = sorted(os.listdir(semantics_folder))

# Ensure the number of images and masks match
if len(image_files) != len(mask_files):
    raise ValueError("The number of images and masks do not match.")

# Apply each mask to the corresponding image
for image_file, mask_file in zip(image_files, mask_files):
    # Construct full paths
    image_path = os.path.join(images_folder, image_file)
    mask_path = os.path.join(semantics_folder, mask_file)
    output_path = os.path.join(output_folder, image_file)

    # Read the image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {image_file} or {mask_file} due to read error.")
        continue

    # Apply the mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imwrite(output_path, masked_image)


print("Mask application completed. Check the 'output' folder.")