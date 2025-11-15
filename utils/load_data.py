import cv2
import os

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43


def load_data(data_dir, img_width=None, img_height=None, num_categories=None):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    
    # Use provided values or defaults
    if img_width is None:
        img_width = IMG_WIDTH
    if img_height is None:
        img_height = IMG_HEIGHT
    if num_categories is None:
        num_categories = NUM_CATEGORIES
    
    images = []
    labels = []
    
    # Iterate over each category directory (0 to NUM_CATEGORIES - 1)
    for category in range(num_categories):
        # Build the path to the category directory
        category_path = os.path.join(data_dir, str(category))
        
        # Check if the directory exists
        if not os.path.isdir(category_path):
            continue
        
        # Iterate over all files in the category directory
        for filename in os.listdir(category_path):
            # Build the full file path
            file_path = os.path.join(category_path, filename)
            
            # Read the image
            img = cv2.imread(file_path)
            
            # Ensure the image was read successfully
            if img is not None:
                # Resize the image to IMG_WIDTH x IMG_HEIGHT
                img_resized = cv2.resize(img, (img_width, img_height))
                
                # Add the image to the list
                images.append(img_resized)
                
                # Add the corresponding label to the list
                labels.append(category)
    
    return (images, labels)
