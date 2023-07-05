import os
from sklearn.model_selection import train_test_split
import shutil

# Set the path to the directory containing the images
data_dir = "fruits-360"

# Set the ratio for train/validation/test split

val_ratio = 0.15
test_ratio = 0.15
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Get the list of image filenames in the directory
image_filenames = os.listdir(data_dir)
for classes in image_filenames:
    img_files = os.listdir(f'{data_dir}/{classes}')

    # Split the image filenames into train/validation/test sets
    train_filenames, val_test_filenames = train_test_split(img_files, test_size=(val_ratio + test_ratio), random_state=42)
    val_filenames, test_filenames = train_test_split(val_test_filenames, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    os.makedirs(f'train/{classes}', exist_ok=True)
    os.makedirs(f'val/{classes}', exist_ok=True)
    os.makedirs(f'test/{classes}', exist_ok=True)


    # Move train files
    for filename in train_filenames:
        src = os.path.join(f'{data_dir}/{classes}', filename)
        dst = os.path.join(f'train/{classes}', filename)
        shutil.move(src, dst)

    # Move validation files
    for filename in val_filenames:
        src = os.path.join(f'{data_dir}/{classes}', filename)
        dst = os.path.join(f'val/{classes}', filename)
        shutil.move(src, dst)

    # Move test files
    for filename in test_filenames:
        src = os.path.join(f'{data_dir}/{classes}', filename)
        dst = os.path.join(f'test/{classes}', filename)
        shutil.move(src, dst)
