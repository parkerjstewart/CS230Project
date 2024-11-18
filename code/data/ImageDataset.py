import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, annotations_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.transform = transform

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Create category mapping
        self.category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}
        
        # Map image filenames to labels
        self.image_to_label = {}
        image_metadata = {img['id']: img for img in annotations['images']}
        
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            file_name = os.path.basename(image_metadata[image_id]['file_name'])
            self.image_to_label[file_name] = category_id
        
        # Get list of image filenames
        self.image_filenames = list(self.image_to_label.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image filename
        file_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, file_name)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        category_id = self.image_to_label[file_name]
        label = self.category_mapping[category_id]
        
        return image, label    

'''

# takes in "test", "val", "test-dev", or "test"
# returns the json object for that dataset
def load_annotations(dataset):
    annotations_path = "../../data/annotations"
    mappings = {"test": "/home/ec2-user/CS230Project/data/annotations/Test-Challenge_poly.json",
                "test-dev": "/home/ec2-user/CS230Project/data/annotations/Test-Dev_poly.json",
                "train": "/home/ec2-user/CS230Project/data/annotations/Train_poly.json",
                "val": "/home/ec2-user/CS230Project/data/annotations/Val_poly.json"}
    
    """Load annotations from a JSON file."""
    with open(mappings.get(dataset), "r") as f:
        annotations = json.load(f)
    return annotations

    self.category_mapping = {cat['id']: cat['name'] for cat in annotations['categories']}
    self.image_to_label = {}
    image_metadata = {img['id']: img for img in annotations['images']}
    

print(load_annotations("test").keys())
print(len(load_annotations("test")["categories"]))
print(type(load_annotations("test")["images"]))
print(type(load_annotations("test")["annotations"]))
print(load_annotations("test")["images"][0])
print(load_annotations("test")["categories"][0])
print(load_annotations("test")["annotations"][0])

'''

