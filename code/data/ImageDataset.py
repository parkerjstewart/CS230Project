import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

'''
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
            category_id = int(annotation['category_id'])
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

class ImageDataset(Dataset):
    def __init__(self, annotations_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.transform = transform

        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # label 0 will be real and label 1 will be fake
        categories = annotations["categories"]
        category_ids_to_name = {category["id"] : category["name"] for category in categories}

        self.image_ids_to_file_names = {}
        for image in annotations["images"]:
            id_num = image["id"]
            file_name = os.path.basename(image["file_name"])
            self.image_ids_to_file_names[id_num] = file_name
        
        self.image_ids_to_labels = {}
        for image in annotations["annotations"]:
            id_num = image["image_id"]
            category_id = image["category_id"]
            if category_ids_to_name[category_id] == "Real":
                self.image_ids_to_labels[id_num] = 0.0
            else:
                self.image_ids_to_labels[id_num] = 1.0
        
        self.image_ids = list(self.image_ids_to_labels.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image filename
        file_name = self.image_ids_to_file_names[self.image_ids[idx]]
        image_path = os.path.join(self.images_dir, file_name)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.image_ids_to_labels[self.image_ids[idx]]
        
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

