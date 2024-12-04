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
