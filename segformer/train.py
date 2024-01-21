import cv2
from torch.utils.data import Dataset
import os
from PIL import Image

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, img_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.feature_extractor = feature_extractor
        frame_path = img_dir + "train/"
        mask_path = img_dir + "mask/"

        self.img_dir = frame_path
        self.ann_dir = mask_path

        # Get the file names list from provided directory
        frame_list = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
        mask_list = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]

        # Separate frame and mask files lists, exclude unnecessary files
        frames_list = [file for file in frame_list if ('_L' not in file) and ('png' in file)]
        masks_list = [file for file in mask_list if ('_L' in file) and ('png' in file)]



        # read images


        self.images = sorted(frames_list)

        # read annotations


        self.annotations = sorted(masks_list)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)




        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

from transformers import SegformerFeatureExtractor

img_dir = '../data/'
feature_extractor = SegformerFeatureExtractor(align=False, reduce_labels=True)

train_dataset = SemanticSegmentationDataset(img_dir=img_dir, feature_extractor=feature_extractor)
valid_dataset = SemanticSegmentationDataset(img_dir=img_dir, feature_extractor=feature_extractor, train=False)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)

encoded_inputs = train_dataset[0]

mask = encoded_inputs["labels"].numpy()
import matplotlib.pyplot as plt
plt.imshow(mask)

print(encoded_inputs)
print(encoded_inputs["labels"].squeeze().unique())
print(encoded_inputs["pixel_values"].shape)
print(encoded_inputs["labels"].shape, encoded_inputs["labels"].squeeze().unique().shape)


from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url

# load id2label mapping from a JSON on the hub
filename = "idlabel.json"

with open(filename, 'r') as j:
    id2label = json.loads(j.read())

id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=len(id2label),
                                                         ignore_mismatched_sizes=True,
                                                         id2label=id2label,
                                                         label2id=label2id,
)


import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from datasets import load_metric
metric = load_metric("mean_iou")

# '''--------------------------------------------------------------'''
# Fine tuning the model

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Initialized!")

for epoch in range(200):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            pred_labels = upsampled_logits.detach().cpu().numpy()
            # note that the metric expects predictions + labels as numpy arrays
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
            metrics = metric._compute(num_labels=len(id2label),
                                      ignore_index=255,
                                      reduce_labels=False,predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy()
                                      )

            print("Loss:", loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])