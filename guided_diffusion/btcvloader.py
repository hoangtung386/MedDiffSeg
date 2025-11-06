import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_btcv_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    num_slices=5, # Number of slices for 2.5D input
):
    """
    For creating a data loader for the BTCV dataset.

    :param data_dir: a pointer to the dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    dataset = BTCVLoader(
        data_dir=data_dir,
        image_size=image_size,
        class_cond=class_cond,
        num_slices=num_slices,
    )

    if deterministic:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    return loader


class BTCVLoader(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False, side_x=256, side_y=256, num_slices=5):
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor()
            ]
        )
        self.image_size = image_size
        self.class_cond = class_cond
        self.side_x = side_x
        self.side_y = side_y
        self.num_slices = num_slices  # Number of slices for 2.5D input

        self.image_paths = []
        self.label_paths = []
        
        images_tr_path = os.path.join(data_dir, 'imagesTr')
        labels_tr_path = os.path.join(data_dir, 'labelsTr')

        if not os.path.isdir(images_tr_path):
            raise RuntimeError(f"imagesTr directory not found at {images_tr_path}")
        if not os.path.isdir(labels_tr_path):
            raise RuntimeError(f"labelsTr directory not found at {labels_tr_path}")

        for case in os.listdir(images_tr_path):
            if case.endswith('.nii.gz'):
                self.image_paths.append(os.path.join(images_tr_path, case))
                self.label_paths.append(os.path.join(labels_tr_path, case))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]

        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        image = image_nii.get_fdata()
        label = label_nii.get_fdata()

        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Choose a random slice
        slice_idx = np.random.randint(0, image.shape[2])
        
        image_slice = image[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]

        # Get 2.5D data (a stack of slices)
        half_slices = self.num_slices // 2
        start_idx = max(0, slice_idx - half_slices)
        end_idx = min(image.shape[2], slice_idx + half_slices + 1)
        
        actual_slices = image[:, :, start_idx:end_idx]
        
        # Pad if necessary (at the beginning or end of the volume)
        pad_before = half_slices - (slice_idx - start_idx)
        pad_after = (half_slices + 1) - (end_idx - slice_idx)

        padded_slices = np.pad(actual_slices, ((0,0), (0,0), (pad_before, pad_after)), 'constant')
        
        image_2_5d = padded_slices

        # Convert to tensors
        image_slice = Image.fromarray(image_slice * 255).convert("L")
        label_slice = Image.fromarray(label_slice).convert("L")

        if self.transform:
            image_slice = self.transform(image_slice)
            label_slice = self.transform(label_slice)

        # Process 2.5D image stack
        # The model expects (B, C, D, H, W), so we need to permute
        # from (H, W, D) to (C, D, H, W)
        image_2_5d = torch.from_numpy(image_2_5d).float().permute(2, 0, 1).unsqueeze(0) # D, H, W -> 1, D, H, W

        return (image_slice, image_2_5d), label_slice, os.path.basename(image_path)
