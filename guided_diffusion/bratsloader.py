import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
    
    def __len__(self):
        return len(self.database) * 155

    def __getitem__(self, x):
        # Determine volume and slice index
        n = x // 155
        slice_idx = x % 155
        filedict = self.database[n]
        path = filedict[self.seqtypes[0]]  # for virtual path

        # Load full 3D volumes for all modalities
        volumes = {}
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            volumes[seqtype] = torch.tensor(nib_img.get_fdata())

        # --- Create 2D data (center slice) ---
        image_2d_modalities = [volumes[s][..., slice_idx] for s in self.seqtypes if s != 'seg']
        image_2d = torch.stack(image_2d_modalities)

        # --- Create 2.5D data (stack of slices from one modality) ---
        # Use flair, fallback to the first available modality
        vol_2_5d = volumes.get('flair', volumes[self.seqtypes[0]])
        num_slices_2_5d = 3
        half_slices = num_slices_2_5d // 2

        slices_for_stack = []
        for i in range(slice_idx - half_slices, slice_idx + half_slices + 1):
            clamped_idx = np.clip(i, 0, vol_2_5d.shape[2] - 1)
            slices_for_stack.append(vol_2_5d[..., clamped_idx])

        image_2_5d = torch.stack(slices_for_stack, dim=0).unsqueeze(0)  # Shape: (1, D, H, W)

        # --- Handle label and test mode ---
        if self.test_flag:
            label_2d = torch.zeros_like(image_2d[:1])
        else:
            label_vol = volumes['seg']
            label_2d = label_vol[..., slice_idx].unsqueeze(0)
            label_2d = torch.where(label_2d > 0, 1, 0).float()

        # --- Apply transformations ---
        if self.transform:
            state = torch.get_rng_state()
            image_2d = self.transform(image_2d)
            if not self.test_flag:
                torch.set_rng_state(state)
                label_2d = self.transform(label_2d)

        # --- Final output structure ---
        batch_image = (image_2d, image_2_5d)
        virtual_path = path.split('.nii')[0] + "_slice" + str(slice_idx) + ".nii"

        if self.test_flag:
            return (batch_image, label_2d, virtual_path)
        else:
            return (batch_image, label_2d, virtual_path)



