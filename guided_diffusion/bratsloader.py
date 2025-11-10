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
                  BraTS2021_00002_seg.nii.gz
                  where the last part before extension is one of t1, t1ce, t2, flair, seg
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
                # Filter out non-.nii/.nii.gz files
                files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]
                
                if len(files) == 0:
                    continue
                    
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    try:
                        seqtype = f.split('_')[2].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    except IndexError:
                        print(f"Warning: Cannot parse filename {f}, skipping...")
                        continue
                
                if len(datapoint) > 0 and set(datapoint.keys()) != self.seqtypes_set:
                    continue
                    
                if set(datapoint.keys()) == self.seqtypes_set:
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
                  BraTS2021_00002_seg.nii.gz
                  where the last part before extension is one of t1, t1ce, t2, flair, seg
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
                # Filter out non-.nii/.nii.gz files
                files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]
                
                if len(files) == 0:
                    continue
                    
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    try:
                        seqtype = f.split('_')[2].split('.')[0]
                        datapoint[seqtype] = os.path.join(root, f)
                    except IndexError:
                        print(f"Warning: Cannot parse filename {f}, skipping...")
                        continue
                
                if len(datapoint) > 0 and set(datapoint.keys()) != self.seqtypes_set:
                    continue
                    
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
        
        print(f"Loaded {len(self.database)} complete patient scans")
    
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

        # --- Create 2D data (center slice) with all 4 modalities ---
        image_2d_modalities = []
        for s in self.seqtypes:
            if s != 'seg':
                image_2d_modalities.append(volumes[s][..., slice_idx])
        image_2d = torch.stack(image_2d_modalities)

        # --- Create 2.5D data (stack of 3 consecutive slices from flair) ---
        vol_2_5d = volumes.get('flair', volumes[self.seqtypes[0]])
        num_slices_2_5d = 3
        half_slices = num_slices_2_5d // 2

        slices_for_stack = []
        for i in range(slice_idx - half_slices, slice_idx + half_slices + 1):
            clamped_idx = np.clip(i, 0, vol_2_5d.shape[2] - 1)
            slices_for_stack.append(vol_2_5d[..., clamped_idx])

        image_2_5d = torch.stack(slices_for_stack, dim=0)  # Shape: (3, H, W)

        # --- Handle label ---
        if self.test_flag:
            label_2d = image_2d  # Return image as label for test mode
        else:
            label_vol = volumes['seg']
            label_2d = label_vol[..., slice_idx].unsqueeze(0)
            label_2d = torch.where(label_2d > 0, 1, 0).float()

        # --- Apply transformations ---
        if self.transform:
            state = torch.get_rng_state()
            image_2d = self.transform(image_2d)
            image_2_5d = self.transform(image_2_5d)
            if not self.test_flag:
                torch.set_rng_state(state)
                label_2d = self.transform(label_2d)

        # --- Final output structure ---
        batch_image = (image_2d, image_2_5d)
        virtual_path = path.split('.nii')[0] + "_slice" + str(slice_idx) + ".nii"

        if self.test_flag:
            return (batch_image, batch_image, virtual_path)
        
        return (batch_image, label_2d, virtual_path)
    