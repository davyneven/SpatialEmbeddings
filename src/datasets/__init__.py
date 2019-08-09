from datasets.CityscapesDataset import CityscapesDataset

def get_dataset(name, dataset_opts):
    if name == "cityscapes": 
        return CityscapesDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))