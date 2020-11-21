from torch.utils.data import Dataset
from os import walk
from PIL import Image
from torchvision import transforms
from pathlib import Path


class PrecomputedAudio(Dataset):
    def __init__(self, path, dpi=50, img_transforms=None):
        # get the files and set the dpi for resolution
        files = Path(path).glob('{}{}*.wav.png'.format(path.name, dpi))
        # file name format: {foldername}{dpi}_xxx_{label}.wav.png, for example, 'test50_5-9032-A-0.wav.png'
        # item format - (file, lab)
        self.items = [(f, int(f.name.split("-")[-1].replace(".wav.png", ""))) for f in files]
        self.length = len(self.items)
        # transforms is quite useful function for preprocessing each image inside training/validation dataset
        if img_transforms == None:
            self.img_transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.img_transforms = img_transforms

    def __getitem__(self, index):
        filename, label = self.items[index]
        img = Image.open(filename).convert('RGB')

        return (self.img_transforms(img), label)

    def __len__(self):

        return self.length