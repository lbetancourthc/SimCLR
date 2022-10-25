from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class PANDA_Dataset(Dataset):
  def __init__(self, data_dir, df):
    self.data_dir = data_dir
    self.df = df

  def __getitem__(self, index):
    idx_1, idx_2 = random.sample(range(0, len(df_train)), 2)
    row_1 = self.df.iloc[idx_1]
    row_2 = self.df.iloc[idx_2]

    image_id_1 = row_1['image_id']
    image_id_2 = row_2['image_id']

    image_1 = Image.open(os.path.join(self.data_dir, image_id_1.split('_')[0], image_id_1+'.jpeg'))
    image_2 = Image.open(os.path.join(self.data_dir, image_id_2.split('_')[0], image_id_2+'.jpeg'))
    image_1 = transforms.PILToTensor()(image_1)
    image_2 = transforms.PILToTensor()(image_2)

    label_1 = row_1['gleason_score']
    label_2 = row_2['gleason_score']
    label_1 = torch.tensor(label_1, dtype=torch.long)
    label_2 = torch.tensor(label_2, dtype=torch.long)

    return image_1.float(), image_2.float()

  def __len__(self):
    return len(self.df)
