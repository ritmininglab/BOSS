import os
import shutil

from torchvision import datasets, transforms
import torchvision

from .vision import VisionDataset

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image
import math
import numpy as np

np.random.seed(42)

def generate_exponential_imbalance(num_classes, initial_sample_size, decay_rate):
    class_sizes = [int(initial_sample_size * math.exp(-decay_rate * i)) for i in range(num_classes)]
    return class_sizes

def generate_mask(class_sizes, targets):
    np.random.seed(42)
    mask = np.zeros(len(targets), dtype=bool)
    for class_idx, num_samples in enumerate(class_sizes):
        class_indices = np.where(np.array(targets) == class_idx)[0]
        sampled_indices = np.random.choice(class_indices, size=num_samples, replace=False)
        mask[sampled_indices] = True
    return mask

class CIFARDataset(object):
    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset

    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False, imbalance=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)

        if imbalance:
            num_classes = 100
            initial_sample_size = 500
            decay_rate = 0.01
            class_sizes = generate_exponential_imbalance(num_classes, initial_sample_size, decay_rate)
            print("Class Sizes:", class_sizes)

            # Generate masking array
            mask = generate_mask(class_sizes, trainset.targets)

            # Apply the mask to the dataset
            trainset.data = trainset.data[mask]
            trainset.targets = np.array(trainset.targets)[mask]

        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean=[0.507, 0.487, 0.441]
        std=[0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

class SVHNDataset(object):
    @staticmethod
    def get_svhn_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform)
        return trainset

    @staticmethod
    def get_svhn_test(path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        return testset

class CINIC10Dataset(object):
    @staticmethod
    def get_cinic10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_cinic10_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        path = os.path.join(path, 'test')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def find_classes(directory: str):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
):
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

def default_loader(path: str) :
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) :
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, target

    def __len__(self):
        return len(self.samples)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


class TinyDataset(object):
    @staticmethod
    def get_tiny_train(path):
        # if transform is None:
        #     transform = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])

        TRAIN_MEAN = [0.4802, 0.4481, 0.3975]
        TRAIN_STD = [0.2302, 0.2265, 0.2262]
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        path = os.path.join(path, 'train')
        trainset = ImageFolder(root=path, transform=train_transform)
        return trainset

    @staticmethod
    def get_tiny_test(path):
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        TRAIN_MEAN = [0.4802, 0.4481, 0.3975]
        TRAIN_STD = [0.2302, 0.2265, 0.2262]

        test_transform = transforms.Compose([
            transforms.Resize(int(64/0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        path = os.path.join(path, 'val')
        testset = ImageFolder(root=path, transform=test_transform)
        return testset