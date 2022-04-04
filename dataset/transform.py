from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from config import Config

train_transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    rand_augment_transform(
        config_str='rand-m7-mstd0.5', 
        hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
    ),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


test_transform = transforms.Compose([
    transforms.Resize(Config.image_size), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])