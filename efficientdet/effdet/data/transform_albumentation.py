import albumentations as A

from albumentations.augmentations.transforms import HorizontalFlip, VerticalFlip, \
    PadIfNeeded, RandomBrightness, Downscale, Blur, RandomFog, RandomRain, RandomShadow

def get_transform():
    transforms = A.Compose([
        RandomShadow(p = 0.1),
        RandomFog(p = 0.2),
        RandomRain(p=0.2),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Blur(blur_limit=7, always_apply=False, p=0.5),
        RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        Downscale(scale_min=0.5, scale_max=0.9, interpolation=0, always_apply=False, p=0.5),
        PadIfNeeded(min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0)]
    )
    return transforms