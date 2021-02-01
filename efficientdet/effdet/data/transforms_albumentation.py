import albumentations as A

from albumentations.augmentations.transforms import (HorizontalFlip, VerticalFlip,
    PadIfNeeded, RandomBrightness, Downscale, Blur, RandomFog, RandomRain, RandomSnow)

def get_transform():
    transforms = A.Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Blur(blur_limit=7, always_apply=False, p=0.5),
        RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        Downscale(scale_min=0.5, scale_max=0.9, interpolation=0, always_apply=False, p=0.5),
        PadIfNeeded(min_height=1024, min_width=1024, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None, always_apply=False, p=1.0),
        RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.2),
        RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.2),
        RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.2)
        ]
    )
    return transforms