from pathlib import Path

try:
    _this_file = Path(__file__)
except NameError:
    _this_file = Path(
        'configs/gr00t/gr00t_eagle_3b_robocasa_posttrain_24x30_official_aug.py')

_base_path = _this_file.with_name(
    'gr00t_eagle_3b_robocasa_posttrain_24x30.py')
_namespace = {'__file__': str(_base_path)}
exec(_base_path.read_text(), _namespace)

for _key, _value in _namespace.items():
    if not _key.startswith('__'):
        globals()[_key] = _value

_train_transforms = train_dataloader['dataset']['datasets'][0]['transforms']
_resize_idx = next(
    idx for idx, transform in enumerate(_train_transforms)
    if transform['type'] == 'ResizeImages')

_train_transforms[_resize_idx:_resize_idx + 1] = [
    dict(type='RandomCropImages', scale=0.95),
    dict(type='ResizeImages', height=224, width=224),
    dict(
        type='ColorJitterImages',
        brightness=0.3,
        contrast=0.4,
        saturation=0.5,
        hue=0.08),
]

del Path
del _base_path
del _key
del _namespace
del _resize_idx
del _this_file
del _train_transforms
del _value
