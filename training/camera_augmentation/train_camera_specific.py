"""Use the official trainer with a camera-keyed dataset wrapper for this experiment."""
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    from camera_specific import CameraSpecificDataset, augmentation_spec
    from lerobot.scripts import lerobot_train as trainer
    profile=os.environ.get('SO101_AUG_PROFILE','combined')
    original=trainer.make_train_eval_datasets
    original_save=trainer.save_checkpoint

    def save_with_metadata(*args,**kwargs):
        result=original_save(*args,**kwargs)
        checkpoint=Path(kwargs['checkpoint_dir'])
        spec=augmentation_spec(profile)
        (checkpoint/'pretrained_model'/'camera_augmentation.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        for filename in ['camera_specific.py','train_camera_specific.py']:
            shutil.copy2(Path(__file__).with_name(filename),checkpoint/filename)
        return result

    def make_datasets(cfg):
        if cfg.dataset.image_transforms.enable:
            raise ValueError('Built-in shared transforms must be disabled to avoid double augmentation')
        train, evaluation=original(cfg)
        spec=augmentation_spec(profile)
        out=Path(cfg.output_dir)
        out.mkdir(parents=True,exist_ok=True)
        (out/'camera_augmentation.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        print('CAMERA_SPECIFIC_AUGMENTATION '+json.dumps(spec),flush=True)
        return CameraSpecificDataset(train,profile=profile),evaluation

    trainer.make_train_eval_datasets=make_datasets
    trainer.save_checkpoint=save_with_metadata
    trainer.main()


if __name__=='__main__':
    main()
