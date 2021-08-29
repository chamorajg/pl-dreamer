import os
import yaml
import os.path as osp
import pytorch_lightning as pl
from dreamer import DreamerTrainer
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    with open("configs.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    pl.seed_everything(config["seed"])
    ckpt_callback = ModelCheckpoint(
                            filename='{epoch}-{loss:.2f}',
                            save_top_k=config["ckpt_callback"]["save_top_k"],
                            monitor=config["ckpt_callback"]["monitor"],
                            mode=config["ckpt_callback"]["mode"],
                            save_on_train_epoch_end=config["ckpt_callback"]["save_on_train_epoch_end"],
                            )
    model = DreamerTrainer(config)
    if config["trainer_params"]["default_root_dir"] == "None":
        config["trainer_params"]["default_root_dir"] = osp.dirname(__file__)
    trainer = pl.Trainer(
                        default_root_dir=config["trainer_params"]["default_root_dir"],
                        gpus=config["trainer_params"]["gpus"],
                        gradient_clip_val=config["trainer_params"]["gradient_clip_val"],
                        callbacks=[ckpt_callback],
                        val_check_interval=config["trainer_params"]["val_check_interval"],
                        max_epochs=config["trainer_params"]["max_epochs"],
                        )
    os.makedirs(f'{trainer.log_dir}/movies', exist_ok=True)
    trainer.fit(model)

if __name__ == '__main__':
    main()