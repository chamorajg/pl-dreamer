import yaml
import pytorch_lightning as pl
from dreamer import DreamerTrainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    with open("configs.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    ckpt_callback = ModelCheckpoint(
                            save_top_k=config["ckpt_callback"]["save_top_k"],
                            monitor=config["ckpt_callback"]["monitor"],
                            mode=config["ckpt_callback"]["mode"],
                            )
    model = DreamerTrainer(config)
    trainer = pl.Trainer(
                        default_root_dir=config["trainer_params"]["default_root_dir"],
                        gpus=config["trainer_params"]["gpus"],
                        gradient_clip_val=config["trainer_params"]["gradient_clip_val"],
                        callbacks=[ckpt_callback],
                        val_check_interval=config["trainer_params"]["val_check_interval"],
                        max_epochs=config["trainer_params"]["max_epochs"],
                        )
    trainer.fit(model)

if __name__ == '__main__':
    main()