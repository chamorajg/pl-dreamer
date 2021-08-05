import yaml
import pytorch_lightning as pl
from dreamer import Dreamer


def main():
    with open("example.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    model = Dreamer(config)
    trainer = pl.Trainer()
    trainer.fit(model)

if __name__ == '__main__':
    main()