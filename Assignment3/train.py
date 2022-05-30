import os

from kws import KWSDataModule
from transformer import LitTransformer

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

if __name__ == '__main__':
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    
    # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    
    #Parameters:
    batch_size = 128
    num_workers = 0
    path="data/speech_commands/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    n_fft=1024
    n_mels=128
    win_length=None
    hop_length=512
    patch_num = 8
    class_dict=CLASS_TO_IDX
    
    num_classes=len(CLASSES)
    lr=0.001
    epochs=30
    depth=12
    embed_dim=128
    head=4
    
    accelerator='gpu'
    devices=1
    max_epochs=30
    precision=16
    
    datamodule = KWSDataModule(batch_size=batch_size, num_workers=num_workers,
                               path=path, n_fft=n_fft, n_mels=n_mels,
                               win_length=win_length, hop_length=hop_length, patch_num = patch_num,
                               class_dict=class_dict)
    datamodule.setup()
    
    data = iter(datamodule.train_dataloader()).next()
    patch_dim = data[0].shape[-1]
    seqlen = data[0].shape[-2]
    # print(patch_dim, seqlen)
    # exit()
    # print('data: {}\ndata.shape(): {}'.format(data[0], data[0].shape))
    # exit()
    
    model_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(path, "checkpoints"),
        filename="kws-best-acc",
        save_top_k=1,
        verbose=True,
        monitor='test_acc',
        mode='max',
    )
    model_checkpoint.FILE_EXTENSION = ".pt"
    model = LitTransformer(num_classes=num_classes, lr=lr, epochs=epochs, 
                           depth=depth, embed_dim=embed_dim, head=head,
                           patch_dim=patch_dim, seqlen=seqlen,)

    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class

    trainer = Trainer(accelerator=accelerator, devices=devices,
                      max_epochs=max_epochs, precision=precision,
                      callbacks=[model_checkpoint])
    
    
    trainer.fit(model, datamodule=datamodule)