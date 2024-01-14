from module.data_module import DataModule
from model.phobert_base import PhoBert_base
from model.phobert_large import *
from model.phobert_lstm import *
from model.phobert_cnn import *
from trainer.fasttext import *
from trainer.phobert import *
import os
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything

class config:
    root_data_dir = './dataset/datashopee/'
    model_type = 'phobert'  
    batch_size = 32
    max_epochs = 50
    drop_out = 0.1
    num_labels = 2
    vector_size = 300  
    num_workers = 2
    fasttext_embedding = 'src/embedding/fasttext_train_dev.model'  # Otherwise specify path to embedding like src/embedding/fasttext_train_dev.model
    seed = 42
    freeze_backbone = False
    val_each_epoch = 2
    learning_rate = 1e-4
    accelarator = "gpu"

    tensorboard = {
        'dir': 'logging',
        'name': 'experiment',
        'version': 0
    }

    ckpt_dir = 'logging/experiment/0/ckpt'

dm = DataModule(root_data_dir=config.root_data_dir,
                    model_type=config.model_type,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    fasttext_embedding=config.fasttext_embedding)

dm.setup('fit')
loss_weight = dm.train_data.class_weights

print("Chon model:\n")
print("(1)PhoBERT(base)")
print("(2)PhoBERT(large)")
print("(3)PhoBERT(base) + LSTM")
print("(4)PhoBERT(base) + CNN")
print("(5)FastText + LSTM")

model_num = int(input("Chon model so: "))

if model_num == 1:
    model = PhoBert_base(freeze_backbone=config.freeze_backbone,drop_out=config.drop_out,num_labels=config.num_labels)
elif model_num == 2:
    model = PhoBert_large(drop_out=config.drop_out,num_labels=config.num_labels)
elif model_num == 3:
    model = PhoBERTLSTM(drop_out=config.drop_out,num_labels=config.num_labels)
elif model_num == 4:
    model = PhoBERTCNN_base(drop_out=config.drop_out,num_labels=config.num_labels)
elif model_num == 5:
    pass
else:
    raise ValueError(f"Not support model")

if model_num == 5:
    system = FastTextLSTMModel(dropout=config.drop_out,
                                num_labels=config.num_labels,
                                hidden_size=config.vector_size,
                                loss_weight=loss_weight)
else:
    system = PhoBERTModel(model=model,
                            num_labels=config.num_labels,
                            loss_weight=loss_weight)
    
checkpoint_callback = ModelCheckpoint(dirpath=config.ckpt_dir, monitor="val_loss",save_top_k=5, mode="min")

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=40)

logger = TensorBoardLogger(save_dir=config.tensorboard['dir'], name=config.tensorboard['name'], version=config.tensorboard['version'])

trainer = Trainer(accelerator=config.accelarator, check_val_every_n_epoch=config.val_each_epoch,gradient_clip_val=1.0,
                max_epochs=config.max_epochs,enable_checkpointing=True, deterministic=True, default_root_dir=config.ckpt_dir,
                callbacks=[checkpoint_callback, early_stopping], logger=logger, accumulate_grad_batches=4,log_every_n_steps=1)

trainer.fit(model=system, datamodule=dm)

trainer.test(model=system, datamodule=dm)