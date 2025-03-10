import torch
from configs.option import get_option
from tools.datasets.datasetsv2 import *
from tools.pl_tool import *
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import os
import time


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option()
    """定义网络"""
    from tools.models.smfanet import *
    from tools.models.pnet import PNet
    from tools.models.scnet import SCNet
    from tools.models.lkfn import LKFN
    from tools.models.competition_backup import FusionNet
    from tools.models.mynet import FusionNet

    # model = FusionNet(
    #     dim=opt.dim,
    #     n_blocks=opt.n_blocks,
    #     upscaling_factor=opt.upscaling_factor,
    #     fmb_params={
    #         "smfa_growth": opt.smfa_growth,
    #         "pcfn_growth": opt.pcfn_growth,
    #         "snfa_dropout": opt.snfa_dropout,
    #         "pcfn_dropout": opt.pcfn_dropout,
    #         "p_rate": opt.p_rate,
    #     },
    # )
    # model = Ensemble()
    # model = LKFN()
    # model = PNet()
    # model = SCNet()
    model = FusionNet(
        dim=opt.dim,
        n_blocks=opt.n_blocks,
        upscaling_factor=opt.upscaling_factor,
    )
    """模型编译"""
    # model = torch.compile(model)
    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.devices,
        strategy="auto",
        max_epochs=opt.epochs,
        precision=opt.precision,
        default_root_dir="./",
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join("./checkpoints", opt.exp_name),
                monitor="metric/valid_psnr",
                mode="max",
                save_top_k=3,
                save_last=True,
                filename="epoch_{epoch}-loss_{metric/valid_psnr:.3f}",
                auto_insert_metric_name=False,
            )
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )
    wandb.finish()
