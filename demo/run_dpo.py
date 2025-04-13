#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行 Direct Preference Optimization (DPO) 训练

使用方法:
    python demo/run_dpo.py --config demo/dpo_test.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
# from trl import DPOTrainer # Import standard DPOTrainer from trl
from llamafactory.train.dpo.trainer import CustomDPOTrainer # Use llamafactory's custom DPO trainer
from llamafactory.train.trainer_utils import create_ref_model
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_dpo_training(config_path, wandb_project=None):
    """运行 DPO 训练"""
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    
    print('=' * 60)
    print(f"正在加载配置: {config_path}")
    print('=' * 60)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            list_value_str = ",".join(map(str, value))
            args.append(f"--{key}={list_value_str}")
        elif value is not None:
            args.append(f"--{key}={value}")
    
    print('=' * 60)
    print("处理参数中...")
    print('=' * 60)
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args) # Removed generating_args
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.remove_unused_columns = False
    training_args.dataloader_num_workers = 0
    print(f"强制设置 remove_unused_columns = {training_args.remove_unused_columns}")
    print(f"强制设置 dataloader_num_workers = {training_args.dataloader_num_workers}")

    device = None
    if torch.cuda.is_available():
        print("使用 CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("使用 MPS (Apple Silicon)")
        device = torch.device("mps")
    else:
        print("使用 CPU")
        device = torch.device("cpu")
        
    print('=' * 60)
    print("准备模型组件...")
    print('=' * 60)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # DPO 训练不需要 valuehead
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # DPO 需要参考模型 (ref_model)
    ref_model = create_ref_model(model_args, finetuning_args) if finetuning_args.stage == "dpo" else None 

    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    # 确保 stage 正确传递 (应该从 dpo_test.yaml 读取为 'dpo')
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage=finetuning_args.stage, **tokenizer_module)
    # DPOTrainer 使用自己的内部数据整理器，我们不需要显式创建 PairwiseDataCollatorWithPadding
    data_collator = PairwiseDataCollatorWithPadding(tokenizer)
    
    if "train_dataset" in dataset_module:
        print(f"训练集样本数: {len(dataset_module['train_dataset'])}")
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"]:
        print(f"验证集样本数: {len(dataset_module['eval_dataset'])}")
    
    print('=' * 60)
    print("设置训练器...")
    print('=' * 60)
    callbacks = [LogCallback()]
    # trainer = DPOTrainer( # Use standard DPOTrainer
    trainer = CustomDPOTrainer( # Use llamafactory's custom DPO trainer
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args, # Custom trainer likely needs finetuning_args
        # beta=finetuning_args.pref_beta, # Pass beta from finetuning_args - Custom trainer might get this via finetuning_args
        # loss_type=finetuning_args.pref_loss, # Pass loss_type from finetuning_args - Custom trainer might get this via finetuning_args
        data_collator=data_collator, # Custom trainer likely needs the data collator explicitly
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
        tokenizer=tokenizer,
        callbacks=callbacks,
        # peft_config=..., # Likely handled internally or via finetuning_args
        # max_prompt_length=data_args.cutoff_len, # Likely handled internally or via finetuning_args
        # max_length=data_args.cutoff_len + 128 # Likely handled internally or via finetuning_args
    )
    
    print('=' * 60)
    print("开始训练...")
    print('=' * 60)
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("保存模型...")
    try:
        # trainer.save_model(training_args.output_dir) # DPOTrainer.save_model 需要 output_dir
        trainer.save_model() # Custom trainer likely uses HuggingFace Trainer's save_model
        trainer.save_state() # Custom trainer likely has save_state
        print(f"模型已保存至: {training_args.output_dir}")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

def main():
    parser = argparse.ArgumentParser(description='运行 Direct Preference Optimization (DPO) 训练')
    parser.add_argument('--config', type=str, default='demo/dpo_test.yaml', help='DPO 训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_dpo_training(args.config, args.wandb_project)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 