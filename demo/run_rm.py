#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行 Reward Modeling (RM) 训练

使用方法:
    python demo/run_rm.py --config demo/rm_test.yaml [--wandb_project PROJECT]
"""

import argparse
import os
import sys
import yaml
import torch
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, PairwiseDataCollatorWithPadding
from llamafactory.train.rm.trainer import PairwiseTrainer # Import PairwiseTrainer instead of RewardTrainer
from llamafactory.train.rm.metric import ComputeAccuracy # Import ComputeAccuracy
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def run_reward_modeling(config_path, wandb_project=None):
    """运行RM训练"""
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
            args.append(f"--{key}={','.join(map(str, value))}")
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
    # RM 训练需要 valuehead 模型
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True) 

    print('=' * 60)
    print("准备数据集...")
    print('=' * 60)
    # 确保 stage 正确传递 (应该从 rm_test.yaml 读取为 'rm')
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage=finetuning_args.stage, **tokenizer_module)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    
    if "train_dataset" in dataset_module:
        print(f"训练集样本数: {len(dataset_module['train_dataset'])}")
    if "eval_dataset" in dataset_module and dataset_module["eval_dataset"]:
        print(f"验证集样本数: {len(dataset_module['eval_dataset'])}")
    
    print('=' * 60)
    print("设置训练器...")
    print('=' * 60)
    callbacks = [LogCallback()]
    trainer = PairwiseTrainer( # Use PairwiseTrainer
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(), # Add compute_metrics
        **dataset_module,
        **tokenizer_module,
    )
    
    print('=' * 60)
    print("开始训练...")
    print('=' * 60)
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    print("保存模型...")
    try:
        trainer.save_model()
        trainer.save_state()
        print(f"模型已保存至: {training_args.output_dir}")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

def main():
    parser = argparse.ArgumentParser(description='运行 Reward Modeling (RM) 训练')
    parser.add_argument('--config', type=str, default='demo/rm_test.yaml', help='RM 训练配置文件路径')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb项目名称')
    args = parser.parse_args()
    run_reward_modeling(args.config, args.wandb_project)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 