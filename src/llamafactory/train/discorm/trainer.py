# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch.distributions import Normal  # 添加Normal用于log_cdf计算
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""Inherits Trainer to compute pairwise loss."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], 
        **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Compute DiscoRM loss using both mean and variance. 
        
        The first n examples are chosen and the last n examples are rejected.
        """
        # 调用模型获取均值和方差
        outputs = model(**inputs, output_hidden_states=True, return_dict=False, use_cache=False)
        # AutoModelForCausalLMWithNormalHead 返回的是(logits, loss, value, variance)
        _, _, values, variances = outputs[:4]
        
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        
        # 分割均值和方差
        chosen_means, rejected_means = torch.split(values, batch_size, dim=0)
        chosen_vars, rejected_vars = torch.split(variances, batch_size, dim=0)
        
        # 获取每个序列最后一个token的均值和方差
        chosen_mean_scores = chosen_means.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_mean_scores = rejected_means.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        chosen_var_scores = chosen_vars.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_var_scores = rejected_vars.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        
        # 去掉不必要的维度
        chosen_mean_scores = chosen_mean_scores.squeeze()
        rejected_mean_scores = rejected_mean_scores.squeeze()
        chosen_var_scores = chosen_var_scores.squeeze()
        rejected_var_scores = rejected_var_scores.squeeze()
        
        # 计算DiscoRM损失: -log P(chosen > rejected)
        # P(chosen > rejected) = Phi((mu_c - mu_r) / sqrt(var_c + var_r))
        delta_mu = chosen_mean_scores - rejected_mean_scores
        sum_var = chosen_var_scores + rejected_var_scores
        
        # 直接在 compute_loss 内部定义和使用 variance_epsilon
        variance_epsilon = 1e-6 
        clamped_sum_var = torch.clamp(sum_var, min=variance_epsilon)  # 增加数值稳定性
        sigma_sum = torch.sqrt(clamped_sum_var)
        
        z_score = delta_mu / sigma_sum
        
        # 使用log_cdf计算log Phi(z)，保证数值稳定性
        log_p_chosen_preferred = Normal(0, 1).log_cdf(z_score.float())
        
        # 最终损失
        loss = -log_p_chosen_preferred.mean()
        
        if return_outputs:
            # 保持与原有返回格式相似，但在评估中可以访问方差信息
            return loss, (loss, chosen_mean_scores, rejected_mean_scores, chosen_var_scores, rejected_var_scores)
        else:
            return loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        
        # 根据predictions的长度判断是否包含方差
        if len(predict_results.predictions) >= 4:
            chosen_mean_scores, rejected_mean_scores, chosen_var_scores, rejected_var_scores = predict_results.predictions[:4]
            
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                res: list[str] = []
                for i in range(len(chosen_mean_scores)):
                    res.append(json.dumps({
                        "chosen_mean": round(float(chosen_mean_scores[i]), 4), 
                        "rejected_mean": round(float(rejected_mean_scores[i]), 4),
                        "chosen_variance": round(float(chosen_var_scores[i]), 4),
                        "rejected_variance": round(float(rejected_var_scores[i]), 4)
                    }))
                writer.write("\n".join(res))
        else:
            # 兼容原有格式
            chosen_scores, rejected_scores = predict_results.predictions[:2]
            
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                res: list[str] = []
                for c_score, r_score in zip(chosen_scores, rejected_scores):
                    res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))
                writer.write("\n".join(res))
                
    @override
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        """重写prediction_step确保在评估时返回方差信息"""
        # 确保模型处于评估模式
        model.eval()
        
        # 准备输入
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            # 计算损失和输出
            loss, outputs_tuple = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            # outputs_tuple = (loss, chosen_mean, rejected_mean, chosen_var, rejected_var)
            if outputs_tuple is not None and len(outputs_tuple) == 5:
                preds = tuple(output.detach() for output in outputs_tuple[1:])  # 跳过loss
            else:
                # 默认处理
                logger.warning_rank0("prediction_step无法从compute_loss获取完整输出")
                batch_size = inputs["input_ids"].size(0) // 2
                dummy_tensor = torch.zeros(batch_size, device=loss.device)
                preds = (dummy_tensor, dummy_tensor)
        
        return loss, preds, None  # 返回 (loss, predictions, labels)
