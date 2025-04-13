# Copyright 2025 the LlamaFactory team.
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

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any
from typing_extensions import override
from trl import AutoModelForCausalLMWithValueHead


class VarianceHead(nn.Module):
    """
    VarianceHead类实现了一个头部，为每个输出token返回一个方差值。
    与ValueHead结合使用，共同表示正态分布的均值和方差。
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # 获取hidden_size，与ValueHead保持一致的逻辑
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        # 输出方差，使用线性层加softplus确保方差为正
        self.var = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus(beta=1.0)  # beta=1.0是默认值，可根据需要调整

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # 确保数据类型一致
        if output.dtype != self.var.weight.dtype:
            output = output.to(self.var.weight.dtype)

        # 通过softplus函数确保方差为正
        var = self.softplus(self.var(output))
        
        return var


class AutoModelForCausalLMWithNormalHead(AutoModelForCausalLMWithValueHead):
    """
    一个自回归模型，除了语言模型头部外，还有一个正态分布头部。
    该类继承自`AutoModelForCausalLMWithValueHead`，保留原来的v_head计算均值，
    并添加var_head计算方差，共同表示一个正态分布。
    """

    def __init__(self, pretrained_model, **kwargs):
        """
        初始化模型，先调用父类初始化v_head，再添加var_head。
        """
        # 先用父类初始化，保留原有的v_head结构
        super().__init__(pretrained_model, **kwargs)
        
        config = getattr(pretrained_model, "config", None)
        if config is None:
            raise ValueError("The model you passed to `AutoModelForCausalLMWithNormalHead` doesn't have a config.")

        # 创建方差头部
        self.var_head = VarianceHead(config, **kwargs)
        
        # 初始化权重
        v_head_init_strategy = getattr(self, "v_head_init_strategy", None)
        v_head_initializer_range = getattr(self, "v_head_initializer_range", 0.2)
        
        if v_head_init_strategy == "normal":
            # 用正态分布初始化方差头部
            self.var_head.var.weight.data.normal_(mean=0.0, std=v_head_initializer_range)
            self.var_head.var.bias.data.zero_()

    @override
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_past_key_values=False,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor, ...], Any]:
        """
        对包装的模型应用前向传播，返回语言模型的logits以及值头的均值和方差。
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入序列token的索引。
            past_key_values (`tuple(tuple(torch.FloatTensor))`, 可选):
                包含模型预先计算的隐藏状态，用于加速序列解码。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, 可选):
                用于避免对padding token执行注意力的掩码。取值范围为[0, 1]。
                - 对于**未被掩码**的token，值为1
                - 对于**被掩码**的token，值为0
            return_past_key_values (bool): 指示是否返回计算的隐藏状态的标志。
            kwargs: 其他传递给被包装模型的关键字参数。
        """
        # 确保输出隐藏状态
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        # PREFIX_TUNING的特殊处理，直接继承自父类
        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        # 调用底层模型获取输出
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # 获取最后一层隐藏状态和其他输出
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        # 设备对齐
        if last_hidden_state.device != self.v_head.summary.weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)

        # 计算均值和方差
        value = self.v_head(last_hidden_state).squeeze(-1)  # 均值
        variance = self.var_head(last_hidden_state).squeeze(-1)  # 方差

        # 如果需要，将logits上升为fp32
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        # 根据return_past_key_values参数决定返回值
        if return_past_key_values:
            return (lm_logits, loss, value, variance, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value, variance)

    @override
    def state_dict(self, *args, **kwargs):
        """
        获取状态字典，添加var_head的参数。
        """
        state_dict = super().state_dict(*args, **kwargs)
        var_head_state_dict = self.var_head.state_dict(*args, **kwargs)
        
        # 添加var_head的参数到状态字典
        for k, v in var_head_state_dict.items():
            state_dict[f"var_head.{k}"] = v
            
        return state_dict 