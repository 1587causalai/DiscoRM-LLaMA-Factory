# DiscoRM 基础模型架构设计与实现

本文档是 Disco Reward Model (DiscoRM) 集成至 LlamaFactory 系列文档的第一部分。本文档详细记录了 DiscoRM 基础模型 (`AutoModelForCausalLMWithNormalHead`) 的设计决策、具体代码实现、与 LlamaFactory 框架的集成方式，以及相关的兼容性考量。

**本文档的目标读者**: 需要理解 DiscoRM 模型如何在 LlamaFactory 中实现、其设计原理以及与标准奖励模型差异的开发者。

**与现有文档的关系**: 本文档**不同于** `奖励模型的训练_with_LLama-Factory.md`。后者描述了 LlamaFactory 中通用的奖励模型训练流程，而本文档专注于 DiscoRM 模型本身的架构设计和代码实现细节。

## 1. 背景与目标

### 1.1. 标准奖励模型 (RM) 的局限
标准的奖励模型通常为给定的输入（prompt+response）输出一个单一的标量奖励值。这表示了模型对该响应"好坏"程度的估计。然而，单一标量值无法表达模型对其预测的**不确定性**。在某些情况下，模型可能对某个奖励预测非常确定，而在另一些情况下则非常不确定。

### 1.2. DiscoRM 的核心思想
Disco Reward Model (DiscoRM) 旨在解决这个问题。它不直接输出标量奖励，而是输出一个**概率分布**来表示奖励。具体来说，我们将其建模为一个**正态分布 (Normal Distribution)**，由**均值 (mean)** 和**方差 (variance)** 两个参数定义：
*   **均值 (μ)**：可以看作是传统 RM 输出的期望奖励值。
*   **方差 (σ²)**：表示模型对该均值预测的不确定性。方差越大，表示模型越不确定。

### 1.3. 集成目标
我们的目标是将 DiscoRM 的这种能力集成到 LlamaFactory 框架中，要求：
1.  **最小改动**: 尽可能复用 LlamaFactory 和 TRL 库的现有代码。
2.  **保持兼容**: 确保新模型能与 LlamaFactory 的加载、微调 (LoRA/Full)、训练参数管理等流程兼容。
3.  **清晰实现**: 代码结构清晰，易于理解和维护。

## 2. 设计决策与演进 (Top-Down)

我们从高层设计出发，逐步细化实现方案：

### 2.1. 基础模型选择
我们选择 TRL 库的 `trl.AutoModelForCausalLMWithValueHead` 作为基础。
*   **原因**: 该类已经实现了在基础语言模型之上添加一个**价值头 (Value Head, `v_head`)** 来输出标量值。这天然地可以作为我们 DiscoRM 所需的**均值 (μ)** 输出。

### 2.2. 如何实现方差输出？
我们考虑了两种主要方案：

1.  **方案一 (重写，已否决)**: 创建一个全新的类 `NewDiscoModel`，不继承 `WithValueHead`，而是包含一个新的 `NormalHead` 模块，该模块同时输出均值和方差。
    *   **缺点**: 需要重新实现大量 `WithValueHead` 中已有的逻辑（如 PEFT 集成、`forward` 方法中的通用处理、`state_dict` 管理等），代码冗余，兼容性风险高。违背了最小改动原则。

2.  **方案二 (继承，已采纳)**: 创建 `AutoModelForCausalLMWithNormalHead` 类，并让它**继承**自 `trl.AutoModelForCausalLMWithValueHead`。
    *   **优点**:
        *   最大限度地复用父类代码和逻辑。
        *   自动继承父类对 PEFT、模型加载/保存、`generate` 等功能的处理。
        *   改动集中且清晰。
    *   **实现**:
        *   保留父类继承下来的 `self.v_head` 用于计算**均值 (μ)**。
        *   **新增**一个独立的模块 `VarianceHead` (`self.var_head`)，专门用于计算**方差 (σ²)**。
    *   **讨论点与权衡**:
        *   **Token-level 输出**: `v_head` 和 `var_head` 都作用于最后一层每个 token 的隐藏状态，因此输出的均值和方差形状为 `(batch_size, sequence_length)`。这意味着下游任务（如损失函数或推理）需要自行实现**聚合逻辑**，例如，通常取**最后一个有效 token** 对应的均值和方差作为序列级别的分布参数。
        *   **参数独立性**: 均值头和方差头是两个独立的线性层。这提供了最大的灵活性（例如可以独立初始化），但没有利用两者可能共享的特征表示。
        *   **备选头部设计 (未采纳)**:
            *   *单一组合头*: 使用单个线性层输出 2 个值，分别对应均值和（Softplus前的）方差。优点是可能更参数高效，缺点是失去独立控制，且需要更多自定义代码。
            *   *共享主干头*: 先用共享层处理隐藏状态，再分出两个小头计算均值和方差。优点是可能提升表示能力，缺点是结构更复杂。
        *   **结论**: 当前方案通过牺牲一定的参数共享可能性，换取了实现的简洁性、灵活性以及对 `trl` 库的最大化复用。

### 2.3. 方差激活函数选择
方差必须是非负的。我们考虑了两种激活函数：

1.  **`torch.exp`**: 计算 `exp(linear_output)`。
    *   **缺点**: 数值不稳定，容易溢出；梯度可能爆炸。
2.  **`nn.Softplus` (已采纳)**: 计算 `log(1 + exp(linear_output))`。
    *   **优点**: 数值稳定，梯度平滑，同样能保证输出为正。是更健壮的选择。
    *   **补充说明**: `nn.Softplus` 有一个 `beta` 参数（默认为 1.0）控制函数的陡峭程度。虽然理论上可调，但在大模型实践中，调整此参数通常被认为是次要的超参数优化。除非遇到特定的数值问题，否则使用默认值是合理的起点，调整学习率和权重初始化等更为关键。

### 2.4. 与 LlamaFactory 集成点
1.  **模型加载 (`loader.py`)**: 需要引入一个控制参数 (`is_disco`)，让加载器知道何时应加载我们新的 `NormalHead` 模型，何时加载标准的 `ValueHead` 模型。
2.  **模型适配 (`patcher.py`)**: `WithValueHead` 模型需要 `patch_valuehead_model` 函数进行适配处理以保证兼容性。由于我们的 `NormalHead` 模型继承自它，因此也需要确保这个适配过程能正确应用于我们的新模型。

## 3. 代码实现细节 (Bottom-Up)

基于上述设计，我们对以下文件进行了修改或创建：

### 3.1. `src/llamafactory/model/model_utils/xhead.py` (新文件)

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Any
from typing_extensions import override
from trl import AutoModelForCausalLMWithValueHead

class VarianceHead(nn.Module):
    """VarianceHead 类，计算方差"""
    def __init__(self, config, **kwargs):
        # ... (初始化 dropout, 获取 hidden_size)
        self.var = nn.Linear(hidden_size, 1) # 线性层
        self.softplus = nn.Softplus(beta=1.0) # Softplus 激活

    def forward(self, hidden_states):
        # ... (应用 dropout, 确保 dtype 一致)
        var = self.softplus(self.var(output)) # 计算 Softplus(Linear(x))
        return var

class AutoModelForCausalLMWithNormalHead(AutoModelForCausalLMWithValueHead):
    """DiscoRM 模型类，继承自 ValueHead 模型"""
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs) # 初始化父类 (v_head)
        # ... (获取 config)
        self.var_head = VarianceHead(config, **kwargs) # 创建 VarianceHead
        # ... (如果需要，根据 v_head_init_strategy 初始化 var_head 权重)

    @override
    def forward(self, ..., return_past_key_values=False, **kwargs) -> Union[Tuple[torch.FloatTensor, ...], Any]:
        # 不调用 super().forward()，而是重写逻辑
        # ... (处理 kwargs, 调用 self.pretrained_model 获取 base_model_output)
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss # 可能为 None
        # ... (确保 hidden_states 与 head 在同一设备)
        value = self.v_head(last_hidden_state).squeeze(-1)     # 计算均值 μ
        variance = self.var_head(last_hidden_state).squeeze(-1) # 计算方差 σ²
        # ... (确保 lm_logits 是 float32)

        if return_past_key_values:
            return (lm_logits, loss, value, variance, base_model_output.past_key_values)
        else:
            return (lm_logits, loss, value, variance) # 返回值增加了 variance

    @override
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs) # 获取父类 state_dict (含 v_head)
        var_head_state_dict = self.var_head.state_dict(*args, **kwargs)
        for k, v in var_head_state_dict.items():
            state_dict[f"var_head.{k}"] = v # 添加 var_head 参数
        return state_dict
```
*   **关键点**: 继承、新增 `VarianceHead`、使用 `Softplus`、重写 `forward` 返回值增加 `variance`、修改 `state_dict` 添加 `var_head` 参数。

### 3.2. `src/llamafactory/model/loader.py` (修改)

*   **`load_model` 函数**:
    *   增加 `is_disco: bool = False` 参数。
    *   在 `if add_valuehead:` 块内，根据 `is_disco` 的值选择加载 `AutoModelForCausalLMWithNormalHead` 或 `AutoModelForCausalLMWithValueHead`。
*   **导入**: 添加 `from .model_utils.xhead import AutoModelForCausalLMWithNormalHead`。

### 3.3. `src/llamafactory/model/patcher.py` (修改)

*   **`patch_valuehead_model` 函数**:
    *   **作用**: 通过猴子补丁为 `WithValueHead` 模型添加/修改方法，以确保与 Transformers 库的兼容性（如正确保存权重）。
    *   **修改**: 将函数签名和内部类型提示中接受的模型类型从 `AutoModelForCausalLMWithValueHead` 扩展为 `Union["AutoModelForCausalLMWithValueHead", "AutoModelForCausalLMWithNormalHead"]`。
    *   **原因**: 我们的 `NormalHead` 模型继承自 `ValueHead`，这些适配逻辑对它同样适用且必要。

### 3.4. `src/llamafactory/model/model_utils/valuehead.py` (修改)

*   **`load_valuehead_params` 函数**:
    *   仅修改了文档字符串，说明加载的权重可能包含 `var_head` 参数（如果检查点是由 `NormalHead` 模型保存的）。加载逻辑本身不变。

## 4. 兼容性分析与使用注意事项

*   **`forward` 输出变化 (最重要)**: `NormalHead` 模型的 `forward` 方法比 `ValueHead` 多返回一个 `variance`。**注意**: 返回的 `value` (μ) 和 `variance` (σ²) 都是 **token-level** 的张量，形状为 `(batch_size, sequence_length)`。任何直接调用 `forward` 并依赖返回值数量/索引的代码（尤其是 **Trainer** 和 **损失函数**）**必须修改**以处理新增的 `variance` 输出，并且需要实现**从 token-level 到 sequence-level 的聚合逻辑**（例如，取最后一个 token 的值）。
*   **检查点不完全兼容**:
    *   `NormalHead` 检查点包含 `var_head.`，不能被 `ValueHead` 加载（会报 `unexpected keys` 错误）。
    *   `ValueHead` 检查点可被 `NormalHead` 加载（仅加载 `v_head` 和基础模型权重），但 `var_head` 需要重新随机初始化或从其他来源加载。
*   **训练流程适配**:
    *   需要实现 DiscoRM 的损失函数，该函数需要同时接收**聚合后的** `value` (μ) 和 `variance` (σ²)。
    *   需要修改或创建新的 Trainer 来处理 `forward` 新增的 `variance` 输出，执行聚合操作，并将聚合后的 μ 和 σ² 传递给 DiscoRM 损失函数。标准的 `RewardTrainer` 不适用。

## 5. 总结与后续步骤

我们成功设计并实现了 DiscoRM 的基础模型架构 `AutoModelForCausalLMWithNormalHead`，并将其集成到了 LlamaFactory 的模型加载和适配流程中。该实现遵循了最小改动和保持兼容性的原则，巧妙复用了 `trl` 的 `WithValueHead` 结构作为均值头，是一个实用且高效的起点。设计中也认识到 token-level 输出和参数独立性等权衡。

**后续步骤**:
1.  **实现 DiscoRM 损失函数**: 在 `src/llamafactory/train/discorm/loss.py` (建议路径) 中创建损失函数模块，需要处理输入的均值和方差（注意聚合）。
2.  **适配 Trainer**: 创建新的 `DiscoTrainer` 类 (建议名称)，继承或修改现有 Trainer，使其能正确处理 `NormalHead` 的 token-level 输出，执行聚合，并调用 DiscoRM 损失。
3.  **配置与运行**: 更新训练脚本/配置文件，使用 `is_disco=True` 并指定新的 Trainer (`discorm`) 和对应的损失函数。

---

➡️ **下一步**: 查阅 `