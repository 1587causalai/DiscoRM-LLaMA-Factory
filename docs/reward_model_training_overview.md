# LlamaFactory 通用奖励模型训练流程概述

本文档旨在**概述** LlamaFactory 框架中**通用**奖励模型（Reward Model, RM）训练流程的关键组件和步骤，重点关注**标准的 Pairwise Loss** 训练方式。本文档提供一个高层次的理解，为深入特定实现（如 DiscoRM）或进行定制开发打下基础。

**重要提示**: 本文档描述的是 LlamaFactory 中**通用**的、基于输出**标量奖励**的 RM 训练流程。如果您正在寻找关于 **Disco Reward Model (DiscoRM)**（输出奖励分布）的具体实现细节，请参阅：

*   `./DiscoRM_基础模型架构设计与实现.md`
*   (后续可能存在的 DiscoRM 损失、Trainer 等相关文档)

## 1. 核心概念：成对比较 (Pairwise Comparison)

通用 RM 训练的核心思想是让模型学会区分"好"的响应和"坏"的响应。这通常通过**成对比较**实现：
*   **数据**: 训练数据包含成对的响应 (`chosen` 和 `rejected`)，它们都对应同一个初始提示 (prompt)。`chosen` 响应被认为是优于 `rejected` 响应的。
*   **目标**: 训练模型，使其为 `chosen` 响应赋予比 `rejected` 响应更高的**标量奖励分数**。

## 2. 关键组件与参数

RM 训练流程涉及以下关键组件和配置参数：

*   **工作流入口 (`run_rm`)**: 通常位于 `src/llamafactory/train/rm/workflow.py` (注意：示例可能基于 `discorm` 目录，但原理相似)。该函数负责协调整个训练流程。
*   **参数对象**: (与您之前看到的类似)
    *   `model_args`: 定义基础模型 (如 Llama 3)、路径、量化等。
    *   `data_args`: 定义数据集 (需要包含 `chosen` 和 `rejected` 字段)、模板、序列长度等。
    *   `training_args`: 定义标准训练参数 (学习率、批大小、优化器、保存策略等)。
    *   `finetuning_args`: 定义微调参数，其中 **`add_valuehead=True`** 是 RM 训练的关键，指示需要在基础模型上附加一个**价值头 (Value Head)**。
*   **模型 (`AutoModelForCausalLMWithValueHead`)**: 由 `loader.py` 加载，并通过 `finetuning_args.add_valuehead=True` 自动附加一个价值头。这个价值头是一个小型神经网络（通常是线性层），附加在基础语言模型的最后一层隐藏状态之上，用于将高维隐藏状态映射到一个**标量奖励值**。
*   **数据集 (`get_dataset`)**: 从 `data_args` 指定的数据源加载数据，并进行预处理，确保每个样本包含 `chosen` 和 `rejected` 对。
*   **数据整理器 (`PairwiseDataCollatorWithPadding`)**: 将来自数据集的 `chosen` 和 `rejected` 样本对整理成批次。关键操作是将 `chosen` 样本和 `rejected` 样本的 `input_ids` 和 `attention_mask` 在批次维度上拼接起来，形成一个 `(2 * batch_size, seq_len)` 的张量。
*   **训练器 (`PairwiseTrainer`)**: 继承自 Hugging Face `Trainer`，并重写了 `compute_loss` 方法以实现 Pairwise Loss 计算逻辑。

## 3. 标准训练流程步骤

典型的通用 RM 训练流程如下：

1.  **参数解析**: 从命令行或配置文件加载 `model_args`, `data_args`, `training_args`, `finetuning_args`。
2.  **组件加载**: 使用加载的参数：
    *   加载 Tokenizer。
    *   加载基础模型，并根据 `add_valuehead=True` **附加价值头**，得到 `AutoModelForCausalLMWithValueHead` 实例。
    *   加载并预处理成对数据集。
    *   初始化 `PairwiseDataCollatorWithPadding`。
    *   初始化 `PairwiseTrainer`。
3.  **训练循环 (由 `PairwiseTrainer.train()` 驱动)**: 对每个训练批次：
    *   **数据整理**: `PairwiseDataCollatorWithPadding` 从数据集中取出 `batch_size` 个样本对，并将它们整理成包含 `chosen` 和 `rejected` 的批次 (`input_ids`, `attention_mask` 的形状为 `(2 * batch_size, seq_len)`）。
    *   **前向传播**: 在 `PairwiseTrainer.compute_loss` 中，将整个批次 (`2 * batch_size`) 输入到 `model` (`AutoModelForCausalLMWithValueHead`)。
        *   模型（包括价值头）为每个输入序列的每个 token 输出一个**标量奖励预测值** (`values`，形状类似 `(2 * batch_size, seq_len, 1)`)。
    *   **分数提取**: 从 `values` 中提取每个序列**最后一个有效 token** 的奖励值，作为该序列的整体得分。
        *   将 `values` 和 `attention_mask` 拆分成 `chosen` 和 `rejected` 两部分（形状变为 `(batch_size, ...)`）。
        *   利用 `attention_mask` 找到每个序列最后一个非 padding token 的索引。
        *   使用 `gather` 操作提取对应的奖励值，得到 `chosen_scores` 和 `rejected_scores` (形状 `(batch_size,)`)。
    *   **损失计算**: 使用 Pairwise Loss 函数（如 Log Sigmoid Loss: `-logsigmoid(chosen_scores - rejected_scores)`) 计算损失。目标是最大化 `chosen_scores` 和 `rejected_scores` 之间的差值。
    *   **反向传播与优化**: `Trainer` 根据计算出的损失执行反向传播，计算梯度，并使用优化器更新模型参数（包括基础模型和价值头的参数，具体取决于微调策略如 LoRA 或 Full-tuning）。
4.  **评估与保存**: 根据 `training_args` 中的设置，定期进行评估（如果提供了评估数据集）并保存模型检查点。

## 4. 总结

LlamaFactory 提供了一套用于训练通用奖励模型的流程，其核心是利用成对数据 (`chosen`/`rejected`) 和一个附加的价值头 (Value Head) 来训练模型区分不同响应的优劣。通过最大化 `chosen` 和 `rejected` 响应之间的预测奖励分数差异，模型学会了评估生成的质量。

**要点回顾**: 成对数据、附加价值头、Pairwise 数据整理器、Pairwise Trainer 中的特定损失计算逻辑。

---

**下一步**: 如果您对输出奖励**分布**而不是标量值的 DiscoRM 感兴趣，请查阅 `./DiscoRM_基础模型架构设计与实现.md` 及后续相关文档。 