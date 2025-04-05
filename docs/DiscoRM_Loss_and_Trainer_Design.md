# DiscoRM：损失函数与 Trainer 设计

**文档目的：** 本文档是 Disco Reward Model (DiscoRM) 集成至 LlamaFactory 系列文档的第二部分，承接定义了模型结构的 `discorm_base_model_design.md`。本文档聚焦于以下两个核心方面：
1.  `AutoModelForCausalLMWithNormalHead` 模型所需的特定损失函数。
2.  为了计算此损失并正确处理模型输出，需要对 LlamaFactory 的 `Trainer` 进行的必要适配。

**目标读者：** 需要理解 DiscoRM 损失函数原理、实现细节，以及 LlamaFactory 训练流程如何为支持 DiscoRM 而修改的开发者。

**与其他文档的关系：**
*   **基础：** `discorm_base_model_design.md` (定义了 `NormalHead` 模型)。
*   **后续：** 关于准备数据和配置 DiscoRM 训练运行的实践指南。

---

## 1. DiscoRM 损失函数

**背景：** 标准的奖励模型 (Reward Model, RM) 通常预测一个单一的标量奖励值。而 DiscoRM (基于 `AutoModelForCausalLMWithNormalHead`) 则不同，它预测与每个"提示-生成对"(prompt-completion pair) $(x, y)$ 相关联的奖励的**均值** ($\mu$) 和**方差** ($\sigma^2$)。这种设计捕捉了奖励的不确定性，因此需要一种不同的损失函数，该函数基于一个响应优于另一个响应的概率。

### 1.1. 数学原理
*   **核心假设：**
    1.  **奖励服从正态分布：** $R(x, y; \psi) \sim \mathcal{N}(\mu_\psi(x, y), \sigma^2_\psi(x, y))$，其中均值 $\mu_\psi$ 和方差 $\sigma^2_\psi$ 由 `NormalHead` 模型 $f_\psi$ 预测。
    2.  **条件独立性：** 对于一个偏好对 $(y_w, y_l)$ (其中 $y_w$ 是更优的响应，$y_l$ 是较差的响应)，给定提示 $x$，它们的奖励 $R_w = R(x, y_w)$ 和 $R_l = R(x, y_l)$ 是条件独立的。
*   **偏好概率推导：** 基于上述假设，两个奖励的差值 $R_d = R_w - R_l$ 也服从正态分布：$R_d \sim \mathcal{N}(\mu_w - \mu_l, \sigma^2_w + \sigma^2_l)$。我们关心的是 $y_w$ 优于 $y_l$ 的概率，即 $P(R_w > R_l)$。利用正态分布的性质，该概率可以表示为：
    \[ P_\psi(y_w \succ y_l | x) = \Phi\left(\frac{\mu_\psi(x, y_w) - \mu_\psi(x, y_l)}{\sqrt{\sigma^2_\psi(x, y_w) + \sigma^2_\psi(x, y_l)}}\right) \]
    其中 $\Phi$ 是标准正态分布的累积分布函数 (CDF)。
*   **损失函数 (负对数似然 NLL)：** 训练的目标是最大化观测到的偏好数据集 $D = \{(x, y_w, y_l)\}$ 的似然，等价于最小化其负对数似然损失：
    \[ \mathcal{L}_{RM}(\psi; D) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \Phi\left(\frac{\mu_w - \mu_l}{\sqrt{\sigma^2_w + \sigma^2_l}}\right) \right] \]

### 1.2. 在 `PairwiseTrainer` 中的实现
核心的 DiscoRM 损失计算逻辑被直接实现在我们修改后的 `PairwiseTrainer` 的 `compute_loss` 方法中 (位于 `src/llamafactory/train/discorm/trainer.py`)。

*   **计算步骤：**
    1.  **获取模型输出：** 调用 `model.forward(**inputs)` 返回包含 `value` (即均值 $\mu$) 和 `variance` (即方差 $\sigma^2$) 的输出。
    2.  **提取偏好对数据：** 对批次中的每个偏好样本 $(x, y_w, y_l)$，分别提取对应的 $(\mu_w, \sigma^2_w)$ 和 $(\mu_l, \sigma^2_l)$。这通常通过张量分割 (split) 和基于序列结束位置的收集 (gather) 操作完成。
    3.  **计算核心组件：** 计算均值差 $\Delta\mu = \mu_w - \mu_l$ 和方差和 $\Sigma^2 = \sigma^2_w + \sigma^2_l$。
    4.  **数值稳定性 (分母处理)：** 在计算 $\sqrt{\Sigma^2}$ 之前，应用 `torch.clamp(\Sigma^2, min=variance_epsilon)`。为了简化配置，这个小的正数 `variance_epsilon` (例如 `1e-6`) 被**直接硬编码**在 `compute_loss` 方法内部，用于防止除以零或极小的数。
    5.  **计算 CDF 输入：** 计算 $z = \frac{\Delta\mu}{\sqrt{\Sigma^2 + \epsilon}}$。
    6.  **数值稳定性 (Log CDF 计算)：** 使用 `torch.distributions.Normal(0, 1).log_cdf(z)` 直接且稳定地计算 $\log \Phi(z)$，避免了 $\log(0)$ 的问题。
    7.  **最终损失：** 将批次内所有样本的 $-\log \Phi(z)$ 值取平均。
*   **超参数：** 损失函数本身不引入新的超参数。用于数值稳定性的 `variance_epsilon` 已作为内部常量处理。

---

## 2. 适配训练流程

**背景：** 为了使用 DiscoRM 损失函数训练 `NormalHead` 模型，标准的 Hugging Face `Trainer` 需要进行修改。我们需要确保：1) 正确计算 DiscoRM 损失；2) 在评估过程中正确处理模型独特的输出（均值和方差）；3) 模型检查点能够被正确地保存。

### 2.1. 适配 `PairwiseTrainer`
*   **策略选择：** 我们选择**修改**现有的 `PairwiseTrainer` (位于 `src/llamafactory/train/discorm/trainer.py`)，而不是创建一个全新的 `DiscoRMTrainer` 类。这种方式保持了与 LlamaFactory 其他部分的兼容性，并遵循了我们偏好的渐进式、保守的开发风格。
*   **关键修改点：**
    *   **重写 `compute_loss` 方法：** 这是最核心的改动。该方法被完全重写，以实现 1.2 节中描述的 DiscoRM NLL 损失计算逻辑。它现在预期模型会返回 `value` 和 `variance`。
    *   **重写 `prediction_step` 方法：** 该方法被调整以处理 `NormalHead` 模型在评估阶段产生的更丰富的输出。它调用 `compute_loss(..., return_outputs=True)` 来获取包含均值和方差的预测结果，并将这个元组 (`chosen_mean, rejected_mean, chosen_var, rejected_var`) 传递给后续的指标计算步骤。
    *   **(可选优化) `save_predictions` 方法：** 更新该方法，使其在检测到方差信息可用时，能够选择性地将方差预测与均值预测一同保存。

### 2.2. 处理评估指标 (`ComputeAccuracy`)
`ComputeAccuracy` 类 (位于 `src/llamafactory/train/discorm/metric.py`) 被用作 `Trainer` 的 `compute_metrics` 函数。
*   **兼容性：** 它**始终**会基于预测的均值 (`chosen_mean`, `rejected_mean`) 计算标准的奖励模型 `accuracy` 指标。这确保了与标准 RM 训练的兼容性，并提供了基本的评估结果。
*   **可选的方差指标：** 如果 `prediction_step` 提供了方差信息 (即输入 `eval_preds.predictions` 的长度大于等于 4)，`ComputeAccuracy` 会**额外**计算并记录 `mean_chosen_variance` 和 `mean_rejected_variance` 这两个指标，以供分析。

### 2.3. 确保检查点正确保存 (针对之前的疑惑点)
*   **问题描述：** 标准的 `Trainer.save_model()` 方法在保存带有自定义头部的模型（如 `WithValueHead` 或我们的 `NormalHead`）时，会将基础模型参数和所有头部参数（`v_head`, `var_head`）合并存储在一个大文件中。这种格式不标准，给后续加载模型带来了困难。
*   **解决方案：** 利用 `TrainerCallback` 回调机制介入保存过程。`FixValueHeadModelCallback` (位于 `src/llamafactory/train/callbacks.py`) 就是为此设计的。
*   **工作流程详解 (Top-Down)：**
    1.  **Trainer 保存：** `Trainer` 首先按照默认行为，保存一个临时的、包含所有参数的合并检查点文件。
    2.  **Callback 触发：** `on_save` 事件被触发，`FixValueHeadModelCallback` 调用核心函数 `fix_valuehead_checkpoint`。
    3.  **`fix_valuehead_checkpoint` 执行：**
        a.  **加载合并状态：** 读取 Trainer 刚保存的那个大文件中的所有参数。
        b.  **参数分离：** 遍历所有参数，将基础模型/适配器参数放入 `decoder_state_dict`，将头部参数（**关键：现在能同时识别 `v_head.` 和 `var_head.` 前缀**）放入 `head_state_dict`。
        c.  **分别保存：**
            *   使用 `decoder_state_dict` 调用 `model.pretrained_model.save_pretrained()` 保存基础模型/适配器部分（符合标准格式）。
            *   将包含 `v_head` 和 `var_head` 参数的 `head_state_dict` 保存到一个**单独的** `adapter_v_head.bin` (或 `.safetensors`) 文件中。
        d.  **清理：** 删除临时的合并检查点文件。
*   **当前状态：** 我们已经仔细检查并更新了 `fix_valuehead_checkpoint` 的逻辑，确保它能够正确处理 `var_head` 参数，并将其与 `v_head` 一同保存到 `adapter_v_head.*` 文件中。这保证了 DiscoRM 模型检查点的完整性和可用性。

---

## 3. 在 LlamaFactory 中集成并启用 DiscoRM

启用 DiscoRM 训练流程非常简单，只需要通过一个标志位和工作流中的少量设置即可完成。

### 3.1. `disco` 标志位
*   我们在 `FinetuningArguments` (位于 `src/llamafactory/hparams/finetuning_args.py`) 中增加了一个布尔类型的参数 `--disco` (默认值为 `False`)。
*   用户可以通过命令行传递 `--disco` 或在配置文件中设置 `disco: true`。
*   这**一个标志位**充当了启用 DiscoRM 模式的总开关。

### 3.2. 工作流激活 (`run_discorm`)
主要的训练脚本入口点 `run_discorm` (位于 `src/llamafactory/train/discorm/workflow.py`) 会利用 `finetuning_args.disco` 的值来：
*   **加载正确的模型：** 调用 `load_model(..., is_disco=finetuning_args.disco)`。如果 `disco` 为 `True`，则加载 `AutoModelForCausalLMWithNormalHead`；否则加载标准的 `WithValueHead` 模型。
*   **实例化正确的 Trainer：** 始终实例化我们修改后的 `PairwiseTrainer`。该 Trainer 能够根据加载的模型类型及其输出，自动处理标准 RM 损失或 DiscoRM 损失。
*   **实例化指标计算器：** 实例化 `ComputeAccuracy`，它能处理标准准确率和可选的方差指标。

### 3.3. 模块导出
必要的组件 (`PairwiseTrainer`, `ComputeAccuracy`) 已经通过 `src/llamafactory/train/discorm/__init__.py` 文件从 `discorm` 模块中导出，确保它们可以在 LlamaFactory 的其他部分被正确引用。

---

## 4. 总结

DiscoRM 的集成涉及定义一个新的、基于概率偏好的损失函数，并适配 `PairwiseTrainer` 来计算该损失、正确处理评估过程以及通过回调机制确保检查点的正确保存。整个流程的设计旨在简化用户操作，通过单一的 `disco` 标志即可在 LlamaFactory 框架内启用。

➡️ **后续步骤：** 准备合适的偏好数据集，并使用 `--disco` 标志配置并开始实际的训练运行。 