# DiscoRM-LLaMA-Factory

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 实现的 Disco Reward Modeling 框架。

## 项目简介

Disco Reward Modeling (DiscoRM) 是一种新型奖励建模方法，专注于提升大型语言模型 (LLMs) 的对齐能力。本项目基于 LLaMA-Factory 构建，为研究人员和开发者提供了一个高效、灵活的框架来实现和测试 DiscoRM。

## 主要特性

- 基于 LLaMA-Factory 的完整 Disco Reward Modeling 实现
- 支持多种预训练模型，包括但不限于 Qwen1.5、LLaMA 系列等
- 兼容 Reward Bench 评估框架
- 提供高效的训练和评估流程

## 快速开始

详细使用文档正在编写中...

## 许可证

本项目遵循 Apache-2.0 许可证

## 致谢

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 开发，感谢原作者的杰出工作。



- 最简单的代码实现 disco-DPO with LLamaFactory



## 核心贡献

reward score 是一个随机变量, 并且随机性来自于不同的个体.  举个例子, 苹果好吃还是香蕉好吃, 不同的人有不同的答案, 对于这样没有标准答案的问题, reponse "苹果" 的奖励和 reponse "香蕉" 的奖励是不同的, 因此他应该具备非常大的方差. 即使评分差异非常大, 加大方差, 我们 disco-Reward 可以让其没有偏好. 
