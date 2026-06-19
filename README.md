# 稀疏推理优化科研项目

武汉大学 · 李清安老师课题组 · LLM 推理优化方向

## 目录结构

```
sparse-research/
├── experiments/       # 实验脚本（按编号：01, 02, 03...）
├── src/               # 可复用的工具代码
├── data/              # 数据集、校准文本
├── outputs/           # 输出图表、实验结果
├── papers/            # 论文 PDF（在 ~/comsoft 下，可软链接过来）
└── setup_5060.sh      # 5060 笔记本一键环境脚本
```

## 环境

| 机器 | 用途 | 环境 |
|------|------|------|
| 轻薄本（当前） | 写代码、轻量实验、画图 | PyTorch CPU |
| 5060 笔记本（宿舍） | GPU 密集型实验 | PyTorch CUDA 12.6 |

激活环境：`source .venv/bin/activate`

**注意**：如果你配了代理（如 Clash），运行前需要先 `unset ALL_PROXY all_proxy`，
否则 httpx 会报错。或者把 ALL_PROXY 的 socks 改成 http。

## 当前进度
- [x] 01: 激活稀疏性初探 — 跑通，看到 SiLU 下天然稀疏性很低（~5-10% near-zero）
- [ ] 02: 复现 CHESS 逐通道阈值剪枝

## 入门路线

### 第一步：亲眼看到稀疏性
```bash
source .venv/bin/activate
cd experiments
python 01_activation_sparsity.py
```
会用 LLaMA-3.2-1B 跑一次推理，抓每层 FFN 的激活值，画分布图。
CPU 就能跑，不需要 GPU。

### 第二步：带着问题读论文
通读何俊辉论文的第 3 章（CHESS），重点关注：
- 为什么要把剪枝建模成优化问题？
- 逐通道阈值是怎么算的？
- 为什么只对部分投影层做剪枝？

### 第三步：复现 CHESS 核心算法
在 experiments/ 下新建 02_chess_*.py，逐步复现。

## 两台机器同步代码

```bash
# 在 GitHub 建私有仓库，然后：
git init
git add -A
git commit -m "init"
git remote add origin git@github.com:你的用户名/sparse-research.git
git push -u origin main

# 另一台机器上：
git clone git@github.com:你的用户名/sparse-research.git
cd sparse-research
bash setup_5060.sh  # 5060 上跑这个安装 GPU 环境
```

## 参考论文

- 何俊辉 — 基于动态稀疏性的大模型推理优化研究
- 何宇昕 — 面向移动端大模型推理的异构计算与权重管理优化研究
- 黄宇帆 — 显存受限环境下的 DNN 推理优化方法研究
