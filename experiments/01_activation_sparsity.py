"""
实验 01：LLM 激活稀疏性初探
目标：加载一个小模型，跑一次推理，抓出 FFN 层的激活值，画分布图，
      亲眼看到"大量激活值堆在零附近"这个现象。

这个脚本 CPU 就能跑，不需要 GPU。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ============================================================
# 配置
# ============================================================
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# 0.5B 参数，SiLU 激活，公开下载，GPU 上秒跑
# 后面可以换 "meta-llama/Llama-3.2-1B"（需要 HF token）

PROMPT = "The capital of France is Paris. The capital of Germany is"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 加载模型
# ============================================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",          # 5060 GPU 模式
    dtype=torch.float16,         # FP16，显存省一半
)
model.eval()
print("Model loaded.\n")

# ============================================================
# 2. 注册 hook，抓激活值
# ============================================================
activations = {}

def make_hook(name):
    """创建一个 hook，把指定层的输出保存到 activations 字典里"""
    def hook(module, inputs, output):
        # output 是 tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            activations[name] = output[0].detach().float().cpu()
        else:
            activations[name] = output.detach().float().cpu()
    return hook

# 在每一层的 FFN（MLP）后面挂 hook
num_layers = model.config.num_hidden_layers
print(f"Model has {num_layers} layers.\n")

for layer_idx in range(num_layers):
    mlp = model.model.layers[layer_idx].mlp
    mlp.register_forward_hook(make_hook(f"layer_{layer_idx}_ffn"))

# ============================================================
# 3. 跑推理
# ============================================================
print(f"Running inference with prompt: '{PROMPT}'")
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

print(f"Inference done. Captured {len(activations)} layers' activations.\n")

# ============================================================
# 4. 分析：画每层 FFN 激活值的分布
# ============================================================
# 抽几层代表性的来画
sample_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, layer_idx in enumerate(sample_layers):
    key = f"layer_{layer_idx}_ffn"
    act = activations[key].flatten().numpy()

    # 只看 [-3, 3] 区间，因为异常值很少
    act_clipped = act[(act > -3) & (act < 3)]

    # 统计零附近的比例（|x| < 0.05 视为"接近零"）
    near_zero_ratio = (np.abs(act) < 0.05).mean()

    ax = axes[i]
    ax.hist(act_clipped, bins=200, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f"Layer {layer_idx} FFN\n{near_zero_ratio*100:.1f}% near zero")
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")

# 隐藏多余子图
axes[-1].set_visible(False)

fig.suptitle("Qwen2.5-0.5B FFN Activation Distributions", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_activation_distribution.png", dpi=150)
print(f"Saved: {OUTPUT_DIR / '01_activation_distribution.png'}")

# ============================================================
# 5. 全层稀疏性统计
# ============================================================
print("\n=== Sparsity by Layer ===")
print(f"{'Layer':<8} {'Near-Zero %':<15} {'Mean Abs':<12} {'Std':<12}")
print("-" * 47)

for layer_idx in range(num_layers):
    key = f"layer_{layer_idx}_ffn"
    act = activations[key].flatten().numpy()
    near_zero = (np.abs(act) < 0.05).mean() * 100
    mean_abs = np.abs(act).mean()
    std = act.std()
    print(f"{layer_idx:<8} {near_zero:<15.2f} {mean_abs:<12.4f} {std:<12.4f}")

# ============================================================
# 6. 全层稀疏性对比图
# ============================================================
near_zero_ratios = []
for layer_idx in range(num_layers):
    key = f"layer_{layer_idx}_ffn"
    act = activations[key].flatten().numpy()
    near_zero_ratios.append((np.abs(act) < 0.05).mean() * 100)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(num_layers), near_zero_ratios, color='steelblue', edgecolor='white')
ax.set_xlabel("Layer Index")
ax.set_ylabel("% of Activations Near Zero")
ax.set_title("Activation Sparsity vs Layer Depth\n(Qwen2.5-0.5B)")
ax.axhline(y=np.mean(near_zero_ratios), color='red', linestyle='--',
           label=f"Mean: {np.mean(near_zero_ratios):.1f}%")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "01_sparsity_by_layer.png", dpi=150)
print(f"\nSaved: {OUTPUT_DIR / '01_sparsity_by_layer.png'}")
print("Done!")
