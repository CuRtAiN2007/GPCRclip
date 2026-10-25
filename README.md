# GPCR-CLIP 最新版提交包

版本：`latest_20260924`

## 模型谱系与父权重

SMO Class F 特化模型的训练谱系为：

```text
DrugCLIP 原版多折权重
  -> 逐 fold GPCR LoRA 微调
  -> GPCR 微调 fold-0 父模型
  -> shared LoRA + A/B1/B2/C/F 类别适配
  -> SMO Class F 特化模型
```

本包包含 GPCR 微调 fold-0 父模型：

`models/base/gpcr_official_expneg_eval_fold0_merged.pt`

该文件约 379 MiB，SHA-256 为 `f43b432b36c28e2409d0239892a36431a707b0dfcf42df4ef6972cda3274a954`。`predict.py` 默认读取该路径，
可运行 SMO Class F 特化模型的推理。

完整方法说明见 `docs/METHODS_FULL.md`。

## 模型层级评测（同口径补充数据）

原版 DrugCLIP、GPCR 微调 fold-0 父模型、最终 Class F 特化模型和 LOCO-F 对照，
在同一套拆分、同一构象生成流程和同一 per-pocket median/MAD z-score 下评测：

| 评价 | 原版 DrugCLIP | GPCR 微调 fold-0 父模型 | 最终 Class F 特化模型 | LOCO-F 对照 |
| --- | ---: | ---: | ---: | ---: |
| Class F 留出 AUC（520 正 / 17 负） | 0.3781 | 0.5436 | 0.8261 | 0.3382 |
| SMO WT holdout AUC（201 正 / 11 负） | 0.7094 | 0.4111 | 0.4315 | 0.6766 |
| known-drug holdout AUC（12 正 / 7 负） | 0.2738 | 0.3095 | 0.0357 | 0.2500 |

逐条评测记录、复现命令和限制见 `results/model_benchmark/`。


这是服务器端最新模型和代码的整理结果：

`9 月 6 日 GPCR fold-0 merged 模型 -> class-balanced LoRA/adapters -> SMO 初筛 -> WT/D473H docking -> MM-GBSA -> short MD`

最终排序模型是 class-balanced `fold_0_seed42_v1/epoch_005` 检索模型。D473H 不是该模型的训练标签，只作为探索性物理证据记录。

## 目录说明

```text
README.md
requirements.txt
environment.yml
src/                  # 模型、训练、筛选与物理计算代码
data/                 # 数据清单、来源说明和可运行示例 LMDB
models/               # 最新适配器及 9 月 6 日父模型
notebooks/            # 可执行快速开始 Notebook
train.py              # class-balanced 训练入口
screen.py             # 从 SMILES/PDB 生成标准结果表
predict.py            # 模型推理入口
results/              # 标准化结果和完整物理筛选结果
logs/                 # 训练、验证和队列日志
docs/                 # Model Card、数据溯源、许可和复现说明
```

## 运行环境

参考服务器环境为 Ubuntu 22.04、Python 3.13.12、PyTorch 2.13.0+cu130。推荐使用：

```bash
conda env create -f environment.yml
conda activate gpcrclip
```

也可以根据 `requirements.txt` 安装固定版本依赖。

## 可运行示例

包内包含一个口袋、五个可成功预处理的示例分子，以及对应 LMDB：

```bash
python predict.py   --molecules-lmdb data/demo/molecules.lmdb   --targets-lmdb data/demo/targets.lmdb   --device cuda:0   --output results/demo_results.csv
```

参考服务器可使用单张 RTX 4090 运行。使用 `--device cpu` 也可执行，但速度较慢。

## 从 SMI 文件开始筛选

```bash
python screen.py   --smiles data/demo/molecules.smi   --pdb data/demo/pdb/SMO_WT_5L7I_A_withVIS.pdb   --ligand VIS   --target SMO_WT   --state active   --device cuda:0   --output results/results.csv
```

`screen.py` 会自动生成分子 LMDB、按配体定义提取口袋、加载 class-balanced 模型并写出 UTF-8 结果表。

## 训练入口

原始训练 LMDB 不随包分发。取得溯源文档中说明的数据后，可运行：

```bash
python train.py   --pairs-lmdb /path/to/pairs.lmdb   --split-manifest /path/to/split_manifest.json   --base-checkpoint models/base/gpcr_official_expneg_eval_fold0_merged.pt   --run-kind fold   --held-out fold_0   --output-dir checkpoints/class_balanced_fold0
```

源服务器训练参数为：5 个 epoch、随机种子 42、batch size 16、梯度累积 4、每个 epoch 3000 步。

训练参数固化文件为 `configs/train_class_balanced_fold0_seed42_v1.yaml`。

## 最终物理筛选证据

- 标准化短名单：`results/results.csv`
- 完整可排序候选表：`results/final_candidates_with_md.csv`
- Top-200 探索性候选表：`results/MCE_STAGE6_TOP200.csv`
- 第一梯队 docking 模型：`results/structures/mce_first_tier_models/`
- Top-200 受体-配体结构：`results/structures/mce_top200/`
- 排序权重与不确定性：`results/RECOMMENDATION_WEIGHTS_20260919.md`、`results/final_candidates_with_md_summary.json`
- MM-GBSA 与 short MD 汇总：`results/post_docking/`

Stage 5C 的 D473H 方向门控结果为 `FAIL`。本包任何记录均不得据此声明已验证的耐药逆转机制。

## 主要限制

检索分数是初筛分数，不是亲和力数值。同口径评测中 Class F 留出 AUC 从原版 DrugCLIP 的 0.3781 提升到最终 Class F 特化模型的 0.8261，但 SMO WT holdout 上原版 DrugCLIP 反而最高（0.7094），说明收益集中在 Class F 同类判别而非通用 GPCR 排序；负样本数量少（17 / 11 / 7），这些数值只能作为 sanity check。最终优先级仍取决于 docking、MM-GBSA、short-MD 稳定性和实验验证。
