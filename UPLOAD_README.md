# 参赛代码上传说明（区赛）

本 ZIP 为平台上传用的参赛代码包。

- 未包含音频、视频或图片文件。
- 已按要求完成匿名化处理，包内不含任何可识别身份的文字、照片或标识。
- 文件根目录直接包含 `README.md`、`requirements.txt`、`src/`、`models/`、`data/`、`results/` 等工程内容。
- 不包含额外的最外层包装文件夹。
- 保留工程运行必需的内部子目录；这是 Python 包导入和模型加载所必需的。
- 单文件小于 500 MB。

模型说明：

- DrugCLIP 原版多折权重：`weights/model_weights/6_folds/fold_0.pt` 至 `fold_5.pt`，未随包分发。
- GPCR 微调 fold-0 父模型：`models/base/gpcr_official_expneg_eval_fold0_merged.pt`，已随包分发。
- SMO Class F 特化模型：`models/components/`，已随包分发。

完整方法见 `docs/METHODS_FULL.md`。
