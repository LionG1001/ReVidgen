# RBench MUSA 适配：第一版 review

本地仓库由官方 [DAGroup-PKU/ReVidgen](https://github.com/DAGroup-PKU/ReVidgen) clone，基线 `b03df27f0376faa148dcd8cd620a1989a32ca979`，分支 `codex/20261110-rbench-musa`。本版用于评审工程接入；尚未进行真实 MUSA 扩展编译、72B judge 加载或视频评分。

## 改动与 review 顺序

| 文件 | 作用 |
| --- | --- |
| `scripts/musa/run.py` | 在 scorer 和 Transformers/Accelerate 导入前设置 MUSA 环境并加载 torchada；设备不可用时报错，随后用 runpy 执行上游入口 |
| `environment/musa/requirements-{vlm,ops}.txt`、`scripts/musa/install.py` | 两套依赖、保留已有 GPU 栈的约束、临时副本构建扩展、安装记录 |
| `scripts/rbench_eval_{5tasks,4embodiments}_musa.sh` | 复用上游命令参数，以本地 Qwen 评分；四类机器人拆为 VQA / motion 两阶段 |
| `scripts/musa/prepare_inputs.py` | 为同一视频创建数字命名的 VQA 视图与带类别前缀的 motion 视图；保留完整 prompt 索引；防止复用其他输入的旧图网格 |
| `scripts/musa/summarize.py` | 检查所有所选样本评分有效且覆盖一致，在副本中规范化结果 ID，再调用上游汇总函数；原始结果保留 |

上游源码仅有以下工程调整：8个 VLM scorer 的 `write_video` 改为在编码函数内导入（常规评分只读视频）；motion amplitude 使用安装后的标准 `groundingdino` 导入路径；GroundingDINO/SAM2 的 setup 增加受 `RBENCH_USE_TORCHADA=1` 控制的早期导入；GroundingDINO 两处 `AT_DISPATCH` 改用 `value.scalar_type()` 以兼容新 PyTorch。公式、prompt、抽帧位置和模型推理调用参数保留上游实现。默认直接执行原 scorer 时不加载 torchada。

```bash
git diff origin/main...HEAD -- eval pkgs scripts environment tests
git status --short
# PR中的所有新增/修改文件均可通过上述diff查看。
python3 -m unittest discover -s tests -v
```

## 依赖与安装

使用容器默认 Python 3.10–3.12，先安装相互匹配的厂商 torch、torch_musa、torchvision 及 MUSA 工具链。安装器读取这些包的现有版本生成精确约束，也保留已安装的 torchaudio/triton；普通 Python 依赖正常解析。不会安装官方 README 中的 CUDA torch 2.5.1，也不创建 venv。

VLM 与 ops 必须在**不同容器的默认 Python**中安装：VLM 使用 Transformers 4.57.1 / Accelerate 1.10.1；Q-Align 需要 Transformers 4.36.1 / Accelerate 0.21.0。共用本仓库和输入/结果挂载，不共用 Python site-packages。两者均固定 torchada 0.1.108；首次导入可能 JIT 编译其兼容算子，因此配置包含 ninja/setuptools 等构建依赖。

```bash
# VLM 容器；--dry-run 只输出安装命令，不安装。
python3 scripts/musa/install.py vlm --dry-run
python3 scripts/musa/install.py vlm
python3 scripts/musa/run.py --check

# 另一个 ops 容器；须具备 MUSA 编译器及系统 C++ 构建工具。
python3 scripts/musa/install.py ops --dry-run
python3 scripts/musa/install.py ops
python3 scripts/musa/run.py --check
```

安装器在 `.musa-install/<唯一目录>/` 保存约束、安装清单和 freeze；ops 将 `pkgs/` 复制到该目录后构建。torchada 会在构建期间原地转写 C++ 源文件，因此仅构建副本。vendored 包以 `--no-deps --no-build-isolation` 安装，运行依赖由 ops 文件明确列出，避免其元数据重新安装 CUDA 相关包。GroundingDINO/SAM2 扩展导入失败会使安装失败。

ops 的依赖清单参考 [NVIDIA RBench 配方](https://github.com/NVIDIA/cosmos-framework/blob/7306dc9c2f99ffecc065006c9813482f70f030e9/.agents/skills/evaluation/cosmos3/generator/rbench/setup_rbench_embodiment_scorer.sh)，保留 Q-Align 的旧 Transformers/PEFT。其运行子集不安装训练/服务包或 bitsandbytes，RBench 使用普通 FP16 路径；为 Python 3.12 改用 scikit-learn 1.5.2、sentencepiece 0.2.x，因此 vendored Q-Align 的严格元数据会与运行子集有差异。真实模型加载仍须核验。

**架构限制：** Decord 0.6.0 的 PyPI wheel 面向 x86_64，当前 ARM64 开发机的 ops 解析会在此失败。x86_64 MUSA 是本版 ops 的目标；ARM64 若需要实际运行 motion，需要另外构建并核验同版本 CPU Decord。本版未用其他解码器替换官方实现。

## 准备输入

先使用 Cosmos 项目的统一推理栈生成视频，再交给 RBench；本仓库不负责加载或生成 Cosmos ckpt。各候选固定 judge 模型、prompt、采样参数、输入集合及 seed。视频放为：

```text
<RBENCH_VIDEO_ROOT>/
  common_manipulation/videos/0001.mp4
  single_arm/videos/single_arm_0001.mp4
  ...
<RBENCH_PROMPT_ROOT>/
  common_manipulation_prompts.json
  single_arm_prompts.json
  ...
```

两种视频命名均可；helper 按原索引创建所需视图。prompt 使用完整、按原索引排列的类别 JSON，不能把抽选后的16例重排成1–16。子集可以只放部分视频。每个 ckpt/EMA/seed 使用独立 `RBENCH_MODEL_NAME`，结果根位于可写挂载；原视频只读，通过软链接访问。视图记录 prompt 哈希及视频路径/大小/mtime；同一标签下输入发生变化会要求换标签，避免沿用旧网格。

Qwen judge 使用本地完整 `Qwen2.5-VL-72B-Instruct` 资产；运行时 `device_map="auto"` 仍由上游及 Accelerate 分配可见卡。ops 还需要官方 README 列出的 BERT、GroundingDINO、SAM2、CoTracker、Q-Align 资产，按原布局放到本仓库 `checkpoints/` 或建立只读软链接。本版不自动下载模型、数据或调用外部 API。

```bash
export RBENCH_MODEL_NAME=h20_iter500_ema_seed0
export RBENCH_VIDEO_ROOT=/path/to/generated/h20_iter500_ema_seed0
export RBENCH_PROMPT_ROOT=/path/to/RBench/prompts
export RBENCH_RESULTS_ROOT=/path/to/writable/results
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-72B-Instruct

# 先预览。预览不会导入 torchada、访问 GPU、创建输入视图或执行 scorer。
RBENCH_DRY_RUN=1 bash scripts/rbench_eval_5tasks_musa.sh

# VLM 容器：5类任务、4类机器人VQA。
bash scripts/rbench_eval_5tasks_musa.sh
bash scripts/rbench_eval_4embodiments_musa.sh

# ops 容器，使用相同源码/结果挂载及上述 model/video/prompt/results 设置。
RBENCH_STAGE=motion bash scripts/rbench_eval_4embodiments_musa.sh
```

探索性子集可设置 `RBENCH_TASKS="common_manipulation"`、`RBENCH_ROBOTS="single_arm"`。默认5类任务/4类机器人；`PYTHON_BIN`可显式指向容器默认 Python。选择可见 GPU 时在进程启动前设置 `MUSA_VISIBLE_DEVICES`。

## 汇总与完整性

上游 VQA、motion、汇总脚本使用的 ID 格式不同：`0001.jpg`、`single_arm_0001` 和数字 ID。以下 helper 在新目录中复制并规范化 ID，再调用上游 `process_model` / `process_single_model` 等函数。它拒绝负分、非数值、重复/缺失样本和 VQA/motion 覆盖不一致，避免上游汇总忽略失败记录后仍产出均值。

```bash
# CPU 汇总，在任一已安装 pandas 的评测容器执行。
python3 scripts/musa/summarize.py tasks \
  --results-root "$RBENCH_RESULTS_ROOT" --model-name "$RBENCH_MODEL_NAME" \
  --output-dir "$RBENCH_RESULTS_ROOT/summary/$RBENCH_MODEL_NAME/tasks"

python3 scripts/musa/summarize.py embodiments \
  --results-root "$RBENCH_RESULTS_ROOT" --model-name "$RBENCH_MODEL_NAME" \
  --output-dir "$RBENCH_RESULTS_ROOT/summary/$RBENCH_MODEL_NAME/embodiments"
```

子集时用 `--categories common_manipulation` 或 `--categories single_arm`。`coverage.json`保存实际样本数；该结果是所选样本/类别的上游指标，不能直接称650例全量 RBench 总分。没有实现新的综合分数或替换评分公式。

## 当前验证范围

- 16项 CPU 工程测试：导入顺序/失败行为、安装预演、输入索引与视图、两个 launcher 预演、真实上游公式的合成结果汇总及 scorer AST 一致性。
- Python 编译、Bash 语法、`git diff --check`和不接触GPU的CLI帮助已检查。
- VLM 依赖解析使用 pip dry-run；未实际安装或替换当前 Python 的包。ops 的架构解析状态见工作区交付记录。
- **未验证：** MUSA/C++扩展编译和导入、真实视频解码、72B judge 加载/推理、跨CUDA/MUSA分数对齐、完整650例评测。运行前先在目标容器完成安装、backend check 和1例评分。
