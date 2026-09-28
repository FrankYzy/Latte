# FFS 训练基线结果（8 × H200）

本目录保存 `baseline` 分支上 Latte FaceForensics（FFS）无条件视频生成训练的基线数据，用于与其他训练框架对比 loss 收敛、训练速度和显存占用。代码改动和运行方法见仓库根目录 [`CHANGELOG.md`](../../CHANGELOG.md)。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `run_10k/` | 1 万步基线：`step_metrics.jsonl`（逐步记录，10013 行）、`log.txt`、`config.yaml`（运行时保存的完整配置）、`loss_curve.png` |
| `run_1k/` | 1000 步短程运行：同上，`step_metrics.jsonl` 为 1000 行 |
| `samples_10k_model/` | 1 万步 checkpoint 的非 EMA 权重生成的视频 `sample_*.mp4`（16 帧，256×256，8 fps）及其全部帧拼图 `frames_*.png` |
| `summary.json` | 由原始数据计算的统计：分段 loss、耗时、显存、两次运行的逐步一致性 |
| `scripts/make_figures.py` | 从本目录原始数据重新生成 `summary.json`、`loss_curve.png` 和 `frames_*.png` |

`step_metrics.jsonl` 的字段含义见 `CHANGELOG.md` 的"`step_metrics.jsonl` 字段"一节。

## 运行条件

| 项目 | 值 |
| --- | --- |
| 代码 | `baseline` 分支 `9078fff`。训练运行时相关改动尚未提交，运行后训练代码、依赖和配置未再改变；仓库中的 `configs/ffs/ffs_train.yaml`、`configs/ffs/ffs_train_10k.yaml` 与两次运行保存的 `config.yaml` 逐项相同 |
| 模型与训练 | Latte-XL/2，16 帧，采样间隔 3，256×256，从头训练；每卡 batch 5，全局 batch 40；AdamW，学习率 1e-4；FP32 + TF32；`global_seed: 3407` |
| 数据读取 | `read_sampled_frames_only: True`（已验证与官方读取逐字节一致），`num_workers: 8` |
| 逐步记录 | `log_step_metrics: True`（实测开销约 0.13%） |
| 硬件 | 单机 8 × NVIDIA H200，驱动 595.71.05 |
| 软件 | Python 3.12，torch 2.7.1+cu118，diffusers 0.24.0，av 13.1.0，decord 0.6.0（完整版本见 `uv.lock`） |
| 启动方式 | `torchrun --standalone --nproc_per_node=8`（`train_scripts/ffs_train_8gpu.sh`） |

| 运行 | 日期 | 配置 | 节点 |
| --- | --- | --- | --- |
| `run_10k` | 2026-09-28 16:04–18:01 | `configs/ffs/ffs_train_10k.yaml` | gpu-lg-cmc-h-h200-0062 |
| `run_1k` | 2026-09-28 14:03–14:18 | `configs/ffs/ffs_train.yaml` | gpu-lg-cmc-h-h200-0828 |

`run_1k` 的 1000 步训练完整，但在第 1000 步保存 checkpoint 时磁盘写满而中止，因此没有 checkpoint，也少了按 epoch 取整本应多跑的约 3 步。`run_10k` 正常结束。

## 主要结果

loss 为 8 个 rank 的平均值。

| 步数 | loss 均值 | 标准差 |
| --- | --- | --- |
| 0–99 | 0.3045 | 0.2562 |
| 100–999 | 0.1300 | 0.0399 |
| 1000–1999 | 0.1119 | 0.0363 |
| 2000–2999 | 0.1052 | 0.0348 |
| 4000–4999 | 0.0988 | 0.0330 |
| 7000–7999 | 0.0952 | 0.0315 |
| 9000–10012 | 0.0918 | 0.0309 |

- 100 步滑动平均：第 1000 步 0.122，结束时 0.095；1 万步时仍在缓慢下降。
- 第 0 步 loss 为 1.0112。

| 性能指标 | `run_10k` | `run_1k` |
| --- | --- | --- |
| 实际每步耗时（相邻 `wall_time` 之差的平均） | 0.674 s | 0.670 s |
| 计算时间中位数（`compute_time_s`） | 0.592 s | 0.592 s |
| 等数据时间占比（`data_time_s` 之和 / `step_time_s` 之和） | 10.8% | 11.5% |
| 峰值显存，已分配 / 保留 | 69,840 / 75,756 MiB | 69,840 / 75,756 MiB |
| 总耗时 | 1.88 h | 0.19 h |

- 可复现性：两次运行在不同节点上，前 1000 步的逐步 loss 完全相等。
- 等数据的时间主要集中在每个 epoch（17 步）的第 0 步，原因是官方 DataLoader 未开 `persistent_workers`。

## 推理样本

- 使用官方 `sample/sample.py` 和 `configs/ffs/ffs_sample.yaml`（DDPM 250 步，FP16，无 CFG），只把 `pretrained_model_path` 改为本地的官方 `maxin-cn/Latte` 目录。官方脚本不固定随机种子，样本不可逐帧复现。
- 使用非 EMA 权重：官方 `train.py` 把 EMA 初始化为随机初始权重，衰减 0.9999，1 万步时 EMA 中仍有约 37%（0.9999^10000）来自随机初始权重，生成结果为噪声。`utils.find_model` 在 checkpoint 含 `ema` 时优先加载 EMA，因此先把 checkpoint 的 `model` 权重单独保存为只含 `model` 的文件，再交给 `sample.py`。
- 结果：可以辨认人脸和嘴部动作，帧间基本连贯，但画面模糊、五官有变形；官方 `ffs.pt` 的生成结果清晰。这符合 1 万步的训练程度，官方未给出 FFS 的训练步数。

## 未包含的内容

- checkpoint（`0010000.pt` 5.39 GB，含 model 和 EMA；`0010000_model_only.pt` 约 2.7 GB）保存在集群存储 `/mnt/shared-storage-gpfs1/ailab-sys/yangzhenyu/latte_results/000-Latte-XL-2-F16S3-ffs/checkpoints/`。
- TensorBoard 日志：内容与 `log.txt` 重复。

## 用于对比时的建议

- 逐步对齐：在当前环境下官方训练可逐位复现，另一框架固定相同种子、数据顺序、时间步和噪声后，可以先比较前几百步的逐步 loss。
- 统计趋势：单步 loss 波动较大（第 100 步以后标准差约 0.03–0.04），应比较滑动平均或分段均值；判断差异是否显著需要改变 `global_seed` 的对照运行。
- 性能：纯计算性能用 `compute_time_s` 的中位数；端到端吞吐用相邻 `wall_time` 之差，并说明受数据加载影响。`data_time_s`、`compute_time_s`、`step_time_s` 分别在各 rank 取最大值，三者不能直接相加。

## 重新生成图表

matplotlib 不在训练环境中，使用 uv 临时环境：

```bash
uv run --no-project --with matplotlib --with numpy --with imageio --with imageio-ffmpeg --with pillow \
    python baseline_results/ffs/scripts/make_figures.py
```
