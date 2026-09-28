# CHANGELOG：`baseline` 分支

`baseline` 分支基于官方 `main`（`Vchitect/Latte` 提交 `ad4b85a`），用作 Latte 训练的性能基线和算法精度基线，与其他训练框架对比 loss 收敛、训练速度和显存占用。已在单机 8 卡上跑通的基线任务是 FaceForensics（FFS）无条件视频生成训练。

改动遵循一个原则：不改变送入模型的数据和参与训练计算的张量。新增能力都有开关，默认关闭时与官方行为一致。

## 相比 `main` 的改动

"影响数值"指是否改变送入模型的数据或参与训练计算的张量。

| 文件 | 改动 | 影响数值 |
| --- | --- | --- |
| `pyproject.toml`、`uv.lock`、`.python-version` | 改用 uv 管理环境，Python 3.12。依赖与官方 `environment.yml` 一致，官方未固定的版本由 `uv.lock` 固定；`av` 固定为 `>=13,<14`，原因见下文"数据加载" | 依赖版本与官方当年环境不同，见"运行环境" |
| `environment.yml` | 顶部加注释说明已改用 uv，文件仅作参考 | 否 |
| `README.md` | Setup 改为 uv 用法，训练示例改为 8 卡脚本 | 否 |
| `train.py` | 新增 `log_step_metrics` 开关（默认关闭）：rank 0 每步向实验目录写一行 `step_metrics.jsonl` | 否，开销约 0.13% |
| `datasets/ffs_datasets.py` | 新增 `read_sampled_frames_only` 开关（默认关闭）：用 decord 只读取采样的 16 帧，不再读取整段视频 | 否，已验证与官方读取逐字节一致 |
| `configs/ffs/ffs_train.yaml` | 数据和 VAE 路径改为本地路径；`pretrained` 置空（与官方一致，从头训练）；打开上述两个开关；`max_train_steps` 1000000 → 1000、`ckpt_every` 10000 → 1000，用于短程收敛观察 | 否（步数只决定训练长度） |
| `configs/ffs/ffs_train_10k.yaml` | 1 万步基线配置，与 `ffs_train.yaml` 只差 `max_train_steps: 10000`、`ckpt_every: 10000` 和 `results_dir` | 否 |
| `configs/ffs/ffs_sample.yaml` | VAE 路径（`pretrained_model_path`）改为本地路径，其余保持官方采样设置 | 否 |
| `train_scripts/ffs_train_8gpu.sh` | 单机 8 卡 `torchrun` 启动脚本，与官方 `slurm_scripts/ffs.slurm` 的单机 8 卡设置一致；可用环境变量 `LATTE_CONFIG` 指定配置、`LATTE_CUDA_DEVICES` 指定 GPU | 否 |
| `bench/verify_ffs_frame_reading.py` | 验证 `read_sampled_frames_only` 与官方读取逐字节一致，并记录两种方式的读取耗时；纯 CPU | 否（不参与训练） |
| `baseline_results/ffs/` | FFS 基线结果数据，见下文"基线结果" | 否（不参与训练） |
| `.gitignore` | 忽略本地开发记录和集群专用文件：`.tmux-remote-gpu/`、`.h-cluster-rjob/`、`train_scripts/*_rjob.sh`、`bench/*_rjob.sh`、`devlog.md`、`results_bench` | 否 |

官方 dataset 文件（除上述开关外）和 `train.py` 的训练逻辑保持原样。

### `step_metrics.jsonl` 字段

每行一个 JSON 对象，对应一个训练步：

| 字段 | 含义 |
| --- | --- |
| `step`、`epoch` | 训练步数和 epoch |
| `wall_time` | 该步结束时的时间戳，相邻两行之差即实际每步耗时 |
| `loss` | 各 rank loss 的平均值 |
| `grad_norm`、`lr` | 梯度范数和学习率 |
| `optimizer_stepped` | 该步是否执行了 `opt.step()`（官方第 0 步不更新参数） |
| `data_time_s` | 等待数据的时间，各 rank 取最大值 |
| `compute_time_s` | 前向、反向和参数更新的时间，各 rank 取最大值 |
| `step_time_s` | 上两项之和 |
| `max_mem_allocated_mib`、`max_mem_reserved_mib` | 该步的峰值显存，各 rank 取最大值 |

打点和保存 checkpoint 的时间不计入下一步。开关打开后，每步会多一次 `cuda.synchronize` 和两次 `all_reduce`；由于官方训练循环每步已有 `loss.item()` 等待 GPU，实测 8 × H200 上每步从 0.5851 s 变为 0.5858 s（+0.13%）。

### 数据加载

官方代码每个样本用 `torchvision.io.read_video` 读取整段视频（FFS 为未压缩的 `rawvideo`，约 43 MB），只使用其中 16 帧。在新版本环境中这一步成为训练的主要瓶颈：

- PyAV 18（FFmpeg 8）在 `read_video` 中为每一帧新建一个多线程的 `SwsContext`，并在帧释放前一直保留，线程和内存随帧数累积；读取 780 帧的视频时，7 核机器上会累计约 4858 个线程而失败。PyAV 将修复列在尚未发布的 v19。因此 `av` 固定为 13.1.0。
- `read_sampled_frames_only` 用 decord 只解码采样的 16 帧。对全部 704 个训练视频（363,313 帧）核实：decord 读出的每一帧与官方 `read_video`（av 13.1.0）逐字节相等，帧数一致；相同随机种子下 `__getitem__` 的输出完全相等。单样本读取耗时中位数从 0.116 s 降到 0.047 s。
- 逐字节一致依赖当前的 av 和 decord 版本（两者的 yuv420p 转 RGB 实现不同）。升级 av 或 decord 后，需要重新运行 `bench/verify_ffs_frame_reading.py` 确认：

  ```bash
  uv run python bench/verify_ffs_frame_reading.py --config ./configs/ffs/ffs_train.yaml --processes 16
  ```

  结果写入 `results_bench/ffs_frame_reading/<时间>/summary.json`，`passed` 为 `true` 表示通过。

两项改动合计效果（8 × H200，每卡 batch 5）：

| 指标 | av 18 + 整段读取 | av 13 + 只读采样帧 |
| --- | --- | --- |
| 实际每步耗时 | 2.511 s | 0.659 s |
| 等数据时间占比 | 76.3% | 10.4% |
| 计算时间中位数 | 0.596 s | 0.590 s |

剩余的等待主要在每个 epoch 的第 0 步：官方 DataLoader 未开 `persistent_workers`，每个 epoch 重启 worker。本分支保持官方做法。

## 运行 FFS 8 卡基线

### 1. 准备环境

安装 [uv](https://docs.astral.sh/uv/) 后，在仓库根目录执行：

```bash
uv sync --frozen
```

会在仓库内创建 `.venv`，安装 `uv.lock` 固定的依赖（Linux 上 torch 使用官方 CUDA 11.8 wheel）。

### 2. 准备数据和 VAE

- 训练数据：从 Hugging Face 数据集 [`maxin-cn/FaceForensics`](https://huggingface.co/datasets/maxin-cn/FaceForensics) 下载 `FaceForensics.zip` 并解压，训练集为 `train/videos/` 下 704 个视频（256×256，25 fps）。
- VAE：从 [`maxin-cn/Latte`](https://huggingface.co/maxin-cn/Latte) 下载仓库，训练使用其中的 `vae/`。

修改 `configs/ffs/ffs_train.yaml`（或 `ffs_train_10k.yaml`）中的路径：

```yaml
data_path: "<数据目录>/train/videos/"
pretrained_model_path: "<maxin-cn/Latte 下载目录>"
results_dir: "./results"
pretrained:            # 保持为空，从头训练
```

### 3. 启动训练

```bash
# 默认配置 configs/ffs/ffs_train.yaml，1000 步，使用 GPU 0-7
uv run bash train_scripts/ffs_train_8gpu.sh

# 1 万步基线
LATTE_CONFIG=./configs/ffs/ffs_train_10k.yaml uv run bash train_scripts/ffs_train_8gpu.sh

# 指定其他 8 张 GPU
LATTE_CUDA_DEVICES=8,9,10,11,12,13,14,15 uv run bash train_scripts/ffs_train_8gpu.sh
```

注意事项：

- DataLoader 使用 8 个 worker（与官方一致），worker 之间通过 `TMPDIR` 下的 AF_UNIX socket 通信。仓库或 `TMPDIR` 路径很长时可能超过 socket 路径长度上限而报错，此时先 `export TMPDIR=/tmp`。
- checkpoint 包含模型和 EMA 权重，保存前确认 `results_dir` 所在磁盘有足够空间。
- 训练按 epoch 取整结束，实际步数略多于 `max_train_steps`（1000 步配置约 1003 步）。

### 4. 输出

每次运行在 `results_dir` 下新建 `NNN-Latte-XL-2-F16S3-ffs/`，包括：

- `log.txt`：训练日志，每 `log_every`（100）步输出平均 loss、梯度范数和吞吐。
- `config.yaml`：本次运行的完整配置。
- `step_metrics.jsonl`：逐步记录，字段见上文。
- TensorBoard 日志和 `checkpoints/`。

### 5. 参考结果

环境：8 × NVIDIA H200，每卡 batch 5（全局 batch 40），FP32 + TF32，从头训练。

| 项目 | 结果 |
| --- | --- |
| loss（100 步滑动平均） | 第 1000 步 0.122，第 1 万步 0.095 |
| 端到端每步耗时 | 0.674 s |
| 等数据时间占比 | 10.8% |
| 纯计算每步耗时 | 0.592 s |
| 峰值显存（已分配 / 保留） | 69,840 / 75,756 MiB |
| 可复现性 | 相同配置和种子在不同节点上运行，前 1000 步的逐步 loss 完全相等 |

在当前环境下，逐步 loss 可以用"差值为 0"判断是否对齐；估计随机波动范围时需要改变 `global_seed`。完整数据见下文"基线结果"。

## 基线结果

`baseline_results/ffs/` 保存 FFS 基线的原始数据和由其生成的图表，运行条件、结果和使用建议见该目录的 [`README.md`](baseline_results/ffs/README.md)：

| 路径 | 内容 |
| --- | --- |
| `run_10k/` | 1 万步基线（`configs/ffs/ffs_train_10k.yaml`）的 `step_metrics.jsonl`、`log.txt`、`config.yaml`、`loss_curve.png` |
| `run_1k/` | 1000 步运行（`configs/ffs/ffs_train.yaml`）的同类文件 |
| `samples_10k_model/` | 用 1 万步 checkpoint 的非 EMA 权重生成的视频 `sample_*.mp4` 及帧拼图 `frames_*.png` |
| `summary.json`、`scripts/make_figures.py` | 统计结果，以及从原始数据重新生成统计和图表的脚本 |

## 运行环境

官方 `environment.yml` 只要求 `python>=3.10`、`pytorch>2.0.0`、`pytorch-cuda>=11.7`，仅固定 `diffusers==0.24.0`，无法严格还原官方当年环境。本分支的环境：

| 项目 | 版本 |
| --- | --- |
| Python | 3.12 |
| torch / torchvision | 2.7.1+cu118 / 0.22.1+cu118 |
| diffusers | 0.24.0（与官方一致） |
| transformers | 4.45.2 |
| av（PyAV） | 13.1.0（捆绑 FFmpeg 7.0） |
| decord | 0.6.0 |
| 启动方式 | `torchrun --standalone`（官方为 slurm `srun`，两者得到的 rank、world size 和随机种子一致） |

论文使用 8 × A100 80G，本分支的参考结果在 8 × H200 上测得。

## 官方代码的已知问题与训练细节

本分支未修改以下行为，对齐 loss 时需要注意：

- 无法正确续训：`resume_from_checkpoint` 分支调用不存在的 `model.load_state`；checkpoint 不保存优化器状态。任务中断后只能从头重跑。
- 第 0 步只做反向、不更新参数（`opt.step()` 的条件是 `train_steps > 0`），第 0、1 步的梯度累加后才第一次更新。
- `start_clip_iter: 20000`：前 2 万步只计算梯度范数，不裁剪。
- `mixed_precision`、xformers、`use_compile` 均关闭，全程 FP32（开启 TF32）。
- 短训练的 EMA 权重不可用：EMA 用 `update_ema(ema, model.module, decay=0)` 初始化为随机初始权重，之后每步按 0.9999 衰减，1 万步时仍有约 37%（0.9999^10000）来自随机初始权重，用它推理只得到噪声。短训练的推理应使用 `model` 权重；对比 EMA 实现时应直接比较 EMA 参数。
- 官方未给出 FFS 的训练步数；1000 步和 1 万步是本分支为对比设定的长度。
