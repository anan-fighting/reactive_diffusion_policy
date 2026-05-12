# Reactive Diffusion Policy (RDP) — 快速入门指南

> 论文：*Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation*
> RSS 2025 Best Student Paper Award Finalist
> arXiv: [2503.02881](https://arxiv.org/abs/2503.02881)

---

## 一、项目概览

**Reactive Diffusion Policy (RDP)** 是一个用于接触丰富型机器人操作任务的视觉-触觉策略学习框架，采用"慢-快"双频控制架构：

- **慢策略（Slow Policy）**：基于视觉图像输入，使用扩散模型（Diffusion Policy，DP）在低频（12fps）生成动作序列。
- **快策略（Fast Policy）**：基于触觉/力觉输入，使用潜在扩散策略（Latent Diffusion Policy，LDP）在高频（24fps）进行反应性修正。
- **非对称 Tokenizer（AT，Asymmetric Tokenizer）**：将高维动作序列压缩为低维潜空间表示，作为连接慢快策略的桥梁，使用 VAE（含可选 VQ 量化）实现。

### 整体架构图（文字描述）

```
传感器输入
  ├── 视觉图像（RealSense相机）────→ 慢策略（DP/LDP，12~24fps）──→ 动作指令
  └── 触觉图像（GelSight/McTAC）──→ 触觉嵌入（PCA）→ 快策略（AT+LDP）──→ 动作修正
      力矩数据（Wrench）──────────→ 快策略（AT+LDP）
```

---

## 二、仓库目录结构详解

```
reactive_diffusion_policy/          # 仓库根目录
│
├── train.py                        # 训练入口（基于 Hydra 配置）
├── train_dp.sh                     # 训练 Diffusion Policy 的脚本
├── train_rdp.sh                    # 训练 RDP（AT + LDP）的两阶段脚本
├── eval.sh                         # 推理/评估脚本
├── eval_real_robot_flexiv.py       # 真实机器人推理主程序
├── teleop.py                       # 遥操作服务端启动程序
├── camera_node_launcher.py         # ROS2 相机节点启动程序
├── record_data.py                  # 数据录制程序
├── post_process_data.py            # 数据后处理（原始数据→Zarr格式）
├── vcamera_server.py               # 虚拟相机服务器
├── requirements.txt                # Python 依赖列表
│
├── assets/                         # 论文图片资源
│
├── data/
│   ├── calibration/                # 机器人/相机标定文件（A_to_B_transform.json）
│   ├── PCA_Transform_GelSight/     # GelSight 触觉传感器的 PCA 变换矩阵
│   └── PCA_Transform_McTAC_v1/    # McTAC 触觉传感器的 PCA 变换矩阵
│
├── docs/                           # 文档目录
│   ├── quick_start.md              # 本文件：快速入门指南
│   ├── customized_deployment_guide.md   # 自定义传感器/机器人/任务指南
│   ├── data_collection_tips.md     # 数据采集技巧
│   ├── franka_setup_instructions.md    # Franka 机器人配置说明
│   ├── tactile_embedding_guide.md  # 触觉嵌入生成指南
│   └── Q&A.md                     # 常见问题解答
│
├── tests/                          # 单元测试
│
├── third_party/                    # 第三方库
│   ├── flexiv_rdk-main/            # Flexiv 机器人 SDK
│   ├── mvcam/                      # MindVision 相机驱动
│   └── mvsdk/                      # MindVision SDK
│
└── reactive_diffusion_policy/      # 核心代码包
    ├── common/                     # 通用工具
    ├── config/                     # Hydra 配置文件
    ├── dataset/                    # 数据集加载
    ├── env/                        # 真实环境接口
    ├── env_runner/                 # 环境运行器（评估）
    ├── model/                      # 神经网络模型
    ├── policy/                     # 策略类
    ├── real_world/                 # 真实世界交互层
    ├── scripts/                    # 辅助脚本
    └── workspace/                  # 训练 Workspace
```

---

## 三、核心代码包详解

### 3.1 `model/` — 神经网络模型

| 子目录/文件 | 内容 |
|---|---|
| `common/` | 归一化器（`normalizer.py`）等通用模型组件 |
| `diffusion/` | 扩散模型核心：`ConditionalUnet1D`（条件1D UNet）、EMA模型、掩码生成器、位置编码 |
| `vae/` | 非对称 Tokenizer（VAE）：编码器/解码器、VQ 量化层（`vector_quantize_pytorch`） |
| `vision/` | 视觉编码器：`MultiImageObsEncoder`（多相机融合）、`TimmObsEncoder`（基于 timm 的特征提取）、随机裁剪/增强 |

### 3.2 `policy/` — 策略类

| 文件 | 说明 |
|---|---|
| `base_image_policy.py` | 策略基类 |
| `diffusion_unet_image_policy.py` | **DP（Diffusion Policy）**：视觉输入 + 条件扩散，低频动作预测 |
| `latent_diffusion_unet_image_policy.py` | **LDP（Latent Diffusion Policy）**：继承 DP，在 AT 压缩的潜空间中做扩散，实现高频反应性控制 |

### 3.3 `workspace/` — 训练工作区

| 文件 | 说明 |
|---|---|
| `base_workspace.py` | 工作区基类（保存/加载 checkpoint） |
| `train_diffusion_unet_image_workspace.py` | 训练 DP 的工作区 |
| `train_at_workspace.py` | 训练 AT（非对称 Tokenizer）的工作区 |

### 3.4 `dataset/` — 数据集

| 文件 | 说明 |
|---|---|
| `base_dataset.py` | 数据集基类 |
| `real_image_tactile_dataset.py` | 真实世界图像+触觉数据集（供 DP 使用）|
| `real_image_tactile_latent_diffusion_dataset.py` | 图像+触觉数据集（供 LDP 使用，含潜空间标签）|

### 3.5 `config/` — Hydra 配置

配置体系基于 [Hydra](https://hydra.cc/)，支持多级覆盖。

```
config/
├── train_diffusion_unet_real_image_workspace.yaml   # DP 训练主配置
├── train_latent_diffusion_unet_real_image_workspace.yaml  # LDP 训练主配置
├── train_at_workspace.yaml                          # AT 训练主配置
├── real_world_env.yaml                              # 真实环境通用配置
├── at/                                              # AT 超参数配置
├── robot/                                           # 机器人配置
└── task/                                            # 任务配置（重要！每个实验对应一个文件）
    ├── real_robot_env.yaml                          # 机器人环境（IP、标定路径等）
    ├── real_peel_two_realsense_one_gelsight_one_mctac_24fps.yaml  # 数据采集任务配置
    ├── real_peel_image_gelsight_emb_at_24fps.yaml   # 训练 AT 的任务配置
    ├── real_peel_image_gelsight_emb_ldp_24fps.yaml  # 训练 LDP 的任务配置
    └── ...（wipe、lift 等任务同理）
```

> **任务配置命名规律：**
> `real_{task}_{sensors}_{method}_{control_freq}.yaml`
> - `task`：peel（剥皮）、wipe（擦拭）、lift（抬起）
> - `sensors`：image（视觉）、gelsight_emb/mctac_emb（触觉嵌入）、wrench（力矩）
> - `method`：dp（扩散策略）、at（非对称Tokenizer）、ldp（潜在扩散策略）

### 3.6 `real_world/` — 真实世界交互层

| 文件/目录 | 说明 |
|---|---|
| `robot/` | 机器人服务端：`bimanual_flexiv_server.py`（双臂Flexiv）、`franka_server.py`（Franka）、`single_flexiv_controller.py` |
| `publisher/` | ROS2 传感器数据发布节点 |
| `simple_camera/` | 相机接口封装 |
| `teleoperation/` | 遥操作相关代码（与 Quest3 通信）|
| `real_world_transforms.py` | 传感器数据坐标变换 |
| `real_inference_util.py` | 推理时的实用函数 |
| `ros_data_converter.py` | ROS2 消息与 Numpy 数据互转 |
| `post_process_utils.py` | 数据后处理工具类 |

### 3.7 `scripts/` — 辅助脚本

| 文件 | 说明 |
|---|---|
| `extract_gelsight_marker_motion.py` | 提取 GelSight 标记点运动 |
| `extract_mctac_marker_motion.py` | 提取 McTAC 标记点运动 |
| `generate_pca_embedding.py` | 生成触觉 PCA 嵌入 |
| `reencode_videos.py` | 视频重编码 |

### 3.8 `common/` — 通用工具

| 文件 | 说明 |
|---|---|
| `replay_buffer.py` | Zarr 格式的 Replay Buffer |
| `normalize_util.py` | 数据归一化工具 |
| `pose_trajectory_interpolator.py` | 位姿轨迹插值 |
| `tactile_marker_utils.py` | 触觉标记点处理 |
| `visualization_utils.py` | 可视化工具 |
| `ensemble.py` | 动作集成（多模型融合） |
| `ring_buffer.py` | 环形缓冲区 |

---

## 四、完整工作流程

### Step 1：环境安装

```bash
# 安装 ROS2 Humble（参考官方文档）

# 创建 Python 虚拟环境
python3 -m venv rdp_venv
source rdp_venv/bin/activate

# 安装 PyTorch（CUDA 11.7）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 安装其他依赖
pip install -r requirements.txt
```

### Step 2：配置任务

1. 编辑 `reactive_diffusion_policy/config/task/real_robot_env.yaml`，配置 `host_ip`、`robot_ip`、`vr_server_ip`、`calibration_path`。
2. 选择或新建任务配置文件（参考 `config/task/` 下的示例文件）。

### Step 3：数据采集

在三个独立终端（建议使用 `tmux`）分别运行：

```bash
# 终端1：启动遥操作服务
python teleop.py task=[任务配置文件名]

# 终端2：启动相机节点
python camera_node_launcher.py task=[任务配置文件名]

# 终端3：启动数据录制
python record_data.py --save_to_disk \
    --save_file_dir [数据保存目录] \
    --save_file_name [录制文件名]
```

### Step 4：数据后处理

修改 `post_process_data.py` 中的关键配置：

| 参数 | 说明 |
|---|---|
| `TAG` | 任务标签（如 `'peel_v3'`） |
| `ACTION_DIM` | 动作维度（如 4） |
| `TEMPORAL_DOWNSAMPLE_RATIO` | 时间降采样比例（DP用2，RDP用1） |
| `SENSOR_MODE` | 传感器模式 |

```bash
python post_process_data.py
```

处理后生成 Zarr 格式数据集，包含图像、触觉嵌入、机器人状态、动作标签等字段。

### Step 5：（可选）生成触觉嵌入

```bash
# 提取标记点运动
python reactive_diffusion_policy/scripts/extract_gelsight_marker_motion.py
python reactive_diffusion_policy/scripts/extract_mctac_marker_motion.py

# 生成 PCA 嵌入
python reactive_diffusion_policy/scripts/generate_pca_embedding.py
```

也可直接使用仓库内预计算的 PCA 矩阵（`data/PCA_Transform_GelSight/`）。

### Step 6：训练

#### 方案 A：仅训练 Diffusion Policy（DP）

```bash
# （可选）配置多卡训练
accelerate config

./train_dp.sh
```

#### 方案 B：训练 Reactive Diffusion Policy（RDP = AT + LDP）

```bash
./train_rdp.sh
```

`train_rdp.sh` 分两个阶段：
1. **阶段1**：训练非对称 Tokenizer（AT）
   - 使用 `--config-name=train_at_workspace`
2. **阶段2**：训练潜在扩散策略（LDP），加载 AT 的 checkpoint
   - 使用 `--config-name=train_latent_diffusion_unet_real_image_workspace`

训练底层调用 `train.py`，由 Hydra 根据 `--config-name` 加载对应 workspace 配置。

### Step 7：推理部署

1. （可选）启动虚拟相机服务器：
   ```bash
   python vcamera_server.py --host_ip [IP] --port [端口] --camera_id [相机ID]
   ```

2. 修改 `eval.sh` 中的任务配置和 checkpoint 路径，然后在三个终端运行：
   ```bash
   # 终端1：遥操作服务
   python teleop.py task=[任务配置]

   # 终端2：相机节点
   python camera_node_launcher.py task=[任务配置]

   # 终端3：推理
   ./eval.sh
   ```

---

## 五、三个实验任务

| 任务 | 配置前缀 | 说明 |
|---|---|---|
| `peel` | `real_peel_*` | 剥皮任务（双臂）|
| `wipe` | `real_wipe_*` | 擦拭任务（双臂）|
| `lift` | `real_lift_*` | 抬起任务（单臂或双臂）|

---

## 六、两种触觉传感器

| 传感器 | 配置关键词 | PCA 矩阵路径 |
|---|---|---|
| GelSight Mini | `gelsight` / `gelsight_emb` | `data/PCA_Transform_GelSight/` |
| McTAC v1 | `mctac` / `mctac_emb` | `data/PCA_Transform_McTAC_v1/` |

触觉嵌入维度默认为 **15**（`PCA_DIM=15`）。

---

## 七、支持的机器人

| 机器人 | 配置文件 | 说明 |
|---|---|---|
| Flexiv Rizon 4（双臂） | `bimanual_flexiv_server.py` | 默认机器人，支持关节力矩传感 |
| Flexiv Rizon 4（单臂） | `single_flexiv_controller.py` | 单臂控制 |
| Franka Research 3 | `franka_server.py` | 参考 `docs/franka_setup_instructions.md` |

---

## 八、预训练模型与数据集

- 📦 数据集：[HuggingFace Dataset](https://huggingface.co/datasets/WendiChen/reactive_diffusion_policy_dataset)
- 🤖 Checkpoints：[HuggingFace Model](https://huggingface.co/WendiChen/reactive_diffusion_policy_model)

---

## 九、参考文档

| 文档 | 内容 |
|---|---|
| `docs/customized_deployment_guide.md` | 如何自定义传感器、机器人和任务 |
| `docs/tactile_embedding_guide.md` | 触觉数据集采集与嵌入生成 |
| `docs/data_collection_tips.md` | 数据采集注意事项 |
| `docs/franka_setup_instructions.md` | Franka 机器人配置 |
| `docs/Q&A.md` | 常见问题 |

---

## 十、关键依赖

- **ROS2 Humble**：传感器数据发布与机器人控制通信
- **PyTorch 1.13.1 + CUDA 11.7**：模型训练与推理
- **Hydra**：配置管理系统
- **Zarr**：高效的分层数组存储格式（用于数据集）
- **diffusers**：扩散模型调度器（DDPM）
- **timm**：视觉特征提取骨干网络
- **Flexiv RDK / deoxys**：机器人控制 SDK

---

*更多细节请参考 [README.md](../README.md) 及 `docs/` 下的各专题文档。*
