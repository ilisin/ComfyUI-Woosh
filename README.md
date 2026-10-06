# ComfyUI-Woosh

在 ComfyUI 中使用 Sony AI [Woosh](https://github.com/SonyResearch/Woosh) 生成文本音效和环境声。支持 **Woosh-DFlow** 和 **Woosh-Flow**，输出标准 `AUDIO`，可连接原生音频预览、保存和视频合成节点。

节点名称：**Woosh 音效 / 环境声生成**；节点 ID：`WooshLocalGenerate`；分类：`audio/Woosh`。

## 功能与运行方式

- 输出 float32、48 kHz、单声道音频，张量形状为 `[batch, 1, samples]`。
- 每段使用官方 128 × 501 潜变量，生成约 5 秒音频。短输出通过裁剪得到；长输出分段生成并交叉淡化拼接。
- 支持固定种子、批量生成和短音效峰值裁剪。
- Woosh 在独立 Python 进程中运行，任务结束、中断或超时后退出并释放模型。每次任务都会重新加载模型。
- 默认生成前卸载 ComfyUI 已加载模型，为 Woosh 腾出显存；目前没有持久模型缓存或原生分层 offload。

## 安装（Windows）

### 1. 安装扩展

将本仓库下载或克隆到 `ComfyUI/custom_nodes/ComfyUI-Woosh`。插件目录下应直接包含 `__init__.py`、`nodes.py` 和 `setup.ps1`，避免 ZIP 解压后多套一层文件夹。

准备 Git 和终端中可运行的 Python。节点本身使用 ComfyUI 已有的 torch/numpy；安装脚本会为 Woosh 创建独立环境。官方 Woosh 固定 torch/torchaudio 2.8.0，因此建议使用这个独立环境。

### 2. 安装环境并下载 DFlow 权重

在插件目录打开 PowerShell，执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -DownloadModels
```

脚本会：

1. 下载固定版本的官方 Woosh 源码到 `runtime/Woosh`。
2. 在 `runtime/` 内创建 bootstrap 环境，通过 uv 创建独立 Python 3.12 环境并安装 CUDA 12.8 依赖。
3. 安装 SOCKS 代理所需的 socksio，并缓存 `roberta-large` 的 tokenizer/config。
4. 下载 DFlow、Woosh-AE、TextConditionerA，写入本机配置 `runtime.json`。

三个模型压缩包合计约 **3.4 GB**；解压后的模型、Python 和 CUDA/PyTorch 依赖还需要额外磁盘空间。**只有显式运行安装/下载脚本才会下载资源，生成节点不会自动下载模型。**生成时后端启用 Hugging Face 离线模式。

只安装环境、不下载权重时，省略 `-DownloadModels`。CPU 环境加上 `-Device cpu`；CPU 推理速度需要自行评估。

### 3. 安装 Flow 或指定模型目录

```powershell
# 安装原始 Flow 模型；已有公共模型会跳过
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -DownloadModels -Model Woosh-Flow

# 自定义权重目录：请替换为自己的路径
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -DownloadModels -ModelRoot "C:/AI/models/woosh"
```

使用自定义目录后，再次运行安装脚本时也应提供相同的 `-ModelRoot`；省略该参数会将配置改回默认目录。

安装完成后重启 ComfyUI，刷新浏览器。升级后若旧节点没有新增参数，可重新添加生成节点。

## 模型下载与目录

默认权重目录为 `ComfyUI/models/woosh`。官方 [v1.0.0 发布页](https://github.com/SonyResearch/Woosh/releases/tag/v1.0.0) 提供以下文件：

| 文件 | 用途 |
|---|---|
| [Woosh-DFlow.zip](https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-DFlow.zip) | 蒸馏文本生成模型 |
| [Woosh-Flow.zip](https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-Flow.zip) | 原始文本生成模型 |
| [Woosh-AE.zip](https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-AE.zip) | 公共音频解码器 |
| [TextConditionerA.zip](https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/TextConditionerA.zip) | 公共文本编码器 |

手动下载时，将压缩包里的相应模型目录放到以下结构。每个模型目录应直接包含 `config.yaml` 和 `weights.safetensors`，避免额外的 `checkpoints/` 层：

```text
ComfyUI/models/woosh/
├── Woosh-DFlow/
│   ├── config.yaml
│   └── weights.safetensors
├── Woosh-Flow/             # 仅使用 Flow 时需要
│   ├── config.yaml
│   └── weights.safetensors
├── Woosh-AE/
│   ├── config.yaml
│   └── weights.safetensors
└── TextConditionerA/
    ├── config.yaml
    └── weights.safetensors
```

### 已有 Woosh 环境

将 [runtime.example.json](runtime.example.json) 复制为 `runtime.json`，替换三个路径：

- `python`：Woosh 环境中的实际 Python 可执行文件。
- `source`：官方源码根目录，其中包含 `woosh/` Python 包。
- `model_root`：包含上述模型子目录的权重根目录。

已有环境需预先缓存 `roberta-large` 的 tokenizer/config。环境变量 `WOOSH_PYTHON`、`WOOSH_SOURCE`、`WOOSH_MODEL_DIR` 可覆盖配置。

没有配置 `model_root` 时，扩展会搜索 ComfyUI 注册的 `woosh` 模型目录及源码的 `checkpoints`。例如，在 `extra_model_paths.yaml` 中添加：

```yaml
woosh_assets:
  base_path: C:/AI
  woosh: models/woosh
```

安装脚本生成的 `runtime.json` 会设置明确的 `model_root`，此时以该目录为准。

## 快速使用

1. 添加 **Woosh 音效 / 环境声生成**，或拖入 [示例工作流](example_workflows/woosh_ambience.json)。
2. 输入英文提示词，先用 DFlow、5 秒、4 步、CFG 4.5、批量 1 测试。
3. 将 `audio` 接到 **Preview Audio** 试听，接到 **Save Audio (Advanced)** 保存。示例保存为 FLAC。
4. 满意后再增加环境声时长或更换种子筛选效果。

[woosh_api.json](example_workflows/woosh_api.json) 是 API 格式的 prompt 示例，不是浏览器拖入用的工作流文件。

## 模型与参数

| 模型 | 推荐起点 | 说明 |
|---|---|---|
| Woosh-DFlow | 4 步、CFG 4.5 | 蒸馏模型，适合快速试提示词和种子 |
| Woosh-Flow | CFG 4.5 | 使用官方 dopri5 自适应 ODE，误差容限为 0.001；没有固定步数设置 |

Flow 可用于质量对比，但不保证每类声音都优于 DFlow。

| 参数 | 作用与建议 |
|---|---|
| `prompt` | 描述音源、音色、空间和变化。先描述一种明确声音，避免堆叠过多效果 |
| `duration_seconds` | 最终输出时长。短输出先生成完整片段再裁剪，通常不会减少单段推理时间 |
| `seed` | 比较参数时固定种子，筛选效果时更换种子；跨设备/版本不保证逐样本一致 |
| `batch_size` | 建议从 1 开始，增加批量会增加显存占用 |
| `cfg` | 默认 4.5；UI 点击可试 3.5 并与 4.5 对比，不是越高越好 |
| `dflow_steps` | 仅 DFlow 使用，推荐 4；4 步采用官方 renoise `[0, 0.5, 0.5, 0.3]`，其他步数使用零 renoise |
| `crossfade_seconds` | 长音频拼接的交叉淡化，环境声可试 0.25–0.5 秒，短音效设为 0 |
| `crop_mode` | `start` 从开头截取；`peak` 围绕最大振幅截取，适合单次点击或撞击 |
| `device` | NVIDIA 显卡使用 `cuda` 或 `auto`；也可选择 `cpu`、`mps` |
| `normalize` | 建议开启，只降低超过 ±1 的峰值，不会自动放大小音量 |
| `release_comfy_models` | 建议开启，生成前卸载 ComfyUI 已加载模型 |
| `timeout_seconds` | 整次后端任务的超时秒数，包含模型加载，默认 1800 |

`peak` 按每个样本单独定位，保留最多 30ms 前导，并在裁剪边缘施加最多 2ms 淡化。最大振幅可能是噪声，它不会自动识别目标事件或改善模型音色。需要多段生成的长环境声仍从开头截取。

## 音效 Demo

以下设置使用批量 1、固定种子 42、normalize 开启；DFlow 使用 4 步。Flow 可以保持同样参数进行试听对比。

| 用途 | 时长 | CFG | 交叉淡化 | 裁剪 |
|---|---:|---:|---:|---|
| UI 按钮点击 | 0.35 秒 | 3.5 | 0 | peak |
| 城市街景环境声 | 先 5 秒，再 30 秒 | 4.5 | 0.5 秒 | start |
| 闯关失败 | 5 秒 | 4.5 | 0 | start |
| 闯关成功 | 5 秒 | 4.5 | 0 | start |
| 开启宝箱 | 5 秒 | 4.5 | 0 | start |

**UI 点击**（已完成两个模型的真实生成测试，可试 0.25–0.5 秒）：

```text
A single short, dry plastic button click. Sharp attack, soft low thump, quick decay.
```

**现代城市街景**：

```text
Ambient sound of a modern city street heard from the sidewalk during daytime. Steady distant traffic, occasional cars passing, scattered pedestrian footsteps, and a gentle breeze. Natural outdoor acoustics, consistent background activity. No intelligible speech, no music.
```

**闯关失败**：

```text
A short arcade game failure sound. Three descending electronic tones ending in a low, muted buzzer. Playful and clearly conveying defeat. Brief decay followed by silence. Isolated sound effect, no speech, no background ambience.
```

**闯关成功**：

```text
A short arcade game level completion sound. Four bright ascending electronic tones ending with a satisfying bell chime and a brief sparkle. Cheerful and rewarding. Isolated sound effect, no speech, no background ambience.
```

**开启宝箱**：

```text
A fantasy treasure chest opening. A metal latch clicks, a heavy wooden lid creaks open, then sparkling magical chimes ring out with a gentle jingle of gold coins. Clear sequence, detailed and satisfying. No speech, no background music.
```

多阶段音效先保留完整 5 秒，再手动裁剪，避免峰值裁剪丢失前面的事件。除点击外，其余 Demo 是建议起点；提示词中的时长、事件数量和静音要求不保证被严格遵守。

## 常见问题

- **切换 Flow 报缺少模型**：下拉框不会下载权重。运行 `setup.ps1 -DownloadModels -Model Woosh-Flow`，并确认权重目录与 `runtime.json` 一致。
- **SOCKS 代理报缺少 socksio**：最新版安装脚本会补装依赖。更新后重新运行安装命令；脚本继承终端代理环境。
- **网络中断**：修复网络后重跑。已完整安装的模型会跳过；中断的压缩包需重新下载，目前没有断点续传。GitHub 和 Hugging Face 均需可访问。
- **离线生成报 tokenizer/config 缺失**：联网运行安装脚本，缓存 `roberta-large` 资源。模型权重之外也需要这些小文件。
- **短音效静音或不完整**：先用 5 秒、`start` 试听完整片段；单次声音再用 `peak` 裁剪。多阶段音效建议手动裁剪。
- **显存不足**：批量设为 1，开启 `release_comfy_models`。该选项不能释放其他程序或插件自行持有的显存，也不保证避免 OOM。
- **进度暂时不动**：加载期间没有分段进度；每完成一个片段才更新。超时限制包含加载过程。

目前只支持文本生成，不含视频条件、参考音频或无缝循环保证。长音频由独立片段拼接，跨段事件可能不连续。英文提示词作为测试起点；中文输入的效果需自行试听。安装脚本面向 Windows；其他平台可以手动配置独立环境，但尚未验证。

## 验证与开发

已通过 21 项测试，并完成真实 ComfyUI 节点加载、示例 API 工作流校验。在 Windows、RTX 4060 Ti、ComfyUI 主环境 PyTorch 2.10.0+cu130 下，独立 Woosh 环境分别完成 DFlow 和 Flow 的真实推理。未进行系统音质评测或显存基准测试。

在仓库目录，使用含 torch/numpy 的 Python 运行：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖音频拼接、裁剪、种子复现、采样调用契约（模拟模型）、子进程回传、诊断、超时和中断，不代替真实推理。安装脚本固定官方源码 `7f238bbd5d6dbc94d27190c07827b2d0eaefa268`；首次安装由官方 uv 配置解析依赖并生成本地 uv.lock。

## 公开仓库与许可

仓库应包含插件源码、安装脚本、示例配置、示例工作流、测试、README 和 LICENSE。`.gitignore` 排除了独立环境、下载权重、音频、缓存及 `runtime.json`；不要强制提交这些本地文件。已被 Git 跟踪的文件不会因新增忽略规则自动移除，应检查暂存文件列表。

插件代码使用 [MIT License](LICENSE)。Woosh 的源码及模型权重保留各自许可；本仓库不包含官方源码副本、权重或官方媒体样本。官方说明开放权重使用 CC-BY-NC，插件的 MIT 许可不会改变模型权重的许可条件。详见 [Woosh 官方许可说明](https://github.com/SonyResearch/Woosh#license)。
