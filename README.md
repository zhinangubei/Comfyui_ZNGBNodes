# Comfyui_ZNGBNodes

一组**容错（null-tolerant）** 的 ComfyUI 自定义节点。核心设计目标：当视频 URL 为空（`None`）时，后续接入的取分量 / 拼接 / 裁剪等节点依然能正常运行而**不报错**，输出 `null` 占位，从而让整条视频处理流程可以顺利走到最终的 Video Combine。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zhinangubei/Comfyui_ZNGBNodes.git
```

安装依赖（在 ComfyUI 所用的 Python 环境下执行）：

```bash
pip install -r Comfyui_ZNGBNodes/requirements.txt
```

> 大部分节点仅依赖 ComfyUI 自带的 `torch` / `torchaudio`，无需额外安装。
> 只有 **gaussian splatting converter（ply/spz）** 节点需要 `3dgsconverter`，它会通过
> `requirements.txt` 从 GitHub 安装（并自动拉取 `taichi`、`plyfile`、`scikit-learn` 等依赖）。

重启 ComfyUI 后，节点会出现在 `ZNGBNodes/video`、`ZNGBNodes/image`、`ZNGBNodes/audio`、`ZNGBNodes/utils`、`ZNGBNodes/3d` 分类下。

## 节点说明

### load video from url（`ZNGBNodes/video`）

从 http(s) 链接下载视频并输出 `VIDEO`。

- 输入：`url`(STRING)、`keep_audio`(BOOLEAN)、`trim_time`(FLOAT，秒，0=不裁剪)
- 输出：`VIDEO`
- `url` 为空 / `none` / `null` 时，输出 `null`。
- 下载文件按 URL 哈希缓存到 `input` 目录，重复运行不重复下载；仅允许 http/https。

### LamaInpainting_zngb（`ZNGBNodes/image`）

使用本地 ModelScope `iic/cv_fft_inpainting_lama` 模型修复蒙版区域。

- 模型目录固定为 `ComfyUI/models/cv_fft_inpainting_lama`，其中必须包含 `pytorch_model.pt`。
- 输入：`img`（`IMAGE`）和 `mask`（`MASK`）；任何大于 0 的蒙版像素都会作为修复区域。
- 输出：与输入图像批次和尺寸相同的 `image`，蒙版外像素保持原值。
- 单张 mask 可广播到整批图像；mask 尺寸不同时会使用最近邻插值对齐图像。
- 模型自动使用 ComfyUI 当前推理设备并在首次加载后缓存。图像和蒙版会按官方普通推理逻辑对称补到 8 的倍数，再裁回原尺寸。

### crop image by bboxes（`ZNGBNodes/image`）

根据 `Qwen2.5-VL Object Detection` 输出的 `BBOX` 列表裁剪原图。

- 输入：`bboxes`（`BBOX`）和 `image`（`IMAGE`）。
- 输出：`imgs_list`，每个检测框对应一个独立的 ComfyUI `IMAGE` 列表项，可保留不同裁剪尺寸。
- 坐标格式为 `[x1, y1, x2, y2]`；越界坐标会裁到图像边界，反向坐标会自动纠正，无效框会跳过。
- 图像为 batch 时，每张图都会应用全部检测框，输出顺序为先图像、后检测框。


## 节点列表

### 1. load video from url（`ZNGBNodes/video`）
从 http(s) 链接下载视频并输出 `VIDEO`。

- 输入：`url`(STRING)、`keep_audio`(BOOLEAN)、`trim_time`(FLOAT，秒，0=不裁剪)
- 输出：`VIDEO`
- `url` 为空 / `none` / `null` 时，输出 `null`。
- 下载文件按 URL 哈希缓存到 `input` 目录，重复运行不重复下载；仅允许 http/https。
- `keep_audio=False` 时去掉音轨；`trim_time>0` 时只保留前 N 秒。

### 2. get video components（`ZNGBNodes/video`）
类似官方 Get Video Components，但允许 `video` 输入为空。

- 输入：`video`(VIDEO，可空)
- 输出：`images`(IMAGE)、`audio`(AUDIO)、`fps`(FLOAT)、`bit_depth`(INT)
- `video` 为空时：`images / audio / bit_depth = null`，但 **`fps = 30.0`**，方便下游帧数 / 时长计算不报错。
- 正常情况下 `bit_depth` 输出 `8`。

### 3. image batch multi（`ZNGBNodes/image`）
把多张图像拼成一个 batch。

- 输入：`inputcount` + 动态 `image_1..image_N`（点击 **Update inputs** 增减输入口）
- 输出：`images`(IMAGE)
- 任意输入可为空并被跳过；尺寸 / 通道数不一致会自动对齐；全部为空则输出 `null`。

### 4. audio concat multi（`ZNGBNodes/audio`）
按输入顺序把多段音频在时间轴上依次拼接（after）。

- 输入：`inputcount` + `sample_rate`(INT，默认 44100) + 动态 `audio_1..audio_N`（点击 **Update inputs** 增减输入口）
- 输出：`audio`(AUDIO)
- 任意输入可为空并被跳过；声道数不同会自动补齐；全部为空则输出 `null`。
- 采样率不一致时，会先把每段音频重采样到 `sample_rate` 再拼接（**时长保持不变**），不再报错。

### 5. get image range from batch（`ZNGBNodes/image`）
从图像 batch 中取一段连续帧

- 输入：`start_index`(INT)、`num_frames`(INT)、`images`(IMAGE，可空)
- 输出：`images`(IMAGE)
- `images` 为空时输出 `null`；`start_index=-1` 表示取最后 `num_frames` 帧。

### 6. resize image（`ZNGBNodes/image`）
缩放图像。

- 输入：`width`、`height`、`upscale_method`、`keep_proportion`、`pad_color`、`crop_position`、`divisible_by`、`image`(可空)、`device`
- 输出：`image`(IMAGE)、`width`(INT)、`height`(INT)
- `image` 为空时，三个输出均为 `null`。
- `keep_proportion` 支持 `stretch / resize / pad / crop`。

### 7. audio crop（`ZNGBNodes/audio`）
按起始时间和时长裁剪音频（单位：秒）。

- 输入：`start_time`(FLOAT，秒)、`duration`(FLOAT，秒，0=到结尾)、`sample_rate`(INT，默认 44100)、`audio`(AUDIO，可空)
- 输出：`audio`(AUDIO)
- `audio` 为空时：`duration>0` 则输出对应时长的**静音**（采样率取 `sample_rate`），否则输出 `null`。

### 8. video clip（`ZNGBNodes/video`）
一站式裁剪节点：按秒裁剪图像帧并对齐音频，专为**唇形对齐**设计，可处理原视频帧率与合成帧率不一致的情况。

- 输入：
  - `source_fps`(FLOAT)：**输入图像的原始帧率**（如 24/25，建议接 `get video components` 的 `fps` 输出）
  - `fps`(FLOAT)：**输出/合成帧率**（如 30）
  - `start` / `end`(FLOAT，秒，步进 0.001 = 毫秒精度，`end<=start` 或 0 表示到结尾)
  - `width` / `height` / `upscale_method` / `keep_proportion`(默认 `pad`) / `pad_color` / `crop_position`(默认 `center`) / `divisible_by`：和 resize image 一致的输出分辨率控制
  - `sample_rate`(INT，默认 44100)：输出音频目标采样率（建议与 audio concat multi 一致）
  - `images`(IMAGE，可空)、`audio`(AUDIO，可空)、`device`
- 输出：`images`(IMAGE)、`audio`(AUDIO)
- 处理流程：
  1. 按 `source_fps` 把 `start/end` 秒换算成帧索引，取出片段，并用 `帧数 / source_fps` 得到真实时长。
  2. 缩放后，把帧从 `source_fps` **重定时**（复制/丢弃）到 `fps`，使帧数 = `真实时长 × fps`，与合成时间轴一致。
  3. 音频按输出时长（`输出帧数 / fps`）裁剪/补齐，使音画在合成时间轴上严格对齐。
- 容错规则：
  - `images` 和 `audio` 都为空 → 两个输出均为 `null`。
  - `images` 有、`audio` 为空 → 输出图像 + 等长**静音**。
  - `images` 为空 → 两个输出均为 `null`。

### 9. float（`ZNGBNodes/utils`）
浮点数输入节点，专为 video clip 的 `start`/`end` 提供毫秒精度数值。

- 输入：`value`(FLOAT，步进 0.0000000001，可输入小数点后 10 位)
- 输出：`float`(FLOAT)
- 输出会自动 `round` 到 **3 位小数**（毫秒精度）。

### 10. gaussian splatting converter (ply/spz)（`ZNGBNodes/3d`）
通过 [`3dgsconverter`](https://github.com/francescofugazzi/3dgsconverter) 对 **3D 高斯泼溅（Gaussian Splatting）模型**做格式转换 / 无损压缩。典型场景：把一个很大的 3DGS `.ply`压成体积很小的 `.spz`。

- 输入：
  - `input_path`(STRING)：源模型路径，支持 `.ply / .spz / .ksplat / .splat / .sog / .parquet`
  - `target_format`：目标格式 `spz`(默认) / `compressed_ply` / `3dgs` / `cc` / `ksplat` / `splat` / `sog` / `parquet`
  - `compression_level`(INT，0–9，默认 9)：压缩等级；**仅对 spz(Gzip/ZSTD，无损)、ksplat、sog 生效**
  - `spz_version`(3/4，默认 3)：仅 spz，3=Gzip，4=ZSTD + 原生 SH4
  - `force`(BOOLEAN，默认 True)：覆盖已存在的输出
  - `rgb`(BOOLEAN)：由球谐 DC 生成显式 RGB（CC/SOG/SPZ 查看器需要）
  - `crop_sh`(BOOLEAN)：只写源中实际存在的球谐系数，**关闭补齐到 SH3 的填充**（防止 SH0 模型转 `3dgs`/`cc` 时体积暴涨）
  - `extra_elements`(BOOLEAN)：保留 PLY 里的相机内外参等额外元素（仅 3dgs/cc）
  - `sh_level`(INT，-1–4，-1=保持源)：球谐阶数上限，越低越小
  - `min_opacity`(INT，0–255，0=全保留)：丢弃低于该不透明度的点，**>0 会丢点、变有损**
- 输出：`output_path`(STRING，保存路径)、`info`(STRING，体积/压缩比摘要)
- 结果固定保存到 `ComfyUI/output/3dgsconver/`，命名为 `3dgsconver_年月日_时分秒 + 扩展名`（如 `3dgsconver_20260716_120130.spz`）。
- 说明：
  - 该节点**仅支持高斯泼溅格式之间互转**，这里的 `.ply` 指高斯泼溅点云 PLY，**不支持** `glb/gltf/obj/fbx` 等网格模型互转。
  - 想无损缩小体积：用 `spz`（`compression_level 9`，可选 `spz_version 4` 更小）或 `compressed_ply`。
  - `3dgs` 是全精度无压缩格式，且默认会把球谐补齐到 SH3，SH0 源模型转它会**变大**——需要保留 PLY 时请配合 `crop_sh` 或改用 `compressed_ply`。
  - 采用子进程调用 CLI，避免把 Taichi/CUDA 初始化引入 ComfyUI 主进程。

### 11. Equirect360ToViews（`ZNGBNodes/image`）
从**等距柱状 360 全景图**中提取若干张普通透视（rectilinear）视角图，沿水平方向（yaw）均匀分布，输出为一个图像 batch。

- 输入：
  - `image`(IMAGE)：等距柱状（equirectangular）360 全景图
  - `num_views`(INT，默认 4，1–64)：沿水平线均匀取的视角数量（4 = 前/右/后/左）
  - `fov`(FLOAT，默认 90，1–179)：每个视角的水平视场角（度）
  - `pitch`(FLOAT，默认 0，-90–90)：俯仰角，+ 向上 / - 向下（度）
  - `yaw_offset`(FLOAT，默认 0，-360–360)：施加到第一个视角的旋转偏移（度）
  - `width` / `height`(INT，默认 1024)：每张输出视角图的分辨率
  - `device`(可选，`cpu`/`gpu`)：采样计算设备
- 输出：`views`(IMAGE，`num_views` 张图组成的 batch)
- `image` 为空时输出 `null`。

### 12. lens distortion correction (OpenCV)（`ZNGBNodes/image`）
使用 OpenCV Brown-Conrady 相机模型校正稳定的径向和切向畸变。它不会改变目标 FOV，和直接降低
`Equirect360ToViews.fov` 不同，而是对离图像中心距离不同的像素施加不同强度的非线性重映射。

- 输入：
  - `images`(IMAGE)：`Equirect360ToViews` 输出的单张图或 batch
  - `source_horizontal_fov`(FLOAT)：必须与 `Equirect360ToViews.fov` 一致
  - `k1 / k2 / k3`：径向畸变系数；先只调 `k1`，不够时再小幅调 `k2`，通常保持 `k3=0`
  - `p1 / p2`：切向畸变系数，只有畸变明显不对称时才调整
  - `center_x / center_y`：归一化畸变中心，默认 `(0.5, 0.5)`
  - `zoom`：校正后的裁边倍率，默认 `1.0`，边缘不理想时再增大
- 输出：`images`(IMAGE)，分辨率和 batch 数量保持不变
- 桶形畸变一般从负 `k1` 开始尝试，例如 `-0.05`、`-0.10`、`-0.20`；若弯曲加重则改用正值。
- 同一张 AI 全景图导出的全部视角必须共享同一组参数，不能逐张自动拟合，否则会破坏 3DGS 所需的多视图一致性。
- 该节点只能修复符合统一相机模型的规律性弯曲，不能恢复 AI 生成的局部家具变形、重复物体、断裂接缝或空间结构错误。

## 典型用法

`load video from url` → `get video components` → 对 `images` / `audio` 做处理（`resize image` / `get image range from batch` / `image batch multi` / `audio concat multi` / `audio crop` / `video clip`）→ Video Combine。

当 URL 为空时，整条链路输出 `null` 占位而不中断，便于把另外准备好的图像 / 音频送入最终合成。

> 多段不同帧率素材统一合成时，建议用 `video clip`：把各段原始帧率接到 `source_fps`、统一合成帧率接到 `fps`，即可在不同原始帧率下保持唇形对齐。
