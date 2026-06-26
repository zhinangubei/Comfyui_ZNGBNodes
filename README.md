# Comfyui_ZNGBNodes

一组**容错（null-tolerant）** 的 ComfyUI 自定义节点。核心设计目标：当视频 URL 为空（`None`）时，后续接入的取分量 / 拼接 / 裁剪等节点依然能正常运行而**不报错**，输出 `null` 占位，从而让整条视频处理流程可以顺利走到最终的 Video Combine。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zhinangubei/Comfyui_ZNGBNodes.git
```

重启 ComfyUI 后，节点会出现在 `ZNGBNodes/video`、`ZNGBNodes/image`、`ZNGBNodes/audio`、`ZNGBNodes/utils` 分类下。


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

## 典型用法

`load video from url` → `get video components` → 对 `images` / `audio` 做处理（`resize image` / `get image range from batch` / `image batch multi` / `audio concat multi` / `audio crop` / `video clip`）→ Video Combine。

当 URL 为空时，整条链路输出 `null` 占位而不中断，便于把另外准备好的图像 / 音频送入最终合成。

> 多段不同帧率素材统一合成时，建议用 `video clip`：把各段原始帧率接到 `source_fps`、统一合成帧率接到 `fps`，即可在不同原始帧率下保持唇形对齐。
