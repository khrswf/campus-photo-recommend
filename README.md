# 校园摄影推荐 · 成员 2 本地演示

克隆后可使用 [公开样例图库](data/sample/catalog.json) 试跑，包含两张 CC0 照片及其来源和标签。需要复现 98 张国科大照片上的三条演示时，请向成员 2 获取组内文件 `校园摄影推荐_组内图库.zip`，解压到仓库根目录。解压后应能找到 `data/formal/catalog.json` 和 `data/formal/images/`。模型权重由下方脚本下载。

## 准备 Windows 环境

已验证的配置为 Windows x64、Python 3.10、NVIDIA RTX 4060 Laptop 和 CUDA 版 PyTorch 2.9.1。把项目放在空间充足的磁盘，在项目根目录打开 PowerShell。首次安装 Python 依赖需要联网。

```powershell
$project = (Get-Location).Path
$installCache = Join-Path $project 'cache/install'
$env:TEMP = Join-Path $installCache 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $installCache 'pip'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip==26.3
& .\.venv\Scripts\python.exe -m pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
```

如果机器已用 `python` 命令指向 Python 3.10，也可以用 `python -m venv .venv`。依赖版本以 `requirements.txt` 为准。已有项目虚拟环境时可直接启动服务。

## 下载模型并生成特征

首次需要联网下载约 2 GiB 的模型。命令将权重存入仓库下的 `cache/models/`。

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\demo\download_models.py
```

然后选择图库。用公开样例时执行以下命令。

```powershell
$env:PHOTO_RECOMMEND_CATALOG = 'data/sample/catalog.json'
& .\.venv\Scripts\python.exe -X utf8 -m photo_recommend.preprocess --catalog $env:PHOTO_RECOMMEND_CATALOG
```

用组内正式图库时，把变量改成 `data/formal/catalog.json` 后，运行同一条预处理命令。若已经从完整本地演示包取得模型和特征缓存，可把其中的 `cache/models/` 和 `cache/features/` 复制到仓库根目录。

## 启动本机接口

在设置了 `PHOTO_RECOMMEND_CATALOG` 的同一个 PowerShell 窗口中运行。未设置变量时，默认使用 `data/formal/catalog.json`。

```powershell
& .\.venv\Scripts\python.exe -m uvicorn photo_recommend.api:app --host 127.0.0.1 --port 8000
```

使用公开样例时，在 `http://127.0.0.1:8000/docs` 的 `POST /recommend` 发送 `{"location_id":"tsinghua_old_library","query":"图书馆入口上方的秋叶和台阶","limit":3}`，预期返回一条 `scene_02`。

使用正式图库时，保持服务窗口运行，在另一个 PowerShell 窗口进入项目根目录，执行以下命令。脚本会检查三条实测响应、照片资源和筛选数量。

```powershell
& .\.venv\Scripts\python.exe -X utf8 .\demo\run_demo.py
```

| 演示 | 预期结果 |
| --- | --- |
| 情人坡红椅人像 | `REF_019`、`REF_053`、`REF_057`，对应三个机位组 |
| 国科大桥双人互动 | `REF_041`、`REF_031`、`REF_040`，全部满足双人条件 |
| 三四食堂单人全身红椅坐姿 | 返回空列表，原因为 `no_confirmed_matches` |

请求见 [demo/requests.json](demo/requests.json)，实测响应见 [demo/responses.json](demo/responses.json)。也可以打开 `http://127.0.0.1:8000/docs` 手动发请求。`GET /health` 用来检查服务是否启动。

## 缓存与评测

缓存按模型内容、依赖版本和图片内容校验。若更新图库或依赖，或响应出现 `features_unavailable`，运行以下命令重建正式图库缓存。

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m photo_recommend.preprocess --catalog data/formal/catalog.json
```

评测使用 4 条调权需求和 6 条留出需求。纯图文匹配与加入美学分的平均 nDCG@3 分别为 0.656 和 0.695。逐条结果见 [A/B 初评数据](reports/AB结果.json)。获取图库与缓存后，可运行 `python -m photo_recommend.evaluate` 复现。

## POST /recommend

```json
{
  "location_id": "qingrenpo_lawn",
  "query": "国科大情人坡草坪上红椅旁的校园人像",
  "limit": 3
}
```

| 请求字段 | 含义 |
| --- | --- |
| `location_id` | 地点编码，例如 `qingrenpo_lawn`。地点对照见所选图库的 `catalog.json`。 |
| `query` | 用于图文排序的中文描述。省略时按美学分排序。 |
| `person_count`、`framing`、`pose` | 可选硬条件，例如 `1`、`"full_body"`、`"sitting"`。 |
| `required_objects`、`forbidden_poses` | 可选字符串数组，例如 `["red_bench"]`。 |
| `limit`、`aesthetic_weight` | 最多返回 1–3 张，默认 3 张。美学分权重范围 0–1，默认 0.2。 |

成功响应使用 HTTP 200，`recommendations` 包含以下字段。这里节选了上面请求的第一条推荐。

```json
{
  "photo_id": "REF_019",
  "resource_id": "REF_019",
  "image_path": "images/REF_019.webp",
  "cosine_similarity": 0.1643,
  "aesthetic_score": 5.0816,
  "score": 0.8907,
  "location_id": "qingrenpo_lawn",
  "viewpoint_group": "红椅_朝圆塔",
  "viewpoint_status": "unknown"
}
```

正式图库暂用原有 `REF_XXX` 编号作为 `resource_id`。`image_path` 相对于当前 `catalog.json` 所在目录。App 根据该路径或资源索引加载对应图片。

想验证无匹配，可用 `{"location_id":"cafeterias_3_4","person_count":1,"framing":"full_body","pose":"sitting","required_objects":["red_bench"]}`。响应仍是 HTTP 200，`recommendations` 为空，`reason` 为 `no_confirmed_matches`，`filter_counts` 给出匹配、待确认和排除数量。`pending` 列出待确认照片及缺失字段。`features_unavailable` 表示有匹配照片，但可用特征缓存不足。字段校验失败返回 HTTP 422，模型或数据读取异常返回 HTTP 503。
