# RAG 知识库问答系统：技术方案 (v0.2，M2 已完成)

> AI 作品集项目 #1。目标是一个**纯 CPU、零付费 API、可离线**的 RAG 问答系统，回答必须带可核验的引用（文档名 + 页码 + 原句）。

## 1. 目标与约束

| 维度 | 要求 |
|---|---|
| 功能 | 上传 txt/md/pdf；问答结果带引用（文档名、页码、chunk id、支撑原句）；网页 demo；REST API |
| 工程 | Docker（CPU）、pytest、CI 配置、英文 README |
| 约束 | 不用 GPU；不用付费 API；模型只放 `MODEL_DIR`（默认 `./models`），不进 `~/.cache`；虚拟环境在项目内 `.venv` |
| 非目标（v0.1） | OCR、多租户和鉴权、海量（>100 万 chunk）检索、流式输出 |

## 2. 架构

```mermaid
flowchart TB
  subgraph Client
    UI[静态网页 index.html]
    CLI[curl / scripts/e2e_demo.sh]
  end
  subgraph API["FastAPI (rag.api)"]
    UP[POST /documents]
    LS[GET/DELETE /documents]
    ASK[POST /ask]
    H[GET /health]
  end
  subgraph Core["RAGService (rag.service)"]
    ING["ingest.py<br/>parse_document → split_text → Chunk(meta)"]
    EMB["embeddings.py<br/>Embedder 协议<br/>FastEmbed(ONNX) / Hash"]
    VS[("store.py<br/>SQLite 元数据 + float32 向量<br/>内存 NumPy 矩阵")]
    RET["retriever.py<br/>向量余弦 + BM25 → 加权 RRF"]
    ANS{"answerer.py<br/>Answerer 协议"}
    EX[Extractive 默认]
    OA["OpenAICompat (可选)<br/>Ollama / llama.cpp / vLLM"]
  end
  UI & CLI --> API
  UP --> ING --> EMB --> VS
  ASK --> RET --> VS
  RET --> ANS
  ANS --> EX
  ANS -.LLM_BACKEND=openai.-> OA
  OA -.失败自动降级.-> EX
```

## 3. 模块划分

| 模块 | 职责 | 关键点 |
|---|---|---|
| `config.py` | pydantic-settings 读取环境变量和 `.env` | `apply_model_cache_env()` 把 HF_HOME / HF_HUB_CACHE / FASTEMBED_CACHE_PATH / SENTENCE_TRANSFORMERS_HOME 全部指向 `MODEL_DIR` |
| `ingest.py` | 解析和分块 | PDF 按页解析，保留页码；分块优先在段落、句子、词边界切分，带 overlap，重叠起点对齐到句首；每个 chunk 带 `doc_id/doc_name/chunk_index/page/start_char` |
| `embeddings.py` | 向量化 | `Embedder` 协议；`FastEmbedEmbedder`（bge-small-en-v1.5，384 维，ONNX 量化版，约 65MB）；`HashEmbedder`（确定性哈希词袋，供测试和离线使用）；输出统一做 L2 归一化 |
| `store.py` | 持久化与检索 | SQLite 单文件：documents/chunks/kv 三张表；启动时把向量加载成 NumPy 矩阵，用暴力点积求余弦；存储 embedder 签名，换模型时会报错，避免向量空间混用；按 sha256 去重 |
| `retriever.py` | 混合检索 | 稠密 top-N 和 BM25 top-N 用加权 RRF 融合（`HYBRID_ALPHA`）；BM25 索引在语料变化时懒重建 |
| `answerer.py` | 生成答案和引用 | 见第 6 节 |
| `service.py` | 编排 | ingest：hash 去重、解析、分块、嵌入、入库；ask：检索后交给 answerer |
| `api.py` | HTTP 层 | app factory（便于测试注入）；415 不支持类型，413 超大文件，422 空文档或参数非法 |
| `static/index.html` | demo | 单文件、无构建；支持上传、列表、删除、提问，引用卡片展示；`/?q=` 深链接 |

## 4. 选型与理由

| 选择 | 备选 | 理由 |
|---|---|---|
| **fastembed (ONNX Runtime)** + `BAAI/bge-small-en-v1.5` | sentence-transformers + torch | 不依赖 torch（省约 700MB+），CPU 推理快；`cache_dir` 可显式指定；模型约 65MB；MTEB 上同量级表现好。中文场景可换 `BAAI/bge-small-zh-v1.5` 或 `paraphrase-multilingual-MiniLM-L12-v2`（fastembed 都支持，改一个环境变量即可） |
| **SQLite + NumPy** 自研向量存储 | Chroma / FAISS | 零额外依赖、单文件持久化、方便调试；10 万 chunk 以内暴力检索仍在毫秒级；接口很薄，后续可换 FAISS，上层不用改 |
| **BM25 (rank-bm25) + RRF 混合** | 纯向量 | 小模型对专有名词、数字（如 "X2"、"14 characters"）召回弱，BM25 能补上；RRF 不需要分数归一化，稳健 |
| **FastAPI** | Flask / Gradio | 自带 OpenAPI 文档、类型校验、TestClient 测试方便 |
| **静态 HTML/JS** | Gradio / Streamlit | 零依赖、与 API 同源部署；展示"真实前后端分离"更像产品 |
| **pypdf** | pdfplumber / PyMuPDF | 纯 Python、许可证友好（PyMuPDF 为 AGPL） |
| **Extractive 默认 + OpenAI 兼容接口** | 内置 llama-cpp-python | 默认零模型、零编译即可演示；OpenAI 兼容协议覆盖 Ollama、llama.cpp server、vLLM、LM Studio，一个实现接入所有本地 LLM |

## 5. 数据流

**上传**：`POST /documents` → 计算 sha256（已存在就直接返回 `duplicate: true`）→ `parse_document` 得到若干 `Page(text, page)` → `chunk_pages` 得到 `Chunk[]` → `embed_documents` → SQLite 事务写入 → 重载内存矩阵。

**问答**：`POST /ask` → `embed_query` → 向量 top-N（N = max(4k, 20)）和 BM25 top-N → 加权 RRF 取 top-k → Answerer → `{answer, backend, citations[]}`。

引用结构（API 契约）：

```json
{"index": 1, "doc_id": "fdaa0ec62cac", "doc_name": "security_policy.pdf",
 "chunk_id": "fdaa0ec62cac:1", "page": 2,
 "snippet": "Security incidents must be reported to ... within 1 hour of discovery.", "score": 0.016}
```

答案正文中的 `[n]` 与 `citations[].index` 一一对应。测试会校验答案里的标记集合与引用集合相等，不允许出现悬空引用。

## 6. 可插拔 Answerer 设计

```python
class Answerer(Protocol):
    name: str
    def answer(self, question: str, hits: list[Hit]) -> Answer: ...   # Answer(answer, citations, backend)
```

- **ExtractiveAnswerer（默认）**：把检索到的 chunk 切成句子（能容忍 Markdown/PDF 的硬换行；跳过标题行和问句，例如 FAQ 中的问题），对每句打分：`0.5 × 词项重叠率 + cos(句向量, 问句向量) + 排名加成`。取最高分 85% 以内的最多 3 句，按原文顺序拼接，并标注 `[n]`；引用的 snippet 就是支撑原句。最高分低于 `ANSWER_MIN_SCORE`（0.58，按 bge-small 校准：无关问题约 0.5，相关问题 ≥ 0.63）时回答 "not found"，不硬凑答案。
- **OpenAICompatAnswerer（可选）**：`POST {LLM_BASE_URL}/chat/completions`；system prompt 要求只依据给定来源回答，并用 `[n]` 标注；上下文按 `[n] (doc, p. X)` 格式拼接。回答后解析 `[n]`，只返回实际被引用的来源。网络错误或模型错误时**自动降级**到 extractive，`backend` 字段标记为 `openai->fallback:extractive`。
- 扩展方式：再实现一个 Answerer（例如 llama-cpp-python 进程内推理），在 `build_answerer` 中注册即可。

## 7. 评测思路（M2 落地）

1. **检索**：为 sample_docs 和一份公开文档集（如某开源项目文档）手工标注 30–50 个 `question → gold chunk/page`，计算 hit@k、MRR；对比 `HYBRID_ALPHA ∈ {0, 0.5, 1}`、chunk_size、embedding 模型。
2. **引用正确性**：自动校验 (a) 每个 `[n]` 都能解析到对应引用；(b) extractive 模式下 snippet 必须是 chunk 原文的子串；(c) LLM 模式下用 NLI 小模型或词项重叠检查答案句是否被引用片段支撑。
3. **拒答**：准备一批无答案问题，统计误答率，用来校准 `ANSWER_MIN_SCORE`。
4. **性能**：ingest 吞吐（chunks/s）、/ask P50/P95 延迟（CPU）、内存占用。
5. 输出 `make eval` 生成 Markdown 报告，写入 README 的结果表。

## 8. 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M1 骨架** | 模块分层、配置、API、extractive 回答器、SQLite 存储、Dockerfile/compose、Makefile、31 个离线测试和 1 个 slow 集成测试 | ✅ 已完成（本次交付） |
| **M2 检索与引用** | 评测集和 `make eval`（hit@k/MRR/引用正确率/拒答率）；chunk、alpha、阈值调参；中文（bge-small-zh + CJK bigram BM25）和多语言模型验证；Ollama 实测 qwen2.5:0.5b/1.5b（门控、事后引用归因）；cross-encoder 重排做过探查，结论是不采用（见 §10.4） | ✅ 已完成（见 §10） |
| **M3 前端 demo** | 引用点击高亮原文和 PDF 页预览、拖拽上传、上传进度、对话历史、LLM/extractive 切换开关；可选流式输出（SSE） | ⏳ |
| **M4 交付** | Docker 镜像实测 build/run（需要有 Docker 的环境）、GitHub Actions 跑通、录屏 GIF、README 完善（评测表、架构图、设计取舍）、可选 HF Spaces（免费 CPU）部署 | ⏳ |

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Extractive 只能摘句，不能跨文档综合 | 答案不够流畅 | 定位为"零成本、可核验"的基线；LLM 后端可选；README 中明确说明 |
| 小 embedding 模型对改写问题召回不足 | 答非所问 | 混合 BM25；M2 加重排；评测驱动调参 |
| 拒答阈值依赖具体模型 | 换模型后误答或误拒 | 已实现：`MODEL_MIN_SCORES` 按模型给默认值，由 `make eval` 校准；`ANSWER_MIN_SCORE` 可覆盖 |
| 评测集小，阈值在同一数据上校准 | 指标偏乐观（过拟合） | M3/M4 扩充题目并划分 dev/test；报告中注明 1 题 ≈ 2.9（en）或 5.3（zh）个百分点 |
| sub-1B LLM 质量差、几乎不会自行拒答 | 幻觉，或出现 "costs [3]" 这类空答案 | 默认仍用 extractive；LLM 必须配合置信度门控；文档建议至少使用 1.5B |
| 扫描版 PDF 无文本 | 上传失败 | 返回 422 并给出明确提示；后续可选 OCR（Tesseract，CPU） |
| 暴力检索规模上限 | 超过 10 万 chunk 后变慢 | `VectorStore` 接口可替换为 FAISS（`IndexFlatIP`/HNSW） |
| HF Hub 不可达或限流 | 首次下载失败 | `make download-model` 预取到 `MODEL_DIR`；运行时设 `HF_HUB_OFFLINE=1`；Docker 支持 `BAKE_MODEL=1` 把模型打进镜像 |
| 当前机器没有 Docker | Dockerfile 未实测 | 已用全新 venv 模拟 Dockerfile 的安装步骤（lock 安装 + 非 editable 安装 + 静态资源打包）并验证通过；CI 中包含 docker build job；M4 在有 Docker 的环境实测 |
| 换 embedding 模型导致向量空间混用 | 检索结果错误 | 存储时记录 embedder 签名，不一致时拒绝启动并提示 |


## 10. M2 实施记录（评测、本地 LLM、中文）

### 10.1 评测集与方法
- `eval/datasets/en.jsonl`：44 题，其中事实型 15、改写问法 13、跨文档 6、不可回答 10。语料是 `sample_docs/en/`，共 5 份文档，含 4 页 PDF。
- `eval/datasets/zh.jsonl`：26 题，其中事实 9、改写 8、跨文档 2、不可回答 7。语料是 `sample_docs/zh/`，共 4 份文档。
- 标注方式是 `(文档名, 证据子串)`，与 chunk 大小解耦：检索到的 chunk 来自金标文档且包含证据文本（忽略空白和大小写），就算命中。`validate_dataset()` 会检查所有证据都真实存在于文档中，单测也覆盖了这一点。
- 指标：
  - 检索：hit@1/3/5、MRR@10；跨文档题另算 evidence recall@5（所有证据都被召回的比例）。
  - 回答：answer acc（未拒答，且包含所有期望事实）；citation acc（回答所引的 chunk 中至少有一个是金标）；explicit [n] 率；无效 [n] 率。
  - 拒答：拒答准确率（不可回答题被拒）、误拒率（可回答题被拒）。
  - 延迟：P50/P95，端到端，CPU。
- 输出：`eval/results/<lang>__<model>.json`（逐题明细）和 `eval/results/REPORT.md`（汇总表加失败清单）。

### 10.2 检索结论（向量 vs BM25 vs 混合）
- 英文 chunk=800：混合 MRR 0.836，与纯向量 0.831 持平、hit@3 都是 97.1；纯 BM25 的 hit@1 最高（76.5），但 hit@5 只有 88.2，改写问法会漏召回。
- 中文 chunk=200：混合 MRR 0.947 最好；BM25 字 bigram 0.939；纯向量 0.912。
- alpha 扫描（0.3–1.0）没有一致的赢家，差异都在 1–2 题以内，因此保持 `HYBRID_ALPHA=0.5`。混合检索的价值在于稳健：它不会像 BM25 那样在改写问法上崩掉，也不会像纯向量那样漏掉专有名词。
- chunk 大小：英文 800 和 400 的 MRR 相近，400 时 hit@3 更低，所以默认保持 800。中文信息密度高，800 字时 4 份文档各只有 1 个 chunk，检索被"送分"，因此中文推荐 `CHUNK_SIZE=200`。

### 10.3 拒答阈值校准
- 对比了三种拒答信号：①最佳句得分（0.5·词项重叠 + 余弦），②chunk 级余弦，③**句级最大余弦**。用 balanced accuracy 衡量：③在英文上 0.856、中文上 0.947，明显好于①（0.78）和②（0.72/0.66）。原因是词项重叠会抬高"只是提到已知实体"的问题，例如 "price of the RoboArm X2"。
- 阈值在 0.40–0.90 之间扫描，取 balanced accuracy 最优平台的中点：bge-small-en 为 0.63–0.64，bge-small-zh 为 0.64，因此默认 **0.64**（`MODEL_MIN_SCORES`）；多语言 MiniLM 的分数尺度不同（en 0.425 / zh 0.545），取 0.49。它在英文上明显弱于 bge-small-en（答案准确率 79.4 vs 88.2）；在中文 chunk=200 时与 bge-small-zh 持平（89.5）。所以默认按语言选用专用模型，混合语种的知识库再考虑多语言模型。
- M1 的 0.58 用的是旧信号，已作废。
- 剩余误判：英文 3/10 个不可回答题没被拒，都是近邻问题：price of X2、X3 maximum speed、ceiling mount，置信度 0.70–0.77；部分改写题被误拒（bumps into a person、thumb drive，置信度约 0.55）。要进一步改善，需要 NLI 或更强的重排模型。

### 10.4 尝试过但没有采用的方案
- 用 cross-encoder 重排给可答性打分：试了 `Xenova/ms-marco-MiniLM-L-6-v2`、`jinaai/jina-reranker-v1-tiny-en`、`BAAI/bge-reranker-v2-m3-int8`（fastembed）。在"可答 vs 不可答"上都不比句级余弦更有区分度：tiny 模型给无关问题打了高分，int8 的 m3 在中文上分数异常。因此没有引入，相关文件已从 models/ 删除。
- jieba 分词：字 bigram 在本语料上 BM25 MRR 已达 0.939，不需要额外依赖。

### 10.5 本地 LLM（Ollama，纯 CPU）
- 安装：从 GitHub release（v0.40.9）下载，并校验 sha256；解压时排除 CUDA/ROCm/Vulkan 运行库，只剩 72MB。装在 `tools/ollama`，`OLLAMA_MODELS=models/ollama`，`HOME=tools/home-ollama`，不需要 root，也不写 `~/.ollama`。脚本是 `scripts/ollama.sh`。
- 拉模型时遇到问题：`ollama pull` 被重定向到 Cloudflare R2 的 blob CDN，而本机 DNS 把它解析成 198.18.0.1（代理的 fake-IP），Ollama 的 SSRF 防护拒绝了这个"非公网"地址。替代方案是从 Hugging Face 官方仓库下载**同一量化版本**（Q4_K_M）的 GGUF，校验 sha256 后用 `ollama create` 和 ChatML 模板导入为 `qwen2.5:0.5b`。脚本会先尝试 pull，失败后自动走这条路。另外导入了 `qwen2.5:1.5b` 作对照。
- Prompt 调优：0.5B 模型会把 `[1] (file.txt)` 这种来源头原样复述；换成 `<source id="1" file="...">` 的 XML 格式后基本不再复述。但模型很少主动写 `[n]`（explicit 率 6–25%），所以加入了**事后引用归因**：把答案逐句与各来源做词项包含度比对（≥50% 就标 `[n]`），并用 `citation_mode=attributed` 标明。
- 结果（`make eval-llm`）：
  - 英文：extractive 的答案准确率 88.2，qwen2.5:0.5b 加门控 47.1，1.5b 加门控 70.6。
  - 中文：extractive 89.5，0.5b 加门控 52.6，1.5b 加门控 68.4。
  - 不加门控时，0.5b 的拒答准确率只有 20%（英文）和 0%（中文），所以**门控是必需的**。
  - 延迟 P50：英文 0.5b 约 1.1s、1.5b 约 1.9s；中文 0.5b 约 0.5s、1.5b 约 0.7s；extractive 约 70–80ms（8 核 CPU）。
- 结论：在纯 CPU 和小模型的约束下，默认使用 extractive；LLM 定位为"可选的流畅表述"，必须配合门控，至少用 1.5B。

### 10.6 中文支持
- 分词：CJK 连续段切成字 bigram（单字段保留 unigram），去掉常见疑问和虚词 bigram；英文部分额外把字母和数字拆开（"ip54" 同时产生 "ip"、"54"）。
- 分句：支持 `。！？；`（后面不需要空格），硬换行拼接时中文之间不插空格。
- 拒答提示语按问题语言切换（"在已上传的文档中没有找到相关信息。"）。
- 指标（bge-small-zh，chunk=200）：混合检索 hit@1 89.5、hit@3 100、MRR 0.947；extractive 答案准确率 89.5、引用正确率 100、拒答准确率 100、误拒率 10.5（2 道改写题："充满一次电能扫多久"、"恢复成刚买时的状态"）。
