# Public Management Literature Analyzer

基于 `.bib/.ris` 文献摘要数据的文本挖掘脚本，核心流程包括：
- 文本清洗与停用词处理
- `TF-IDF (1-3gram)` 特征表示
- `NMF` 主题建模
- 基于 seed 词典的 `topic -> dimension` 映射
- 导出结构化结果与图表

当前主脚本：`Final_260214.py`

## 1. 环境要求

- Python `3.14.2`
- scikit-learn `1.8.0`

主要依赖：
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `gensim`（用于 `C_v` / `U_mass` 连贯性计算）
- `seaborn`（可选，用于更美观图表）
- `bibtexparser`（可选，解析 `.bib`）
- `rispy`（可选，解析 `.ris`）

## 2. 安装依赖

```bash
pip install pandas numpy matplotlib scikit-learn gensim seaborn bibtexparser rispy
```

## 3. 数据输入

脚本支持输入：
- `.bib`
- `.ris`

默认输入文件在脚本中配置为：

```python
target_filename = "ktss.bib"
```

每条记录至少应包含摘要（`abstract`），并默认过滤摘要长度 `< 30` 的记录。

## 4. 运行方式

在项目根目录执行：

```bash
python Final_260214.py
```

脚本会顺序运行 3 个版本：
- Main analysis：`domain-generic stopwords ON` + `seed boost ON`（主结果）
- Robustness A：`domain-generic stopwords OFF` + `seed boost ON`
- Robustness B：`domain-generic stopwords ON` + `seed boost OFF`

## 5. 方法说明

### 5.1 文本预处理
- 摘要清洗：去换行、压缩多余空格
- 停用词：`ENGLISH_STOP_WORDS` + 自定义学术高频词
- 可选领域低区分词：`digital/transformation/implementation/...`

### 5.2 特征表示
- `TfidfVectorizer`
- `ngram_range=(1,3)`
- `max_features=8000`
- `max_df=0.98`
- `min_df=2`
- `sublinear_tf=True`

### 5.3 主题建模与映射
- 使用 `sklearn.decomposition.NMF` 提取主题
- 默认基准主题数：`N_TOPICS = 15`（当关闭自动选 K 时生效）
- 默认开启基于 coherence 的自动选 K（`AUTO_SELECT_TOPICS_BY_COHERENCE = True`）
- 候选范围：`TOPIC_CANDIDATES = 5..20`
- 对每个 `K` 使用多随机种子重复拟合（默认 `COHERENCE_NUM_SEEDS = 30`）
- coherence 评估词数：`COHERENCE_TOP_N_TERMS = 20`
- coherence 指标：`C_v` 与 `U_mass`（按 `K` 汇总均值和标准差）
- 网格搜索 NMF 初始化：`COHERENCE_NMF_INIT = "nndsvdar"`
- 默认选 K 规则：`COHERENCE_SELECTION_RULE = "penalized_cv"`
- 复杂度惩罚：`score = C_v_mean - λ * (K-5)/(20-5)`，默认 `COHERENCE_COMPLEXITY_LAMBDA = 0.25`
- 最小主题规模约束：默认 `COHERENCE_MIN_TOPIC_DOCS = 3`，且要求跨 seed 的可行率 `>= COHERENCE_MIN_FEASIBLE_RATE = 0.8`
- 使用 5 维 seed 词典进行 topic -> dimension 映射
- 支持 specificity-weighted seed boosting

## 6. 输出文件

所有 analysis CSV 默认输出到新目录：`analysis_outputs/`。

每个运行版本会输出三类表：
- `analysis_report*.csv`（文献级结果：Topic_ID / Dimension / Objective / Method / Result）
- `*_coherence_scan.csv`（每个候选 K 的 `C_v/U_mass` 均值和标准差）
- `*_coherence_scan_raw.csv`（每个候选 K × seed 的原始分数）
- `*_topic_top_terms.csv`（每个 topic 的 top terms，含 `K`、`Term_Rank`、`Term_Weight`）
- `*_topic_dimension_crosswalk.csv`（topic 到 dimension 的映射，含 `WeightedOverlap`）

当前文件名如下：
- `analysis_outputs/analysis_report.csv`（Main analysis）
- `analysis_outputs/analysis_report_no_domain_stopwords.csv`（Robustness A）
- `analysis_outputs/analysis_report_no_seed_boost.csv`（Robustness B）

图表仍输出到 `Figure/`：
- `Figure/Fig1.png`（Top TF-IDF 关键词）
- `Figure/Fig2.png`（Topic 分布）
- `Figure/Fig3.png`（Dimension 分布）
- `Figure/Fig4_topic_coherence.png`（K 与 Coh(T) 曲线）

## 7. 项目结构（关键）

```text
.
|-- Final_260214.py
|-- ktss.bib
|-- analysis_outputs/
|-- Figure/
|-- README.md
```

## 8. 常见调整

- 修改输入文件：编辑 `target_filename`
- 修改主题数：编辑 `N_TOPICS`
- 调整 seed 强度：编辑 `SEED_BOOST`
- 关闭绘图：将 `run_pipeline_variant(..., plot=False)`
