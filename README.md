# Public Management Literature Analyzer

基于 `.bib/.ris` 文献摘要数据的文本挖掘脚本，核心流程为：
- 文本清洗与停用词处理
- `TF-IDF (1-3gram)` 特征表示
- `NMF` 主题建模
- 基于 seed 词典的主题-维度映射
- 导出结构化分析结果与图表

当前主脚本：`Final_260214.py`

## 1. 环境要求

- Python `3.14.2`（当前项目环境）
- scikit-learn `1.8.0`

主要依赖：
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `seaborn`（可选，用于更美观图表）
- `bibtexparser`（可选，解析 `.bib`）
- `rispy`（可选，解析 `.ris`）

## 2. 安装依赖

```bash
pip install pandas numpy matplotlib scikit-learn seaborn bibtexparser rispy
```

## 3. 数据输入

脚本支持输入：
- `.bib`
- `.ris`

默认输入文件在脚本中配置为：

```python
target_filename = "ktss.bib"
```

每条记录至少应包含摘要（`abstract`），且默认会过滤长度小于 30 的摘要。

## 4. 运行方式

在项目根目录执行：

```bash
python Final_260214.py
```

脚本会顺序运行 3 个版本：
- Main analysis：领域通用低区分词作为停用词，seed boost 关闭
- Robustness A：不加入领域通用低区分停用词，seed boost 关闭
- Robustness B：加入领域通用低区分停用词，seed boost 开启

## 5. 方法说明

### 5.1 文本预处理
- 摘要字段清洗：去换行、压缩多余空格
- 停用词：`ENGLISH_STOP_WORDS` + 自定义学术高频词
- 可选额外停用词：`digital/transformation/implementation/...`（用于降低低区分词主导）

### 5.2 特征表示
- `TfidfVectorizer`
- `ngram_range=(1,3)`
- `max_features=8000`
- `max_df=0.98`
- `min_df=2`
- `sublinear_tf=True`

### 5.3 主题建模与映射
- 使用 `sklearn.decomposition.NMF` 进行主题提取
- 默认主题数：`N_TOPICS = 10`
- 使用 5 维 seed 词典进行 topic -> dimension 映射
- 可选 seed boosting（按 token 在维度中的特异性进行加权）

## 6. 输出文件

默认会生成（或更新）：
- `analysis_report.csv`
- `analysis_report_no_domain_stopwords.csv`
- `analysis_report_seed_boost.csv`
- `Figure/Fig1.png`（Top TF-IDF 关键词）
- `Figure/Fig2.png`（Topic 分布）
- `Figure/Fig3.png`（Dimension 分布）

## 7. 项目结构（当前）

```text
.
├─ Final_260214.py
├─ ktss.bib
├─ Figure/
├─ analysis_report.csv
├─ Final_1213.ipynb
└─ ...
```

## 8. 常见调整

- 修改输入文件：编辑 `target_filename`
- 修改主题数：编辑 `N_TOPICS`
- 调整 seed 强度：编辑 `SEED_BOOST`
- 关闭绘图：将 `run_pipeline_variant(..., plot=False)`

