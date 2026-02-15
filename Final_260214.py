# ===========================
# Public Management Literature Analyzer (BIB/RIS)
# - TF-IDF (1-3gram) + seed-features aligned with vectorizer analyzer
# - Specificity-weighted seeds: weight = 1 / (#dims sharing the token)
# - NMF topic modeling (n_topics can be > 5)
# - Many-to-one mapping with ensure_all_dims
# ===========================

import os
import re
from datetime import datetime
from multiprocessing import freeze_support
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

# ========== optional parsers + display ==========
bibtexparser = None
rispy = None
try:
    import bibtexparser
except ImportError:
    pass

try:
    import rispy
except ImportError:
    pass

# Compatibility shim for older gensim with newer scipy builds.
try:
    import scipy.linalg.special_matrices as _scipy_special_matrices
    if not hasattr(_scipy_special_matrices, "triu"):
        _scipy_special_matrices.triu = np.triu
except Exception:
    pass

gensim_coherence_model = None
gensim_dictionary = None
try:
    from gensim.models.coherencemodel import CoherenceModel as gensim_coherence_model
    from gensim.corpora import Dictionary as gensim_dictionary
except ImportError:
    pass

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

# ===========================
# FIGURE-ONLY MODIFICATIONS START
# ===========================

# Seaborn theme for publication
if sns is not None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="Arial",
        font_scale=1.1
    )

# Matplotlib rcParams for journal-ready figures
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],     # force Arial (if not installed, matplotlib will fallback)
    "axes.unicode_minus": False,

    # Vector font embedding (important for journal PDFs)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Figure export defaults
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Clean spines
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Line widths
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,

    # Typography (paper-like)
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

def _pub_ax(ax, grid_axis="x"):
    """Lightweight publication styling for an axis."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

def _annotate_barh(ax, pad=0.15, fmt="{:d}"):
    """Annotate horizontal bar values."""
    for p in ax.patches:
        w = p.get_width()
        if np.isnan(w):
            continue
        ax.text(w + pad, p.get_y() + p.get_height()/2, fmt.format(int(round(w))),
                va="center", ha="left", fontsize=11)

def _annotate_bar(ax, pad=0.15, fmt="{:d}"):
    """Annotate vertical bar values."""
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        ax.text(p.get_x() + p.get_width()/2, h + pad, fmt.format(int(round(h))),
                va="bottom", ha="center", fontsize=11)

# ===========================
# FIGURE-ONLY MODIFICATIONS END
# ===========================


# ========== stopwords (avoid putting 'public/government/governance' here) ==========
PM_STOP_WORDS_BASE = set(ENGLISH_STOP_WORDS)
PM_STOP_WORDS_BASE.update([
    "study","paper","article","research","review","result","results","finding","findings",
    "method","methods","methodology","approach","approaches","model","models","framework",
    "analysis","data","dataset","experiment","experimental","significant","demonstrate",
    "show","shown","present","provide","provides","discuss","discussion","conclusion",
    "based","use","used","using","new","novel","various","different"
])

# These terms are intentionally treated as low-discrimination domain-generic words.
# Purpose: reduce their dominant effect so topic terms better reflect substantive differences.
PM_DOMAIN_LOW_DISCRIM_WORDS = {
    "digital", "transformation", "implementation", "sector", "egovernment", "electronic"
}

PM_STOP_WORDS_WITH_DOMAIN = sorted(PM_STOP_WORDS_BASE | PM_DOMAIN_LOW_DISCRIM_WORDS)
PM_STOP_WORDS_WITHOUT_DOMAIN = sorted(PM_STOP_WORDS_BASE)

# Main analysis default (can be overridden per run variant).
PM_STOP_WORDS = PM_STOP_WORDS_WITH_DOMAIN

def _get_stop_words(include_domain_low_discrim: bool = True):
    return PM_STOP_WORDS_WITH_DOMAIN if include_domain_low_discrim else PM_STOP_WORDS_WITHOUT_DOMAIN

# ========== 5-dimension seeds ==========
DIM_SEEDS = {
    "Public service delivery": [
        "service delivery","public service","service provision","citizen","citizens",
        "user experience","access","accessibility","convenience","satisfaction",
        "one stop","one-stop","portal","frontline","service quality"
    ],
    "Government organizational structure": [
        "organizational structure","restructuring","organization","hierarchy",
        "coordination","cross department","cross-department","interagency","inter-agency",
        "horizontal","integration","silo","network governance","collaboration"
    ],
    "Government operating logic and processes": [
        "workflow","workflows","workflow automation","approval workflow",
        "process","processes","service process","business process","business processes",
        "process automation","process standardization","standard operating procedures","sop",
        "procedure","procedures","case processing","case handling","transaction processing",
        "automation","automated workflow","streamlining","streamlined",
        "process reengineering","business process reengineering","bpr",
        "turnaround time","processing time","cycle time"
    ],
    "Departmental responsibilities and administrative authority": [
        "administrative power","authority","administrative authority","jurisdiction","delegation",
        "allocation of authority","distribution of power",
        "discretion","administrative discretion","bureaucratic discretion",
        "accountability","accountability mechanisms",
        "roles and responsibilities","role clarity","mandate",
        "decision rights","decision making authority","administrative control","power asymmetry",
        "street level","level bureaucracy","street level bureaucracy",
        "division of labor","responsibility allocation","allocation of responsibilities",
        "departmental responsibilities","functional responsibilities"
    ],
    "Leadership": [
        "leadership","political support","top management","senior management",
        "strategic vision","commitment","champion","governance capacity",
        "transformational leadership","executive","mayor","party secretary"
    ]
}

def _to_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join([str(i) for i in x if i is not None])
    return str(x)

def _clean_abstract(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _parse_bib_fallback(filepath: str):
    """Very small fallback parser for .bib when bibtexparser is unavailable."""
    raw = open(filepath, "r", encoding="utf-8", errors="ignore").read()
    entries = []
    for m in re.finditer(r"@\w+\s*\{.*?\n\}", raw, flags=re.DOTALL):
        block = m.group(0)
        fields = {}
        for key in ("title", "abstract", "year"):
            fm = re.search(
                rf"{key}\s*=\s*(\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}}|\"(?:[^\"]|\\\")*\"|[^,\n]+)",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if fm:
                val = fm.group(1).strip().strip(",")
                if val.startswith("{") and val.endswith("}"):
                    val = val[1:-1]
                if val.startswith("\"") and val.endswith("\""):
                    val = val[1:-1]
                fields[key] = val.strip()
        if fields:
            entries.append(fields)
    return entries

FIGURE_DIR = "Figure"
ANALYSIS_OUTPUT_DIR = "analysis_outputs"

def _get_unique_path(
    filepath: str,
    default_dir: str = FIGURE_DIR,
    always_timestamp: bool = False,
) -> str:
    """Return a path for output.

    - If always_timestamp=False: return original path when not existing; add timestamp only on collision.
    - If always_timestamp=True: always append timestamp (and optional index) to filename.
    Bare filenames are saved into default_dir.
    """
    if not os.path.dirname(filepath):
        os.makedirs(default_dir, exist_ok=True)
        target = os.path.join(default_dir, filepath)
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        target = filepath

    if (not always_timestamp) and (not os.path.exists(target)):
        return target

    folder, name = os.path.split(target)
    stem, ext = os.path.splitext(name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(folder, f"{stem}_{ts}{ext}")
    idx = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem}_{ts}_{idx}{ext}")
        idx += 1
    return candidate


def _ensure_parent_dir(filepath: str):
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_csv_with_fallback(df: pd.DataFrame, output_file: str):
    _ensure_parent_dir(output_file)
    try:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"[Saved] {output_file}")
        return output_file
    except PermissionError:
        fallback_path = _get_unique_path(output_file, default_dir=ANALYSIS_OUTPUT_DIR)
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"[Warn] Target file is in use; saved to fallback path: {fallback_path}")
        return fallback_path


class LiteratureAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = []
        self.df = None

        self.vectorizer = None
        self.feature_names = None
        self.tfidf_matrix = None
        self.tfidf_matrix_base = None

        self.nmf_model = None
        self.topic_values = None

        # seed-features aligned with vectorizer analyzer
        self.seed_features_by_dim = None

        # token specificity weights
        self.seed_token_dim_count = None
        self.seed_token_weight = None

        # topic->dimension map
        self.topic_to_dimension = None
        self.topic_top_terms_df = None
        self.topic_dimension_df = None
        self.n_topics = None
        self.coherence_seed_scores_df = None
        self.coherence_scores_df = None
        self.selected_n_topics = None

    def load_data(self, min_abstract_len=30):
        if not self.filepath or not os.path.exists(self.filepath):
            print(f"[Error] File not found: {self.filepath}")
            return False

        ext = os.path.splitext(self.filepath)[1].lower()
        print(f"[Info] Reading file: {self.filepath}")

        try:
            if ext == ".ris":
                if rispy is None:
                    print("[Error] rispy is not installed; cannot parse .ris in this environment.")
                    return False
                with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                    entries = rispy.load(f)
                for e in entries:
                    title = _to_str(e.get("primary_title") or e.get("title") or "No Title")
                    abstract = _to_str(e.get("abstract") or e.get("notes") or "")
                    year = _to_str(e.get("year") or e.get("publication_year") or e.get("date") or "")
                    self.data.append({"title": title.strip(), "abstract": _clean_abstract(abstract), "year": year.strip()})

            elif ext == ".bib":
                if bibtexparser is not None:
                    with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                        bib_database = bibtexparser.load(f)
                    entries = bib_database.entries
                else:
                    print("[Warn] bibtexparser is not installed; using fallback .bib parser.")
                    entries = _parse_bib_fallback(self.filepath)
                for e in entries:
                    title = _to_str(e.get("title", "No Title"))
                    abstract = _to_str(e.get("abstract", ""))
                    year = _to_str(e.get("year", ""))
                    self.data.append({"title": title.strip(), "abstract": _clean_abstract(abstract), "year": year.strip()})

            else:
                print("[Error] Unsupported format. Only .ris / .bib are supported.")
                return False

            self.df = pd.DataFrame(self.data)
            self.df["abstract"] = self.df["abstract"].fillna("").astype(str)
            self.df = self.df[self.df["abstract"].str.len() >= min_abstract_len].copy()
            self.df.reset_index(drop=True, inplace=True)

            print(f"[OK] Valid records: {len(self.df)} (abstract_len >= {min_abstract_len})")
            display(self.df[["title", "year"]].head(5))
            return True
        except Exception as e:
            print(f"[Error] Failed to load data: {e}")
            return False

    def build_tfidf(self, max_features=8000, max_df=0.98, min_df=2, ngram_range=(1, 3), stop_words=None):
        print("\n[Info] Building TF-IDF (1-3gram)...")
        if stop_words is None:
            stop_words = PM_STOP_WORDS
        stop_words_set = set(stop_words)
        domain_sw_on = PM_DOMAIN_LOW_DISCRIM_WORDS.issubset(stop_words_set)
        print(
            "[Info] Domain-generic stopwords "
            "(digital/transformation/implementation/...): "
            f"{'ON' if domain_sw_on else 'OFF'}"
        )

        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["abstract"])
        self.tfidf_matrix_base = self.tfidf_matrix.copy()
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"[OK] TF-IDF shape: {self.tfidf_matrix.shape}")

        self._prepare_seed_features_and_weights()

    def _prepare_seed_features_and_weights(self):
        analyzer = self.vectorizer.build_analyzer()
        vocab = set(self.feature_names)

        # seed-features per dimension
        self.seed_features_by_dim = {}
        for dim, seed_phrases in DIM_SEEDS.items():
            feats = set()
            for phrase in seed_phrases:
                # Avoid over-dominance from very generic tokens in one dimension.
                seed_token_blacklist = {"public", "service"}
                for tok in analyzer(phrase):
                    if tok in seed_token_blacklist:
                        continue
                    if tok in vocab:
                        feats.add(tok)
            self.seed_features_by_dim[dim] = feats

        # token -> how many dims contain it
        counts = {}
        for dim, feats in self.seed_features_by_dim.items():
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
        self.seed_token_dim_count = counts
        self.seed_token_weight = {f: 1.0 / counts[f] for f in counts}

        print("\n[Info] Seed-feature matches (hits + sample)")
        for dim in DIM_SEEDS.keys():
            hits = sorted(list(self.seed_features_by_dim[dim]))
            print(f"  - {dim}: hits={len(hits)} | {hits[:12]}")

        shared = sorted([(f, c) for f, c in counts.items() if c >= 2], key=lambda x: (-x[1], x[0]))
        if shared:
            print("\n[Info] Shared seed-features (dim_count>=2) top15:")
            print(shared[:15])

    def analyze_keywords(self, top_n=15, plot=True):
        if self.tfidf_matrix is None:
            print("[Warn] Please run build_tfidf() first.")
            return []

        mean_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[-top_n:][::-1]
        keywords = [self.feature_names[i] for i in top_idx]
        scores = [mean_tfidf[i] for i in top_idx]

        if plot:
            fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
            if sns is not None:
                sns.barplot(x=scores, y=keywords, ax=ax, edgecolor="white", linewidth=0.6)
            else:
                ypos = np.arange(len(keywords))
                ax.barh(ypos, scores, edgecolor="white", linewidth=0.6)
                ax.set_yticks(ypos)
                ax.set_yticklabels(keywords)
                ax.invert_yaxis()
            ax.set_title(f"Top {top_n} Keywords (TF-IDF)")
            ax.set_xlabel("Mean TF-IDF")
            ax.set_ylabel("")
            _pub_ax(ax, grid_axis="x")
            fig.tight_layout()
            fig_path = _get_unique_path("Fig1.png", always_timestamp=True)
            plt.savefig(fig_path, dpi=600, bbox_inches="tight")
            print(f"[Saved] {fig_path}")
            plt.show()

        return keywords

    def _boost_seed_features(self, boost_dict, verbose=True):
        """
        For each seed token, apply only one scale factor:
        max(dim_boost * specificity_weight) among all matched dimensions.
        """
        if self.seed_features_by_dim is None:
            if verbose:
                print("[Warn] seed_features_by_dim is empty. Run build_tfidf() first.")
            return 0

        term_to_idx = {t: i for i, t in enumerate(self.feature_names)}
        col_scale = {}

        for dim, feats in self.seed_features_by_dim.items():
            b = float(boost_dict.get(dim, 1.0))
            if b <= 1.0:
                continue
            for f in feats:
                j = term_to_idx.get(f, None)
                if j is None:
                    continue
                w = float(self.seed_token_weight.get(f, 1.0))
                s = b * w
                if j not in col_scale or s > col_scale[j]:
                    col_scale[j] = s

        if not col_scale:
            if verbose:
                print("[Warn] No seed-features matched for boosting.")
            return 0

        mat = self.tfidf_matrix.tocsc(copy=True)
        for j, s in col_scale.items():
            if s > 1.0:
                mat[:, j] = mat[:, j].multiply(s)
        self.tfidf_matrix = mat.tocsr()

        if verbose:
            print(f"[OK] Seed boosting done: boosted_cols={len(col_scale)}")
        return len(col_scale)

    def _prepare_coherence_inputs(self):
        if self.vectorizer is None or self.feature_names is None or self.df is None:
            print("[Warn] Please run build_tfidf() first.")
            return None, None
        if gensim_coherence_model is None or gensim_dictionary is None:
            raise ImportError(
                "gensim is required for coherence-based topic-number determination. "
                "Install it via: python -m pip install gensim. "
                "If installation fails on Python 3.14, use a Python 3.10-3.12 environment."
            )

        analyzer = self.vectorizer.build_analyzer()
        vocab = set(self.feature_names.tolist())

        texts = []
        for txt in self.df["abstract"].astype(str).tolist():
            toks = [tok for tok in analyzer(txt) if tok in vocab]
            if toks:
                texts.append(toks)

        if not texts:
            print("[Warn] Empty tokenized corpus for coherence.")
            return None, None

        dictionary = gensim_dictionary(texts)
        if len(dictionary) == 0:
            print("[Warn] Empty dictionary for coherence.")
            return None, None

        return texts, dictionary

    def _get_topic_top_terms_from_components(self, components, top_n_terms=20):
        if components is None or self.feature_names is None:
            return []
        n = max(2, int(top_n_terms))
        topics = []
        for comp in components:
            top_idx = comp.argsort()[::-1][:n]
            topics.append([self.feature_names[i] for i in top_idx])
        return topics

    def _compute_topic_diversity(self, topics):
        if not topics:
            return 0.0
        flat_terms = []
        for t in topics:
            flat_terms.extend([w for w in t if w])
        total = len(flat_terms)
        if total == 0:
            return 0.0
        unique = len(set(flat_terms))
        return float(unique) / float(total)

    def _topic_similarity_bidirectional(self, topic_sets_a, topic_sets_b):
        if not topic_sets_a or not topic_sets_b:
            return 0.0

        def _max_jaccard(src, dst):
            vals = []
            for sa in src:
                if not sa:
                    vals.append(0.0)
                    continue
                best = 0.0
                for sb in dst:
                    union = len(sa | sb)
                    if union == 0:
                        continue
                    j = float(len(sa & sb)) / float(union)
                    if j > best:
                        best = j
                vals.append(best)
            return float(np.mean(vals)) if vals else 0.0

        ab = _max_jaccard(topic_sets_a, topic_sets_b)
        ba = _max_jaccard(topic_sets_b, topic_sets_a)
        return 0.5 * (ab + ba)

    def _compute_seed_stability_scores(self, topics_by_seed):
        n = len(topics_by_seed)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        sim_mat = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                s = self._topic_similarity_bidirectional(topics_by_seed[i], topics_by_seed[j])
                sim_mat[i, j] = s
                sim_mat[j, i] = s

        stabilities = []
        for i in range(n):
            vals = [sim_mat[i, j] for j in range(n) if j != i]
            stabilities.append(float(np.mean(vals)) if vals else 1.0)
        return stabilities

    def determine_optimal_topics_by_coherence(
        self,
        k_values,
        n_seeds=30,
        top_n_terms=20,
        use_seed_boost=False,
        seed_boost=None,
        nmf_init="nndsvdar",
        random_seed_start=42,
        coherence_processes=1,
        selection_rule="coh_stab_plateau_div",
        coh_high_std_multiplier=1.0,
        stab_plateau_tol=0.015,
        stab_plateau_window=2,
        plateau_min_k=10,
        min_topic_docs=3,
        min_feasible_rate=0.8,
        plot=True,
    ):
        if self.tfidf_matrix_base is None:
            print("[Warn] Please run build_tfidf() first.")
            return None, None

        ks = sorted(set([int(k) for k in k_values if int(k) >= 2]))
        if not ks:
            print("[Warn] Invalid k_values for coherence scan.")
            return None, None
        n_seeds = max(1, int(n_seeds))
        top_n_terms = max(2, int(top_n_terms))
        init_choice = str(nmf_init).strip()
        if init_choice not in {"random", "nndsvd", "nndsvda", "nndsvdar"}:
            init_choice = "nndsvdar"
        random_seed_start = int(random_seed_start)
        coherence_processes = max(1, int(coherence_processes))
        selection_rule = str(selection_rule).strip().lower()
        if selection_rule != "coh_stab_plateau_div":
            selection_rule = "coh_stab_plateau_div"
        coh_high_std_multiplier = max(0.0, float(coh_high_std_multiplier))
        stab_plateau_tol = max(0.0, float(stab_plateau_tol))
        stab_plateau_window = max(1, int(stab_plateau_window))
        if plateau_min_k is None:
            plateau_min_k = int(min(ks) + 2)
        else:
            plateau_min_k = int(plateau_min_k)
        min_topic_docs = max(1, int(min_topic_docs))
        min_feasible_rate = float(min_feasible_rate)
        if min_feasible_rate < 0.0:
            min_feasible_rate = 0.0
        if min_feasible_rate > 1.0:
            min_feasible_rate = 1.0

        if use_seed_boost and seed_boost is None:
            seed_boost = {d: 2.5 for d in DIM_SEEDS.keys()}

        texts, dictionary = self._prepare_coherence_inputs()
        if texts is None or dictionary is None:
            print("[Warn] Failed to prepare coherence inputs.")
            return None, None

        raw_rows = []
        print(
            f"\n[Info] Topic-number determination by coherence-stability-diversity workflow "
            f"| K candidates={ks} | seeds={n_seeds} | init={init_choice} | processes={coherence_processes}"
        )
        print(
            f"[Info] Selection rule: high coherence (within {coh_high_std_multiplier:.2f}*std of best) "
            f"+ stability plateau (rolling |delta| <= {stab_plateau_tol:.4f}, window={stab_plateau_window}, "
            f"K >= {plateau_min_k}) + diversity tie-break"
        )

        for k in ks:
            seed_rows = []
            topics_by_seed = []
            cnpmi_list = []
            diversity_list = []
            max_share_list = []
            for seed_idx in range(n_seeds):
                seed = random_seed_start + seed_idx
                self.tfidf_matrix = self.tfidf_matrix_base.copy()
                if use_seed_boost:
                    self._boost_seed_features(seed_boost, verbose=False)

                nmf_tmp = NMF(
                    n_components=int(k),
                    random_state=seed,
                    init=init_choice,
                    max_iter=1500
                )
                topic_values_tmp = nmf_tmp.fit_transform(self.tfidf_matrix)
                topics = self._get_topic_top_terms_from_components(
                    nmf_tmp.components_,
                    top_n_terms=top_n_terms,
                )
                topic_sets = [set(t) for t in topics]
                topic_assign = topic_values_tmp.argmax(axis=1)
                topic_counts = np.bincount(topic_assign, minlength=int(k))
                min_topic_size = int(topic_counts.min()) if len(topic_counts) > 0 else 0
                small_topic_ratio = float((topic_counts < int(min_topic_docs)).mean()) if len(topic_counts) > 0 else 1.0
                n_docs = int(topic_counts.sum()) if len(topic_counts) > 0 else 0
                max_topic_share = float(topic_counts.max()) / float(n_docs) if n_docs > 0 else 1.0
                feasible = bool(min_topic_size >= int(min_topic_docs))
                cnpmi_score = float(gensim_coherence_model(
                    topics=topics,
                    texts=texts,
                    dictionary=dictionary,
                    coherence="c_npmi",
                    processes=coherence_processes,
                ).get_coherence())
                diversity_score = self._compute_topic_diversity(topics)
                cnpmi_list.append(cnpmi_score)
                diversity_list.append(diversity_score)
                max_share_list.append(max_topic_share)
                topics_by_seed.append(topic_sets)

                seed_rows.append({
                    "K": int(k),
                    "Seed": int(seed),
                    "C_NPMI": cnpmi_score,
                    "Topic_Diversity": diversity_score,
                    "Top_n_terms": int(top_n_terms),
                    "NMF_Init": init_choice,
                    "Coherence_Processes": int(coherence_processes),
                    "MinTopicDocs": int(min_topic_size),
                    "SmallTopicRatio": small_topic_ratio,
                    "MaxTopicShare": max_topic_share,
                    "ConstraintFeasible": feasible,
                    "SeedBoost_ON": bool(use_seed_boost),
                })

            stability_scores = self._compute_seed_stability_scores(topics_by_seed)
            for i, row in enumerate(seed_rows):
                stab = float(stability_scores[i]) if i < len(stability_scores) else 0.0
                row["Topic_Stability"] = stab
                row["Redundancy"] = 1.0 - float(row["Topic_Diversity"])
                raw_rows.append(row)

            print(
                f"  - K={k}: C_NPMI={np.mean(cnpmi_list):.6f}+/-{np.std(cnpmi_list):.6f}, "
                f"Diversity={np.mean(diversity_list):.6f}, "
                f"Stability={np.mean(stability_scores):.6f}, "
                f"MaxTopicShare={np.mean(max_share_list):.6f}"
            )

        self.tfidf_matrix = self.tfidf_matrix_base.copy()
        raw_df = pd.DataFrame(raw_rows)
        summary_df = (
            raw_df.groupby("K", as_index=False)
            .agg(
                C_NPMI_Mean=("C_NPMI", "mean"),
                C_NPMI_Std=("C_NPMI", "std"),
                Topic_Diversity_Mean=("Topic_Diversity", "mean"),
                Topic_Diversity_Std=("Topic_Diversity", "std"),
                Redundancy_Mean=("Redundancy", "mean"),
                Redundancy_Std=("Redundancy", "std"),
                Topic_Stability_Mean=("Topic_Stability", "mean"),
                Topic_Stability_Std=("Topic_Stability", "std"),
                MaxTopicShare_Mean=("MaxTopicShare", "mean"),
                MaxTopicShare_Std=("MaxTopicShare", "std"),
                MinTopicDocs_Mean=("MinTopicDocs", "mean"),
                MinTopicDocs_Min=("MinTopicDocs", "min"),
                SmallTopicRatio_Mean=("SmallTopicRatio", "mean"),
                Constraint_Feasible_Rate=("ConstraintFeasible", "mean"),
                Runs=("Seed", "count"),
            )
            .sort_values("K")
            .reset_index(drop=True)
        )
        for col in [
            "C_NPMI_Std",
            "Topic_Diversity_Std",
            "Redundancy_Std",
            "Topic_Stability_Std",
            "MaxTopicShare_Std",
        ]:
            summary_df[col] = summary_df[col].fillna(0.0)

        summary_df["Constraint_Passed"] = summary_df["Constraint_Feasible_Rate"] >= float(min_feasible_rate)

        feasible_df = summary_df[summary_df["Constraint_Passed"]].copy()
        if feasible_df.empty:
            print(
                f"[Warn] No K satisfies min-topic-size constraint "
                f"(min_topic_docs={min_topic_docs}, min_feasible_rate={min_feasible_rate}). "
                "Falling back to unconstrained selection."
            )
            feasible_df = summary_df.copy()

        best_coh_row = feasible_df.sort_values(
            ["C_NPMI_Mean", "C_NPMI_Std", "K"],
            ascending=[False, True, True]
        ).iloc[0]
        best_coh_val = float(best_coh_row["C_NPMI_Mean"])
        best_coh_std = float(best_coh_row["C_NPMI_Std"])
        coh_threshold = best_coh_val - coh_high_std_multiplier * best_coh_std

        summary_df["High_Coherence"] = summary_df["C_NPMI_Mean"] >= float(coh_threshold)
        summary_df["Stability_Delta_Abs"] = summary_df["Topic_Stability_Mean"].diff().abs()
        summary_df["Stability_Delta_Rolling"] = summary_df["Stability_Delta_Abs"].rolling(
            window=stab_plateau_window,
            min_periods=stab_plateau_window,
        ).mean()
        summary_df["Stability_Plateau"] = (
            (summary_df["K"].astype(int) >= int(plateau_min_k))
            & (summary_df["Stability_Delta_Rolling"].fillna(np.inf) <= float(stab_plateau_tol))
        )
        summary_df["Selection_Eligible"] = (
            summary_df["Constraint_Passed"]
            & summary_df["High_Coherence"]
            & summary_df["Stability_Plateau"]
        )

        final_pool = summary_df[summary_df["Selection_Eligible"]].copy()
        reason = (
            f"high coherence (C_NPMI >= {coh_threshold:.6f}) + stability plateau "
            f"(rolling_delta <= {stab_plateau_tol:.4f}, window={stab_plateau_window}, K>={plateau_min_k})"
        )
        if final_pool.empty:
            final_pool = summary_df[
                summary_df["Constraint_Passed"]
                & summary_df["High_Coherence"]
            ].copy()
            reason = (
                f"fallback: high coherence only (C_NPMI >= {coh_threshold:.6f}); "
                "stability plateau condition not met"
            )
        if final_pool.empty:
            final_pool = feasible_df.copy()
            reason = "fallback: feasible set only; high coherence condition not met"
        if final_pool.empty:
            final_pool = summary_df.copy()
            reason = "fallback: global candidate set (all K)"

        best_row = final_pool.sort_values(
            ["C_NPMI_Mean", "Topic_Stability_Mean", "Topic_Diversity_Mean", "Redundancy_Mean", "K"],
            ascending=[False, False, False, True, True]
        ).iloc[0]

        best_k = int(best_row["K"])
        best_cnpmi = float(best_row["C_NPMI_Mean"])
        best_div = float(best_row["Topic_Diversity_Mean"])
        best_red = float(best_row["Redundancy_Mean"])
        best_stab = float(best_row["Topic_Stability_Mean"])
        best_max_share = float(best_row["MaxTopicShare_Mean"])
        print(
            f"[OK] Selected K={best_k} ({reason}, C_NPMI={best_cnpmi:.6f}, "
            f"Stability={best_stab:.6f}, Diversity={best_div:.6f}, Redundancy={best_red:.6f}, "
            f"MaxShare={best_max_share:.6f})"
        )

        self.coherence_seed_scores_df = raw_df
        summary_df["Selection_Rule"] = selection_rule
        summary_df["Coh_High_Std_Multiplier"] = float(coh_high_std_multiplier)
        summary_df["Coh_High_Threshold"] = float(coh_threshold)
        summary_df["Stab_Plateau_Tol"] = float(stab_plateau_tol)
        summary_df["Stab_Plateau_Window"] = int(stab_plateau_window)
        summary_df["Plateau_Min_K"] = int(plateau_min_k)
        summary_df["Min_Topic_Docs_Threshold"] = int(min_topic_docs)
        summary_df["Min_Feasible_Rate_Threshold"] = float(min_feasible_rate)
        summary_df["Selected_K"] = (summary_df["K"].astype(int) == int(best_k))
        self.coherence_scores_df = summary_df
        self.selected_n_topics = best_k

        if plot:
            fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
            x = summary_df["K"].astype(int).tolist()
            y_coh = summary_df["C_NPMI_Mean"].astype(float).tolist()
            e_coh = summary_df["C_NPMI_Std"].astype(float).tolist()
            y_div = summary_df["Topic_Diversity_Mean"].astype(float).tolist()
            y_stab = summary_df["Topic_Stability_Mean"].astype(float).tolist()
            best_row_plot = summary_df[summary_df["K"].astype(int) == int(best_k)].iloc[0]
            best_coh_plot = float(best_row_plot["C_NPMI_Mean"])

            ax.errorbar(x, y_coh, yerr=e_coh, marker="o", linewidth=1.4, capsize=2.5, label="C_NPMI (mean+-std)")
            ax.axvline(best_k, color="tab:red", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.axhline(coh_threshold, color="tab:green", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.scatter([best_k], [best_coh_plot], marker="*", s=150, color="tab:green", zorder=5, label="Selected K")
            ax.annotate(
                f"Selected K={best_k}",
                xy=(best_k, best_coh_plot),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=10,
                color="tab:green",
            )
            ax.set_title("Topic Coherence by K (High Coherence + Stability Plateau + Diversity Tie-break)")
            ax.set_xlabel("K (Number of Topics)")
            ax.set_ylabel("C_NPMI")
            ax2 = ax.twinx()
            ax2.plot(x, y_stab, marker="d", linewidth=1.0, alpha=0.75, color="tab:brown", label="Stability mean")
            ax2.plot(x, y_div, marker="^", linewidth=1.0, alpha=0.75, color="tab:purple", label="Diversity mean")
            ax2.set_ylabel("Stability / Diversity")
            _pub_ax(ax, grid_axis="both")
            fig.tight_layout()
            fig_path = _get_unique_path("Fig4_topic_coherence.png", always_timestamp=True)
            plt.savefig(fig_path, dpi=600, bbox_inches="tight")
            print(f"[Saved] {fig_path}")
            plt.show()

        return best_k, summary_df
    def extract_topics_guided(self, n_topics=12, seed_boost=None, use_seed_boost=False, top_terms=12, plot=True):
        if self.tfidf_matrix is None:
            print("[Warn] Please run build_tfidf() first.")
            return

        if self.tfidf_matrix_base is None:
            print("[Warn] Base TF-IDF matrix missing. Rebuild TF-IDF first.")
            return

        # Always restart from base TF-IDF to avoid cumulative boosts across repeated runs.
        self.tfidf_matrix = self.tfidf_matrix_base.copy()

        if use_seed_boost:
            if seed_boost is None:
                seed_boost = {d: 2.5 for d in DIM_SEEDS.keys()}
            print(f"\n[Info] NMF: n_topics={n_topics} (specificity-weighted seed boost ON)")
            self._boost_seed_features(seed_boost)
        else:
            print(f"\n[Info] NMF: n_topics={n_topics} (seed boost OFF; seeds used for mapping only)")

        self.nmf_model = NMF(
            n_components=n_topics,
            random_state=42,
            init="nndsvdar",
            max_iter=1500
        )
        self.topic_values = self.nmf_model.fit_transform(self.tfidf_matrix)
        self.n_topics = int(n_topics)
        self.df["Topic_ID"] = (self.topic_values.argmax(axis=1) + 1).astype(int)

        topic_term_rows = []
        print("-" * 70)
        for k, comp in enumerate(self.nmf_model.components_, start=1):
            top_idx = comp.argsort()[::-1][:top_terms]
            words = []
            for r, i in enumerate(top_idx, start=1):
                term = self.feature_names[i]
                weight = float(comp[i])
                words.append(term)
                topic_term_rows.append({
                    "K": int(n_topics),
                    "Topic_ID": int(k),
                    "Term_Rank": int(r),
                    "Term": term,
                    "Term_Weight": weight
                })
            print(f"Topic {k}: {', '.join(words)}")
        print("-" * 70)
        self.topic_top_terms_df = pd.DataFrame(topic_term_rows)

        if plot:
            fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=300)
            if sns is not None:
                sns.countplot(x="Topic_ID", data=self.df, ax=ax, edgecolor="white", linewidth=0.6)
            else:
                vc = self.df["Topic_ID"].value_counts().sort_index()
                ax.bar(vc.index.astype(str), vc.values, edgecolor="white", linewidth=0.6)
            ax.set_title("Document Distribution by Topic (NMF)")
            ax.set_xlabel("Topic ID")
            ax.set_ylabel("Count")
            _pub_ax(ax, grid_axis="y")
            _annotate_bar(ax, pad=0.25, fmt="{:d}")
            fig.tight_layout()
            fig_path = _get_unique_path("Fig2.png", always_timestamp=True)
            plt.savefig(fig_path, dpi=600, bbox_inches="tight")
            print(f"[Saved] {fig_path}")
            plt.show()

    def map_topics_to_dimensions_many(self, top_k=35, min_score=1.0, ensure_all_dims=True, plot=True):
        """
        score(topic, dim) = sum(specificity_weight(token) for token in intersection)
        """
        if self.nmf_model is None:
            print("[Warn] Please run extract_topics_guided() first.")
            return None
        if self.seed_features_by_dim is None:
            print("[Warn] Please run build_tfidf() first.")
            return None

        dims = list(DIM_SEEDS.keys())
        dim_seed_sets = {d: set([x.lower() for x in self.seed_features_by_dim.get(d, set())]) for d in dims}

        topic_top_terms = {}
        for topic_idx, comp in enumerate(self.nmf_model.components_, start=1):
            top_idx = comp.argsort()[::-1][:top_k]
            topic_top_terms[topic_idx] = [self.feature_names[i] for i in top_idx]

        topics = list(topic_top_terms.keys())

        score = np.zeros((len(topics), len(dims)), dtype=float)
        for i, t in enumerate(topics):
            terms_lower = set([x.lower() for x in topic_top_terms[t]])
            for j, d in enumerate(dims):
                inter = terms_lower & dim_seed_sets[d]
                score[i, j] = sum(float(self.seed_token_weight.get(tok, 1.0)) for tok in inter)

        base_map = {}
        for i, t in enumerate(topics):
            best_j = int(score[i].argmax())
            best_dim = dims[best_j]
            best_score = float(score[i, best_j])
            base_map[t] = best_dim if best_score >= float(min_score) else "Unmapped"

        mapping = dict(base_map)

        if ensure_all_dims:
            impossible = [dims[j] for j in range(len(dims)) if score[:, j].max() < float(min_score)]
            if impossible:
                print(f"[Warn] Cannot cover these dimensions (all weighted scores < {min_score}): {impossible}")
            else:
                pairs = []
                for i, t in enumerate(topics):
                    for j, d in enumerate(dims):
                        pairs.append((float(score[i, j]), t, d))
                pairs.sort(reverse=True, key=lambda x: x[0])

                used_topics, used_dims = set(), set()
                for s, t, d in pairs:
                    if s < float(min_score):
                        break
                    if t in used_topics or d in used_dims:
                        continue
                    mapping[t] = d
                    used_topics.add(t)
                    used_dims.add(d)
                    if len(used_dims) == len(dims):
                        break

        rows = []
        for i, t in enumerate(topics):
            mapped = mapping[t]
            best_s = 0.0
            if mapped in dims:
                j = dims.index(mapped)
                best_s = float(score[i, j])
            rows.append({
                "Topic_ID": t,
                "Mapped_Dimension": mapped,
                "WeightedOverlap": round(best_s, 3),
                "Top_terms_sample": ", ".join(topic_top_terms[t][:12])
            })

        map_df = pd.DataFrame(rows).sort_values(["Mapped_Dimension", "WeightedOverlap"], ascending=[True, False])
        print(f"\n[Info] Topic -> Dimension (weighted overlap, ensure_all_dims={ensure_all_dims})")
        display(map_df)

        self.df["Topic_ID"] = self.df["Topic_ID"].astype(int)
        self.df["Dimension"] = self.df["Topic_ID"].map(mapping).astype("string").str.strip()

        show_dims = dims + ["Unmapped"]
        counts = self.df["Dimension"].value_counts(dropna=False).reindex(show_dims, fill_value=0)
        print("\n[Info] Dimension counts (show zeros):")
        print(counts)

        topic_dim_counts = pd.Series(mapping).value_counts().reindex(show_dims, fill_value=0)
        print("\n[Info] #Topics mapped into each dimension:")
        print(topic_dim_counts)

        if plot:
            counts_plot = counts.copy()
            if "Unmapped" in counts_plot.index and int(counts_plot["Unmapped"]) == 0:
                counts_plot = counts_plot.drop(index=["Unmapped"])

            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
            y = counts_plot.index.tolist()
            x = counts_plot.values.astype(int)

            ax.barh(y, x, edgecolor="white", linewidth=0.6)
            ax.set_title("Document Distribution by Dimension")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.invert_yaxis()
            _pub_ax(ax, grid_axis="x")
            _annotate_barh(ax, pad=0.25, fmt="{:d}")

            fig.tight_layout()
            fig_path = _get_unique_path("Fig3.png", always_timestamp=True)
            plt.savefig(fig_path, dpi=600, bbox_inches="tight")
            print(f"[Saved] {fig_path}")
            plt.show()

        self.topic_to_dimension = mapping
        self.topic_dimension_df = map_df.copy()
        return mapping, map_df

    def extract_key_sentences(self, sample_n=3):
        if self.df is None or self.df.empty:
            print("[Warn] No data available.")
            return

        regex = {
            "Objective": re.compile(
                r"(aims?\s+to|objective\s+is\s+to|purpose\s+is\s+to|we\s+investigate|this\s+paper\s+examines|focus(?:es)?\s+on|"
                r"\u7814\u7a76\u76ee\u7684|\u7814\u7a76\u76ee\u6807|\u672c\u6587\u65e8\u5728|\u672c\u7814\u7a76\u65e8\u5728|"
                r"\u76ee\u7684\u662f|\u76ee\u7684\u5728\u4e8e|\u805a\u7126\u4e8e|\u5173\u6ce8)",
                re.IGNORECASE
            ),
            "Method": re.compile(
                r"(we\s+propose|we\s+develop|we\s+use|we\s+utili[sz]e|method|approach|framework|case\s+study|interviews?|"
                r"coding|qualitative|quantitative|survey|regression|fsqca|experiment|"
                r"\u7814\u7a76\u65b9\u6cd5|\u65b9\u6cd5|\u91c7\u7528|\u57fa\u4e8e|\u901a\u8fc7|\u6784\u5efa|"
                r"\u5b9e\u8bc1|\u95ee\u5377|\u8bbf\u8c08|\u56de\u5f52|\u5b9e\u9a8c|\u6a21\u578b)",
                re.IGNORECASE
            ),
            "Result": re.compile(
                r"(we\s+find|results?\s+show|we\s+show|we\s+demonstrate|our\s+findings\s+indicate|this\s+study\s+reveals|"
                r"improv(?:e|es|ed)|outperform(?:s|ed)?|"
                r"\u7814\u7a76\u53d1\u73b0|\u7ed3\u679c\u8868\u660e|\u8868\u660e|\u663e\u793a|\u63ed\u793a|"
                r"\u63d0\u5347\u4e86|\u63d0\u9ad8\u4e86|\u4f18\u4e8e)",
                re.IGNORECASE
            ),
        }

        rows = []
        for text in self.df["abstract"].astype(str).tolist():
            obj = met = res = ""
            sents = [s.strip() for s in re.split(r"(?<=[\.\?\!\u3002\uff01\uff1f])\s*", text) if s.strip()]
            for s in sents:
                if not obj and regex["Objective"].search(s):
                    obj = s.strip()
                if not met and regex["Method"].search(s):
                    met = s.strip()
                if not res and regex["Result"].search(s):
                    res = s.strip()
            rows.append({"Objective": obj, "Method": met, "Result": res})

        struct_df = pd.DataFrame(rows)
        for c in ["Objective", "Method", "Result"]:
            if c in self.df.columns:
                self.df.drop(columns=[c], inplace=True)
        self.df = pd.concat([self.df, struct_df], axis=1)

        print("\n[Info] Key sentence sample:")
        display(self.df[["title", "Objective", "Method", "Result"]].head(sample_n))

    def save_results(self, output_file="analysis_report.csv"):
        if self.df is None or self.df.empty:
            print("[Warn] No results to save.")
            return
        _write_csv_with_fallback(self.df, output_file)

    def save_topic_top_terms(self, output_file="topic_top_terms.csv", run_name=None):
        if self.topic_top_terms_df is None or self.topic_top_terms_df.empty:
            print("[Warn] No topic top terms to save. Run extract_topics_guided() first.")
            return
        out_df = self.topic_top_terms_df.copy()
        if run_name is not None and str(run_name).strip():
            out_df.insert(0, "Run_Name", str(run_name))
        _write_csv_with_fallback(out_df, output_file)

    def save_topic_dimension_crosswalk(self, output_file="topic_dimension_crosswalk.csv", run_name=None):
        if self.topic_dimension_df is None or self.topic_dimension_df.empty:
            print("[Warn] No topic-dimension crosswalk to save. Run map_topics_to_dimensions_many() first.")
            return
        out_df = self.topic_dimension_df.copy()
        if self.n_topics is not None and "K" not in out_df.columns:
            out_df.insert(0, "K", int(self.n_topics))
        if run_name is not None and str(run_name).strip():
            out_df.insert(0, "Run_Name", str(run_name))
        _write_csv_with_fallback(out_df, output_file)

    def save_coherence_scores(self, output_file="topic_coherence_scan.csv", raw_output_file=None, run_name=None):
        if self.coherence_scores_df is None or self.coherence_scores_df.empty:
            print("[Warn] No topic-quality scan results to save. Run determine_optimal_topics_by_coherence() first.")
            return
        out_df = self.coherence_scores_df.copy()
        if run_name is not None and str(run_name).strip():
            out_df.insert(0, "Run_Name", str(run_name))
        _write_csv_with_fallback(out_df, output_file)

        if raw_output_file and self.coherence_seed_scores_df is not None and not self.coherence_seed_scores_df.empty:
            raw_df = self.coherence_seed_scores_df.copy()
            if run_name is not None and str(run_name).strip():
                raw_df.insert(0, "Run_Name", str(run_name))
            _write_csv_with_fallback(raw_df, raw_output_file)



# ===========================
# RUN
# ===========================
target_filename = "ktss.bib"  # ??????
N_TOPICS = 15
OUTPUT_DIR = ANALYSIS_OUTPUT_DIR
AUTO_SELECT_TOPICS_BY_COHERENCE = True
TOPIC_CANDIDATES = list(range(5, 17))
COHERENCE_NUM_SEEDS = 30
COHERENCE_TOP_N_TERMS = 20
COHERENCE_NMF_INIT = "nndsvdar"
COHERENCE_RANDOM_SEED_START = 42
COHERENCE_PROCESSES = 1
COHERENCE_SELECTION_RULE = "coh_stab_plateau_div"  # currently only "coh_stab_plateau_div"
COHERENCE_HIGH_STD_MULTIPLIER = 1.0
COHERENCE_STAB_PLATEAU_TOL = 0.010
COHERENCE_STAB_PLATEAU_WINDOW = 2
COHERENCE_PLATEAU_MIN_K = 10
COHERENCE_MIN_TOPIC_DOCS = 3
COHERENCE_MIN_FEASIBLE_RATE = 0.8
RUN_ROBUSTNESS_VARIANTS = False
RUN_MAIN_PLOT = True

SEED_BOOST = {
    "Public service delivery": 3.0,
    "Government organizational structure": 3.0,
    "Government operating logic and processes": 4.0,
    "Departmental responsibilities and administrative authority": 4.5,
    "Leadership": 2.8
}


def run_pipeline_variant(
    run_name,
    include_domain_low_discrim_stopwords=True,
    use_seed_boost=True,
    output_file="analysis_report.csv",
    auto_select_topics=AUTO_SELECT_TOPICS_BY_COHERENCE,
    topic_candidates=TOPIC_CANDIDATES,
    coherence_num_seeds=COHERENCE_NUM_SEEDS,
    coherence_top_n_terms=COHERENCE_TOP_N_TERMS,
    coherence_nmf_init=COHERENCE_NMF_INIT,
    coherence_random_seed_start=COHERENCE_RANDOM_SEED_START,
    coherence_processes=COHERENCE_PROCESSES,
    coherence_selection_rule=COHERENCE_SELECTION_RULE,
    coherence_high_std_multiplier=COHERENCE_HIGH_STD_MULTIPLIER,
    coherence_stab_plateau_tol=COHERENCE_STAB_PLATEAU_TOL,
    coherence_stab_plateau_window=COHERENCE_STAB_PLATEAU_WINDOW,
    coherence_plateau_min_k=COHERENCE_PLATEAU_MIN_K,
    coherence_min_topic_docs=COHERENCE_MIN_TOPIC_DOCS,
    coherence_min_feasible_rate=COHERENCE_MIN_FEASIBLE_RATE,
    plot=False,
):
    print("\n" + "=" * 86)
    print(f"[Run] {run_name}")
    print(
        "[Run] Rationale: include domain-generic stopwords to reduce dominance of "
        "low-discrimination terms (e.g., digital/transformation)."
    )
    print(f"[Run] domain-generic stopwords included: {include_domain_low_discrim_stopwords}")
    print(f"[Run] seed boosting enabled: {use_seed_boost}")
    print(f"[Run] coherence-based K selection enabled: {auto_select_topics}")
    print(
        f"[Run] K-selection rule: {coherence_selection_rule} "
        f"(coh_std_mult={coherence_high_std_multiplier}, "
        f"stab_plateau_tol={coherence_stab_plateau_tol}, "
        f"stab_plateau_window={coherence_stab_plateau_window}, "
        f"plateau_min_k={coherence_plateau_min_k})"
    )
    print(
        f"[Run] Min-topic-size constraint: min_topic_docs={coherence_min_topic_docs}, "
        f"min_feasible_rate={coherence_min_feasible_rate}"
    )

    analyzer = LiteratureAnalyzer(target_filename)

    if analyzer.load_data(min_abstract_len=30):
        analyzer.build_tfidf(
            max_features=8000,
            max_df=0.98,
            min_df=2,
            ngram_range=(1, 3),
            stop_words=_get_stop_words(include_domain_low_discrim_stopwords),
        )

        analyzer.analyze_keywords(top_n=15, plot=plot)

        n_topics_to_use = int(N_TOPICS)
        if auto_select_topics:
            best_k, _ = analyzer.determine_optimal_topics_by_coherence(
                k_values=topic_candidates,
                n_seeds=coherence_num_seeds,
                top_n_terms=coherence_top_n_terms,
                use_seed_boost=use_seed_boost,
                seed_boost=SEED_BOOST,
                nmf_init=coherence_nmf_init,
                random_seed_start=coherence_random_seed_start,
                coherence_processes=coherence_processes,
                selection_rule=coherence_selection_rule,
                coh_high_std_multiplier=coherence_high_std_multiplier,
                stab_plateau_tol=coherence_stab_plateau_tol,
                stab_plateau_window=coherence_stab_plateau_window,
                plateau_min_k=coherence_plateau_min_k,
                min_topic_docs=coherence_min_topic_docs,
                min_feasible_rate=coherence_min_feasible_rate,
                plot=plot,
            )
            if best_k is not None:
                n_topics_to_use = int(best_k)
            print(f"[Run] Using n_topics={n_topics_to_use} (coherence-selected)")
        else:
            print(f"[Run] Using n_topics={n_topics_to_use} (fixed)")

        analyzer.extract_topics_guided(
            n_topics=n_topics_to_use,
            seed_boost=SEED_BOOST,
            use_seed_boost=use_seed_boost,
            top_terms=12,
            plot=plot,
        )

        analyzer.map_topics_to_dimensions_many(
            top_k=60,
            min_score=1.0,
            ensure_all_dims=True,
            plot=plot,
        )

        analyzer.extract_key_sentences(sample_n=3)
        analyzer.save_results(output_file)
        base, _ = os.path.splitext(output_file)
        if auto_select_topics:
            analyzer.save_coherence_scores(
                output_file=f"{base}_coherence_scan.csv",
                raw_output_file=f"{base}_coherence_scan_raw.csv",
                run_name=run_name,
            )
        analyzer.save_topic_top_terms(
            output_file=f"{base}_topic_top_terms.csv",
            run_name=run_name,
        )
        analyzer.save_topic_dimension_crosswalk(
            output_file=f"{base}_topic_dimension_crosswalk.csv",
            run_name=run_name,
        )


def main():
    variants = [
        {
            "run_name": "Main analysis (domain-generic stopwords ON, seed boost ON)",
            "include_domain_low_discrim_stopwords": True,
            "use_seed_boost": True,
            "output_file": os.path.join(OUTPUT_DIR, "analysis_report.csv"),
            "plot": bool(RUN_MAIN_PLOT),
        },
    ]

    if RUN_ROBUSTNESS_VARIANTS:
        variants.extend([
            {
                "run_name": "Robustness A (domain-generic stopwords OFF, seed boost ON)",
                "include_domain_low_discrim_stopwords": False,
                "use_seed_boost": True,
                "output_file": os.path.join(OUTPUT_DIR, "analysis_report_no_domain_stopwords.csv"),
                "plot": False,
            },
            {
                "run_name": "Robustness B (domain-generic stopwords ON, seed boost OFF)",
                "include_domain_low_discrim_stopwords": True,
                "use_seed_boost": False,
                "output_file": os.path.join(OUTPUT_DIR, "analysis_report_no_seed_boost.csv"),
                "plot": False,
            },
        ])

    for cfg in variants:
        run_pipeline_variant(**cfg)


if __name__ == "__main__":
    freeze_support()
    main()


