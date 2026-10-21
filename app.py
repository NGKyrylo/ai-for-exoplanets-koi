"""
Gradio app: Research (EDA) + Model card + Predict.

Run: python app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from features import FEATURES, NEGATIVE, POSITIVE, TARGET

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "model" / "model.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
SAMPLE_BATCH = DATA / "sample_batch.csv"
TRAIN_CSV = DATA / "koi_train.csv"
CAND_CSV = DATA / "koi_candidates.csv"
WORKFLOW_IMG = ROOT / "assets" / "orange_workflow.png"

PRED_NOTE = """
Outputs per row: **prediction**, **p_confirmed**, **confidence** (= max class probability).  
Sorting by `p_confirmed` ranks “most planet-like” first.

**Score reading tip:** hard label uses a 0.5 cut
(`CONFIRMED` if `p_confirmed ≥ 0.5`, else `FALSE POSITIVE`).  
If `p_confirmed` is roughly **0.30–0.60**, treat the row as uncertain / needs review.
Clearer cases usually sit near 0–0.2 or 0.8–1.0.  
(Why the mid-range is messy — see **Model → How to read the model**.)
"""

MODEL_READING = """
### How to read the model

- The classifier outputs class probabilities; the displayed label is only a **0.5 threshold**
  on `p_confirmed`.
- Overall test metrics can look strong while individual rows in the **~0.30–0.60** band stay
  ambiguous: CONFIRMED and FALSE POSITIVE overlap in the moderate region of period/depth/SNR
  (as the Research charts show).
- **`confidence` near 0.5** means the model is undecided — useful for triage, not a failure by itself.
- That grey zone could likely be narrowed with a deeper feature study
  (e.g. measurement uncertainties `*_err*`, extra transit/stellar diagnostics).
  In this time-limited student project I kept a compact core feature set on purpose.
"""

DISCLAIMER = """
> **Attribution**  
> Research, feature design, Orange modeling, and interpretation are the work of student  
> **Kyrylo Yeromenko** (NASA Kepler KOI · exploratory analysis in R/Orange · CatBoost classification).  
> This interactive dashboard was assembled with Cursor assistance as a demo shell around  
> that student work and a reproduced `model.joblib` from the Orange training workflow.  
> Data: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (KOI cumulative).
"""

FINDINGS = """
### Research notes

**Terms**
- **Period** — how *often* the brightness dip repeats (orbital period), not how long one dip lasts.
- **Depth** — how *strongly* the star dims (ppm); extreme depths are often false positives.
- **SNR** — how *clearly* the dip stands out from noise; high SNR ≠ automatically a planet.
- **Duration** — how long a single dip lasts (hours).

**What to look for in the charts**
- Class balance, period–depth overlap, boxplots/histograms for period/depth/SNR, and the binned median trend line.
"""

CONCLUSIONS = """
### Conclusions from the charts

1. **FALSE POSITIVE dominate the sample**, so any classifier must handle imbalance and avoid
   pretending every strong dip is a planet.
2. **CONFIRMED** objects usually sit in a **moderate** period/depth region; **FALSE POSITIVE**
   are more often extreme (very short/long periods, very deep dips) and more scattered.
3. **Depth** is the clearest single separator among the viewed features; **SNR** helps but
   high SNR alone does not prove a planet.
4. Even so, **classes overlap** — especially in the “typical” middle of feature space — so
   a multi-feature model is needed, and mid-range probabilities should be read cautiously.
5. Unsupervised clustering on the same core features also pulled out extreme groups that were
   almost entirely FALSE POSITIVE, while the large typical group stayed mixed. That supports
   the supervised model, but does not replace it.
"""

# Gradio 6 rewrites custom CSS by prefixing selectors with
# `.gradio-container… .contain …`. Starting rules with `.gradio-container`
# creates a broken nested selector that never matches. Use short selectors.
APP_CSS = """
img {
  max-width: 100% !important;
  height: auto !important;
}
.prose, .markdown-text, .md {
  overflow-wrap: anywhere;
  word-break: break-word;
}
/* Plot cards: less empty chrome around charts */
.plot-row .block {
  padding: 0.35rem !important;
}
@media (max-width: 768px) {
  /* Phone: stack every Gradio row (charts + button groups) vertically. */
  .row, .plot-row {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
  }
  .row > *, .plot-row > * {
    min-width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 auto !important;
    width: 100% !important;
  }
  button {
    width: 100% !important;
    min-height: 44px !important;
  }
  .table-wrap, table, .dataframe {
    display: block !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    max-width: 100% !important;
  }
  .prose table {
    display: block !important;
    overflow-x: auto !important;
  }
  .tab-nav, .tabs > div:first-child {
    flex-wrap: wrap !important;
    gap: 0.25rem !important;
  }
  .plotly-graph-div, .js-plotly-plot {
    width: 100% !important;
  }
}
"""

# Unscoped styles via launch(head=…): bypasses Gradio 6 CSS rewriter.
# Cap width so 16:9 / ultrawide monitors don't stretch charts into a ribbon.
APP_HEAD = """
<style>
.gradio-container {
  max-width: 1100px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-left: 1.1rem !important;
  padding-right: 1.1rem !important;
}
@media (max-width: 768px) {
  .gradio-container {
    max-width: 100% !important;
    padding-left: 0.55rem !important;
    padding-right: 0.55rem !important;
  }
  div.row, div.plot-row {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
  }
  div.row > *, div.plot-row > * {
    min-width: 100% !important;
    max-width: 100% !important;
    flex: 1 1 auto !important;
    width: 100% !important;
  }
}
</style>
"""

# Back-compat aliases (older launch snippets / docs)
MOBILE_CSS = APP_CSS
MOBILE_HEAD = APP_HEAD


def _style_fig(fig, height: int = 400):
    """Readable Plotly layout for capped desktop width + phone stack."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        autosize=True,
        margin=dict(l=52, r=18, t=58, b=52),
        font=dict(size=12, color="#e7e7ea"),
        title=dict(font=dict(size=14, color="#f2f2f4")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            x=0,
            font=dict(size=11, color="#d0d0d6"),
            bgcolor="rgba(0,0,0,0)",
        ),
        colorway=["#3ddc97", "#ff6b6b", "#f0a202"],
    )
    fig.update_xaxes(
        automargin=True,
        title_font=dict(size=12),
        tickfont=dict(size=11),
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    fig.update_yaxes(
        automargin=True,
        title_font=dict(size=12),
        tickfont=dict(size=11),
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
    )
    return fig


def load_clean() -> pd.DataFrame:
    return pd.read_csv(DATA / "koi_clean.csv")


def load_candidates() -> pd.DataFrame:
    return pd.read_csv(CAND_CSV)


def load_labeled() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def _labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TARGET].isin([POSITIVE, NEGATIVE])].copy()


def research_figures():
    df = load_clean()
    labeled = _labeled(df)
    colors = {POSITIVE: "#3ddc97", NEGATIVE: "#ff6b6b", "CANDIDATE": "#f0a202"}

    counts = df[TARGET].value_counts().rename_axis(TARGET).reset_index(name="count")
    fig_bar = px.bar(
        counts, x=TARGET, y="count", color=TARGET,
        title="Disposition counts (full cleaned table)",
        color_discrete_map=colors,
        labels={"count": "Amount", TARGET: "Disposition"},
    )
    fig_bar.update_layout(showlegend=False)
    _style_fig(fig_bar, height=420)

    sample = labeled.sample(min(4000, len(labeled)), random_state=0)
    fig_scatter = px.scatter(
        sample, x="koi_period", y="koi_depth", color=TARGET, opacity=0.45,
        log_x=True, log_y=True,
        title="Period vs depth (CONFIRMED vs FALSE POSITIVE)",
        color_discrete_map=colors,
        labels={"koi_period": "Period (days, log)", "koi_depth": "Depth (ppm, log)"},
    )
    _style_fig(fig_scatter, height=420)

    fig_box_depth = px.box(
        labeled, x=TARGET, y="koi_depth", color=TARGET, log_y=True,
        title="Depth by disposition (log)",
        color_discrete_map=colors,
        labels={"koi_depth": "Depth (ppm, log)", TARGET: "Disposition"},
    )
    fig_box_depth.update_layout(showlegend=False)
    _style_fig(fig_box_depth, height=400)

    fig_box_period = px.box(
        labeled, x=TARGET, y="koi_period", color=TARGET, log_y=True,
        title="Period by disposition (log)",
        color_discrete_map=colors,
        labels={"koi_period": "Period (days, log)", TARGET: "Disposition"},
    )
    fig_box_period.update_layout(showlegend=False)
    _style_fig(fig_box_period, height=400)

    fig_box_snr = px.box(
        labeled, x=TARGET, y="koi_model_snr", color=TARGET, log_y=True,
        title="SNR by disposition (log)",
        color_discrete_map=colors,
        labels={"koi_model_snr": "SNR (log)", TARGET: "Disposition"},
    )
    fig_box_snr.update_layout(showlegend=False)
    _style_fig(fig_box_snr, height=400)

    # Plotly histogram + log_x often renders empty; bin on log10 values instead.
    hist_df = labeled.copy()
    hist_df["log_period"] = np.log10(hist_df["koi_period"])
    hist_df["log_depth"] = np.log10(hist_df["koi_depth"])

    fig_hist_period = px.histogram(
        hist_df, x="log_period", color=TARGET, barmode="overlay", opacity=0.55,
        nbins=40,
        title="Period histogram: CONFIRMED vs FALSE POSITIVE",
        color_discrete_map=colors,
        labels={"log_period": "log10(Period days)", "count": "Amount"},
    )
    _style_fig(fig_hist_period, height=400)

    fig_hist_depth = px.histogram(
        hist_df, x="log_depth", color=TARGET, barmode="overlay", opacity=0.55,
        nbins=40,
        title="Depth histogram: CONFIRMED vs FALSE POSITIVE",
        color_discrete_map=colors,
        labels={"log_depth": "log10(Depth ppm)", "count": "Amount"},
    )
    _style_fig(fig_hist_depth, height=400)

    # binned median depth vs period (lab line chart idea)
    line_df = (
        labeled.assign(
            period_bin=lambda d: pd.qcut(d["koi_period"], 20, duplicates="drop")
        )
        .groupby([TARGET, "period_bin"], observed=True)
        .agg(period_med=("koi_period", "median"), depth_med=("koi_depth", "median"))
        .reset_index()
    )
    fig_line = px.line(
        line_df, x="period_med", y="depth_med", color=TARGET, markers=True,
        log_x=True,
        title="Median depth vs period (binned)",
        color_discrete_map=colors,
        labels={"period_med": "Median period in bin", "depth_med": "Median depth"},
    )
    _style_fig(fig_line, height=400)

    return {
        "bar": fig_bar,
        "scatter": fig_scatter,
        "box_depth": fig_box_depth,
        "box_period": fig_box_period,
        "box_snr": fig_box_snr,
        "hist_depth": fig_hist_depth,
        "hist_period": fig_hist_period,
        "line": fig_line,
    }


def model_panel():
    metrics = load_metrics()
    bundle = try_load_model()
    if not metrics or bundle is None:
        return None

    auc = metrics.get("auc", 0)
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_weighted", 0)
    n_train = metrics.get("n_train", "?")
    n_test = metrics.get("n_test", "?")
    params = metrics.get("params") or bundle.get("params", {})

    md = f"""
### Model card

The classifier was designed and evaluated in **Orange**, then **reproduced here** as
`model.joblib` (same 9 features and CatBoost hyperparameters) for the Gradio demo.

**Model selection:** several learners were compared in Orange. One of the evaluation
workflows tested **Random Forest** and **Gradient Boosting** in parallel on the same
train/test split (see screenshot below). **Gradient Boosting (CatBoost)** was kept as
the final model (higher AUC / accuracy on the held-out test). A side branch in the
workflow also prepares the **CANDIDATE** table for later demo prediction — candidates
were never used as training labels.

**Orange pipeline (training branch):** Select Columns → keep CONFIRMED / FALSE POSITIVE →
drop missing feature rows → Data Sampler → learner → Test & Score → Confusion Matrix.

| Metric (hold-out test, reproduced joblib) | Value |
|-------------------------------------------|-------|
| AUC | **{auc:.3f}** |
| Classification accuracy (CA) | **{acc:.3f}** ({acc*100:.1f}%) |
| F1 (weighted) | **{f1:.3f}** |
| Train / test size | {n_train} / {n_test} |

**Hyperparameters (CatBoost):** iterations={params.get('iterations', 100)},
learning_rate={params.get('learning_rate', 0.3)}, depth={params.get('depth', 6)},
l2_leaf_reg={params.get('l2_leaf_reg', 3)}, random_seed={params.get('random_seed', 42)}.

**Features (no leakage):** {', '.join(f'`{f}`' for f in FEATURES)}  
Ignored on purpose: names/IDs, `koi_score`, `koi_fpflag_*`, and derived
`koi_prad` / `koi_teq` / `koi_insol` as model inputs.
"""

    labels = metrics.get("confusion_matrix_labels", [NEGATIVE, POSITIVE])
    cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {c}" for c in labels],
            y=[f"True {c}" for c in labels],
            colorscale="Oranges",
            text=cm,
            texttemplate="%{text}",
            textfont=dict(color="#1a1a1c"),
            showscale=False,
        )
    )
    fig_cm.update_layout(title="Confusion matrix (test set)")
    _style_fig(fig_cm, height=400)

    imp = metrics.get("feature_importances") or {}
    if not imp and bundle is not None:
        imp = {
            k: float(v)
            for k, v in zip(FEATURES, bundle["model"].get_feature_importance())
        }
    imp_s = pd.Series(imp).sort_values(ascending=True)
    fig_imp = px.bar(
        imp_s, orientation="h",
        title="Feature importance (CatBoost)",
        labels={"index": "feature", "value": "importance"},
    )
    fig_imp.update_layout(showlegend=False)
    _style_fig(fig_imp, height=400)

    return md, fig_cm, fig_imp


def try_load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def _order_pred_cols(out: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in ("prediction", "p_confirmed", "confidence", TARGET) if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def predict_frame(df: pd.DataFrame, bundle) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise gr.Error(f"Missing columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise gr.Error("Non-numeric or empty values in feature columns.")

    model = bundle["model"]
    classes = list(bundle.get("classes", model.classes_))
    proba = model.predict_proba(X)
    idx_pos = classes.index(POSITIVE) if POSITIVE in classes else 1
    pred = np.array(classes)[proba.argmax(axis=1)]
    p_conf = proba[:, idx_pos]
    confidence = proba.max(axis=1)

    out = df.copy()
    out["prediction"] = pred
    out["p_confirmed"] = np.round(p_conf, 3)
    out["confidence"] = np.round(confidence, 3)
    return _order_pred_cols(out)


def manual_predict(*values):
    bundle = try_load_model()
    if bundle is None:
        raise gr.Error("Model file missing.")
    row = pd.DataFrame([dict(zip(FEATURES, values))])
    out = predict_frame(row, bundle)
    r = out.iloc[0]
    summary = (
        f"**{r['prediction']}** · P(confirmed)={r['p_confirmed']:.3f} · "
        f"confidence={r['confidence']:.3f}"
    )
    return summary, out


def _random_from(df: pd.DataFrame, source_label: str):
    bundle = try_load_model()
    if bundle is None:
        raise gr.Error("Model file missing.")
    if df.empty:
        raise gr.Error(f"No rows available for {source_label}.")
    row = df.sample(1, random_state=None).iloc[0]
    pred_in = pd.DataFrame([row[FEATURES].to_dict()])
    if "kepoi_name" in row.index:
        pred_in["kepoi_name"] = row["kepoi_name"]
    out = predict_frame(pred_in, bundle)
    if TARGET in row.index:
        out[TARGET] = row[TARGET]
        out = _order_pred_cols(out)
    name = row["kepoi_name"] if "kepoi_name" in row.index and pd.notna(row["kepoi_name"]) else "?"
    r = out.iloc[0]
    disp_bit = f" · {TARGET}={r[TARGET]}" if TARGET in out.columns else ""
    summary = (
        f"Random **{name}** ({source_label}) → **{r['prediction']}** · "
        f"P(confirmed)={r['p_confirmed']:.3f} · confidence={r['confidence']:.3f}{disp_bit}"
    )
    return summary, out


def random_candidate():
    return _random_from(load_candidates(), "NASA CANDIDATE · unlabeled")


def random_labeled():
    return _random_from(load_labeled(), "labeled train table · CONFIRMED/FP")


def _batch_from_df(df: pd.DataFrame, sort_by: str, ascending: bool, keep_disposition: bool = False):
    bundle = try_load_model()
    if bundle is None:
        raise gr.Error("Model file missing.")
    work = df.copy()
    disposition = work[TARGET].to_numpy() if keep_disposition and TARGET in work.columns else None
    out = predict_frame(work, bundle)
    if disposition is not None:
        out[TARGET] = disposition
        out = _order_pred_cols(out)
    if sort_by not in out.columns:
        sort_by = "p_confirmed"
    out = out.sort_values(sort_by, ascending=ascending)
    path = ROOT / "model" / "last_batch_result.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out, str(path)


def batch_predict(file, sort_by, ascending):
    if file is None:
        raise gr.Error("Upload a CSV, or use one of the preset batch buttons.")
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    return _batch_from_df(df, sort_by, ascending, keep_disposition=TARGET in df.columns)


def batch_candidates(sort_by, ascending):
    return _batch_from_df(load_candidates(), sort_by, ascending, keep_disposition=False)


def batch_labeled(sort_by, ascending):
    return _batch_from_df(load_labeled(), sort_by, ascending, keep_disposition=True)


def _gradio_major() -> int:
    return int(gr.__version__.split(".")[0])


def build_app() -> gr.Blocks:
    defaults = load_clean()[FEATURES].median(numeric_only=True)
    figs = research_figures()
    model_bits = model_panel()
    model_ready = MODEL_PATH.exists()

    # Gradio <6: css on Blocks. Gradio 6+: css belongs on launch() (see __main__).
    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Source Sans 3"),
    )
    blocks_kw: dict = {"title": "AI for Exoplanets — KOI", "theme": theme}
    if _gradio_major() < 6:
        blocks_kw["css"] = APP_CSS

    with gr.Blocks(**blocks_kw) as demo:
        gr.Markdown(
            "# AI for Exoplanets Discovery\n"
            "NASA Kepler KOI · interactive research + CatBoost planet vs false-positive demo"
        )
        gr.Markdown(DISCLAIMER)
        gr.Markdown(
            "**Status:** model loaded (`challenge/model/model.joblib`)."
            if model_ready
            else "**Status:** model file missing."
        )

        with gr.Tab("Research"):
            gr.Markdown(FINDINGS)
            gr.Markdown("#### Class balance & period–depth map")
            # Desktop: side-by-side. Phone (≤768px): CSS stacks .plot-row vertically.
            with gr.Row(equal_height=False, elem_classes=["plot-row"]):
                gr.Plot(figs["bar"], show_label=False)
                gr.Plot(figs["scatter"], show_label=False)
            gr.Markdown("#### Boxplots")
            with gr.Row(equal_height=False, elem_classes=["plot-row"]):
                gr.Plot(figs["box_period"], show_label=False)
                gr.Plot(figs["box_depth"], show_label=False)
                gr.Plot(figs["box_snr"], show_label=False)
            gr.Markdown("#### Histograms & trend line")
            with gr.Row(equal_height=False, elem_classes=["plot-row"]):
                gr.Plot(figs["hist_period"], show_label=False)
                gr.Plot(figs["hist_depth"], show_label=False)
            gr.Plot(figs["line"], show_label=False)
            gr.Markdown(CONCLUSIONS)
            gr.Markdown(
                "Source: NASA Exoplanet Archive KOI cumulative · "
                "comparison charts use CONFIRMED vs FALSE POSITIVE only."
            )

        with gr.Tab("Model"):
            if model_bits is None:
                gr.Markdown("Model metrics not available yet.")
            else:
                md, fig_cm, fig_imp = model_bits
                gr.Markdown(md)
                if WORKFLOW_IMG.exists():
                    gr.Markdown(
                        "#### Orange workflow screenshot\n"
                        "Parallel Random Forest vs Gradient Boosting evaluation; "
                        "final choice: **Gradient Boosting (CatBoost)**. "
                        "Lower branch exports CANDIDATE rows for demo use only."
                    )
                    gr.Image(
                        str(WORKFLOW_IMG),
                        label="Orange workflow",
                        show_label=True,
                        container=True,
                    )
                with gr.Row(equal_height=False, elem_classes=["plot-row"]):
                    gr.Plot(fig_cm, show_label=False)
                    gr.Plot(fig_imp, show_label=False)
                gr.Markdown(MODEL_READING)

        with gr.Tab("Predict"):
            gr.Markdown(PRED_NOTE)
            with gr.Tab("Manual"):
                # Full-width number fields (one per feature), stacked.
                inputs = [
                    gr.Number(label=f, value=float(defaults[f])) for f in FEATURES
                ]
                btn_m = gr.Button("Predict", variant="primary")
                out_m = gr.Markdown()
                table_m = gr.Dataframe(wrap=True)
                btn_m.click(manual_predict, inputs, [out_m, table_m])

            with gr.Tab("Random"):
                gr.Markdown(
                    "Draw one random row and score it. "
                    "**Labeled** rows also keep NASA `koi_disposition` for a quick sanity check."
                )
                with gr.Row():
                    btn_rc = gr.Button("Random CANDIDATE", variant="primary")
                    btn_rl = gr.Button("Random labeled (CONFIRMED / FP)")
                out_r = gr.Markdown()
                table_r = gr.Dataframe(wrap=True)
                btn_rc.click(random_candidate, [], [out_r, table_r])
                btn_rl.click(random_labeled, [], [out_r, table_r])

            with gr.Tab("Batch CSV"):
                gr.Markdown(
                    "Run the model on a whole table. Presets use the project CSVs; "
                    "or upload your own file with the 9 feature columns."
                )
                sort_by = gr.Dropdown(
                    ["p_confirmed", "confidence"], value="p_confirmed", label="Sort by"
                )
                asc = gr.Checkbox(label="Ascending", value=False)
                with gr.Row():
                    btn_bc = gr.Button("Use CANDIDATES table", variant="primary")
                    btn_bl = gr.Button("Use labeled table (CONFIRMED / FP)")
                file_in = gr.File(label="Or upload your CSV", file_types=[".csv"])
                btn_bu = gr.Button("Run uploaded CSV")
                table_b = gr.Dataframe(wrap=True)
                file_out = gr.File(label="Download result")
                btn_bc.click(batch_candidates, [sort_by, asc], [table_b, file_out])
                btn_bl.click(batch_labeled, [sort_by, asc], [table_b, file_out])
                btn_bu.click(batch_predict, [file_in, sort_by, asc], [table_b, file_out])

    return demo


demo = build_app()

if __name__ == "__main__":
    # Local default; Hugging Face Spaces also runs this module.
    launch_kw: dict = {
        "server_name": "0.0.0.0",
        "server_port": 7865,
        # head= bypasses Gradio 6's CSS selector rewriter (width cap + phone stack).
        "head": APP_HEAD,
    }
    if _gradio_major() >= 6:
        launch_kw["css"] = APP_CSS
    demo.launch(**launch_kw)
