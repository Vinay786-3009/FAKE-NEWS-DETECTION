"""
Fake News Detection & News Credibility Analyzer
Streamlit application - run with:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics, nlp_analysis, prediction
from src.preprocessing import clean_text

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection & News Credibility Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

MENU = ["Home", "News Analyzer", "NLP Analysis", "Model Performance",
        "Dataset Analytics", "Batch Analysis", "About"]


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def kpi_card(label: str, value, help_text: str = ""):
    st.markdown(
        f"""
        <div style="background:#f5f7fb;border-radius:10px;padding:16px 18px;
                    border:1px solid #e3e8f0;height:100%">
          <div style="font-size:0.85rem;color:#5a6b85;margin-bottom:4px">{label}</div>
          <div style="font-size:1.6rem;font-weight:700;color:#1b2a4a">{value}</div>
          <div style="font-size:0.75rem;color:#8a97ad">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:#5a6b85'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"

if "goto" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("goto")

nav_index = MENU.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in MENU else 0

with st.sidebar:
    st.markdown("### 📰 Fake News Analyzer")
    st.caption("NLP + Machine Learning toolkit")
    page = st.radio("Navigation", MENU, index=nav_index, key="nav_radio", label_visibility="collapsed")
    st.session_state["nav_page"] = page
    st.divider()
    st.caption(
        "⚠️ Predictions are ML-based pattern assessments, not verified facts."
    )

# ===========================================================================
# HOME
# ===========================================================================
if page == "Home":
    st.markdown(
        """
        <div style="text-align:center;padding:32px 8px">
          <h1>FAKE NEWS DETECTION<br>&amp; NEWS CREDIBILITY ANALYZER</h1>
          <p style="font-size:1.05rem;color:#5a6b85;max-width:760px;margin:auto">
            An NLP and machine learning application that analyzes linguistic
            patterns in news content and provides an ML-based classification
            and risk assessment.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "#### 🤖 NLP Analysis\nTokenization, cleaning, keyword "
            "extraction, sentiment and sensational-language signals."
        )
    with c2:
        st.markdown(
            "#### 🧠 Machine Learning\nTF-IDF features with Logistic "
            "Regression / Linear SVM, evaluated with real metrics."
        )
    with c3:
        st.markdown(
            "#### 🔍 Explainable Prediction\nSee which words pushed the "
            "model towards its decision for every article."
        )

    st.markdown("")
    if st.button("🚀 Start Analyzing", use_container_width=True):
        st.session_state["goto"] = "News Analyzer"
        st.rerun()

    st.info(
        "This tool does **not** verify facts against authoritative sources. "
        "It classifies linguistic patterns learned from its training dataset.",
        icon="⚠️",
    )

# ===========================================================================
# NEWS ANALYZER
# ===========================================================================
elif page == "News Analyzer":
    header("🔎 News Analyzer",
           "Enter a headline and/or article. The model classifies linguistic "
           "patterns as REAL or POTENTIALLY FAKE.")

    title_in = st.text_input("Enter Headline (optional)", key="headline")
    text_in = st.text_area("Enter News Article", height=220, key="article")

    col_btn, col_clear, _ = st.columns([1, 1, 3])
    analyze = col_btn.button("Analyze News", type="primary")
    if col_clear.button("Clear"):
        st.session_state["headline"] = ""
        st.session_state["article"] = ""
        st.session_state.pop("news_res", None)
        st.rerun()

    if analyze:
        if not title_in.strip() and not text_in.strip():
            st.error("Please enter a headline or an article before analyzing.")
            st.session_state.pop("news_res", None)
        else:
            with st.spinner("Analyzing..."):
                st.session_state["news_res"] = prediction.predict_news(title_in, text_in)

    if "news_res" in st.session_state:
        res = st.session_state["news_res"]
        if not res["ok"]:
            st.warning(res["error"], icon="⚠️")
        else:
            pred_label = res["prediction"]
            conf = res.get("prediction_confidence", res["confidence"])
            cred = res["credibility"]

                # ----- Result card -----
                if pred_label == "REAL":
                    color, emoji = "#1f7a4d", "✅"
                else:
                    color, emoji = "#b3392f", "⚠️"
                st.markdown(
                    f"""
                    <div style="border-radius:12px;padding:26px;border:2px solid {color};
                                background:linear-gradient(135deg,#ffffff,#f2f6ff);
                                text-align:center">
                      <div style="font-size:1rem;color:#5a6b85">Prediction</div>
                      <div style="font-size:2.2rem;font-weight:800;color:{color}">
                        {emoji} {pred_label}
                      </div>
                      <div style="margin-top:10px;color:#394b68">
                        Model Confidence: <b>{conf*100:.1f}%</b> &nbsp;|&nbsp;
                        Risk Level: <b>{res['risk']}</b>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This is an ML classification of linguistic patterns, "
                    "NOT a verified truth judgement or official credibility "
                    "rating."
                )

                # ----- Credibility gauge -----
                lc, rc = st.columns([1, 1])
                with lc:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=cred,
                        title={"text": "Credibility Score (model confidence)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1b2a4a"},
                            "steps": [
                                {"range": [0, 25], "color": "#e26a5f"},
                                {"range": [25, 50], "color": "#f0a35e"},
                                {"range": [50, 70], "color": "#f5d76e"},
                                {"range": [70, 85], "color": "#a8d5a2"},
                                {"range": [85, 100], "color": "#7cc47f"},
                            ],
                        },
                    ))
                    fig.update_layout(height=280, margin=dict(t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                with rc:
                    st.markdown("#### Why this score?")
                    st.markdown(
                        f"- Credibility score **{cred}/100** reflects how "
                        "confident the model is that the text matches the "
                        "patterns of its REAL training class.\n"
                        f"- Risk level: **{res['risk']}** "
                        "(0–25 Very High, 26–50 High, 51–70 Moderate, "
                        "71–85 Low, 86–100 Very Low).\n"
                        "- The score is a **model confidence indicator**, not "
                        "an official credibility rating."
                    )

                # ----- Why should you be cautious? -----
                expl = res["explanation"]
                st.markdown("### Why did the model make this prediction?")
                if pred_label == "POTENTIALLY FAKE":
                    st.markdown("**Why should you be cautious?**")
                st.markdown(
                    "The indicators below are *model-based* signals: words "
                    "whose weights pushed this specific article towards a "
                    "class. They do **not** prove the article is fake or real."
                )
                fcol, rcol = st.columns(2)
                with fcol:
                    st.markdown("**Indicators towards POTENTIALLY FAKE**")
                    if expl["local_fake"]:
                        lf = pd.DataFrame(expl["local_fake"],
                                          columns=["Feature", "Weight"])
                        fig_f = px.bar(lf.sort_values("Weight"),
                                       x="Weight", y="Feature",
                                       orientation="h",
                                       title="Contributing features",
                                       color_discrete_sequence=["#b3392f"])
                        fig_f.update_layout(height=340)
                        st.plotly_chart(fig_f, use_container_width=True)
                    else:
                        st.caption("No strong single-word signals in this text.")
                with rcol:
                    st.markdown("**Indicators towards REAL**")
                    if expl["local_real"]:
                        lr = pd.DataFrame(expl["local_real"],
                                          columns=["Feature", "Weight"])
                        fig_r = px.bar(lr.sort_values("Weight"),
                                       x="Weight", y="Feature",
                                       orientation="h",
                                       title="Contributing features",
                                       color_discrete_sequence=["#1f7a4d"])
                        fig_r.update_layout(height=340)
                        st.plotly_chart(fig_r, use_container_width=True)
                    else:
                        st.caption("No strong single-word signals in this text.")

                # Sensational-language context
                sens = nlp_analysis.sensational_indicators(title_in, text_in)
                with st.expander("📋 Sensational Language Indicators "
                                 "(context, not proof)"):
                    st.markdown(
                        "Excessive punctuation, capitals and charged "
                        "vocabulary can be associated with low-quality or "
                        "sensational content, but are **not** proof of "
                        "misinformation."
                    )
                    flags = sens["flags"]
                    st.markdown(
                        f"- Exclamations: **{sens['exclamations']}**"
                        f"{' ⚠️' if flags['excessive_exclamations'] else ''}\n"
                        f"- Question marks: **{sens['questions']}**"
                        f"{' ⚠️' if flags['excessive_questions'] else ''}\n"
                        f"- Uppercase words: **{len(sens['uppercase_words'])}**"
                        f"{' ⚠️' if flags['heavy_capitals'] else ''}\n"
                        f"- Repeated punctuation (e.g. '!!!'): "
                        f"**{len(sens['repeated_punctuation'])}**"
                        f"{' ⚠️' if flags['repeated_punctuation'] else ''}\n"
                        f"- Sensational vocabulary: "
                        f"{', '.join(sens['sensational_words']) or '—'}"
                    )

# ===========================================================================
# NLP ANALYSIS
# ===========================================================================
elif page == "NLP Analysis":
    header("🤖 NLP Text Analysis",
           "Linguistic statistics, sentiment, keywords and sensational-"
           "language signals for any text you paste below.")

    title_in = st.text_input("Headline (optional)", key="nlp_headline")
    text_in = st.text_area("Article text", height=200, key="nlp_text",
                           value=st.session_state.get("article", ""))
    c_run, c_clr, _ = st.columns([1, 1, 3])
    run_nlp = c_run.button("Run NLP Analysis", type="primary")
    if c_clr.button("Clear Text", key="clear_nlp"):
        st.session_state["nlp_headline"] = ""
        st.session_state["nlp_text"] = ""
        st.session_state.pop("nlp_res", None)
        st.rerun()

    if run_nlp:
        full_text = f"{title_in} {text_in}".strip()
        if not full_text:
            st.warning("Please enter some text first.", icon="⚠️")
            st.session_state.pop("nlp_res", None)
        else:
            st.session_state["nlp_res"] = (title_in, text_in, full_text)

    if "nlp_res" in st.session_state:
        title_in, text_in, full_text = st.session_state["nlp_res"]
        stats = nlp_analysis.text_statistics(title_in, text_in)

        st.markdown("### 📊 Text Statistics")
            k = st.columns(5)
            items = [
                ("Words", stats["word_count"]),
                ("Characters", stats["char_count"]),
                ("Sentences", stats["sentence_count"]),
                ("Avg sentence length", stats["avg_sentence_length"]),
                ("Avg word length", stats["avg_word_length"]),
            ]
            for col, (label, val) in zip(k, items):
                with col:
                    kpi_card(label, val)
            k2 = st.columns(5)
            items2 = [
                ("Uppercase words", stats["uppercase_words"]),
                ("Exclamation marks", stats["exclamation_marks"]),
                ("Question marks", stats["question_marks"]),
                ("URLs", stats["urls"]),
                ("Numbers", stats["numbers"]),
            ]
            for col, (label, val) in zip(k2, items2):
                with col:
                    kpi_card(label, val)

            # ---------------- Sentiment ----------------
            sent = nlp_analysis.analyze_sentiment(full_text)
            st.markdown("### 🎭 Sentiment")
            sc, sc2 = st.columns([1, 2])
            with sc:
                color = {"Positive": "#1f7a4d", "Negative": "#b3392f",
                         "Neutral": "#7a8aa0"}[sent["label"]]
                st.markdown(
                    f"""
                    <div style="border-radius:10px;padding:18px;text-align:center;
                                border:2px solid {color}">
                      <div style="color:#5a6b85;font-size:0.9rem">Sentiment</div>
                      <div style="font-size:1.8rem;font-weight:800;color:{color}">
                        {sent['label']}</div>
                      <div style="color:#394b68">Score: {sent['score']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Sentiment is a linguistic characteristic and does "
                           "not determine whether a news article is true or "
                           "false.")
            with sc2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=sent["score"],
                    title={"text": "Sentiment score (-1 to 1)"},
                    gauge={"axis": {"range": [-1, 1]},
                           "bar": {"color": "#1b2a4a"},
                           "steps": [
                               {"range": [-1, -0.05], "color": "#e26a5f"},
                               {"range": [-0.05, 0.05], "color": "#d7dde8"},
                               {"range": [0.05, 1], "color": "#7cc47f"}]},
                ))
                fig.update_layout(height=260, margin=dict(t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            if sent["positive"] or sent["negative"]:
                with st.expander("Matched sentiment words"):
                    c1, c2 = st.columns(2)
                    c1.markdown("**Positive:** " +
                                (", ".join(sent["positive"]) or "—"))
                    c2.markdown("**Negative:** " +
                                (", ".join(sent["negative"]) or "—"))

            # ---------------- Keywords ----------------
            st.markdown("### 🔑 Top 10 Keywords")
            kw = nlp_analysis.extract_keywords(full_text, top_n=10)
            if kw:
                kw_df = pd.DataFrame(kw, columns=["Keyword", "Frequency"])
                kc1, kc2 = st.columns([2, 1])
                with kc1:
                    fig = px.bar(kw_df.sort_values("Frequency"), x="Frequency",
                                 y="Keyword", orientation="h",
                                 title="Keyword importance (frequency)",
                                 color_discrete_sequence=["#4a6fa5"])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                with kc2:
                    for i, (word, count) in enumerate(kw, start=1):
                        st.markdown(f"{i}. **{word}** ({count})")
            else:
                st.caption("No keywords could be extracted from this text.")

            # ---------------- Charts ----------------
            st.markdown("### 📈 Text Distribution Charts")
            ch1, ch2 = st.columns(2)
            with ch1:
                sents_len = nlp_analysis.sentence_lengths(full_text)
                if sents_len:
                    fig = px.histogram(x=sents_len, nbins=15,
                                       labels={"x": "Words per sentence"},
                                       title="Sentence length distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No sentences detected.")
            with ch2:
                wf = nlp_analysis.word_frequencies(full_text, top_n=15)
                if wf:
                    wf_df = pd.DataFrame(wf, columns=["Word", "Count"])
                    fig = px.bar(wf_df.sort_values("Count"), x="Count",
                                 y="Word", orientation="h",
                                 title="Most common words",
                                 color_discrete_sequence=["#6a8caf"])
                    st.plotly_chart(fig, use_container_width=True)

            # ---------------- Sensational language ----------------
            st.markdown("### 🚨 Sensational Language Indicators")
            sens = nlp_analysis.sensational_indicators(title_in, text_in)
            st.markdown(
                "These linguistic patterns can be associated with low-quality "
                "or sensational content, but they are **not** proof of "
                "misinformation."
            )
            f1, f2, f3 = st.columns(3)
            with f1:
                kpi_card("Exclamations", sens["exclamations"])
            with f2:
                kpi_card("Question marks", sens["questions"])
            with f3:
                kpi_card("Repeated punctuation", len(sens["repeated_punctuation"]))
            f4, f5, f6 = st.columns(3)
            with f4:
                kpi_card("Uppercase words", len(sens["uppercase_words"]))
            with f5:
                kpi_card("Sensational words", len(sens["sensational_words"]))
            with f6:
                kpi_card("Indicator intensity", f"{sens['intensity']}/100")
            st.caption("Flags triggered: " +
                       (", ".join(k.replace("_", " ").title()
                                  for k, v in sens["flags"].items() if v)
                        or "none"))

# ===========================================================================
# MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":
    header("📊 Model Performance",
           "Metrics computed on the held-out test split of the training "
           "dataset (never hard-coded).")

    metrics = analytics.load_metrics()
    if not metrics:
        st.warning("Model metrics not found. Run `python src/train_model.py` "
                   "first, then restart the app.", icon="⚠️")
    else:
        best = metrics["best_model"]
        st.markdown("### KPI Summary — best model: "
                    f"**{metrics['best_model_name']}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Accuracy", f"{best['accuracy']*100:.1f}%",
                     "Percentage of correctly classified articles.")
        with c2:
            kpi_card("Precision", f"{best['precision']*100:.1f}%",
                     "How often articles predicted FAKE were actually FAKE.")
        with c3:
            kpi_card("Recall", f"{best['recall']*100:.1f}%",
                     "How many FAKE articles the model detected.")
        with c4:
            kpi_card("F1 Score", f"{best['f1']*100:.1f}%",
                     "Balance between precision and recall.")

        # ----- Model comparison -----
        st.markdown("### 🏆 Model Comparison")
        comp = pd.DataFrame(metrics["comparison"])
        best_row = comp["f1"].idxmax()
        comp["Best"] = ["⭐" if i == best_row else "" for i in comp.index]
        styler = comp.style.format(
            {"accuracy": "{:.4f}", "precision": "{:.4f}",
             "recall": "{:.4f}", "f1": "{:.4f}"}
        )
        if hasattr(styler, "map"):  # pandas >= 2.1
            styler = styler.map(
                lambda _: "background-color:#eaf4ea",
                subset=pd.IndexSlice[[best_row], :],
            )
        else:
            styler = styler.applymap(
                lambda _: "background-color:#eaf4ea",
                subset=pd.IndexSlice[[best_row], :],
            )
        st.dataframe(styler, use_container_width=True)

        # ----- Confusion matrix -----
        st.markdown("### 🔢 Confusion Matrix")
        cm = metrics["confusion_matrix"]
        labels = ["FAKE", "REAL"]  # sklearn alphabetical order
        fig = px.imshow(
            cm, x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            text_auto=True, color_continuous_scale="Blues",
            title="Confusion Matrix (positive class = FAKE)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What do the cells mean? (simple language)"):
            cv = analytics.confusion_values(cm)
            st.markdown(
                f"""
                - **Predicted FAKE & actually FAKE ({cv['tp']}):**
                  fake articles the model caught (True Positive).
                - **Predicted REAL & actually FAKE ({cv['fn']}):**
                  fake articles the model missed (False Negative, risky!).
                - **Predicted FAKE & actually REAL ({cv['fp']}):**
                  real articles wrongly flagged (False Positive).
                - **Predicted REAL & actually REAL ({cv['tn']}):**
                  real articles correctly recognised (True Negative).
                """
            )

        # ----- Actual vs Predicted chart -----
        st.markdown("### Actual vs Predicted")
        cvv = analytics.confusion_values(cm)
        actual_fake = cvv["tp"] + cvv["fn"]
        actual_real = cvv["fp"] + cvv["tn"]
        pred_fake = cvv["tp"] + cvv["fp"]
        pred_real = cvv["fn"] + cvv["tn"]
        avp = pd.DataFrame({
            "Category": ["Actual FAKE", "Actual REAL",
                         "Predicted FAKE", "Predicted REAL"],
            "Count": [actual_fake, actual_real, pred_fake, pred_real],
        })
        fig = px.bar(avp, x="Category", y="Count", color="Category",
                     title="Actual vs Predicted article counts (test set)")
        st.plotly_chart(fig, use_container_width=True)

        # ----- Model & training details -----
        with st.expander("Training details"):
            ds = metrics["dataset"]
            vs = metrics["vectorizer_settings"]
            st.markdown(
                f"""
                - Training samples: **{ds['train_samples']}**,
                  test samples: **{ds['test_samples']}**
                - TF-IDF features: **{ds['features']}**
                - Class balance: {ds['class_balance']}
                - Vectorizer: max_features={vs['max_features']},
                  ngram_range={tuple(vs['ngram_range'])},
                  min_df={vs['min_df']}, max_df={vs['max_df']},
                  sublinear_tf={vs['sublinear_tf']}
                - Random state: {metrics['random_state']} (reproducible split)
                """
            )

# ===========================================================================
# DATASET ANALYTICS
# ===========================================================================
elif page == "Dataset Analytics":
    header("🗂️ Dataset Analytics",
           "Overview of the training dataset (computed live from the CSV).")

    df = analytics.load_dataset()
    if df.empty:
        st.warning("Dataset not found at data/news_dataset.csv. Run "
                   "`python src/generate_dataset.py` first.", icon="⚠️")
    else:
        ov = analytics.dataset_overview(df)
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Total articles", ov["total"])
        with c2:
            kpi_card("Real articles", f"{ov['real']} ({ov['real_pct']}%)")
        with c3:
            kpi_card("Fake articles", f"{ov['fake']} ({ov['fake_pct']}%)")

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = px.pie(
                names=["REAL", "FAKE"], values=[ov["real"], ov["fake"]],
                hole=0.45, title="REAL vs FAKE articles",
                color_discrete_sequence=["#1f7a4d", "#b3392f"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            sub = analytics.subject_distribution(df)
            if not sub.empty:
                fig = px.bar(
                    sub, x="Subject", y="Count", color="Label", barmode="group",
                    title="Articles by category/subject",
                    color_discrete_map={"REAL": "#1f7a4d", "FAKE": "#b3392f"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("The dataset has no 'subject' column.")

        with st.expander("Preview dataset"):
            st.dataframe(df.head(30), use_container_width=True)

# ===========================================================================
# BATCH ANALYSIS (upload + download)
# ===========================================================================
elif page == "Batch Analysis":
    header("📤 Batch Analysis",
           "Upload a CSV containing 'text' (and optionally 'title') columns "
           "and run predictions on every row.")

    uploaded = st.file_uploader("Upload News Dataset (CSV)",
                                type=["csv"], key="batch_csv")
    if uploaded is None:
        st.info("Waiting for a file. The CSV must contain at least a "
                "'text' column; a 'title' column is optional.")
    else:
        try:
            df_in = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            st.error("The file could not be decoded. Please upload a UTF-8 "
                     "CSV file.")
            st.stop()
        except Exception:
            st.error("The file could not be read as CSV. Please upload a "
                     "valid .csv file.")
            st.stop()

        # ----- Validation -----
        df_in.columns = [str(c).strip().lower() for c in df_in.columns]
        if df_in.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()
        if "text" not in df_in.columns:
            st.error("Missing required column: 'text'. Found columns: " +
                     ", ".join(df_in.columns))
            st.stop()

        df_in = df_in.rename(columns={"title": "title"})
        if "title" not in df_in.columns:
            df_in["title"] = ""
        df_in["text"] = df_in["text"].fillna("")
        df_in["title"] = df_in["title"].fillna("")

        before = len(df_in)
        df_in = df_in.drop_duplicates(subset=["title", "text"])
        dup_removed = before - len(df_in)
        if dup_removed:
            st.info(f"Removed {dup_removed} duplicate row(s).")

        usable = df_in["text"].str.strip().str.len() > 0
        if not usable.any():
            st.error("Every row has an empty 'text' value. Nothing to "
                     "analyze.")
            st.stop()
        if (~usable).any():
            st.warning(f"{int((~usable).sum())} row(s) have empty text and "
                       "will be skipped.")

        try:
            model, vectorizer = prediction.load_model()
        except prediction.ModelNotTrainedError as exc:
            st.error(str(exc))
            st.stop()

        rows = df_in[usable].reset_index(drop=True)
        st.markdown(f"Analyzing **{len(rows)}** article(s)...")

        # Combine title and text and clean
        combined_texts = [
            (str(t) + " " + str(b)).strip()
            for t, b in zip(rows["title"], rows["text"])
        ]
        cleaned_texts = [clean_text(ct) for ct in combined_texts]

        valid_indices = [
            i for i, c in enumerate(cleaned_texts)
            if len(c) >= prediction.MIN_TEXT_LENGTH
        ]

        preds = ["TEXT TOO SHORT"] * len(rows)
        confs = [None] * len(rows)
        creds = [None] * len(rows)
        risks = [""] * len(rows)

        if valid_indices:
            valid_cleaned = [cleaned_texts[i] for i in valid_indices]
            batch_preds, batch_probs = prediction.predict_batch(valid_cleaned)
            for idx, p, prob in zip(valid_indices, batch_preds, batch_probs):
                is_fake = (p == "FAKE")
                preds[idx] = "POTENTIALLY FAKE" if is_fake else "REAL"
                if not np.isnan(prob):
                    p_fake = float(prob)
                    pred_conf = p_fake if is_fake else (1.0 - p_fake)
                    confs[idx] = round(pred_conf * 100, 1)
                    score = prediction.credibility_score(p_fake)
                    creds[idx] = score
                    risks[idx] = prediction.risk_level(score)

        rows["prediction"] = preds
        rows["model_confidence"] = [f"{c}%" if c is not None else "" for c in confs]
        rows["credibility_score"] = creds
        rows["risk_level"] = risks

        # ----- Summary -----
        n_pred = rows[rows["prediction"].isin(["REAL", "POTENTIALLY FAKE"])]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            kpi_card("Rows analyzed", len(rows))
        with b2:
            kpi_card("Predicted REAL", int((n_pred["prediction"] == "REAL").sum()))
        with b3:
            kpi_card("Predicted POTENTIALLY FAKE",
                     int((n_pred["prediction"] == "POTENTIALLY FAKE").sum()))
        with b4:
            kpi_card("Too short / skipped",
                     int((rows["prediction"] == "TEXT TOO SHORT").sum()))

        if not n_pred.empty:
            fig = px.pie(names=n_pred["prediction"].value_counts().index,
                         values=n_pred["prediction"].value_counts().values,
                         hole=0.4, title="Batch prediction distribution",
                         color_discrete_sequence=["#1f7a4d", "#b3392f"])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Results")
        st.dataframe(rows, use_container_width=True)

        # ----- Download -----
        out_buf = io.StringIO()
        rows.to_csv(out_buf, index=False)
        st.download_button(
            "⬇️ Download Results (CSV)",
            data=out_buf.getvalue(),
            file_name="fake_news_batch_results.csv",
            mime="text/csv",
        )
        st.caption("Predictions are ML-based pattern assessments, not "
                   "verified facts.")

# ===========================================================================
# ABOUT
# ===========================================================================
elif page == "About":
    header("ℹ️ About",
           "Fake News Detection & News Credibility Analyzer")

    st.markdown(
        """
        ### Purpose
        This project demonstrates an end-to-end **NLP + Machine Learning**
        pipeline: text preprocessing, TF-IDF feature engineering, supervised
        classification, model evaluation and explainability — wrapped in an
        interactive Streamlit dashboard.

        ### Technology stack
        - **Python** · Pandas · NumPy
        - **Scikit-learn** (TF-IDF, Logistic Regression, LinearSVC, calibration)
        - **NLTK** (tokenization, stopwords, lemmatization)
        - **Plotly / Matplotlib** (visual analytics)
        - **Streamlit** (web application)
        - **joblib** (model persistence)

        ### Dataset
        The bundled `data/news_dataset.csv` is a **synthetic demonstration
        dataset** generated by `src/generate_dataset.py`. It contains
        template-based articles — neutral, source-attributed reporting for
        the REAL class and sensational-style writing for the FAKE class.
        No real-world news claims are fabricated and labelled as factual.
        You can replace it with a public dataset such as the Kaggle
        *Fake and Real News Dataset* and retrain with
        `python src/train_model.py`.

        ### Machine learning approach
        1. Combine title + body, clean the text (lowercase, HTML/URL removal,
           punctuation, tokenization, stopword removal, lemmatization).
        2. Vectorize with TF-IDF (word 1–2 grams, sublinear TF).
        3. Train **Logistic Regression** and **LinearSVC** (calibrated) on a
           stratified, reproducible split.
        4. Compare Accuracy / Precision / Recall / F1 and keep the best model
           by F1 score.
        5. Explain predictions using per-feature linear weights.

        ### ⚠️ Limitations (important)
        - This system does **NOT** verify facts against authoritative sources.
        - It only predicts patterns learned from its training dataset.
        - A prediction of **REAL** does not mean the article is factually
          verified.
        - A prediction of **POTENTIALLY FAKE** does not prove the article is
          false.
        - The bundled dataset is synthetic; metrics reflect that corpus and
          will differ on real-world data.
        - Users should verify important claims using reliable primary sources
          and professional fact-checking organizations.

        ### 🔮 Future enhancements
        - Fact-checking API integration
        - Source credibility analysis & news-source reputation
        - Real-time news verification
        - Multilingual fake-news detection
        - Transformer models such as BERT
        - Browser extension
        """
    )
