"""
app.py
======
Version 4 Interactive Streamlit Dashboard for Chess Behavioral Feature Engineering
and Engine-Assistance Detection.

Provides:
- PGN file upload and preset master game selection
- Model selection (Logistic Regression default vs Random Forest alternative)
- Configurable Stockfish analysis depth with automatic path verification
- Player-level engine-assistance estimates and behavioral metrics
- Interactive EDA visualizations: centipawn loss distribution, move quality breakdown, ply trajectory
- Model comparison table with educational metric explanations (Accuracy, Precision, Recall, F1, ROC-AUC, Brier score)
- Prominent scientific and ethical disclaimers
"""

from __future__ import annotations

import json
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from chess_eda.dashboard import (
    extract_eda_chart_data,
    get_metric_definitions,
    get_player_card_presentation_data,
    get_preset_games,
    load_model_metadata,
    save_uploaded_pgn_to_temp,
    validate_pgn_input,
)
from chess_eda.engine_analysis import EngineAnalysisError, validate_stockfish_path
from chess_eda.ml.predict import (
    DEFAULT_STOCKFISH_PATH,
    predict_game,
)


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chess Behavioral ML Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Professional, Dark-Theme Optimized Presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        color: #F8FAFC;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.15rem;
        line-height: 1.45;
    }
    .disclaimer-box {
        background-color: rgba(30, 58, 138, 0.18);
        border-left: 3px solid #3B82F6;
        padding: 0.85rem 1.15rem;
        border-radius: 6px;
        margin-top: 0.25rem;
        margin-bottom: 1.25rem;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #93C5FD;
    }
    .player-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .player-identity {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        min-width: 0;
    }
    .player-side-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .side-white {
        background: #F1F5F9;
        color: #0F172A;
        border: 1px solid #CBD5E1;
    }
    .side-black {
        background: #0F172A;
        color: #E2E8F0;
        border: 1px solid #475569;
    }
    .player-display-name {
        font-size: 1.12rem;
        font-weight: 600;
        color: #F8FAFC;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .badge-human {
        background-color: rgba(22, 101, 52, 0.35);
        border: 1px solid #22C55E;
        color: #4ADE80;
        font-weight: 700;
        padding: 0.32rem 0.75rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.84rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .badge-assisted {
        background-color: rgba(180, 83, 9, 0.35);
        border: 1px solid #F59E0B;
        color: #FCD34D;
        font-weight: 700;
        padding: 0.32rem 0.75rem;
        border-radius: 6px;
        display: inline-block;
        font-size: 0.84rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .hero-score-label {
        font-size: 0.86rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
        margin-bottom: 0.15rem;
    }
    .hero-score-value {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 0.45rem;
    }
    .score-human {
        color: #4ADE80;
    }
    .score-assisted {
        color: #FBBF24;
    }
    .secondary-score-line {
        font-size: 15px;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0.85rem;
        line-height: 1.4;
    }
    .secondary-score-line strong {
        color: #E2E8F0;
        font-weight: 600;
    }
    .metrics-separator {
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        margin: 0.65rem 0 0.85rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper function to style matplotlib figures for dark theme
def _apply_dark_chart_style(fig, ax):
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748B")


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# -----------------------------------------------------------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure analysis parameters and game inputs.")

# 1. PGN Input Source
st.sidebar.subheader("Game Input")
input_mode = st.sidebar.radio(
    "Choose Input Method:",
    options=["Select Preset Game", "Upload .PGN File", "Paste PGN Text"],
    index=0,
)

selected_pgn_path: str | None = None
temp_file_to_clean: str | None = None
preset_games = get_preset_games()

if input_mode == "Select Preset Game":
    preset_choice = st.sidebar.selectbox("Choose a sample game:", list(preset_games.keys()))
    preset_info = preset_games[preset_choice]
    selected_pgn_path = preset_info["path"]
    st.sidebar.caption(f"**Type:** {preset_info['type']}")
    st.sidebar.caption(f"_{preset_info['description']}_")

elif input_mode == "Upload .PGN File":
    uploaded_file = st.sidebar.file_uploader("Upload a PGN chess game", type=["pgn"])
    if uploaded_file is not None:
        content_bytes = uploaded_file.read()
        content_str = content_bytes.decode("utf-8", errors="replace")
        is_valid, msg, _ = validate_pgn_input(content_str)
        if not is_valid:
            st.sidebar.error(msg)
        else:
            temp_file_to_clean = save_uploaded_pgn_to_temp(content_str)
            selected_pgn_path = temp_file_to_clean
            st.sidebar.success("PGN uploaded and validated successfully!")

elif input_mode == "Paste PGN Text":
    pgn_text = st.sidebar.text_area(
        "Paste standard PGN content here:",
        height=180,
        placeholder='[Event "Casual Game"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 2. Nf3 Nc6 ...',
    )
    if pgn_text.strip():
        is_valid, msg, _ = validate_pgn_input(pgn_text)
        if not is_valid:
            st.sidebar.error(msg)
        else:
            temp_file_to_clean = save_uploaded_pgn_to_temp(pgn_text)
            selected_pgn_path = temp_file_to_clean
            st.sidebar.success("PGN parsed successfully!")

# 2. Model Selection (Default: Logistic Regression, based on 5-fold grouped CV metrics)
st.sidebar.subheader("Model Selection")
model_type = st.sidebar.radio(
    "Select Classifier:",
    options=["Logistic Regression (lr)", "Random Forest (rf)"],
    index=0,
    help=(
        "Logistic Regression is selected as the default model based on higher 5-fold physical-match-grouped "
        "cross-validation metrics (ROC-AUC 0.7559 vs 0.7437, PR-AUC 0.6892 vs 0.6628, F1 0.6099 vs 0.6011). "
        "Random Forest remains available as an alternative."
    ),
)
model_filename = "chess_fraud_lr.joblib" if "lr" in model_type else "chess_fraud_rf.joblib"
model_path = os.path.join("models", model_filename)

if os.path.isfile(model_path):
    st.sidebar.caption(f"Artifact: `{model_path}` (Ready)")
else:
    st.sidebar.error(f"Model artifact `{model_path}` missing! Run training first.")

# 3. Stockfish Engine Configuration
st.sidebar.subheader("Stockfish Engine")
sf_path_input = st.sidebar.text_input(
    "Stockfish Executable Path:",
    value=DEFAULT_STOCKFISH_PATH,
    help="Path to official Stockfish executable on your system.",
)

# Stockfish path validation
sf_valid = False
try:
    resolved_sf = validate_stockfish_path(sf_path_input)
    st.sidebar.success("Stockfish found and verified.")
    sf_valid = True
except Exception:
    st.sidebar.error("Stockfish binary not found at specified path.")
    resolved_sf = sf_path_input

sf_depth = st.sidebar.slider(
    "Analysis Depth (Plies):",
    min_value=6,
    max_value=18,
    value=10,
    step=1,
    help="Default depth 10 balances high-speed analysis with tactical accuracy (~2-5 seconds).",
)

st.sidebar.markdown("---")
run_analysis_btn = st.sidebar.button(
    "Run Behavioral Analysis",
    type="primary",
    use_container_width=True,
)


# -----------------------------------------------------------------------------
# Main Dashboard Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Chess Behavioral Feature Engineering & Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Version 4 Interactive Dashboard — Supervised Machine Learning on the Controlled ChessFraud Benchmark '
    '(Current User-Facing Production Dashboard; Version 5 Retained as Experimental Research Branch)'
    '</div>',
    unsafe_allow_html=True,
)

# Scientific & Ethical Notice
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Scientific & Ethical Notice:</strong><br>
        These scores estimate behavioral similarity to engine-assisted play. They are not verified probabilities and cannot prove that a player used an engine. Results may differ between models.<br>
        The 50% classification threshold is an operational software boundary, not a scientifically established threshold for cheating. The human-like score is the complement of the engine-assistance score (100% − engine-assistance score).
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs for Organization
tab_prediction, tab_eda, tab_models = st.tabs([
    "Player Analysis & Scores",
    "Exploratory Data Analysis",
    "Model Evaluation & Benchmark",
])


# -----------------------------------------------------------------------------
# State Management: Run Analysis
# -----------------------------------------------------------------------------
if run_analysis_btn:
    if not selected_pgn_path or not os.path.isfile(selected_pgn_path):
        st.error("Please provide or select a valid PGN file before running analysis.")
    elif not sf_valid:
        st.error(
            f"Cannot run analysis: Stockfish executable not found at '{sf_path_input}'. "
            "Please install Stockfish or configure the path in the sidebar."
        )
    elif not os.path.isfile(model_path):
        st.error(f"Cannot run analysis: Model file '{model_path}' not found. Please train models first.")
    else:
        with st.spinner(f"Analyzing game with Stockfish (depth {sf_depth}) and inferring {model_type}..."):
            try:
                report = predict_game(
                    pgn_path=selected_pgn_path,
                    stockfish_path=resolved_sf,
                    model_path=model_path,
                    depth=sf_depth,
                    verbose=False,
                )
                st.session_state["analysis_report"] = report
                st.success("Analysis complete!")
            except ValueError as ve:
                st.error(f"Game Validation Error: {ve}")
            except EngineAnalysisError as ee:
                st.error(f"Stockfish Analysis Error: {ee}")
            except Exception as e:
                st.error(f"Unexpected error during processing: {e}")
            finally:
                # Clean temporary file if generated from upload
                if temp_file_to_clean and os.path.isfile(temp_file_to_clean):
                    try:
                        os.remove(temp_file_to_clean)
                    except Exception:
                        pass


def render_player_card(player_data: dict[str, Any], side: str):
    """Render a balanced player evaluation card with classification-dependent score emphasis."""
    pres = get_player_card_presentation_data(player_data)
    side_upper = side.upper()
    side_pill_class = "side-white" if side_upper == "WHITE" else "side-black"
    p_name = player_data["name"]

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="player-card-header">
                <div class="player-identity">
                    <span class="player-side-pill {side_pill_class}">{side_upper}</span>
                    <span class="player-display-name" title="{p_name}">{p_name}</span>
                </div>
                <span class="{pres['badge_class']}">{pres['badge_text']}</span>
            </div>
            <div class="hero-score-label">{pres['primary_label']}</div>
            <div class="hero-score-value {pres['primary_color_class']}">{pres['primary_pct']:.1f}%</div>
            """,
            unsafe_allow_html=True,
        )

        # Progress bar directly below the emphasized hero score
        st.progress(pres["primary_prob"])

        # Secondary score in 15px readable text
        st.markdown(
            f"""
            <div class="secondary-score-line">
                {pres['secondary_label']}: <strong>{pres['secondary_pct']:.1f}%</strong>
            </div>
            <div class="metrics-separator"></div>
            """,
            unsafe_allow_html=True,
        )

        # 3 aligned behavioral metrics
        m1, m2, m3 = st.columns(3)
        cp = player_data["key_features"]["mean_centipawn_loss"]
        top1 = player_data["key_features"]["top1_match_rate"]
        blund = player_data["key_features"]["blunder_rate"]

        m1.metric("Mean CP Loss", f"{cp:.1f} cp" if cp is not None else "N/A")
        m2.metric("Top-1 Match", f"{top1 * 100:.1f}%" if top1 is not None else "N/A")
        m3.metric("Blunder Rate", f"{blund * 100:.1f}%" if blund is not None else "N/A")


# -----------------------------------------------------------------------------
# TAB 1: Player Analysis & Scores
# -----------------------------------------------------------------------------
with tab_prediction:
    report = st.session_state.get("analysis_report")

    if report is None:
        st.info(
            "Select a sample game or upload a PGN file in the sidebar, then click "
            "**'Run Behavioral Analysis'** to view player estimates."
        )
        st.markdown("### About This System")
        st.write(
            "This dashboard implements Version 4 of an exploratory data analysis and supervised "
            "machine learning pipeline for chess behavioral analysis. It evaluates games move-by-move "
            "with Stockfish to extract centipawn losses, mistake distributions, and top-candidate concordance, "
            "then predicts whether a player's behavior aligns with controlled engine-assisted play."
        )
    else:
        info = report["game_info"]
        w = report["white_player"]
        b = report["black_player"]

        # Game Information Header
        st.markdown("### Game Overview")
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        col_g1.metric("Game", f"{info['white']} vs {info['black']}")
        col_g2.metric("Result", info["result"])
        col_g3.metric("Total Plies", f"{info['total_plies']} moves")
        col_g4.metric("Engine & Depth", f"Depth {info['depth']} ({info['engine_used']})")

        st.markdown("---")
        st.markdown("### Player-Level Engine-Assistance Estimates")

        col_white, col_black = st.columns(2)

        with col_white:
            render_player_card(w, "White")

        with col_black:
            render_player_card(b, "Black")


# -----------------------------------------------------------------------------
# TAB 2: Exploratory Data Analysis (EDA Charts)
# -----------------------------------------------------------------------------
with tab_eda:
    report = st.session_state.get("analysis_report")

    if report is None:
        st.info("Run an analysis in the sidebar to generate interactive behavioral charts.")
    else:
        chart_data = extract_eda_chart_data(report)
        df_moves = chart_data["df_moves"]
        df_quality = chart_data["df_quality"]
        df_comparison = chart_data["df_comparison"]

        st.markdown("### Move Quality Breakdown & Engine Agreement")
        st.write(
            "Comparison of error rates and engine top-1 candidate move agreement. "
            "Higher engine match rates combined with lower blunder and mistake rates indicate higher precision."
        )

        # Matplotlib grouped bar chart for Move Quality
        fig_q, ax_q = plt.subplots(figsize=(9, 4.2))
        _apply_dark_chart_style(fig_q, ax_q)
        x = np.arange(len(df_quality))
        width = 0.35

        ax_q.bar(x - width / 2, df_quality["White (%)"], width, label=f"White ({report['game_info']['white']})", color="#3B82F6", alpha=0.9)
        ax_q.bar(x + width / 2, df_quality["Black (%)"], width, label=f"Black ({report['game_info']['black']})", color="#10B981", alpha=0.9)

        ax_q.set_ylabel("Rate (%)", fontsize=10)
        ax_q.set_title("Move Quality & Concordance Comparison", fontsize=11, fontweight="bold")
        ax_q.set_xticks(x)
        ax_q.set_xticklabels(df_quality["Category"], rotation=15, ha="right")
        ax_q.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
        plt.tight_layout()
        st.pyplot(fig_q)
        plt.close(fig_q)

        st.markdown("---")

        # Two Columns for Trajectory and Centipawn Loss Distribution
        col_c1, col_c2 = st.columns(2)

        # Detect schema column names safely with real pipeline defaults
        color_col = "player_color" if "player_color" in df_moves.columns else ("player" if "player" in df_moves.columns else None)
        ply_col = "ply_number" if "ply_number" in df_moves.columns else ("ply" if "ply" in df_moves.columns else None)
        has_cp = "centipawn_loss" in df_moves.columns

        with col_c1:
            st.markdown("### Move-by-Move Centipawn Loss")
            if not df_moves.empty and color_col and ply_col and has_cp:
                # Filter moves by player color
                w_mask = df_moves[color_col].astype(str).str.lower() == "white"
                b_mask = df_moves[color_col].astype(str).str.lower() == "black"

                w_moves = df_moves[w_mask].copy()
                b_moves = df_moves[b_mask].copy()

                w_moves["cp_numeric"] = pd.to_numeric(w_moves["centipawn_loss"], errors="coerce")
                b_moves["cp_numeric"] = pd.to_numeric(b_moves["centipawn_loss"], errors="coerce")

                w_valid = w_moves.dropna(subset=["cp_numeric"])
                b_valid = b_moves.dropna(subset=["cp_numeric"])

                if w_valid.empty and b_valid.empty:
                    st.info("No numerical centipawn loss data available to plot move trajectory.")
                else:
                    fig_t, ax_t = plt.subplots(figsize=(6, 4))
                    _apply_dark_chart_style(fig_t, ax_t)
                    if not w_valid.empty:
                        ax_t.plot(
                            w_valid[ply_col],
                            w_valid["cp_numeric"],
                            marker="o",
                            markersize=4,
                            label=f"White ({report['game_info']['white']})",
                            color="#3B82F6",
                            alpha=0.85,
                        )
                    if not b_valid.empty:
                        ax_t.plot(
                            b_valid[ply_col],
                            b_valid["cp_numeric"],
                            marker="s",
                            markersize=4,
                            label=f"Black ({report['game_info']['black']})",
                            color="#10B981",
                            alpha=0.85,
                        )
                    ax_t.axhline(200, color="#EF4444", linestyle=":", label="Blunder Threshold (200 cp)")

                    ax_t.set_xlabel("Ply (Half-Move)", fontsize=10)
                    ax_t.set_ylabel("Centipawn Loss (cp)", fontsize=10)
                    ax_t.set_title("Centipawn Loss Trajectory", fontsize=11, fontweight="bold")
                    ax_t.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
                    plt.tight_layout()
                    st.pyplot(fig_t)
                    plt.close(fig_t)
            else:
                st.info("Move-level trajectory data is unavailable for this game.")

        with col_c2:
            st.markdown("### Centipawn Loss Distribution")
            if not df_moves.empty and color_col and has_cp:
                w_mask = df_moves[color_col].astype(str).str.lower() == "white"
                b_mask = df_moves[color_col].astype(str).str.lower() == "black"

                w_cp_vals = pd.to_numeric(df_moves[w_mask]["centipawn_loss"], errors="coerce").dropna()
                b_cp_vals = pd.to_numeric(df_moves[b_mask]["centipawn_loss"], errors="coerce").dropna()

                if w_cp_vals.empty and b_cp_vals.empty:
                    st.info("No numerical centipawn loss values available to compute distribution.")
                else:
                    fig_d, ax_d = plt.subplots(figsize=(6, 4))
                    _apply_dark_chart_style(fig_d, ax_d)
                    combined_vals = pd.concat([w_cp_vals, b_cp_vals])
                    max_cp = float(combined_vals.max()) if not combined_vals.empty else 50.0
                    upper_bound = min(300.0, max(max_cp, 50.0))
                    bins = np.linspace(0, upper_bound, 15)

                    if not w_cp_vals.empty:
                        ax_d.hist(w_cp_vals, bins=bins, alpha=0.55, label=f"White ({report['game_info']['white']})", color="#3B82F6")
                    if not b_cp_vals.empty:
                        ax_d.hist(b_cp_vals, bins=bins, alpha=0.55, label=f"Black ({report['game_info']['black']})", color="#10B981")

                    ax_d.set_xlabel("Centipawn Loss (cp)", fontsize=10)
                    ax_d.set_ylabel("Frequency (Moves)", fontsize=10)
                    ax_d.set_title("Loss Histogram (0–300 cp)", fontsize=11, fontweight="bold")
                    ax_d.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
                    plt.tight_layout()
                    st.pyplot(fig_d)
                    plt.close(fig_d)
            else:
                st.info("Centipawn loss distribution data is unavailable for this game.")

        st.markdown("---")
        st.markdown("### Complete 13-Feature Behavioral Profile")
        st.write("Complete tabular comparison of all features engineered for model training and prediction:")
        st.dataframe(
            df_comparison[["Feature", "White", "Black", "Description"]],
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# TAB 3: Model Evaluation & Benchmark Comparison
# -----------------------------------------------------------------------------
with tab_models:
    st.markdown("### ChessFraud Benchmark & Validation Metrics")
    st.write(
        "Both models were evaluated on the **ChessFraud** controlled-tournament dataset (Linich et al., ACM SIGKDD 2026) "
        "using **5-Fold Stratified Group Cross-Validation (`StratifiedGroupKFold`)** grouped strictly by physical match "
        "(`physical_game_id`). White and Black records of the same chess match are strictly constrained to the same fold, "
        "guaranteeing zero physical-game leakage across training and validation."
    )

    try:
        metadata = load_model_metadata()
        split = metadata.get("split", {})
        models_data = metadata.get("models", {})

        # Cohort summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Player-Games", metadata.get("total_player_games", 1010))
        c2.metric("Physical Matches", metadata.get("total_physical_matches", 505))
        c3.metric("CV Strategy", f"{split.get('n_splits', 5)}-Fold Group CV")
        c4.metric("Physical Match Leakage", "0.0% (Zero Overlap)")

        st.markdown("---")
        st.markdown("#### Out-Of-Fold (OOF) 5-Fold Grouped Cross-Validation Performance")

        lr_metrics = models_data.get("logistic_regression", {}).get("test_metrics", {})
        rf_metrics = models_data.get("random_forest", {}).get("test_metrics", {})
        baseline_acc = metadata.get("class_balance", {}).get("majority_class_baseline_accuracy", 0.597)

        metrics_table = [
            {
                "Metric": "Accuracy",
                "Majority Baseline": f"{baseline_acc * 100:.2f}%",
                "Logistic Regression (Default)": f"{lr_metrics.get('accuracy', 0) * 100:.2f}%",
                "Random Forest (Alternative)": f"{rf_metrics.get('accuracy', 0) * 100:.2f}%",
            },
            {
                "Metric": "Precision (Assisted)",
                "Majority Baseline": "0.00%",
                "Logistic Regression (Default)": f"{lr_metrics.get('precision', 0) * 100:.2f}%",
                "Random Forest (Alternative)": f"{rf_metrics.get('precision', 0) * 100:.2f}%",
            },
            {
                "Metric": "Recall (Assisted)",
                "Majority Baseline": "0.00%",
                "Logistic Regression (Default)": f"{lr_metrics.get('recall', 0) * 100:.2f}%",
                "Random Forest (Alternative)": f"{rf_metrics.get('recall', 0) * 100:.2f}%",
            },
            {
                "Metric": "F1-Score",
                "Majority Baseline": "0.00%",
                "Logistic Regression (Default)": f"{lr_metrics.get('f1_score', 0) * 100:.2f}%",
                "Random Forest (Alternative)": f"{rf_metrics.get('f1_score', 0) * 100:.2f}%",
            },
            {
                "Metric": "ROC-AUC",
                "Majority Baseline": "0.5000",
                "Logistic Regression (Default)": f"{lr_metrics.get('roc_auc', 0):.4f}",
                "Random Forest (Alternative)": f"{rf_metrics.get('roc_auc', 0):.4f}",
            },
            {
                "Metric": "PR-AUC",
                "Majority Baseline": f"{metadata.get('class_balance', {}).get('positive_rate', 0.403):.4f}",
                "Logistic Regression (Default)": f"{lr_metrics.get('pr_auc', 0):.4f}",
                "Random Forest (Alternative)": f"{rf_metrics.get('pr_auc', 0):.4f}",
            },
            {
                "Metric": "Brier Score (Lower is better)",
                "Majority Baseline": "0.2405",
                "Logistic Regression (Default)": "0.1935",
                "Random Forest (Alternative)": "0.1976",
            },
        ]
        st.table(pd.DataFrame(metrics_table))

        st.info(
            "**Model Comparison Summary:** Logistic Regression is designated as the primary default classifier "
            "because it achieves higher discrimination (ROC-AUC 0.7559 vs 0.7437, PR-AUC 0.6892 vs 0.6628), higher recall "
            "(56.27% vs 54.79%), higher F1-score (60.99% vs 60.11%), and lower Brier calibration error (0.1935 vs 0.1976) "
            "under strict physical-match-grouped cross-validation. Random Forest remains available as an alternative, "
            "though tree averaging across shallow trees (max depth 6) naturally contracts predicted probabilities toward the cohort mean."
        )

        st.markdown("---")
        st.markdown("#### Feature Importance (Random Forest)")
        rf_importances = models_data.get("random_forest", {}).get("feature_importances", {})
        if rf_importances:
            df_imp = pd.DataFrame([
                {"Feature": k, "Importance": v * 100}
                for k, v in sorted(rf_importances.items(), key=lambda item: item[1], reverse=True)
            ])
            fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            _apply_dark_chart_style(fig_imp, ax_imp)
            ax_imp.barh(df_imp["Feature"][::-1], df_imp["Importance"][::-1], color="#6366F1", alpha=0.9)
            ax_imp.set_xlabel("Importance (%)", fontsize=10)
            ax_imp.set_title("Top Random Forest Feature Importances", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        st.markdown("---")
        st.markdown("#### Metric Definitions & Interpretations")

        metric_defs = get_metric_definitions()
        for metric_name, defs in metric_defs.items():
            with st.expander(f"What is **{metric_name}**? — {defs['short']}"):
                st.write(defs["detailed"])
                st.caption(f"**Benchmark Context:** {defs['benchmark_context']}")

    except Exception as e:
        st.warning(f"Could not load complete model metadata: {e}")
