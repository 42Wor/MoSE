"""MoSE: Mixture of Smaller Experts Interactive Research Dashboard (app.py).

An interactive visualization web app built with Streamlit and Plotly for the
MoSE (Mixture of Smaller Experts) Causal Language Model.

Features:
- Prompt Infilling & Completion Playground (Autoregressive & FIM support)
- Interactive Token Ribbon with multi-layer path tracking
- Publication-quality Layer-by-Layer Routing Heatmaps (Plotly SVG/PDF export)
- Global Expert Load Balancing Histogram & Normalized Routing Entropy
- Per-Expert Vocabulary Association Table (Top-5 Attracted Tokens)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from config import Config
from evaluate import load_model_and_tokenizer
from model import MoSELanguageModel

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="MoSE Explorer | Mixture of Smaller Experts",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for research dashboard
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #616161;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: rgba(30, 136, 229, 0.08);
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1565C0;
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #757575;
        letter-spacing: 0.5px;
    }
    .token-card {
        background-color: rgba(126, 87, 194, 0.08);
        border: 1px solid rgba(126, 87, 194, 0.25);
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
    }
    .layer-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Model & Tokenizer Loader (Cached)
# ==============================================================================

@st.cache_resource(show_spinner="Loading MoSE weights and BPE tokenizer into memory...")
def get_model_and_tokenizer(checkpoint_path: str = "mose_python_model.pt", device: str = "cpu"):
    """Loads and caches the MoSE model and tokenizer on CPU."""
    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return model, tokenizer, config


# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_entropy(counts: np.ndarray) -> Tuple[float, str]:
    """Computes normalized routing entropy across experts."""
    total = counts.sum()
    if total == 0:
        return 0.0, "Inactive"
    probs = counts / total
    # Avoid log(0)
    nz = probs[probs > 0]
    raw_entropy = -np.sum(nz * np.log(nz))
    max_entropy = np.log(len(counts))
    norm_entropy = float((raw_entropy / max_entropy) * 100.0)

    if norm_entropy >= 80.0:
        status = "Highly Balanced Load"
    elif norm_entropy >= 65.0:
        status = "Healthy Specialization Balance"
    elif norm_entropy >= 45.0:
        status = "Moderate Specialization"
    else:
        status = "Dominant Expert Concentration"

    return norm_entropy, status


def build_expert_vocab_df(trace_result: Dict[str, any], layer_filter: Optional[int] = None) -> pd.DataFrame:
    """Constructs the Per-Expert Vocabulary Association DataFrame."""
    num_experts = 16
    expert_tokens = {e: {} for e in range(num_experts)}
    expert_weights = {e: [] for e in range(num_experts)}
    expert_ranks = {e: {"top1": 0, "top2": 0} for e in range(num_experts)}

    token_labels = trace_result["token_labels"]

    for t_idx, t_trace in enumerate(trace_result["trace"]):
        tok = token_labels[t_idx]
        for l_idx, exp_list in t_trace.items():
            if layer_filter is not None and l_idx != layer_filter:
                continue
            for rank_idx, (exp_id, wt) in enumerate(exp_list):
                expert_tokens[exp_id][tok] = expert_tokens[exp_id].get(tok, 0) + 1
                expert_weights[exp_id].append(wt)
                if rank_idx == 0:
                    expert_ranks[exp_id]["top1"] += 1
                else:
                    expert_ranks[exp_id]["top2"] += 1

    rows = []
    for e in range(num_experts):
        counts = expert_tokens[e]
        total_hits = sum(counts.values())
        avg_wt = float(np.mean(expert_weights[e])) if expert_weights[e] else 0.0

        # Top-5 most frequent tokens
        sorted_tokens = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[:5]
        top_5_display = ", ".join([f"`{tok}` ({cnt}x)" for tok, cnt in sorted_tokens]) if sorted_tokens else "None"

        # Specialization tag based on top tokens
        specialization_tag = "General"
        all_toks = " ".join([t for t, _ in sorted_tokens]).lower()
        if any(kw in all_toks for kw in ["def", "class", "return", "import", "for", "while", "if", "try"]):
            specialization_tag = "Keywords & Control Flow"
        elif any(sym in all_toks for sym in ["(", ")", ":", ",", ".", "[", "]", "{", "}"]):
            specialization_tag = "Syntax & Punctuation"
        elif any(op in all_toks for op in ["+", "-", "*", "/", "=", "==", "<", ">"]):
            specialization_tag = "Operators & Math"
        elif any(ind in all_toks for ind in ["↵", "\\n", "    ", "  "]):
            specialization_tag = "Whitespace & Indentation"
        elif total_hits > 0:
            specialization_tag = "Identifiers & Names"

        rows.append({
            "Expert": f"E{e}",
            "Primary Role": specialization_tag,
            "Total Hits": total_hits,
            "Top-1 Hits": expert_ranks[e]["top1"],
            "Top-2 Hits": expert_ranks[e]["top2"],
            "Avg Routing Wt": f"{avg_wt * 100:.1f}%" if total_hits > 0 else "0.0%",
            "Top-5 Attracted Tokens": top_5_display,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# Presets
# ==============================================================================

PRESETS = {
    "Function Definition (compute)": "def compute(x, y):\n    return x + y",
    "FIM Infilling (<PRE>...<SUF>...<MID>)": "<PRE>def add(a, b):\n    <SUF>\n    return result\n<MID>",
    "Class Architecture": "class DataBuffer:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.items = []",
    "Control Flow & Loops": "import math\nfor i in range(10):\n    if i % 2 == 0:\n        print(math.sqrt(i))",
    "Exception Handling": "try:\n    value = int(user_input)\n    result = 100 / value\nexcept ZeroDivisionError:\n    result = 0",
}


# ==============================================================================
# Main Streamlit Application
# ==============================================================================

def main():
    # 1. Header & Research Overview
    st.markdown('<div class="main-header">MoSE: Mixture of Smaller Experts Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Interactive inspection dashboard for token-by-token expert routing, dynamic activation heatmaps, '
        'and load-balancing entropy verification.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load Model & Tokenizer
    model, tokenizer, config = get_model_and_tokenizer()

    # Architecture Overview Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{config.n_layers}</div><div class="metric-label">Transformer Layers</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{config.num_experts}</div><div class="metric-label">Micro-Experts / Layer</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">Top-{config.top_k}</div><div class="metric-label">Dynamic Routing</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">1 Shared</div><div class="metric-label">Persistent Expert</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">~75%</div><div class="metric-label">Active Sparsity</div></div>',
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{config.vocab_size:,}</div><div class="metric-label">BPE Vocab</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # Sidebar Controls
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration & Controls")

        st.subheader("Code Template Presets")
        preset_choice = st.selectbox(
            "Load Sample Snippet:",
            options=list(PRESETS.keys()),
            index=0,
        )

        st.markdown("---")
        st.subheader("Autoregressive Generation")
        max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=64, value=20, step=1)
        temperature = st.slider("Sampling Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05)
        top_k_sample = st.slider("Top-k Vocabulary Filter", min_value=1, max_value=100, value=40, step=1)

        st.markdown("---")
        st.subheader("Heatmap Visualization")
        colorscale = st.selectbox(
            "Plotly Color Scheme:",
            options=["Viridis", "Plasma", "Cividis", "Blues", "YlGnBu", "Teal", "Turbo"],
            index=0,
        )
        display_mode = st.radio(
            "Heatmap Value Representation:",
            options=["Sparse Top-2 Routing Weights (Active)", "Dense Router Softmax Probabilities (All 16)"],
            index=0,
        )

        st.markdown("---")
        st.caption("🔬 **MoSE Research Artifact**: Plots can be exported directly as vector SVGs for LaTeX inclusion.")

    # ==========================================================================
    # Input Panel: Python Code & FIM Infilling Playground
    # ==========================================================================
    st.subheader("1. Prompt Infilling & Code Playground")

    # Keep user input in state so presets update smoothly
    if "input_code" not in st.session_state or st.session_state.get("last_preset") != preset_choice:
        st.session_state["input_code"] = PRESETS[preset_choice]
        st.session_state["last_preset"] = preset_choice

    user_code = st.text_area(
        "Enter Python Code or Fill-in-the-Middle (FIM) prompt:",
        value=st.session_state["input_code"],
        height=130,
        key="code_area",
        help="Input Python code. To test infilling, use sentinels: <PRE>prefix<SUF>suffix<MID>",
    )

    col_btn1, col_btn2, col_info = st.columns([1.5, 2.2, 4.3])
    with col_btn1:
        run_trace = st.button("🔍 Trace Prompt Routing", use_container_width=True, type="primary")
    with col_btn2:
        run_gen = st.button("⚡ Autoregressively Infill / Complete", use_container_width=True)
    with col_info:
        if "<PRE>" in user_code and "<MID>" in user_code:
            st.info("💡 **FIM Prompt detected**: Model will predict the code intended between `<PRE>` and `<SUF>`.")
        else:
            st.caption("Press **Trace Prompt Routing** for prompt inspection, or **Autoregressively Infill** for next tokens.")

    # Session storage for inference results
    if "trace_result" not in st.session_state:
        # Run default prompt on initial load
        enc = tokenizer.encode(st.session_state["input_code"])
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=config.device)
        st.session_state["trace_result"] = model.inspect_forward(input_ids, tokenizer=tokenizer)
        st.session_state["prompt_len"] = len(enc.ids)
        st.session_state["generated_tokens"] = []

    if run_trace:
        enc = tokenizer.encode(user_code)
        if len(enc.ids) == 0:
            st.warning("Please enter valid Python code.")
            return
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=config.device)
        with st.spinner("Analyzing multi-layer routing decisions..."):
            st.session_state["trace_result"] = model.inspect_forward(input_ids, tokenizer=tokenizer)
            st.session_state["prompt_len"] = len(enc.ids)
            st.session_state["generated_tokens"] = []
            st.session_state["input_code"] = user_code

    if run_gen:
        enc = tokenizer.encode(user_code)
        if len(enc.ids) == 0:
            st.warning("Please enter valid Python code.")
            return
        input_ids = torch.tensor([enc.ids], dtype=torch.long, device=config.device)
        eos_id = tokenizer.token_to_id(config.eos_token)
        with st.spinner(f"Autoregressively generating up to {max_new_tokens} tokens..."):
            gen_res = model.inspect_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k_sample,
                eos_id=eos_id,
                tokenizer=tokenizer,
            )
            st.session_state["trace_result"] = gen_res
            st.session_state["prompt_len"] = gen_res.get("prompt_len", len(enc.ids))
            st.session_state["generated_tokens"] = gen_res.get("generated_tokens", [])
            st.session_state["input_code"] = user_code

    trace_result = st.session_state["trace_result"]
    prompt_len = st.session_state.get("prompt_len", len(trace_result["tokens"]))
    total_tokens = len(trace_result["tokens"])
    num_generated = total_tokens - prompt_len

    # Generated text output if available
    if num_generated > 0:
        st.markdown("---")
        st.subheader("⚡ Model Completion Output")
        full_text = "".join(trace_result["tokens"])
        gen_text = "".join(trace_result["tokens"][prompt_len:])

        col_out1, col_out2 = st.columns([1, 1])
        with col_out1:
            st.markdown(f"**Generated Continuation** ({num_generated} tokens):")
            st.code(gen_text, language="python")
        with col_out2:
            st.markdown(f"**Full Sequence with Prompt** ({total_tokens} tokens total):")
            st.code(full_text, language="python")

    st.markdown("---")

    # ==========================================================================
    # Section 1: Interactive Token Ribbon
    # ==========================================================================
    st.subheader("2. Interactive Token Ribbon")
    st.caption("Click any token in the sequence ribbon to inspect its exact routing path across all 4 layers.")

    tokens = trace_result["tokens"]
    token_labels = trace_result["token_labels"]

    # Option to select token
    token_display_options = []
    for i, t in enumerate(token_labels):
        tag = "⚡" if i >= prompt_len else ""
        token_display_options.append(f"[{i}] {tag}{t}")

    selected_token_idx = st.pills(
        "Token Sequence:",
        options=list(range(total_tokens)),
        format_func=lambda idx: token_display_options[idx],
        default=0,
        label_visibility="collapsed",
    )

    if selected_token_idx is None:
        selected_token_idx = 0

    # Display routing path for selected token
    sel_token_str = tokens[selected_token_idx]
    sel_token_lbl = token_labels[selected_token_idx]
    sel_token_id = trace_result["token_ids"][selected_token_idx]
    is_gen = selected_token_idx >= prompt_len

    st.markdown(
        f"""
        <div class="token-card">
            <b>Selected Token</b>: <code>{sel_token_lbl}</code> &nbsp;|&nbsp;
            <b>Position</b>: <code>{selected_token_idx + 1} / {total_tokens}</code> &nbsp;|&nbsp;
            <b>BPE Token ID</b>: <code>{sel_token_id}</code> &nbsp;|&nbsp;
            <b>Source</b>: <span style="color: {'#7E57C2' if is_gen else '#1E88E5'}; font-weight: 600;">{'Generated Continuation' if is_gen else 'Prompt Token'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Layer Cards side-by-side
    layer_cols = st.columns(config.n_layers)
    token_trace = trace_result["trace"][selected_token_idx]

    for l_idx in range(config.n_layers):
        exp_list = token_trace[l_idx]
        exp_A, wt_A = exp_list[0]
        exp_B, wt_B = exp_list[1]
        raw_A = trace_result["layer_routing"][l_idx]["raw_weights"][selected_token_idx, 0]
        raw_B = trace_result["layer_routing"][l_idx]["raw_weights"][selected_token_idx, 1]

        with layer_cols[l_idx]:
            st.markdown(f"**Layer {l_idx}**")
            st.markdown(
                f"""
                <div style="background-color: rgba(0,0,0,0.03); border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                    <div><b>Top-1: Expert {exp_A}</b> ({wt_A*100:.1f}%)</div>
                    <div style="font-size: 0.75rem; color: #666;">Softmax prob: {raw_A:.3f}</div>
                    <div style="background:#e0e0e0; border-radius:3px; height:6px; margin: 4px 0 8px 0;">
                        <div style="background:#1E88E5; width:{wt_A*100:.1f}%; height:6px; border-radius:3px;"></div>
                    </div>
                    <div><b>Top-2: Expert {exp_B}</b> ({wt_B*100:.1f}%)</div>
                    <div style="font-size: 0.75rem; color: #666;">Softmax prob: {raw_B:.3f}</div>
                    <div style="background:#e0e0e0; border-radius:3px; height:6px; margin: 4px 0 2px 0;">
                        <div style="background:#7E57C2; width:{wt_B*100:.1f}%; height:6px; border-radius:3px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ==========================================================================
    # Sections 2 & 3: Heatmap and Global Load Chart (2 Columns)
    # ==========================================================================
    col_heatmap, col_load = st.columns([3, 2])

    with col_heatmap:
        st.subheader("3. Layer-by-Layer Heatmap")
        
        layer_options = [f"Layer {l}" for l in range(config.n_layers)] + ["All Layers (Aggregated)"]
        selected_layer_str = st.selectbox(
            "Select Transformer Depth to Inspect:",
            options=layer_options,
            index=0,
        )

        # Build Heatmap matrix
        expert_labels = [f"E{e}" for e in range(config.num_experts)]
        x_ticks = [f"[{i}] {lbl}" for i, lbl in enumerate(token_labels)]

        if "All Layers" in selected_layer_str:
            # Average across all layers
            if "Sparse" in display_mode:
                z_matrix = np.mean(
                    [trace_result["layer_routing"][l]["sparse_matrix"] for l in range(config.n_layers)],
                    axis=0,
                )
            else:
                z_matrix = np.mean(
                    [trace_result["layer_routing"][l]["gate_probs"].T for l in range(config.n_layers)],
                    axis=0,
                )
            title_suffix = "Aggregated Across All 4 Layers"
            current_layer_idx = None
        else:
            current_layer_idx = int(selected_layer_str.split(" ")[1])
            if "Sparse" in display_mode:
                z_matrix = trace_result["layer_routing"][current_layer_idx]["sparse_matrix"]
            else:
                z_matrix = trace_result["layer_routing"][current_layer_idx]["gate_probs"].T
            title_suffix = f"Layer {current_layer_idx}"

        # Create interactive Plotly Heatmap
        hover_template = (
            "<b>Token:</b> %{x}<br>"
            "<b>Expert:</b> %{y}<br>"
            "<b>Routing Weight:</b> %{z:.3f}<br>"
            "<extra></extra>"
        )

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=x_ticks,
                y=expert_labels,
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0 if "Sparse" in display_mode else float(np.max(z_matrix) + 1e-4),
                hovertemplate=hover_template,
                colorbar=dict(
                    title="Routing Wt" if "Sparse" in display_mode else "Softmax Prob",
                    thickness=14,
                    len=0.9,
                ),
            )
        )

        # Highlight currently selected token column if valid
        if selected_token_idx < len(x_ticks):
            fig_heatmap.add_vline(
                x=selected_token_idx,
                line_width=2,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Selected [{token_labels[selected_token_idx]}]",
                annotation_position="top left",
            )

        fig_heatmap.update_layout(
            title=f"Micro-Expert Activation Heatmap ({title_suffix})",
            xaxis_title="Input Token Sequence",
            yaxis_title="Micro-Experts (E0 - E15)",
            margin=dict(l=40, r=40, t=50, b=50),
            height=460,
            yaxis=dict(autorange="reversed"),  # E0 at the top, E15 at the bottom
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True,
            config={
                "toImageButtonOptions": {
                    "format": "svg",
                    "filename": f"mose_layer_{selected_layer_str.replace(' ', '_').lower()}",
                    "height": 550,
                    "width": 950,
                    "scale": 2,
                }
            },
        )
        st.caption("📷 Hover over any cell to inspect exact Top-2 weights. Use the camera icon in the chart toolbar to download an SVG figure.")

    with col_load:
        st.subheader("4. Global Load Balancing")

        # Compute utilization counts
        load_counts = np.zeros(config.num_experts)
        for t_idx, t_trace in enumerate(trace_result["trace"]):
            for l_idx, exp_list in t_trace.items():
                if current_layer_idx is not None and l_idx != current_layer_idx:
                    continue
                for exp_id, _ in exp_list:
                    load_counts[exp_id] += 1

        total_allocations = load_counts.sum()
        pct_allocations = (load_counts / max(total_allocations, 1)) * 100.0
        entropy_val, entropy_status = compute_entropy(load_counts)

        # Entropy Badge Card
        entropy_color = "#2E7D32" if entropy_val >= 65.0 else ("#F57C00" if entropy_val >= 50.0 else "#C62828")
        st.markdown(
            f"""
            <div style="background-color: rgba(46, 125, 50, 0.08); border-left: 4px solid {entropy_color}; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px;">
                <div style="font-size: 0.85rem; color: #555;">Routing Entropy ({'Current Layer' if current_layer_idx is not None else 'Global'}):</div>
                <div style="font-size: 1.45rem; font-weight: 700; color: {entropy_color};">
                    {entropy_val:.1f}% &nbsp;<span style="font-size: 0.95rem; font-weight: normal;">({entropy_status})</span>
                </div>
                <div style="font-size: 0.78rem; color: #777; margin-top: 4px;">
                    Theoretical maximum: 100% (uniform routing across all 16 micro-experts).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bar chart for Expert Utilization
        df_load = pd.DataFrame({
            "Expert": expert_labels,
            "Utilization (%)": pct_allocations,
            "Activations": load_counts.astype(int),
        })

        fig_bar = px.bar(
            df_load,
            x="Utilization (%)",
            y="Expert",
            orientation="h",
            color="Utilization (%)",
            color_continuous_scale="Viridis",
            text="Utilization (%)",
            title=f"Expert Dispatch Distribution ({'Layer ' + str(current_layer_idx) if current_layer_idx is not None else 'Global'})",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            height=370,
            margin=dict(l=20, r=40, t=50, b=40),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            config={
                "toImageButtonOptions": {
                    "format": "svg",
                    "filename": "mose_expert_utilization_chart",
                    "height": 450,
                    "width": 650,
                    "scale": 2,
                }
            },
        )

    st.markdown("---")

    # ==========================================================================
    # Section 5: Per-Expert Vocabulary Association Table
    # ==========================================================================
    st.subheader("5. Per-Expert Vocabulary Association Table")
    st.caption("Identifies the Top-5 most frequently routed Python syntax tokens for each of the 16 micro-experts in this session.")

    col_filter, col_download = st.columns([3, 1])
    with col_filter:
        table_layer_choice = st.radio(
            "Filter Associations By Transformer Depth:",
            options=["All Layers", "Layer 0", "Layer 1", "Layer 2", "Layer 3"],
            horizontal=True,
            index=0,
        )
    with col_download:
        st.write("")

    table_layer_idx = None if table_layer_choice == "All Layers" else int(table_layer_choice.split(" ")[1])
    vocab_df = build_expert_vocab_df(trace_result, layer_filter=table_layer_idx)

    st.dataframe(
        vocab_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Expert": st.column_config.TextColumn("Expert", width="small"),
            "Primary Role": st.column_config.TextColumn("Syntactic / Semantic Role", width="medium"),
            "Total Hits": st.column_config.NumberColumn("Total Activations", width="small"),
            "Top-1 Hits": st.column_config.NumberColumn("Top-1", width="small"),
            "Top-2 Hits": st.column_config.NumberColumn("Top-2", width="small"),
            "Avg Routing Wt": st.column_config.TextColumn("Avg Weight", width="small"),
            "Top-5 Attracted Tokens": st.column_config.TextColumn("Top-5 Attracted Tokens (Count)", width="large"),
        },
    )

    # Download raw routing trace as JSON for research papers
    st.markdown("---")
    st.subheader("6. Research Export Tools")
    c_dl1, c_dl2 = st.columns([1, 1])
    with c_dl1:
        # Convert trace to serializable dict
        serializable_trace = []
        for t_idx, t_item in enumerate(trace_result["trace"]):
            serializable_trace.append({
                "token_index": t_idx,
                "token_id": trace_result["token_ids"][t_idx],
                "token_string": trace_result["tokens"][t_idx],
                "routing": {
                    f"layer_{l}": [{"expert": exp, "weight": round(wt, 4)} for exp, wt in exp_list]
                    for l, exp_list in t_item.items()
                },
            })
        import json
        trace_json = json.dumps(serializable_trace, indent=2)
        st.download_button(
            label="📥 Download Routing Trace (JSON)",
            data=trace_json,
            file_name="mose_routing_trace.json",
            mime="application/json",
            use_container_width=True,
        )

    with c_dl2:
        vocab_csv = vocab_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Vocabulary Association Table (CSV)",
            data=vocab_csv,
            file_name="mose_expert_vocabulary_table.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

