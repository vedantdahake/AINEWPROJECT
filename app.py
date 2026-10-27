"""
Neural Network X-Ray: Interactive Neural Network Visualizer
Educational dashboard built with Streamlit, NumPy, and Plotly.
Provides genuine calculations, transparent forward/backward passes,
neuron arithmetic breakdowns, prediction tables, and learning comparisons.
"""

import streamlit as st
import numpy as np
import pandas as pd

from neural_network import NeuralNetwork
from datasets import get_dataset, DATASETS
from visualizations import (
    create_network_diagram,
    create_loss_accuracy_chart,
    create_decision_boundary_plot,
    create_experiment_comparison_chart
)

# Page configuration
st.set_page_config(
    page_title="Neural Network X-Ray",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-navy aesthetic
st.markdown("""
<style>
    /* Dark theme foundation */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header typography */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Highlight gradient text */
    .hero-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Explanation highlight cards */
    .info-card {
        background: #111827;
        border-left: 4px solid #38BDF8;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: #CBD5E1;
        line-height: 1.5;
    }
    .formula-box {
        background: #030712;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #38BDF8;
        font-size: 1.0rem;
        line-height: 1.6;
        margin: 0.75rem 0;
    }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Session State Initialization
# ==========================================
def initialize_app_state():
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = 'XOR'
    if 'hidden_size' not in st.session_state:
        st.session_state.hidden_size = 4
    if 'activation' not in st.session_state:
        st.session_state.activation = 'sigmoid'
    if 'seed' not in st.session_state:
        st.session_state.seed = 42
    if 'learning_rate' not in st.session_state:
        st.session_state.learning_rate = 0.5
    if 'selected_sample_idx' not in st.session_state:
        st.session_state.selected_sample_idx = 1  # (0, 1) -> 1
    if 'selected_layer' not in st.session_state:
        st.session_state.selected_layer = 'hidden'
    if 'selected_neuron_idx' not in st.session_state:
        st.session_state.selected_neuron_idx = 0
    if 'model' not in st.session_state:
        st.session_state.model = NeuralNetwork(
            input_size=2,
            hidden_size=st.session_state.hidden_size,
            output_size=1,
            activation=st.session_state.activation,
            seed=st.session_state.seed
        )
    if 'comparison_runs' not in st.session_state:
        st.session_state.comparison_runs = {}


initialize_app_state()

# Fetch active dataset
X, y, dataset_info = get_dataset(st.session_state.dataset_name)


# Helper function to reset/recreate model
def reset_model():
    st.session_state.model = NeuralNetwork(
        input_size=2,
        hidden_size=st.session_state.hidden_size,
        output_size=1,
        activation=st.session_state.activation,
        seed=st.session_state.seed
    )


# ==========================================
# SIDEBAR: SECTION A (Experiment Setup)
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Experiment Setup")
    st.caption("Configure network hyperparameters and controls.")

    # Dataset Selection
    dataset_choice = st.selectbox(
        "Dataset",
        options=list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(st.session_state.dataset_name),
        help="Select classification problem. XOR is non-linearly separable."
    )
    if dataset_choice != st.session_state.dataset_name:
        st.session_state.dataset_name = dataset_choice
        st.session_state.selected_sample_idx = 0
        X, y, dataset_info = get_dataset(dataset_choice)
        reset_model()
        st.rerun()

    # Network Architecture Controls
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.metric("Input Size", "2", help="Two input features (x₁, x₂)")
    with col_arch2:
        st.metric("Output Size", "1", help="Binary probability (ŷ)")

    hidden_count = st.slider(
        "Hidden Neurons",
        min_value=2,
        max_value=8,
        value=st.session_state.hidden_size,
        step=1,
        help="Number of neurons in the intermediate hidden layer."
    )
    if hidden_count != st.session_state.hidden_size:
        st.session_state.hidden_size = hidden_count
        if st.session_state.selected_neuron_idx >= hidden_count and st.session_state.selected_layer == 'hidden':
            st.session_state.selected_neuron_idx = 0
        reset_model()
        st.rerun()

    act_choice = st.selectbox(
        "Hidden Activation Function",
        options=['sigmoid', 'tanh', 'relu'],
        index=['sigmoid', 'tanh', 'relu'].index(st.session_state.activation),
        format_func=lambda s: s.upper(),
        help="Non-linear function applied to hidden neuron weighted sums."
    )
    if act_choice != st.session_state.activation:
        st.session_state.activation = act_choice
        reset_model()
        st.rerun()

    lr = st.slider(
        "Learning Rate (η)",
        min_value=0.01,
        max_value=2.0,
        value=float(st.session_state.learning_rate),
        step=0.05,
        help="Step size for gradient descent parameter updates."
    )
    st.session_state.learning_rate = lr

    epochs_to_train = st.number_input(
        "Training Epochs per Click",
        min_value=10,
        max_value=2000,
        value=500,
        step=50,
        help="Number of complete passes over the dataset."
    )

    seed_val = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.seed,
        step=1,
        help="Ensures identical initial weights for reproducible experiments."
    )
    if seed_val != st.session_state.seed:
        st.session_state.seed = seed_val
        reset_model()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🕹️ Training Controls")

    # Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Train Network", use_container_width=True, type="primary"):
            st.session_state.model.train(X, y, epochs=epochs_to_train, learning_rate=lr)
            st.rerun()
    with col_btn2:
        if st.button("⏯️ Train 1 Step", use_container_width=True):
            st.session_state.model.train_step(X, y, learning_rate=lr)
            st.rerun()

    if st.button("🔄 Reset Parameters", use_container_width=True):
        reset_model()
        st.rerun()

    # Architecture summary badge
    st.markdown("---")
    st.markdown(f"""
    <div style="background:#111827; border: 1px solid #1E293B; border-radius: 8px; padding: 10px; text-align: center;">
        <span style="font-size:0.8rem; color:#94A3B8;">ACTIVE ARCHITECTURE</span><br>
        <b style="font-size:1.15rem; color:#38BDF8;">2 → {st.session_state.hidden_size} → 1</b><br>
        <span style="font-size:0.8rem; color:#A855F7;">Hidden: {st.session_state.activation.upper()} | Output: SIGMOID</span>
    </div>
    """, unsafe_allow_html=True)

    # Dataset preview
    st.markdown("---")
    st.markdown(f"**Preview: {dataset_info['name']}**")
    df_preview = pd.DataFrame(
        np.hstack([X, y]),
        columns=['x₁', 'x₂', 'y (Target)']
    )
    st.dataframe(df_preview, hide_index=True, use_container_width=True)


# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.markdown('<div class="hero-title">Neural Network X-Ray</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Interactive visualizer uncovering the internal arithmetic, forward pass, and gradient descent updates of an artificial neural network.</div>', unsafe_allow_html=True)

# Top Status Metrics Bar
history = st.session_state.model.history
curr_probs, _ = st.session_state.model.predict(X)
curr_loss = st.session_state.model.compute_bce_loss(y, curr_probs)
curr_acc = st.session_state.model.compute_accuracy(y, curr_probs)
current_epoch = len(history)

init_loss = history[0]['loss'] if history else curr_loss
loss_delta = curr_loss - init_loss

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Epoch</div>
        <div class="metric-value">{current_epoch}</div>
        <div class="metric-sub">Total training cycles</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">BCE Loss</div>
        <div class="metric-value">{curr_loss:.4f}</div>
        <div class="metric-sub">Initial: {init_loss:.4f} ({loss_delta:+.4f})</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Accuracy</div>
        <div class="metric-value" style="color:#10B981;">{curr_acc:.1f}%</div>
        <div class="metric-sub">On {len(X)} training samples</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    if current_epoch == 0:
        status_text = "Initialized (Untrained)"
        status_color = "#94A3B8"
    elif curr_loss < 0.1:
        status_text = "Converged (Minimal Error)"
        status_color = "#10B981"
    elif loss_delta < 0:
        status_text = "Learning (Loss Decreasing)"
        status_color = "#38BDF8"
    else:
        status_text = "Stagnant / Adjust Learning Rate"
        status_color = "#F59E0B"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Status</div>
        <div class="metric-value" style="font-size:1.15rem; color:{status_color};">{status_text}</div>
        <div class="metric-sub">Calculated via live NumPy state</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_xray, tab_predict, tab_train, tab_experiments, tab_viva = st.tabs([
    "🧠 Network X-Ray & Inside a Neuron",
    "🎯 Prediction Lab",
    "📈 Training & Parameter Updates",
    "🔬 Learning Experiments",
    "🎓 Student Viva Guide"
])


# ==========================================
# TAB 1: SECTION B & SECTION C
# ==========================================
with tab_xray:
    st.markdown("### 🔍 Live Network Architecture & Signal Flow")
    st.caption("Select a data sample to feed into the inputs, and select any neuron to X-Ray its arithmetic.")

    # Input Sample Selection Bar
    sample_options = [
        f"Sample #{i+1}: Input = [{X[i, 0]:.1f}, {X[i, 1]:.1f}]  →  Target y = {int(y[i, 0])}"
        for i in range(len(X))
    ]
    selected_sample = st.selectbox(
        "Select Active Input Sample to Trace Signals:",
        options=range(len(X)),
        format_func=lambda idx: sample_options[idx],
        index=st.session_state.selected_sample_idx,
        help="The chosen sample's features enter at the input layer and travel through the network."
    )
    st.session_state.selected_sample_idx = selected_sample

    # Neuron Selection Controls
    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        layer_select = st.radio(
            "Select Layer to Inspect:",
            options=['Hidden Layer', 'Output Layer'],
            horizontal=True,
            index=0 if st.session_state.selected_layer == 'hidden' else 1
        )
        st.session_state.selected_layer = 'hidden' if layer_select == 'Hidden Layer' else 'output'

    with col_n2:
        if st.session_state.selected_layer == 'hidden':
            neuron_options = [f"Hidden Neuron h{j+1}" for j in range(st.session_state.hidden_size)]
            chosen_neuron = st.selectbox(
                "Select Neuron to Inspect:",
                options=range(st.session_state.hidden_size),
                format_func=lambda j: neuron_options[j],
                index=min(st.session_state.selected_neuron_idx, st.session_state.hidden_size - 1)
            )
            st.session_state.selected_neuron_idx = chosen_neuron
        else:
            st.selectbox("Select Neuron to Inspect:", options=[0], format_func=lambda _: "Output Neuron ŷ")
            st.session_state.selected_neuron_idx = 0

    # Draw Network Diagram
    fig_network = create_network_diagram(
        network_obj=st.session_state.model,
        sample_index=st.session_state.selected_sample_idx,
        X=X,
        selected_layer=st.session_state.selected_layer,
        selected_neuron_idx=st.session_state.selected_neuron_idx
    )
    st.plotly_chart(fig_network, use_container_width=True)

    # Color Legend Explanation
    st.markdown("""
    <div style="background:#111827; border:1px solid #1E293B; border-radius:8px; padding:10px 14px; font-size:0.88rem; color:#94A3B8; display:flex; justify-content:space-around; flex-wrap:wrap; gap:10px;">
        <div><span style="color:#38BDF8; font-weight:bold;">● Blue Connection:</span> Positive Weight (excites next neuron)</div>
        <div><span style="color:#F43F5E; font-weight:bold;">● Red/Pink Connection:</span> Negative Weight (inhibits next neuron)</div>
        <div><span style="color:#FACC15; font-weight:bold;">● Gold Ring:</span> Currently Selected Neuron for X-Ray</div>
        <div><span style="color:#E2E8F0; font-weight:bold;">Line Thickness:</span> Weight Magnitude |w|</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # SECTION C: INSIDE A NEURON
    st.markdown("### 🔬 Section C: Inside a Neuron")
    st.markdown(
        "Here is the transparent arithmetic performed inside the selected neuron for the current input. "
        "Every single number is generated live from the network's forward pass."
    )

    neuron_calc = st.session_state.model.get_neuron_calculation(
        sample_index=st.session_state.selected_sample_idx,
        layer=st.session_state.selected_layer,
        neuron_index=st.session_state.selected_neuron_idx,
        X=X
    )

    col_c1, col_c2 = st.columns([1.3, 1])

    with col_c1:
        st.markdown(f"#### Calculation Table: **{neuron_calc['layer_name']}**")
        
        # Build Table of inputs, weights, products
        table_rows = []
        for lbl_in, val_in, lbl_w, val_w, prod in zip(
            neuron_calc['input_labels'],
            neuron_calc['inputs'],
            neuron_calc['weight_labels'],
            neuron_calc['weights'],
            neuron_calc['products']
        ):
            table_rows.append({
                "Input": lbl_in,
                "Value (x)": f"{val_in:.4f}",
                "Weight (w)": f"{val_w:+.4f}",
                "Product (x × w)": f"{prod:+.4f}"
            })
        
        df_calc = pd.DataFrame(table_rows)
        st.dataframe(df_calc, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div style="background:#030712; border:1px solid #1E293B; border-radius:8px; padding:12px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94A3B8;">Sum of Products Σ(x × w):</span>
                <b style="color:#38BDF8;">{sum(neuron_calc['products']):+.4f}</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94A3B8;">Neuron Bias (b):</span>
                <b style="color:#C084FC;">{neuron_calc['bias']:+.4f}</b>
            </div>
            <div style="display:flex; justify-content:space-between; border-top:1px solid #1E293B; padding-top:6px; margin-top:4px;">
                <span style="color:#E2E8F0; font-weight:600;">Weighted Sum (z):</span>
                <b style="color:#38BDF8; font-size:1.1rem;">{neuron_calc['weighted_sum']:+.4f}</b>
            </div>
            <div style="display:flex; justify-content:space-between; border-top:1px solid #1E293B; padding-top:6px; margin-top:4px;">
                <span style="color:#E2E8F0; font-weight:600;">Final Output a = {neuron_calc['activation_name']}(z):</span>
                <b style="color:#10B981; font-size:1.15rem;">{neuron_calc['output']:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown("#### Substituted Formula & Explanation")
        st.markdown(f"""
        <div class="formula-box">
{neuron_calc['formula_str']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <b>Plain English Explanation:</b><br>
            {neuron_calc['explanation']}
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "💡 **Student Viva Tip:**\n"
            "- **Weights** determine the importance of each incoming signal.\n"
            "- **Bias** shifts the activation curve so the neuron can activate even when all inputs are zero.\n"
            "- **Activation functions** introduce non-linearity, allowing neural networks to solve problems like XOR."
        )


# ==========================================
# TAB 2: SECTION D (Prediction Lab)
# ==========================================
with tab_predict:
    st.markdown("### 🎯 Section D: Prediction Lab")
    st.caption("Inspect predictions across all training rows. Values are computed live by the model.")

    probs_all, classes_all = st.session_state.model.predict(X, threshold=0.5)

    # Build predictions dataframe
    pred_data = []
    for idx in range(len(X)):
        exp_val = int(y[idx, 0])
        prob_val = float(probs_all[idx, 0])
        pred_cls = int(classes_all[idx, 0])
        is_correct = (pred_cls == exp_val)
        status_str = "✅ Correct" if is_correct else "❌ Incorrect"

        pred_data.append({
            "Sample #": f"Row {idx+1}",
            "Input 1 (x₁)": f"{X[idx, 0]:.1f}",
            "Input 2 (x₂)": f"{X[idx, 1]:.1f}",
            "Expected (y)": exp_val,
            "Predicted Probability (ŷ)": f"{prob_val:.4f}",
            "Predicted Class (Threshold 0.5)": pred_cls,
            "Accuracy Status": status_str
        })

    df_predictions = pd.DataFrame(pred_data)
    st.dataframe(
        df_predictions,
        hide_index=True,
        use_container_width=True
    )

    col_d1, col_d2 = st.columns([1.2, 1])

    with col_d1:
        st.markdown("#### 2D Decision Surface & Separation")
        st.caption("Visualizes how the hidden layer bends the 2D plane to separate classes.")
        fig_boundary = create_decision_boundary_plot(
            network_obj=st.session_state.model,
            X=X,
            y=y,
            title=f"Decision Boundary for {dataset_info['name']}"
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

    with col_d2:
        st.markdown("#### Classification Threshold Logic")
        st.markdown(r"""
        <div class="info-card">
            <b>How Probabilities Become Binary Decisions:</b><br>
            The output neuron computes a continuous probability $\hat{y} \in (0, 1)$ via the sigmoid function:
            <br><br>
            $$\text{Class} = \begin{cases} 1 & \text{if } \hat{y} \ge 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$
            <br>
            Before training, the network's weights are random, so its predictions may be completely wrong.
            As training progresses, gradient descent adjusts weights to push $\hat{y}$ toward 1 for positive examples and toward 0 for negative examples.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#111827; border:1px solid #1E293B; border-radius:8px; padding:12px;">
            <b>Why XOR requires a Hidden Layer:</b><br>
            <span style="color:#94A3B8; font-size:0.9rem;">
            A single neuron without a hidden layer can only draw a straight line (linear separator). 
            In XOR, Class 1 points [(0,1), (1,0)] and Class 0 points [(0,0), (1,1)] cross diagonally 
            and cannot be separated by any single straight line. 
            The hidden layer creates a non-linear feature transformation!
            </span>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 3: SECTIONS E & F (Training & Updates)
# ==========================================
with tab_train:
    st.markdown("### 📈 Section E: Training Loss & Accuracy")
    st.caption("Loss measures prediction error. Training adjusts parameters to minimize this error.")

    # Plot Loss & Accuracy
    fig_loss = create_loss_accuracy_chart(st.session_state.model.history)
    st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown(r"""
    <div class="info-card">
        <b>What This Loss Curve Tells Us:</b><br>
        • <b>Binary Cross-Entropy Loss:</b> $L = -\frac{1}{N}\sum [y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$. 
        If the network is confident and wrong, loss is very large. If predictions match expected targets, loss approaches 0.<br>
        • <b>Training Accuracy:</b> The percentage of the 4 training examples classified correctly. 
        Note: Training loss does not always drop monotonically every single epoch; learning rate and weight landscape can cause fluctuations.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # SECTION F: WEIGHT AND BIAS UPDATES
    st.markdown("### 🔄 Section F: Weight and Bias Updates (Gradient Descent)")
    st.caption(
        "Inspect the exact gradient and update for every parameter from the most recent gradient descent step. "
        "Rule: new = old − (learning_rate × gradient)."
    )

    if not st.session_state.model.last_updates:
        # Compute one forward and backward pass to have immediate values to display
        _, cache = st.session_state.model.forward(X)
        grads = st.session_state.model.backward(y, cache)
        last_lr = st.session_state.learning_rate
        # Create preview of what next step will be without changing weights
        param_table_data = []
        for i in range(st.session_state.model.input_size):
            for j in range(st.session_state.model.hidden_size):
                old_w = st.session_state.model.W1[i, j]
                g = grads['dW1'][i, j]
                chg = -last_lr * g
                param_table_data.append({
                    "Parameter": f"W1 [x{i+1} → h{j+1}]",
                    "Old Value": f"{old_w:+.4f}",
                    "Gradient (∂L/∂w)": f"{g:+.5f}",
                    "Learning Rate (η)": f"{last_lr:.2f}",
                    "Change (-η × grad)": f"{chg:+.5f}",
                    "Updated Value": f"{old_w + chg:+.4f}"
                })
        for j in range(st.session_state.model.hidden_size):
            old_b = st.session_state.model.b1[0, j]
            g = grads['db1'][0, j]
            chg = -last_lr * g
            param_table_data.append({
                "Parameter": f"b1 [h{j+1} Bias]",
                "Old Value": f"{old_b:+.4f}",
                "Gradient (∂L/∂b)": f"{g:+.5f}",
                "Learning Rate (η)": f"{last_lr:.2f}",
                "Change (-η × grad)": f"{chg:+.5f}",
                "Updated Value": f"{old_b + chg:+.4f}"
            })
        for j in range(st.session_state.model.hidden_size):
            old_w = st.session_state.model.W2[j, 0]
            g = grads['dW2'][j, 0]
            chg = -last_lr * g
            param_table_data.append({
                "Parameter": f"W2 [h{j+1} → ŷ]",
                "Old Value": f"{old_w:+.4f}",
                "Gradient (∂L/∂w)": f"{g:+.5f}",
                "Learning Rate (η)": f"{last_lr:.2f}",
                "Change (-η × grad)": f"{chg:+.5f}",
                "Updated Value": f"{old_w + chg:+.4f}"
            })
        old_b2 = st.session_state.model.b2[0, 0]
        g2 = grads['db2'][0, 0]
        chg2 = -last_lr * g2
        param_table_data.append({
            "Parameter": "b2 [Output Bias]",
            "Old Value": f"{old_b2:+.4f}",
            "Gradient (∂L/∂b)": f"{g2:+.5f}",
            "Learning Rate (η)": f"{last_lr:.2f}",
            "Change (-η × grad)": f"{chg2:+.5f}",
            "Updated Value": f"{old_b2 + chg2:+.4f}"
        })
    else:
        # Use recorded updates from actual step
        up = st.session_state.model.last_updates
        last_lr = up['learning_rate']
        param_table_data = []
        for i in range(st.session_state.model.input_size):
            for j in range(st.session_state.model.hidden_size):
                param_table_data.append({
                    "Parameter": f"W1 [x{i+1} → h{j+1}]",
                    "Old Value": f"{up['W1_old'][i, j]:+.4f}",
                    "Gradient (∂L/∂w)": f"{up['dW1'][i, j]:+.5f}",
                    "Learning Rate (η)": f"{last_lr:.2f}",
                    "Change (-η × grad)": f"{up['W1_change'][i, j]:+.5f}",
                    "Updated Value": f"{up['W1_new'][i, j]:+.4f}"
                })
        for j in range(st.session_state.model.hidden_size):
            param_table_data.append({
                "Parameter": f"b1 [h{j+1} Bias]",
                "Old Value": f"{up['b1_old'][0, j]:+.4f}",
                "Gradient (∂L/∂b)": f"{up['db1'][0, j]:+.5f}",
                "Learning Rate (η)": f"{last_lr:.2f}",
                "Change (-η × grad)": f"{up['b1_change'][0, j]:+.5f}",
                "Updated Value": f"{up['b1_new'][0, j]:+.4f}"
            })
        for j in range(st.session_state.model.hidden_size):
            param_table_data.append({
                "Parameter": f"W2 [h{j+1} → ŷ]",
                "Old Value": f"{up['W2_old'][j, 0]:+.4f}",
                "Gradient (∂L/∂w)": f"{up['dW2'][j, 0]:+.5f}",
                "Learning Rate (η)": f"{last_lr:.2f}",
                "Change (-η × grad)": f"{up['W2_change'][j, 0]:+.5f}",
                "Updated Value": f"{up['W2_new'][j, 0]:+.4f}"
            })
        param_table_data.append({
            "Parameter": "b2 [Output Bias]",
            "Old Value": f"{up['b2_old'][0, 0]:+.4f}",
            "Gradient (∂L/∂b)": f"{up['db2'][0, 0]:+.5f}",
            "Learning Rate (η)": f"{last_lr:.2f}",
            "Change (-η × grad)": f"{up['b2_change'][0, 0]:+.5f}",
            "Updated Value": f"{up['b2_new'][0, 0]:+.4f}"
        })

    df_params = pd.DataFrame(param_table_data)
    st.dataframe(df_params, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <b>Simple English Summary of Gradient Descent:</b><br>
        1. <b>Gradient (∂L/∂w):</b> Tells us the slope of the loss function with respect to this weight. 
        If the gradient is positive, increasing the weight increases error; so we must decrease the weight.<br>
        2. <b>Negative Sign:</b> Moving in the opposite direction of the gradient steps downhill towards minimal loss.<br>
        3. <b>Learning Rate (η):</b> Controls how big of a step we take. If η is too small, learning takes forever; if too big, the parameters overshoot and oscillate.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 4: SECTION G (Learning Experiments)
# ==========================================
with tab_experiments:
    st.markdown("### 🔬 Section G: Learning Experiments & Comparison")
    st.caption("Compare two experiments side-by-side to understand how activation functions and learning rates impact convergence.")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown("#### 🧪 Experiment A")
        exp_a_act = st.selectbox("Activation Function (A)", ['sigmoid', 'tanh', 'relu'], index=0, key="exp_a_act")
        exp_a_lr = st.slider("Learning Rate (A)", 0.05, 2.0, 0.5, 0.05, key="exp_a_lr")
        exp_a_epochs = st.number_input("Epochs (A)", min_value=50, max_value=2000, value=500, step=50, key="exp_a_ep")

    with col_exp2:
        st.markdown("#### 🧪 Experiment B")
        exp_b_act = st.selectbox("Activation Function (B)", ['sigmoid', 'tanh', 'relu'], index=1, key="exp_b_act")
        exp_b_lr = st.slider("Learning Rate (B)", 0.05, 2.0, 1.0, 0.05, key="exp_b_lr")
        exp_b_epochs = st.number_input("Epochs (B)", min_value=50, max_value=2000, value=500, step=50, key="exp_b_ep")

    if st.button("▶️ Run Comparative Experiment", type="primary", use_container_width=True):
        with st.spinner("Training both models from identical initial seeds..."):
            # Model A
            model_a = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation=exp_a_act, seed=42)
            hist_a = model_a.train(X, y, epochs=exp_a_epochs, learning_rate=exp_a_lr)
            pred_a, _ = model_a.predict(X)
            acc_a = model_a.compute_accuracy(y, pred_a)
            loss_a = model_a.compute_bce_loss(y, pred_a)

            # Model B
            model_b = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation=exp_b_act, seed=42)
            hist_b = model_b.train(X, y, epochs=exp_b_epochs, learning_rate=exp_b_lr)
            pred_b, _ = model_b.predict(X)
            acc_b = model_b.compute_accuracy(y, pred_b)
            loss_b = model_b.compute_bce_loss(y, pred_b)

            st.session_state.comparison_runs = {
                'hist_a': hist_a,
                'hist_b': hist_b,
                'label_a': f"{exp_a_act.upper()}, η={exp_a_lr}",
                'label_b': f"{exp_b_act.upper()}, η={exp_b_lr}",
                'loss_a': loss_a,
                'loss_b': loss_b,
                'acc_a': acc_a,
                'acc_b': acc_b,
                'act_a': exp_a_act,
                'act_b': exp_b_act
            }

    if st.session_state.comparison_runs:
        c_data = st.session_state.comparison_runs
        fig_comp = create_experiment_comparison_chart(
            hist_a=c_data['hist_a'],
            hist_b=c_data['hist_b'],
            label_a=c_data['label_a'],
            label_b=c_data['label_b']
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Experiment A ({c_data['label_a']})</div>
                <div class="metric-value">{c_data['loss_a']:.4f} Loss</div>
                <div class="metric-sub" style="color:#10B981;">Training Accuracy: {c_data['acc_a']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Experiment B ({c_data['label_b']})</div>
                <div class="metric-value">{c_data['loss_b']:.4f} Loss</div>
                <div class="metric-sub" style="color:#10B981;">Training Accuracy: {c_data['acc_b']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Plain-English analysis of results
        st.markdown("#### 💡 Comparative Observations in Plain English:")
        if c_data['loss_b'] < c_data['loss_a']:
            better = "Experiment B"
            worse = "Experiment A"
        else:
            better = "Experiment A"
            worse = "Experiment B"

        st.markdown(f"""
        <div class="info-card">
            <b>Why did {better} perform differently than {worse}?</b><br>
            • <b>Activation Function Dynamics:</b> Tanh is zero-centered with steeper gradients (range -1 to 1) 
            compared to Sigmoid (range 0 to 1), which prevents saturation and vanishing gradients during early epochs.<br>
            • <b>Learning Rate Effect:</b> A higher learning rate (when stable) takes bolder steps across flat loss plateaus, 
            allowing the network to rapidly break the symmetry of XOR.<br>
            • <b>Dead Neurons with ReLU:</b> If ReLU is used with high learning rates, some neurons can output 0 for all samples (gradient = 0), 
            causing those neurons to never recover ("dying ReLU").
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 5: STUDENT VIVA & PRESENTATION GUIDE
# ==========================================
with tab_viva:
    st.markdown("### 🎓 College Presentation & Viva Voce Cheatsheet")
    st.caption("Clear, confident explanations for the 8 core concepts required by your college teacher.")

    viva_qas = [
        (
            "1. What do weights and biases actually do?",
            r"""
            • <b>Weights ($w$):</b> Control the strength and polarity of connection between neurons. 
            A positive weight means higher input increases activation; a negative weight suppresses activation.<br>
            • <b>Biases ($b$):</b> Act as a threshold or offset. Without a bias, if all inputs are zero, the weighted sum is forced to be zero ($z = 0$). 
            The bias allows the neuron to shift its activation boundary independently of the inputs.
            """
        ),
        (
            "2. How does an individual neuron calculate its output?",
            r"""
            A neuron computes its output in two distinct, sequential steps:<br>
            1. <b>Linear Combination (Weighted Sum $z$):</b><br>
            $$z = (x_1 \times w_1) + (x_2 \times w_2) + \dots + (x_n \times w_n) + b$$<br>
            2. <b>Non-Linear Activation ($a$):</b><br>
            $$a = f(z)$$ where $f$ is an activation function like Sigmoid, ReLU, or Tanh.
            """
        ),
        (
            "3. How do activation functions affect the output?",
            r"""
            If we did not have activation functions, stacking 100 linear layers would still only produce a single linear equation ($y = W_{total} x + b_{total}$).<br>
            • <b>Sigmoid:</b> Squashes $z$ into $(0, 1)$, ideal for probabilities.<br>
            • <b>Tanh:</b> Squashes $z$ into $(-1, 1)$, zero-centered, which helps hidden layers learn faster.<br>
            • <b>ReLU:</b> Outputs $\max(0, z)$, avoiding saturation for positive values.
            """
        ),
        (
            "4. How do multiple layers process information?",
            r"""
            • <b>Input Layer:</b> Simply holds raw feature values ($x_1, x_2$).<br>
            • <b>Hidden Layer:</b> Acts as a feature extractor, bending and warping the input space into new coordinates.<br>
            • <b>Output Layer:</b> Reads the hidden representations and makes a final decision. 
            For XOR, the hidden layer transforms points that cannot be separated by a straight line into a higher-dimensional representation where a straight boundary can separate them.
            """
        ),
        (
            "5. How are predictions generated?",
            r"""
            Predictions are generated via <b>Forward Propagation</b>:<br>
            Raw inputs pass into Layer 1 $\to$ activations $A_1$ are computed $\to$ $A_1$ acts as inputs for the Output Layer $\to$ 
            Output neuron produces probability $\hat{y} = \sigma(z_2)$.<br>
            Finally, if $\hat{y} \ge 0.5$, we classify as <b>1</b>, else <b>0</b>.
            """
        ),
        (
            "6. How does loss measure prediction error?",
            r"""
            We use <b>Binary Cross-Entropy (BCE) Loss</b>:<br>
            $$L = -[y \ln(\hat{y}) + (1-y)\ln(1-\hat{y})]$$<br>
            • If expected $y=1$ and predicted $\hat{y}=0.99$, $-\ln(0.99) \approx 0.01$ (low penalty).<br>
            • If expected $y=1$ and predicted $\hat{y}=0.01$, $-\ln(0.01) \approx 4.60$ (severe penalty).<br>
            BCE strongly penalizes confident wrong predictions.
            """
        ),
        (
            "7. How does training update weights and biases?",
            r"""
            Training uses <b>Backpropagation and Gradient Descent</b>:<br>
            1. The chain rule of calculus calculates the partial derivative of loss with respect to each weight ($\frac{\partial L}{\partial w}$) and bias ($\frac{\partial L}{\partial b}$).<br>
            2. We update each parameter by stepping opposite to the gradient:<br>
            $$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$
            $$b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b}$$
            where $\eta$ is the learning rate.
            """
        ),
        (
            "8. How does the loss graph show learning progress?",
            r"""
            • A downward sloping loss curve shows that weight updates are successfully lowering prediction errors.<br>
            • Plateaus indicate flat gradient regions where the network is slowly escaping saddle points (common in XOR).<br>
            • A sudden drop in loss accompanied by accuracy jumping to 100% indicates that the network found the non-linear decision boundary for XOR.
            """
        )
    ]

    for title, ans in viva_qas:
        with st.expander(f"📌 {title}", expanded=True):
            st.markdown(ans, unsafe_allow_html=True)
