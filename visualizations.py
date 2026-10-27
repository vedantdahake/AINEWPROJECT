"""
Plotly interactive visualizations for Neural Network X-Ray.
Includes:
- Interactive Neural Network Architecture diagram with actual weights, activations, and legend.
- Interactive Training Loss & Accuracy curve.
- 2D Decision Boundary visualization.
- Experiment Comparison curves.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Color palette for modern dark-themed dashboard
COLOR_BG = "#0B0F19"
COLOR_CARD = "#111827"
COLOR_GRID = "#1E293B"
COLOR_TEXT = "#E2E8F0"
COLOR_MUTED = "#94A3B8"
COLOR_POS_WEIGHT = "#38BDF8"    # Electric sky blue
COLOR_NEG_WEIGHT = "#F43F5E"    # Rose red / violet
COLOR_NEURON_INPUT = "#3B82F6"  # Blue
COLOR_NEURON_HIDDEN = "#8B5CF6" # Purple
COLOR_NEURON_OUTPUT = "#10B981" # Emerald / cyan
COLOR_HIGHLIGHT = "#FACC15"     # Gold glow for selected neuron


def create_network_diagram(
    network_obj,
    sample_index: int = 0,
    X: np.ndarray = None,
    selected_layer: Optional[str] = None,
    selected_neuron_idx: Optional[int] = None
) -> go.Figure:
    """
    Build an interactive Plotly diagram representing the neural network graph.
    Displays:
    - Input, Hidden, and Output layers.
    - Actual weights on connections (color & width indicate sign and magnitude).
    - Actual activations on neurons for the selected data sample.
    - Highlight on selected neuron.
    """
    if X is None:
        X = np.array([[0.0, 0.0]])
    
    # Run forward pass for this specific sample
    A2, cache = network_obj.forward(X)
    x_sample = cache['X'][sample_index]
    a1_sample = cache['A1'][sample_index]
    a2_sample = cache['A2'][sample_index]

    n_inputs = network_obj.input_size
    n_hidden = network_obj.hidden_size
    n_output = network_obj.output_size

    # Layer X coordinates
    layer_x = {'input': 0.0, 'hidden': 1.0, 'output': 2.0}

    # Helper to compute Y coordinates centered around 0
    def get_y_coords(count: int, spacing: float = 1.0) -> List[float]:
        total_height = (count - 1) * spacing
        start_y = total_height / 2.0
        return [start_y - i * spacing for i in range(count)]

    y_inputs = get_y_coords(n_inputs, spacing=1.2)
    y_hidden = get_y_coords(n_hidden, spacing=0.9)
    y_output = get_y_coords(n_output, spacing=1.0)

    fig = go.Figure()

    # 1. Draw connections: Input -> Hidden
    for i in range(n_inputs):
        for j in range(n_hidden):
            w = network_obj.W1[i, j]
            color = COLOR_POS_WEIGHT if w >= 0 else COLOR_NEG_WEIGHT
            width = max(1.0, min(8.0, 1.5 + abs(w) * 1.8))
            hover_text = f"Input x{i+1} → Hidden h{j+1}<br>Weight: <b>{w:+.4f}</b>"

            # Connection line
            fig.add_trace(go.Scatter(
                x=[layer_x['input'], layer_x['hidden']],
                y=[y_inputs[i], y_hidden[j]],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                opacity=0.75
            ))

            # Weight label positioned at 1/3 point along the connection
            mid_x = layer_x['input'] + (layer_x['hidden'] - layer_x['input']) * 0.32
            mid_y = y_inputs[i] + (y_hidden[j] - y_inputs[i]) * 0.32
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='markers',
                marker=dict(size=14, color=COLOR_CARD, line=dict(color=color, width=1.5)),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))

    # 2. Draw connections: Hidden -> Output
    for j in range(n_hidden):
        for k in range(n_output):
            w = network_obj.W2[j, k]
            color = COLOR_POS_WEIGHT if w >= 0 else COLOR_NEG_WEIGHT
            width = max(1.0, min(8.0, 1.5 + abs(w) * 1.8))
            hover_text = f"Hidden h{j+1} → Output ŷ{k+1}<br>Weight: <b>{w:+.4f}</b>"

            # Connection line
            fig.add_trace(go.Scatter(
                x=[layer_x['hidden'], layer_x['output']],
                y=[y_hidden[j], y_output[k]],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False,
                opacity=0.75
            ))

            # Weight label at 2/3 point along the connection
            mid_x = layer_x['hidden'] + (layer_x['output'] - layer_x['hidden']) * 0.68
            mid_y = y_hidden[j] + (y_output[k] - y_hidden[j]) * 0.68
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='markers',
                marker=dict(size=14, color=COLOR_CARD, line=dict(color=color, width=1.5)),
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))

    # 3. Draw Input Nodes
    input_texts = [f"x{i+1}: <b>{x_sample[i]:.2f}</b>" for i in range(n_inputs)]
    fig.add_trace(go.Scatter(
        x=[layer_x['input']] * n_inputs,
        y=y_inputs,
        mode='markers+text',
        marker=dict(
            size=44,
            color=COLOR_NEURON_INPUT,
            line=dict(color="#60A5FA", width=2.5)
        ),
        text=[f"x{i+1}" for i in range(n_inputs)],
        textfont=dict(color="#FFFFFF", size=13, family="monospace"),
        textposition="middle center",
        hoverinfo='text',
        hovertext=[f"Input Neuron x{i+1}<br>Value: {x_sample[i]:.4f}" for i in range(n_inputs)],
        name="Input Neurons"
    ))

    # 4. Draw Hidden Nodes
    hidden_colors = []
    hidden_borders = []
    hidden_border_widths = []
    for j in range(n_hidden):
        is_selected = (selected_layer == 'hidden' and selected_neuron_idx == j)
        hidden_colors.append(COLOR_HIGHLIGHT if is_selected else COLOR_NEURON_HIDDEN)
        hidden_borders.append("#FFFFFF" if is_selected else "#C084FC")
        hidden_border_widths.append(4.0 if is_selected else 2.0)

    fig.add_trace(go.Scatter(
        x=[layer_x['hidden']] * n_hidden,
        y=y_hidden,
        mode='markers+text',
        marker=dict(
            size=48,
            color=hidden_colors,
            line=dict(color=hidden_borders, width=hidden_border_widths)
        ),
        text=[f"h{j+1}" for j in range(n_hidden)],
        textfont=dict(color="#000000" if any(c == COLOR_HIGHLIGHT for c in hidden_colors) else "#FFFFFF", size=13, family="monospace"),
        textposition="middle center",
        hoverinfo='text',
        hovertext=[
            f"Hidden Neuron h{j+1}<br>Bias: {network_obj.b1[0, j]:+.3f}<br>Activation a: <b>{a1_sample[j]:.4f}</b>"
            for j in range(n_hidden)
        ],
        name="Hidden Neurons"
    ))

    # 5. Draw Output Nodes
    output_colors = []
    output_borders = []
    output_border_widths = []
    for k in range(n_output):
        is_selected = (selected_layer == 'output' and selected_neuron_idx == k)
        output_colors.append(COLOR_HIGHLIGHT if is_selected else COLOR_NEURON_OUTPUT)
        output_borders.append("#FFFFFF" if is_selected else "#34D399")
        output_border_widths.append(4.0 if is_selected else 2.5)

    fig.add_trace(go.Scatter(
        x=[layer_x['output']] * n_output,
        y=y_output,
        mode='markers+text',
        marker=dict(
            size=50,
            color=output_colors,
            line=dict(color=output_borders, width=output_border_widths)
        ),
        text=[f"ŷ" for _ in range(n_output)],
        textfont=dict(color="#000000" if any(c == COLOR_HIGHLIGHT for c in output_colors) else "#FFFFFF", size=14, family="monospace"),
        textposition="middle center",
        hoverinfo='text',
        hovertext=[
            f"Output Neuron ŷ<br>Bias: {network_obj.b2[0, k]:+.3f}<br>Predicted Prob: <b>{a2_sample[k]:.4f}</b>"
            for k in range(n_output)
        ],
        name="Output Neuron"
    ))

    # Annotations for values above/below neurons
    # Layer column headers
    fig.add_annotation(x=layer_x['input'], y=max(y_inputs) + 0.9, text="<b>INPUT LAYER</b><br>(Features)", showarrow=False, font=dict(color="#60A5FA", size=13))
    fig.add_annotation(x=layer_x['hidden'], y=max(y_hidden) + 0.9, text=f"<b>HIDDEN LAYER</b><br>({network_obj.activation_name.upper()})", showarrow=False, font=dict(color="#C084FC", size=13))
    fig.add_annotation(x=layer_x['output'], y=max(y_output) + 0.9, text="<b>OUTPUT LAYER</b><br>(SIGMOID)", showarrow=False, font=dict(color="#34D399", size=13))

    # Numerical activation badges below each neuron
    for i, (y, val) in enumerate(zip(y_inputs, x_sample)):
        fig.add_annotation(x=layer_x['input'], y=y - 0.45, text=f"<b>{val:.1f}</b>", showarrow=False, font=dict(color=COLOR_TEXT, size=11))
    for j, (y, val) in enumerate(zip(y_hidden, a1_sample)):
        fig.add_annotation(x=layer_x['hidden'], y=y - 0.42, text=f"a=<b>{val:.3f}</b>", showarrow=False, font=dict(color=COLOR_TEXT, size=11))
    for k, (y, val) in enumerate(zip(y_output, a2_sample)):
        fig.add_annotation(x=layer_x['output'], y=y - 0.45, text=f"p=<b>{val:.3f}</b>", showarrow=False, font=dict(color=COLOR_TEXT, size=12))

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.4, 2.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color=COLOR_TEXT, size=11)
        ),
        height=480,
    )

    return fig


def create_loss_accuracy_chart(history: List[Dict[str, Any]]) -> go.Figure:
    """
    Plot training loss and training accuracy over epochs.
    Uses actual history records from model training.
    """
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No training data yet. Click 'Train Network' or 'Train One Step'.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color=COLOR_MUTED, size=14))
        fig.update_layout(paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, height=350)
        return fig

    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    accuracies = [h['accuracy'] for h in history]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Loss line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            name="Binary Cross-Entropy Loss",
            mode='lines',
            line=dict(color=COLOR_POS_WEIGHT, width=2.5),
            hoverinfo='x+y',
            hovertemplate="Epoch: %{x}<br>Loss: <b>%{y:.5f}</b><extra></extra>"
        ),
        secondary_y=False
    )

    # Accuracy line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=accuracies,
            name="Training Accuracy (%)",
            mode='lines',
            line=dict(color="#10B981", width=2.0, dash='dot'),
            hoverinfo='x+y',
            hovertemplate="Epoch: %{x}<br>Accuracy: <b>%{y:.1f}%</b><extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT)),
        font=dict(family="sans-serif", color=COLOR_TEXT),
        hovermode="x unified",
        height=360,
    )

    fig.update_xaxes(
        title="Training Epoch",
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        color=COLOR_MUTED
    )
    fig.update_yaxes(
        title="BCE Loss",
        gridcolor=COLOR_GRID,
        zerolinecolor=COLOR_GRID,
        color=COLOR_POS_WEIGHT,
        secondary_y=False
    )
    fig.update_yaxes(
        title="Training Accuracy (%)",
        range=[-5, 105],
        showgrid=False,
        color="#10B981",
        secondary_y=True
    )

    return fig


def create_decision_boundary_plot(
    network_obj,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "2D Decision Surface"
) -> go.Figure:
    """
    Generate a 2D contour plot showing model predictions over the input space,
    overlaying the actual training samples.
    """
    # Create coordinate grid
    res = 50
    x_span = np.linspace(-0.2, 1.2, res)
    y_span = np.linspace(-0.2, 1.2, res)
    xx, yy = np.meshgrid(x_span, y_span)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    probs, _ = network_obj.predict(grid_points)
    probs_grid = probs.reshape(xx.shape)

    fig = go.Figure()

    # Contour surface
    fig.add_trace(go.Contour(
        x=x_span,
        y=y_span,
        z=probs_grid,
        colorscale=[
            [0.0, "#1E1B4B"],   # Dark indigo (Class 0)
            [0.5, "#4338CA"],   # Decision boundary region
            [1.0, "#065F46"]    # Dark emerald (Class 1)
        ],
        contours=dict(
            start=0.0,
            end=1.0,
            size=0.1,
            showlabels=True,
            labelfont=dict(size=10, color="#FFFFFF")
        ),
        opacity=0.85,
        hoverinfo='none',
        showscale=False
    ))

    # Add 0.5 decision threshold boundary line
    fig.add_trace(go.Contour(
        x=x_span,
        y=y_span,
        z=probs_grid,
        contours=dict(
            type="constraint",
            operation="=",
            value=0.5,
            coloring="none",
        ),
        line=dict(color="#FACC15", width=2.5, dash="dash"),
        showscale=False,
        name="Decision Boundary (0.5)"
    ))

    # Actual dataset points
    y_flat = y.ravel().astype(int)
    for class_val, color, symbol, label in [
        (0, "#F43F5E", "circle", "Class 0"),
        (1, "#10B981", "diamond", "Class 1")
    ]:
        mask = (y_flat == class_val)
        if np.any(mask):
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(size=16, color=color, symbol=symbol, line=dict(color="#FFFFFF", width=2)),
                name=label,
                hovertemplate="x₁: %{x}<br>x₂: %{y}<br>Target: " + str(class_val) + "<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLOR_TEXT, size=14)),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(title="Input 1 (x₁)", range=[-0.2, 1.2], gridcolor=COLOR_GRID, color=COLOR_MUTED),
        yaxis=dict(title="Input 2 (x₂)", range=[-0.2, 1.2], gridcolor=COLOR_GRID, color=COLOR_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color=COLOR_TEXT)),
        height=380,
    )

    return fig


def create_experiment_comparison_chart(
    hist_a: List[Dict[str, Any]],
    hist_b: List[Dict[str, Any]],
    label_a: str,
    label_b: str
) -> go.Figure:
    """
    Overlays loss curves from two distinct experiments for direct educational comparison.
    """
    fig = go.Figure()

    if hist_a:
        fig.add_trace(go.Scatter(
            x=[h['epoch'] for h in hist_a],
            y=[h['loss'] for h in hist_a],
            name=f"Exp A ({label_a})",
            mode='lines',
            line=dict(color="#38BDF8", width=2.5),
            hovertemplate="Epoch %{x}: Loss %{y:.4f}<extra></extra>"
        ))

    if hist_b:
        fig.add_trace(go.Scatter(
            x=[h['epoch'] for h in hist_b],
            y=[h['loss'] for h in hist_b],
            name=f"Exp B ({label_b})",
            mode='lines',
            line=dict(color="#F43F5E", width=2.5, dash='dash'),
            hovertemplate="Epoch %{x}: Loss %{y:.4f}<extra></extra>"
        ))

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        margin=dict(l=40, r=40, t=30, b=30),
        xaxis=dict(title="Epoch", gridcolor=COLOR_GRID, color=COLOR_MUTED),
        yaxis=dict(title="Binary Cross-Entropy Loss", gridcolor=COLOR_GRID, color=COLOR_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=COLOR_TEXT)),
        hovermode="x unified",
        height=380
    )

    return fig
