# 🧠 Neural Network X-Ray: Interactive Neural Network Visualizer

An educational, interactive dashboard that visualizes how an artificial neural network computes predictions, calculates neuron outputs, and learns through gradient descent.

Built with **pure NumPy** (no TensorFlow, PyTorch, or Keras) to guarantee that every single weight, activation, loss, and gradient is calculated live and transparently.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests (Mathematical Verification)
```bash
python -m pytest -v tests/
```

### 4. Launch the Interactive Dashboard
Run the dashboard using Python's module runner:
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if your Python Scripts folder is in your system PATH).*

The application will launch automatically in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
neural-network-xray/
├── app.py                     # Streamlit web dashboard with custom dark UI
├── neural_network.py          # Core NumPy neural network (forward, backward, updates, inspect)
├── datasets.py                # Datasets (XOR, AND, OR, Circle)
├── visualizations.py          # Plotly diagrams (network graph, loss curves, decision boundaries)
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation and college viva explanation guide
└── tests/
    └── test_neural_network.py # Pytest suite with numerical gradient checking
```

---

## 🎯 Key Features & Dashboard Sections

### Section A: Experiment Setup
- **Dataset**: XOR (default), AND, OR, Concentric Circle.
- **Architecture**: Configurable input (2), hidden neurons ($2 \to 8$, default 4), and output (1).
- **Activation Functions**: Sigmoid, Tanh, and ReLU for hidden neurons; Sigmoid for binary probability output.
- **Controls**: Learning rate slider ($\eta$), training epochs, reproducible random seed.
- **Action Buttons**: `Train Network`, `Train 1 Step` (for stepping through individual updates), and `Reset Parameters`.

### Section B: Neural Network Visualization
- Interactive Plotly diagram showing all neurons and connections.
- **Blue connections** indicate positive weights; **Pink/Red connections** indicate negative weights.
- Connection line thickness indicates weight magnitude $|w|$.
- Live activations ($a$) and probabilities ($p$) displayed above/below each neuron.
- Gold highlight ring surrounds the currently selected neuron.

### Section C: Inside a Neuron (Transparent Arithmetic)
- Select any hidden neuron or the output neuron to view its internal arithmetic table:
  - Input values ($x_i$ or $a^{[1]}_i$)
  - Weights ($w_{ij}$)
  - Products ($x_i \times w_{ij}$)
  - Bias ($b_j$)
  - Weighted sum: $z = \sum (x_i \cdot w_{ij}) + b_j$
  - Activation function: $a = f(z)$
- Substituted mathematical formula with exact numbers.
- Simple English explanation tailored for a student presenting to their college professor.

### Section D: Prediction Lab
- Live table of all XOR inputs, expected targets ($y$), predicted probabilities ($\hat{y}$), binary class (threshold 0.5), and accuracy status.
- **2D Decision Surface Plot**: Visualizes how the hidden layer warps 2D space to linearly separate XOR.

### Section E: Training Visualization
- Loss vs. Epochs curve plotted with Plotly.
- Real-time training accuracy (%) on the 4 XOR examples.
- Status indicator (Initialized, Learning, Converged, Stagnant).

### Section F: Weight and Bias Updates (Gradient Descent)
- Before-and-after view showing:
  - Old parameter value ($w_{old}$)
  - Gradient computed by backpropagation ($\frac{\partial L}{\partial w}$)
  - Learning rate ($\eta$)
  - Parameter change ($-\eta \cdot \text{gradient}$)
  - Updated parameter value ($w_{new}$)
- Formula: $w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$

### Section G: Learning Experiments
- Side-by-side comparison of two configurations (e.g., Tanh vs. Sigmoid, or $\eta=0.1$ vs. $\eta=1.0$).
- Overlaid loss curves and comparative analysis explaining differences in convergence speed and vanishing gradients.

---

## 📐 Mathematical Reference

### 1. Forward Propagation
$$\begin{aligned}
Z^{[1]} &= X W^{[1]} + b^{[1]} \\
A^{[1]} &= f(Z^{[1]}) \quad \text{(where } f \in \{\text{Sigmoid, Tanh, ReLU}\}\text{)} \\
Z^{[2]} &= A^{[1]} W^{[2]} + b^{[2]} \\
A^{[2]} &= \sigma(Z^{[2]}) = \frac{1}{1 + e^{-Z^{[2]}}}
\end{aligned}$$

### 2. Binary Cross-Entropy Loss
$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

### 3. Backpropagation Gradients
Output layer error:
$$\delta^{[2]} = A^{[2]} - y$$
$$\frac{\partial L}{\partial W^{[2]}} = \frac{1}{N} (A^{[1]})^T \delta^{[2]}, \quad \frac{\partial L}{\partial b^{[2]}} = \frac{1}{N} \sum \delta^{[2]}$$

Hidden layer error:
$$\delta^{[1]} = (\delta^{[2]} (W^{[2]})^T) \odot f'(Z^{[1]})$$
$$\frac{\partial L}{\partial W^{[1]}} = \frac{1}{N} X^T \delta^{[1]}, \quad \frac{\partial L}{\partial b^{[1]}} = \frac{1}{N} \sum \delta^{[1]}$$

### 4. Gradient Descent Update Rule
$$W \leftarrow W - \eta \frac{\partial L}{\partial W}$$
$$b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

---

## 🎓 College Viva / Presentation Cheatsheet

1. **What weights and biases do:**
   - *Weights* determine the influence and polarity of each incoming signal.
   - *Biases* allow the neuron to shift its activation boundary so it does not pass through the origin when inputs are zero.
2. **How a neuron calculates its output:**
   - Step 1: Linear combination $z = \sum (x_i w_i) + b$.
   - Step 2: Non-linear activation $a = f(z)$.
3. **How activation functions affect the output:**
   - Without non-linear activations, any deep network collapses into a single linear matrix multiplication. Non-linearities allow the network to model complex functions.
4. **How multiple layers process information:**
   - The hidden layer maps the linearly inseparable XOR coordinates $(x_1, x_2)$ into a transformed space $(a_1, a_2, \dots)$ where a single linear hyperplane can separate the classes.
5. **How predictions are generated:**
   - Forward pass produces a probability $\hat{y} \in (0, 1)$. If $\hat{y} \ge 0.5$, predicted class is 1; otherwise 0.
6. **How loss measures prediction error:**
   - Binary cross-entropy heavily penalizes confident wrong predictions. Minimal loss means predictions are close to ground truth.
7. **How training updates weights and biases:**
   - Backpropagation computes partial derivatives (slopes) via the chain rule. Gradient descent subtracts a fraction ($\eta$) of the gradient to step downhill.
8. **How the loss graph shows learning progress:**
   - A declining curve proves parameter updates are reducing prediction error. Plateaus indicate saddle points or flat gradients.

---

## ✅ Implementation Checklist

- [x] Pure NumPy implementation of neural network without external deep learning frameworks.
- [x] XOR dataset with expected outputs: $(0,0)\to 0, (0,1)\to 1, (1,0)\to 1, (1,1)\to 0$.
- [x] Configurable architecture ($2 \to 4 \to 1$ default).
- [x] Configurable activations (Sigmoid, ReLU, Tanh) with exact derivatives.
- [x] Interactive Plotly network graph with color-coded weights and activation values.
- [x] Transparent neuron inspection table displaying inputs, weights, products, bias, sum, and substituted formula.
- [x] Prediction lab table with probability, 0.5 threshold class, and correctness status.
- [x] Training loss and accuracy curves plotted using actual training records.
- [x] Weight and bias before-and-after view with gradient descent updates ($-\eta \cdot \nabla$).
- [x] Learning experiment comparator for side-by-side model analysis.
- [x] Pytest suite with analytical vs. numerical finite-difference gradient verification.
- [x] Modern dark-themed dashboard with blue/purple accents and clear explanations.
