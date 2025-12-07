# LSTM Architecture Quick Reference

## Visual Architecture

```
Input at time t
     |
     v
┌─────────────────────────────────────────────────────────┐
│                      LSTM Cell                          │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ Forget Gate│  │ Input Gate │  │ Output Gate│       │
│  │            │  │            │  │            │       │
│  │  f_t = σ() │  │  i_t = σ() │  │  o_t = σ() │       │
│  └────┬───────┘  └────┬───────┘  └────┬───────┘       │
│       │               │               │                │
│       v               v               │                │
│  c_prev ──[×]─────[×]──c_tilde       │                │
│            │       │                  │                │
│            └───[+]─┘                  │                │
│                │                      │                │
│                v                      v                │
│              c_new ──[tanh]──────[×]──────> h_new      │
│                                                         │
└─────────────────────────────────────────────────────────┘
     │                                   │
     v                                   v
Cell state to t+1              Hidden state to t+1
                               (also output)
```

## Mathematical Equations

### Gate Computations

**Forget Gate** (what to forget from cell state):
```
f_t = σ(W_f @ x_t + U_f @ h_{t-1} + b_f)
```

**Input Gate** (what new information to add):
```
i_t = σ(W_i @ x_t + U_i @ h_{t-1} + b_i)
```

**Candidate Cell State** (new information):
```
c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
```

**Output Gate** (what to output):
```
o_t = σ(W_o @ x_t + U_o @ h_{t-1} + b_o)
```

### State Updates

**Cell State Update** (combine old and new):
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

**Hidden State Update** (filtered output):
```
h_t = o_t ⊙ tanh(c_t)
```

Where:
- `⊙` denotes element-wise multiplication
- `σ` is the sigmoid function
- `@` is matrix multiplication

## Parameters

### For each gate (4 gates total):
- **W**: Input weight matrix `(hidden_size, input_size)`
- **U**: Recurrent weight matrix `(hidden_size, hidden_size)`
- **b**: Bias vector `(hidden_size, 1)`

### Total parameters (no output projection):
```
params = 4 × (hidden_size × input_size +     # W matrices
              hidden_size × hidden_size +     # U matrices
              hidden_size)                    # b vectors

       = 4 × hidden_size × (input_size + hidden_size + 1)
```

### Example (input=32, hidden=64):
```
params = 4 × 64 × (32 + 64 + 1)
       = 4 × 64 × 97
       = 24,832 parameters
```

## Initialization Strategy

| Parameter | Method | Value | Reason |
|-----------|--------|-------|--------|
| `W_f, W_i, W_c, W_o` | Xavier | U(-√(6/(in+out)), √(6/(in+out))) | Maintain activation variance |
| `U_f, U_i, U_c, U_o` | Orthogonal | SVD-based orthogonal matrix | Prevent gradient explosion/vanishing |
| `b_f` | Constant | **1.0** | Help learn long-term dependencies |
| `b_i, b_c, b_o` | Constant | 0.0 | Standard initialization |

## Key Design Features

### 1. Additive Cell State Update
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ↑               ↑
     forget          add new
```
- **Additive** (not multiplicative like vanilla RNN)
- Allows gradient to flow unchanged through time
- Solves vanishing gradient problem

### 2. Gated Control
All gates use sigmoid activation (output in [0, 1]):
- Acts as "soft switch"
- 0 = block completely
- 1 = pass completely
- Learnable control over information flow

### 3. Separate Memory and Output
- **Cell state (c)**: Long-term memory
- **Hidden state (h)**: Filtered output
- Allows model to remember without outputting

## Forward Pass Algorithm

```python
# Initialize states
h_0 = zeros(hidden_size, batch_size)
c_0 = zeros(hidden_size, batch_size)

# Process sequence
for t in range(seq_len):
    x_t = sequence[:, t, :]

    # Compute gates
    f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
    i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
    c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
    o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)

    # Update states
    c_t = f_t * c_{t-1} + i_t * c̃_t
    h_t = o_t * tanh(c_t)

    outputs[t] = h_t
```

## Shape Flow Example

Input configuration:
- `batch_size = 2`
- `seq_len = 10`
- `input_size = 32`
- `hidden_size = 64`

Shape transformations:
```
x_t:       (2, 32)      Input at time t
h_{t-1}:   (64, 2)      Previous hidden (transposed)
c_{t-1}:   (64, 2)      Previous cell (transposed)

W_f @ x_t:              (64, 32) @ (32, 2) = (64, 2)
U_f @ h_{t-1}:          (64, 64) @ (64, 2) = (64, 2)
b_f:                    (64, 1) → broadcast to (64, 2)

f_t:       (64, 2)      Forget gate activations
i_t:       (64, 2)      Input gate activations
c̃_t:       (64, 2)      Candidate cell state
o_t:       (64, 2)      Output gate activations

c_t:       (64, 2)      New cell state
h_t:       (64, 2)      New hidden state

output_t:  (2, 64)      Transposed for output
```

## Activation Functions

### Sigmoid (for gates)
```
σ(x) = 1 / (1 + e^(-x))
```
- Range: (0, 1)
- Smooth, differentiable
- Used for gating (soft on/off)

### Tanh (for cell state and output)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Range: (-1, 1)
- Zero-centered
- Used for actual values

## Usage Examples

### 1. Sequence Classification
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=10)
output = lstm.forward(sequence, return_sequences=False)
# output shape: (batch, 10) - class logits
```

### 2. Sequence-to-Sequence
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=32)
outputs = lstm.forward(sequence, return_sequences=True)
# outputs shape: (batch, seq_len, 32)
```

### 3. State Extraction
```python
lstm = LSTM(input_size=32, hidden_size=64)
outputs, h, c = lstm.forward(sequence,
                             return_sequences=True,
                             return_state=True)
# outputs: (batch, seq_len, 64)
# h: (batch, 64) - final hidden state
# c: (batch, 64) - final cell state
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Vanishing gradients | ✓ Orthogonal initialization for U matrices |
| Exploding gradients | ✓ Gradient clipping (not implemented) |
| Can't learn long dependencies | ✓ Forget bias = 1.0 |
| Unstable training | ✓ Xavier initialization for W matrices |
| NaN in forward pass | ✓ Numerically stable sigmoid |

## Comparison with Vanilla RNN

| Feature | Vanilla RNN | LSTM |
|---------|-------------|------|
| State update | Multiplicative | Additive |
| Memory mechanism | Single hidden state | Separate cell & hidden |
| Gradient flow | Exponential decay | Controlled by gates |
| Long-term dependencies | Poor | Good |
| Parameters | O(h²) | O(4h²) |
| Computational cost | 1x | ~4x |

## Implementation Files

1. **lstm_baseline.py**: Core implementation
   - `LSTMCell` class (single time step)
   - `LSTM` class (sequence processing)
   - Initialization functions
   - Test suite

2. **lstm_baseline_demo.py**: Usage examples
   - Sequence classification
   - Sequence-to-sequence
   - State persistence
   - Initialization importance

3. **LSTM_BASELINE_SUMMARY.md**: Comprehensive documentation
   - Implementation details
   - Test results
   - Design decisions

## References

- Original LSTM Paper: Hochreiter & Schmidhuber (1997)
- Forget gate: Gers et al. (2000)
- Orthogonal initialization: Saxe et al. (2013)
- Xavier initialization: Glorot & Bengio (2010)

---

**Implementation**: NumPy-only, educational
**Quality**: Production-ready
**Status**: Complete and tested
**Use case**: Baseline for Relational RNN comparison
