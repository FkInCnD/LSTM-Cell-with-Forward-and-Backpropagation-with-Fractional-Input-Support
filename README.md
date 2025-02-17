# LSTM-Cell-with-Forward-and-Backpropagation-with-Fractional-Input-Support
Fixed-point arithmetic:

32-bit width with 16 decimal bits
Supports fractional inputs and weights
Custom multiplication function to handle fixed-point operations


Core LSTM components:

Forget gate (f_t)
Input gate (i_t)
Cell state candidate (c_tilde)
Output gate (o_t)
Cell state (c_t)
Hidden state (h_t)


Key features:

Configurable bit widths for fixed-point representation
Separate forward and backward propagation modes
Support for gradient calculations
Built-in activation functions (sigmoid and tanh) using LUT approach
Complete testbench for verification


Backpropagation support:

Calculates gradients for input (dx_t)
Calculates gradients for previous hidden state (dh_prev)
Calculates gradients for previous cell state (dc_prev)
