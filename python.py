import numpy as np

class FractionalLSTMCell:
    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM cell with given dimensions
        
        Args:
            input_size (int): Size of input vector
            hidden_size (int): Size of hidden state vector
        """
        # Initialize weight matrices and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier/Glorot initialization for weights
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Input gate weights
        self.Wii = np.random.randn(hidden_size, input_size) * scale
        self.Whi = np.random.randn(hidden_size, hidden_size) * scale
        self.bi = np.zeros(hidden_size)
        
        # Forget gate weights
        self.Wif = np.random.randn(hidden_size, input_size) * scale
        self.Whf = np.random.randn(hidden_size, hidden_size) * scale
        self.bf = np.zeros(hidden_size)
        
        # Cell gate weights
        self.Wig = np.random.randn(hidden_size, input_size) * scale
        self.Whg = np.random.randn(hidden_size, hidden_size) * scale
        self.bg = np.zeros(hidden_size)
        
        # Output gate weights
        self.Wio = np.random.randn(hidden_size, input_size) * scale
        self.Who = np.random.randn(hidden_size, hidden_size) * scale
        self.bo = np.zeros(hidden_size)

        # Initialize gradients
        self.init_gradients()
        
    def init_gradients(self):
        """Initialize gradient matrices for all weights and biases"""
        self.dWii = np.zeros_like(self.Wii)
        self.dWhi = np.zeros_like(self.Whi)
        self.dbi = np.zeros_like(self.bi)
        
        self.dWif = np.zeros_like(self.Wif)
        self.dWhf = np.zeros_like(self.Whf)
        self.dbf = np.zeros_like(self.bf)
        
        self.dWig = np.zeros_like(self.Wig)
        self.dWhg = np.zeros_like(self.Whg)
        self.dbg = np.zeros_like(self.bg)
        
        self.dWio = np.zeros_like(self.Wio)
        self.dWho = np.zeros_like(self.Who)
        self.dbo = np.zeros_like(self.bo)
    
    def sigmoid(self, x):
        """Sigmoid activation function with handling for fractional inputs"""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function with handling for fractional inputs"""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass of LSTM cell
        
        Args:
            x (np.array): Input vector
            h_prev (np.array): Previous hidden state
            c_prev (np.array): Previous cell state
            
        Returns:
            tuple: (hidden state, cell state, cache for backprop)
        """
        # Input gate
        i = self.sigmoid(np.dot(self.Wii, x) + np.dot(self.Whi, h_prev) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wif, x) + np.dot(self.Whf, h_prev) + self.bf)
        
        # Cell gate
        g = self.tanh(np.dot(self.Wig, x) + np.dot(self.Whg, h_prev) + self.bg)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wio, x) + np.dot(self.Who, h_prev) + self.bo)
        
        # Cell state
        c = f * c_prev + i * g
        
        # Hidden state
        h = o * self.tanh(c)
        
        # Cache for backprop
        cache = (x, h_prev, c_prev, i, f, g, o, c)
        
        return h, c, cache
    
    def backward(self, dh, dc, cache):
        """
        Backward pass of LSTM cell
        
        Args:
            dh (np.array): Gradient of loss with respect to hidden state
            dc (np.array): Gradient of loss with respect to cell state
            cache (tuple): Cached values from forward pass
            
        Returns:
            tuple: (dx, dh_prev, dc_prev) gradients for backpropagation
        """
        x, h_prev, c_prev, i, f, g, o, c = cache
        
        # Gradient through tanh
        tanh_c = self.tanh(c)
        dt_c = dh * o * (1 - tanh_c ** 2)
        dc = dc + dt_c
        
        # Output gate
        do = dh * tanh_c
        do_input = do * o * (1 - o)
        self.dWio += np.outer(do_input, x)
        self.dWho += np.outer(do_input, h_prev)
        self.dbo += do_input
        
        # Cell gate
        di = dc * g
        di_input = di * i * (1 - i)
        self.dWii += np.outer(di_input, x)
        self.dWhi += np.outer(di_input, h_prev)
        self.dbi += di_input
        
        # Forget gate
        df = dc * c_prev
        df_input = df * f * (1 - f)
        self.dWif += np.outer(df_input, x)
        self.dWhf += np.outer(df_input, h_prev)
        self.dbf += df_input
        
        # Input modulation gate
        dg = dc * i
        dg_input = dg * (1 - g ** 2)
        self.dWig += np.outer(dg_input, x)
        self.dWhg += np.outer(dg_input, h_prev)
        self.dbg += dg_input
        
        # Compute gradients for inputs
        dx = (np.dot(self.Wii.T, di_input) + 
              np.dot(self.Wif.T, df_input) +
              np.dot(self.Wig.T, dg_input) + 
              np.dot(self.Wio.T, do_input))
        
        dh_prev = (np.dot(self.Whi.T, di_input) + 
                  np.dot(self.Whf.T, df_input) +
                  np.dot(self.Whg.T, dg_input) + 
                  np.dot(self.Who.T, do_input))
        
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev
    
    def update_params(self, learning_rate):
        """
        Update parameters using accumulated gradients
        
        Args:
            learning_rate (float): Learning rate for gradient descent
        """
        # Update input gate parameters
        self.Wii -= learning_rate * self.dWii
        self.Whi -= learning_rate * self.dWhi
        self.bi -= learning_rate * self.dbi
        
        # Update forget gate parameters
        self.Wif -= learning_rate * self.dWif
        self.Whf -= learning_rate * self.dWhf
        self.bf -= learning_rate * self.dbf
        
        # Update cell gate parameters
        self.Wig -= learning_rate * self.dWig
        self.Whg -= learning_rate * self.dWhg
        self.bg -= learning_rate * self.dbg
        
        # Update output gate parameters
        self.Wio -= learning_rate * self.dWio
        self.Who -= learning_rate * self.dWho
        self.bo -= learning_rate * self.dbo
        
        # Reset gradients
        self.init_gradients()