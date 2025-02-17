// Fixed point parameters
`define FIXED_POINT_WIDTH 32
`define DECIMAL_BITS 16
`define INTEGER_BITS 15  // 1 bit for sign

module lstm_cell (
    input wire clk,
    input wire rst,
    input wire enable,
    input wire forward_mode, // 1 for forward prop, 0 for backprop
    
    // Input ports - all are fixed point numbers
    input wire signed [`FIXED_POINT_WIDTH-1:0] x_t,      // Current input
    input wire signed [`FIXED_POINT_WIDTH-1:0] h_prev,   // Previous hidden state
    input wire signed [`FIXED_POINT_WIDTH-1:0] c_prev,   // Previous cell state
    
    // Weight matrices (packed arrays)
    input wire signed [`FIXED_POINT_WIDTH-1:0] W_f [`FIXED_POINT_WIDTH-1:0], // Forget gate weights
    input wire signed [`FIXED_POINT_WIDTH-1:0] W_i [`FIXED_POINT_WIDTH-1:0], // Input gate weights
    input wire signed [`FIXED_POINT_WIDTH-1:0] W_c [`FIXED_POINT_WIDTH-1:0], // Cell state weights
    input wire signed [`FIXED_POINT_WIDTH-1:0] W_o [`FIXED_POINT_WIDTH-1:0], // Output gate weights
    
    // Bias terms
    input wire signed [`FIXED_POINT_WIDTH-1:0] b_f,      // Forget gate bias
    input wire signed [`FIXED_POINT_WIDTH-1:0] b_i,      // Input gate bias
    input wire signed [`FIXED_POINT_WIDTH-1:0] b_c,      // Cell state bias
    input wire signed [`FIXED_POINT_WIDTH-1:0] b_o,      // Output gate bias
    
    // Output ports
    output reg signed [`FIXED_POINT_WIDTH-1:0] h_t,      // Current hidden state
    output reg signed [`FIXED_POINT_WIDTH-1:0] c_t,      // Current cell state
    
    // Backpropagation gradients (only valid when forward_mode = 0)
    output reg signed [`FIXED_POINT_WIDTH-1:0] dx_t,     // Gradient w.r.t. input
    output reg signed [`FIXED_POINT_WIDTH-1:0] dh_prev,  // Gradient w.r.t. previous hidden state
    output reg signed [`FIXED_POINT_WIDTH-1:0] dc_prev   // Gradient w.r.t. previous cell state
);

    // Internal signals for gates
    reg signed [`FIXED_POINT_WIDTH-1:0] f_t;    // Forget gate
    reg signed [`FIXED_POINT_WIDTH-1:0] i_t;    // Input gate
    reg signed [`FIXED_POINT_WIDTH-1:0] c_tilde; // Candidate cell state
    reg signed [`FIXED_POINT_WIDTH-1:0] o_t;    // Output gate
    
    // Fixed point multiplication
    function [`FIXED_POINT_WIDTH-1:0] fixed_mult;
        input [`FIXED_POINT_WIDTH-1:0] a;
        input [`FIXED_POINT_WIDTH-1:0] b;
        reg [2*`FIXED_POINT_WIDTH-1:0] temp;
        begin
            temp = a * b;
            fixed_mult = temp[`FIXED_POINT_WIDTH-1+`DECIMAL_BITS:`DECIMAL_BITS];
        end
    endfunction
    
    // Sigmoid activation function using LUT
    function [`FIXED_POINT_WIDTH-1:0] sigmoid;
        input [`FIXED_POINT_WIDTH-1:0] x;
        // Implementation using look-up table or piece-wise linear approximation
        // This is a simplified version - you would want a more accurate implementation
        reg [`FIXED_POINT_WIDTH-1:0] result;
        begin
            if (x[`FIXED_POINT_WIDTH-1]) // Negative input
                result = {`FIXED_POINT_WIDTH{1'b0}}; // Close to 0
            else
                result = {1'b0, {(`FIXED_POINT_WIDTH-1){1'b1}}}; // Close to 1
            sigmoid = result;
        end
    endfunction
    
    // Hyperbolic tangent function using LUT
    function [`FIXED_POINT_WIDTH-1:0] tanh;
        input [`FIXED_POINT_WIDTH-1:0] x;
        // Implementation using look-up table or piece-wise linear approximation
        reg [`FIXED_POINT_WIDTH-1:0] result;
        begin
            if (x[`FIXED_POINT_WIDTH-1]) // Negative input
                result = {1'b1, {(`FIXED_POINT_WIDTH-1){1'b0}}}; // Close to -1
            else
                result = {1'b0, {(`FIXED_POINT_WIDTH-1){1'b1}}}; // Close to 1
            tanh = result;
        end
    endfunction

    // Forward propagation
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            h_t <= 0;
            c_t <= 0;
            dx_t <= 0;
            dh_prev <= 0;
            dc_prev <= 0;
        end
        else if (enable && forward_mode) begin
            // Calculate gates
            f_t <= sigmoid(fixed_mult(W_f[0], x_t) + fixed_mult(W_f[1], h_prev) + b_f);
            i_t <= sigmoid(fixed_mult(W_i[0], x_t) + fixed_mult(W_i[1], h_prev) + b_i);
            c_tilde <= tanh(fixed_mult(W_c[0], x_t) + fixed_mult(W_c[1], h_prev) + b_c);
            o_t <= sigmoid(fixed_mult(W_o[0], x_t) + fixed_mult(W_o[1], h_prev) + b_o);
            
            // Update cell state and hidden state
            c_t <= fixed_mult(f_t, c_prev) + fixed_mult(i_t, c_tilde);
            h_t <= fixed_mult(o_t, tanh(c_t));
        end
        else if (enable && !forward_mode) begin
            // Backpropagation implementation
            // Note: This is a simplified version of backprop
            // You would need to implement the full gradient calculations
            
            // Calculate gradients for gates
            reg signed [`FIXED_POINT_WIDTH-1:0] do_t;
            reg signed [`FIXED_POINT_WIDTH-1:0] dc_tilde;
            reg signed [`FIXED_POINT_WIDTH-1:0] di_t;
            reg signed [`FIXED_POINT_WIDTH-1:0] df_t;
            
            // Gradient calculations (simplified)
            do_t = fixed_mult(h_t, (1 - o_t));
            dc_tilde = fixed_mult(i_t, (1 - fixed_mult(c_tilde, c_tilde)));
            di_t = fixed_mult(c_tilde, (1 - i_t));
            df_t = fixed_mult(c_prev, (1 - f_t));
            
            // Calculate input gradients
            dx_t <= fixed_mult(W_f[0], df_t) + 
                   fixed_mult(W_i[0], di_t) + 
                   fixed_mult(W_c[0], dc_tilde) + 
                   fixed_mult(W_o[0], do_t);
                   
            // Calculate previous hidden state gradients
            dh_prev <= fixed_mult(W_f[1], df_t) + 
                      fixed_mult(W_i[1], di_t) + 
                      fixed_mult(W_c[1], dc_tilde) + 
                      fixed_mult(W_o[1], do_t);
                      
            // Calculate previous cell state gradients
            dc_prev <= fixed_mult(f_t, (1 - fixed_mult(c_prev, c_prev)));
        end
    end

endmodule

// Testbench module
module lstm_cell_tb;
    reg clk;
    reg rst;
    reg enable;
    reg forward_mode;
    
    // Test signals
    reg signed [`FIXED_POINT_WIDTH-1:0] x_t;
    reg signed [`FIXED_POINT_WIDTH-1:0] h_prev;
    reg signed [`FIXED_POINT_WIDTH-1:0] c_prev;
    
    // Weight matrices
    reg signed [`FIXED_POINT_WIDTH-1:0] W_f [`FIXED_POINT_WIDTH-1:0];
    reg signed [`FIXED_POINT_WIDTH-1:0] W_i [`FIXED_POINT_WIDTH-1:0];
    reg signed [`FIXED_POINT_WIDTH-1:0] W_c [`FIXED_POINT_WIDTH-1:0];
    reg signed [`FIXED_POINT_WIDTH-1:0] W_o [`FIXED_POINT_WIDTH-1:0];
    
    // Bias terms
    reg signed [`FIXED_POINT_WIDTH-1:0] b_f;
    reg signed [`FIXED_POINT_WIDTH-1:0] b_i;
    reg signed [`FIXED_POINT_WIDTH-1:0] b_c;
    reg signed [`FIXED_POINT_WIDTH-1:0] b_o;
    
    // Output signals
    wire signed [`FIXED_POINT_WIDTH-1:0] h_t;
    wire signed [`FIXED_POINT_WIDTH-1:0] c_t;
    wire signed [`FIXED_POINT_WIDTH-1:0] dx_t;
    wire signed [`FIXED_POINT_WIDTH-1:0] dh_prev;
    wire signed [`FIXED_POINT_WIDTH-1:0] dc_prev;
    
    // Instantiate LSTM cell
    lstm_cell lstm_inst (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .forward_mode(forward_mode),
        .x_t(x_t),
        .h_prev(h_prev),
        .c_prev(c_prev),
        .W_f(W_f),
        .W_i(W_i),
        .W_c(W_c),
        .W_o(W_o),
        .b_f(b_f),
        .b_i(b_i),
        .b_c(b_c),
        .b_o(b_o),
        .h_t(h_t),
        .c_t(c_t),
        .dx_t(dx_t),
        .dh_prev(dh_prev),
        .dc_prev(dc_prev)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize inputs
        rst = 1;
        enable = 0;
        forward_mode = 1;
        x_t = 0;
        h_prev = 0;
        c_prev = 0;
        
        // Initialize weights and biases
        // Add your initialization here
        
        #10 rst = 0;
        enable = 1;
        
        // Add your test vectors here
        
        // Test forward propagation
        #20 x_t = 32'h0000_4000; // 0.25 in fixed point
        
        // Test backward propagation
        #40 forward_mode = 0;
        
        // Add more test cases as needed
        
        #100 $finish;
    end
    
    // Monitor results
    initial begin
        $monitor("Time=%0t rst=%b enable=%b forward_mode=%b x_t=%h h_t=%h c_t=%h",
                 $time, rst, enable, forward_mode, x_t, h_t, c_t);
    end
    
endmodule