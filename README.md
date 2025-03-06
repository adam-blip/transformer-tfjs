# Optimized Transformer Language Model

An interactive transformer-based language model implemented in TensorFlow.js for the browser. This project allows you to train and experiment with different transformer architecture variants directly in your browser.

## Features

- Train a transformer language model on provided text data
- Configure key hyperparameters through the UI
- Experiment with different attention mechanisms (Linear, Flash, Standard)
- Generate text from your trained model
- Download trained models for later use

## Project Structure

```
/
├── index.html                  // Main HTML page
├── styles.css                  // CSS styling
├── js/
│   ├── config.js               // Configuration constants
│   ├── app.js                  // Main application logic and initialization
│   ├── tokenizer.js            // Tokenization utilities (imports gpt3encoder.js)
│   ├── attention.js            // Attention mechanisms (linear, flash, standard)
│   ├── layers.js               // Custom layer definitions
│   ├── model.js                // Model construction and loading
│   ├── training.js             // Training and evaluation logic
│   ├── generation.js           // Text generation utilities
│   └── utils.js                // Helper functions and memory management
├── data/
│   └── crime-and-punishment_small.txt.js  // Text dataset
└── gpt3encoder.js              // GPT-3 tokenizer implementation
```

## Getting Started

1. Clone the repository
2. Open `index.html` in a modern browser (Chrome or Firefox recommended)
3. Configure model parameters in the UI
4. Click "Start Training" to begin training
5. Use the "Generate" button to sample text from the model

## Attention Mechanisms

### Standard Attention
The original scaled dot-product attention with O(n²) complexity as described in "Attention Is All You Need" (Vaswani et al., 2017).

### Linear Attention
An approximation of attention using low-rank matrices to achieve O(n) complexity, inspired by "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020).

### Flash Attention
A block-wise attention computation that optimizes memory access patterns, based on "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022).

## Normalization and Activation Functions

The model supports different normalization techniques:
- **LayerNorm**: Standard layer normalization
- **RMSNorm**: Root Mean Square normalization (more efficient)

And different activation functions:
- **GELU**: Gaussian Error Linear Unit (used in many modern transformers)
- **SwiGLU**: Swish-Gated Linear Unit (better performance, used in PaLM)
- **ReLU**: Rectified Linear Unit (simpler, faster)

## Browser Compatibility

This application uses TensorFlow.js and requires a modern browser with WebGL support for best performance. WebGL2-capable browsers will achieve the best training speed.

## Memory Management

The application includes memory optimization features such as:
- Float16 precision (when supported by the browser)
- Gradient accumulation 
- Automatic memory cleanup
- Block-wise attention computation

## License

MIT License

## Acknowledgments

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- TensorFlow.js team