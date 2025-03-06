/**
 * Transformer Configuration
 * A centralized place for all model hyperparameters and configuration settings.
 */
const CONFIG = {
    // Model architecture
    dModel: 128,               // Embedding dimension
    nHeads: 8,                 // Number of attention heads
    ffnDim: 512,               // Feed-forward dimension
    dropout: 0.1,              // Dropout rate
    blocks: 6,                 // Number of transformer blocks
    
    // Training parameters
    batchSize: 4,              // Batch size
    inputSize: 256,            // Input sequence length
    epochs: 5000,              // Total epochs
    stepsPerEpoch: 2,          // Steps per epoch
    learningRate: 0.0003,      // Learning rate
    
    // Prediction parameters
    predictionEpochs: 50,      // How often to generate samples
    predictionLength: 20,      // Length of predictions
    temperature: 0.8,          // Sampling temperature
    topK: 40,                  // Top-K sampling
    topP: 0.9,                 // Top-P (nucleus) sampling
    
    // Memory optimization
    useFloat16: true,          // Use 16-bit floating-point
    gradientAccumulation: 2,   // Gradient accumulation steps
    
    // Model variants
    attentionType: 'linear',   // 'linear', 'flash', 'standard'
    normType: 'layerNorm',     // 'layerNorm', 'rmsnorm'
    activationType: 'gelu',    // 'gelu', 'swish', 'relu'
    
    // Memory management
    memoryCleanupFrequency: 5, // Clean memory every N steps
  };
  
  // Export configuration
  window.CONFIG = CONFIG;