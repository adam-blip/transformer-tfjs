/**
 * Model Construction
 * Handles building and loading the transformer model.
 */

// Global model variables
let model;
let attentionModel;
let optimizer;
let modelSaveName = 'optimized_transformer_' + new Date().toISOString().slice(0, 10);

// Export model variables and functions
window.modelManager = {
  get model() { return model; },
  set model(m) { model = m; },
  get attentionModel() { return attentionModel; },
  set attentionModel(am) { attentionModel = am; },
  initializeOptimizer,
  buildModel,
  perplexity,
  get modelSaveName() { return modelSaveName; }
};

/**
 * Perplexity metric for language model evaluation
 */
function perplexity(yTrue, yPred) {
  return tf.tidy(() => {
    const crossEntropy = tf.losses.softmaxCrossEntropy(yTrue, yPred);
    return tf.exp(crossEntropy);
  });
}

/**
 * Initialize model optimizer with learning rate schedule
 */
async function initializeOptimizer() {
  // Set up optimizer with warmup and weight decay
  const lrSchedule = {
    onEpochBegin: (epoch) => {
      const warmupEpochs = 100;
      const decayRate = 0.98;
      
      if (epoch < warmupEpochs) {
        const lr = CONFIG.learningRate * (epoch + 1) / warmupEpochs;
        optimizer.learningRate = lr;
      } else {
        const lr = CONFIG.learningRate * Math.pow(decayRate, epoch - warmupEpochs);
        optimizer.learningRate = Math.max(lr, 0.00001);
      }
      
      if (epoch % 50 === 0) {
        utils.log(`Epoch ${epoch}: Learning rate = ${optimizer.learningRate}`);
      }
    }
  };
  
  // Initialize optimizer with AdamW
  optimizer = tf.train.adam(CONFIG.learningRate, 0.9, 0.98, 1e-6);
  
  return lrSchedule;
}

/**
 * Build the complete transformer model
 */
async function buildModel() {
  await utils.log('Building transformer model...');
  
  try {
    model = await tf.loadLayersModel('indexeddb://' + modelSaveName);
    await utils.log('Successfully loaded saved model');
    return model;
  } catch(err) {
    await utils.log('Building new model: ' + err);
    
    // Input
    const inputs = tf.input({shape: [null], name: 'input'});
    
    // Token embedding
    let x = tf.layers.embedding({
      inputDim: tokenizer.vocabularySize + 1,
      outputDim: CONFIG.dModel,
      name: 'token_embedding'
    }).apply(inputs);
    
    // Positional embedding with rotary implementation
    const pos = new layers.lambdaLayer({
      name: 'rotary_embedding',
      lambdaFunction: `
        // Get sequence length
        const seqLen = input[0].shape[1];
        
        // Generate position indices
        const positions = tf.range(0, seqLen);
        
        // Calculate rotational frequencies
        const freqs = tf.exp(
          tf.mul(
            tf.range(0, input[0].shape[2] / 2, 1),
            tf.log(tf.scalar(10000)) / (input[0].shape[2] / 2)
          )
        );
        
        // Create embeddings
        const sinusoidalPos = tf.expandDims(positions, -1) * tf.expandDims(freqs, 0);
        const cosPos = tf.cos(sinusoidalPos);
        const sinPos = tf.sin(sinusoidalPos);
        
        // Interleave the sin and cos - fix the concat to use axis 1 instead of -1
        const posEmbedding = tf.concat([cosPos, sinPos], 1); 
        
        result = posEmbedding;
      `,
      lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
    }).apply(inputs);
    
    // Apply positional encoding
    x = tf.layers.add().apply([x, pos]);
    
    // Apply dropout to embeddings
    x = tf.layers.dropout({rate: CONFIG.dropout}).apply(x);
    
    // Transformer blocks
    for (let i = 0; i < CONFIG.blocks; i++) {
      const skipConnection = x;
      
      // Choose normalization type
      let norm;
      if (CONFIG.normType === 'rmsnorm') {
        // RMSNorm implementation
        norm = new layers.lambdaLayer({
          name: `rms_norm_${i}`,
          lambdaFunction: `
            // RMSNorm implementation
            const variance = tf.mean(tf.square(input[0]), 2, true);
            const normalizedX = tf.div(input[0], tf.sqrt(tf.add(variance, 1e-6)));
            
            // Scale parameter (gamma) - trainable
            const gamma = tf.variable(
              tf.ones([1, 1, input[0].shape[input[0].shape.length - 1]]),
              true,
              \`gamma_\${i}\`
            );
            
            result = tf.mul(normalizedX, gamma);
          `,
          lambdaOutputShape: x.shape
        }).apply(x);
      } else {
        // Standard LayerNorm
        norm = tf.layers.layerNormalization({
          name: `layer_norm_${i}`,
          epsilon: 1e-6
        }).apply(x);
      }
      
      // Multi-head attention
      const qkv = tf.layers.dense({
        units: CONFIG.dModel * 3,
        name: `qkv_proj_${i}`
      }).apply(norm);
      
      // Split QKV into separate tensors
      const queries = new layers.lambdaLayer({
        name: `queries_${i}`,
        lambdaFunction: `
          const qkvs = input[0];
          const dModel = ${CONFIG.dModel};
          result = tf.slice(qkvs, [0, 0, 0], [-1, -1, dModel]);
        `,
        lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
      }).apply(qkv);
      
      const keys = new layers.lambdaLayer({
        name: `keys_${i}`,
        lambdaFunction: `
          const qkvs = input[0];
          const dModel = ${CONFIG.dModel};
          result = tf.slice(qkvs, [0, 0, dModel], [-1, -1, dModel]);
        `,
        lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
      }).apply(qkv);
      
      const values = new layers.lambdaLayer({
        name: `values_${i}`,
        lambdaFunction: `
          const qkvs = input[0];
          const dModel = ${CONFIG.dModel};
          result = tf.slice(qkvs, [0, 0, dModel * 2], [-1, -1, dModel]);
        `,
        lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
      }).apply(qkv);
      
      // Apply attention
      x = attentionModel.apply([queries, keys, values]);
      
      // Apply dropout
      x = tf.layers.dropout({rate: CONFIG.dropout}).apply(x);
      
      // Add residual connection
      x = tf.layers.add().apply([x, skipConnection]);
      
      // Second normalization
      const skipConnection2 = x;
      
      if (CONFIG.normType === 'rmsnorm') {
        norm = new layers.lambdaLayer({
          name: `rms_norm_${i}_2`,
          lambdaFunction: `
            // RMSNorm implementation
            const variance = tf.mean(tf.square(input[0]), 2, true);
            const normalizedX = tf.div(input[0], tf.sqrt(tf.add(variance, 1e-6)));
            
            // Scale parameter (gamma) - trainable
            const gamma = tf.variable(
              tf.ones([1, 1, input[0].shape[input[0].shape.length - 1]]),
              true,
              \`gamma_\${i}_2\`
            );
            
            result = tf.mul(normalizedX, gamma);
          `,
          lambdaOutputShape: x.shape
        }).apply(x);
      } else {
        norm = tf.layers.layerNormalization({
          name: `layer_norm_${i}_2`,
          epsilon: 1e-6
        }).apply(x);
      }
      
      // Feed-forward network with chosen activation
      let ffnOutput;
      if (CONFIG.activationType === 'swish') {
        // SwiGLU implementation
        const swiglu = new layers.lambdaLayer({
          name: `swiglu_${i}`,
          lambdaFunction: `
            // Implement SwiGLU activation function
            const up = tf.layers.dense({
              units: ${CONFIG.ffnDim},
              name: \`ffn_up_\${i}\`
            }).apply(input[0]);
            
            const gate = tf.layers.dense({
              units: ${CONFIG.ffnDim},
              name: \`ffn_gate_\${i}\`
            }).apply(input[0]);
            
            // Swish activation: x * sigmoid(x)
            const swish = tf.mul(gate, tf.sigmoid(gate));
            
            // Gated output
            result = tf.mul(up, swish);
          `,
          lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.ffnDim]
        }).apply(norm);
        
        ffnOutput = tf.layers.dense({
          units: CONFIG.dModel,
          name: `ffn_down_${i}`
        }).apply(swiglu);
      } else if (CONFIG.activationType === 'gelu') {
        // Standard GELU implementation
        const ffnUp = tf.layers.dense({
          units: CONFIG.ffnDim,
          name: `ffn_up_${i}`
        }).apply(norm);
        
        const gelu = new layers.lambdaLayer({
          name: `gelu_${i}`,
          lambdaFunction: `
            result = tf.mul(
              tf.scalar(0.5),
              tf.mul(
                input[0],
                tf.add(
                  tf.scalar(1),
                  tf.tanh(
                    tf.mul(
                      tf.scalar(0.797884560),
                      tf.add(
                        input[0],
                        tf.mul(
                          tf.scalar(0.044715),
                          tf.pow(input[0], tf.scalar(3))
                        )
                      )
                    )				
                  )
                )
              )
            )
          `,
          lambdaOutputShape: ffnUp.shape
        }).apply(ffnUp);
        
        ffnOutput = tf.layers.dense({
          units: CONFIG.dModel,
          name: `ffn_down_${i}`
        }).apply(gelu);
      } else {
        // ReLU activation
        ffnOutput = tf.layers.dense({
          units: CONFIG.ffnDim,
          activation: 'relu',
          name: `ffn_up_${i}`
        }).apply(norm);
        
        ffnOutput = tf.layers.dense({
          units: CONFIG.dModel,
          name: `ffn_down_${i}`
        }).apply(ffnOutput);
      }
      
      // Apply dropout
      ffnOutput = tf.layers.dropout({rate: CONFIG.dropout}).apply(ffnOutput);
      
      // Add residual connection
      x = tf.layers.add().apply([ffnOutput, skipConnection2]);
    }
    
    // Final layer normalization
    if (CONFIG.normType === 'rmsnorm') {
      x = new layers.lambdaLayer({
        name: 'final_rms_norm',
        lambdaFunction: `
          // RMSNorm implementation
          const variance = tf.mean(tf.square(input[0]), 2, true);
          const normalizedX = tf.div(input[0], tf.sqrt(tf.add(variance, 1e-6)));
          
          // Scale parameter (gamma) - trainable
          const gamma = tf.variable(
            tf.ones([1, 1, input[0].shape[input[0].shape.length - 1]]),
            true,
            'final_gamma'
          );
          
          result = tf.mul(normalizedX, gamma);
        `,
        lambdaOutputShape: x.shape
      }).apply(x);
    } else {
      x = tf.layers.layerNormalization({
        name: 'final_layer_norm',
        epsilon: 1e-6
      }).apply(x);
    }
    
    // Output projection
    const outputs = tf.layers.dense({
      units: tokenizer.vocabularySize,
      activation: 'softmax',
      name: 'output_projection'
    }).apply(x);
    
    // Create the model
    const model = tf.model({
      inputs: inputs,
      outputs: outputs
    });
    
    // Use mixed precision if configured
    if (CONFIG.useFloat16) {
      await tf.ready();
      if (tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED')) {
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
        await utils.log('Using 16-bit floating point for better performance');
      } else {
        await utils.log('16-bit floating point not supported in this browser');
      }
    }
    
    // Compile the model
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy', perplexity]
    });
    
    model.summary();
    return model;
  }
}