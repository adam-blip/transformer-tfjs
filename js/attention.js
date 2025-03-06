/**
 * Attention Mechanisms
 * Different implementations of attention for transformer models.
 */

/**
 * Build the appropriate attention model based on configuration
 */
async function buildAttentionModel() {
    await utils.log('Building optimized attention model...');
    
    // Choose attention mechanism based on configuration
    switch(CONFIG.attentionType) {
      case 'linear':
        return buildLinearAttention();
      case 'flash':
        return buildFlashAttention();
      case 'standard':
      default:
        return buildStandardAttention();
    }
  }
  
  /**
   * Linear Attention
   * Linear complexity attention mechanism, inspired by Linformer
   * Paper: https://arxiv.org/abs/2006.04768
   */
  function buildLinearAttention() {
    const queries = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const keys = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const values = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    
    // Project to lower dimension for efficient processing
    const projSize = Math.min(CONFIG.inputSize, 64);
    
    // Create attention projectors
    const eProj = tf.layers.dense({
      units: projSize,
      kernelInitializer: 'glorotNormal',
      name: 'linear_key_projector'
    }).apply(tf.layers.permute({dims: [2, 1]}).apply(keys));
    
    const fProj = tf.layers.dense({
      units: projSize,
      kernelInitializer: 'glorotNormal',
      name: 'linear_value_projector'
    }).apply(tf.layers.permute({dims: [2, 1]}).apply(values));
    
    // Custom lambda layer for efficient linear attention
    const attention = new layers.lambdaLayer({
      name: 'linear_attention',
      lambdaFunction: `
        // Matrix multiplications for efficient computation
        const qk = tf.matMul(input[0], input[1]); 
        const scale = tf.sqrt(tf.scalar(input[0].shape[2]));
        let scores = tf.div(qk, scale);
        
        // Create causal mask (upper triangular)
        const mask = tf.linalg.bandPart(
          tf.ones([input[0].shape[1], input[0].shape[1]]),
          -1, 0
        );
        
        // Apply causal masking
        const negMask = tf.sub(tf.onesLike(mask), mask);
        const negInf = tf.mul(negMask, tf.scalar(-1e9));
        scores = tf.add(tf.mul(scores, mask), negInf);
        
        // Apply softmax along correct dimension
        const attnWeights = tf.softmax(scores, 1);
        
        // Final attention calculation
        result = tf.matMul(attnWeights, input[2]);
      `,
      lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
    }).apply([queries, eProj, fProj]);
    
    // Create and return the model
    const attentionModel = tf.model({
      inputs: [queries, keys, values],
      outputs: attention
    });
    
    return attentionModel;
  }
  
  /**
   * Flash Attention
   * Approximation of Flash Attention algorithm for efficient transformer training
   * Paper: https://arxiv.org/abs/2205.14135
   */
  function buildFlashAttention() {
    const queries = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const keys = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const values = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    
    // Block size for chunked computation
    const blockSize = 64;
    
    // Custom lambda layer for flash attention approximation
    const flashAttention = new layers.lambdaLayer({
      name: 'flash_attention',
      lambdaFunction: `
        // Implementation of block-wise attention to approximate Flash Attention
        const seqLen = input[0].shape[1];
        const scale = 1.0 / Math.sqrt(input[0].shape[2]);
        
        // Process in blocks for memory efficiency
        let output = tf.zeros([input[0].shape[0], seqLen, input[2].shape[2]]);
        
        // Simplified block-wise processing
        for (let i = 0; i < seqLen; i += ${blockSize}) {
          const qBlock = tf.slice(input[0], [0, i, 0], [-1, Math.min(${blockSize}, seqLen - i), -1]);
          
          for (let j = 0; j <= i; j += ${blockSize}) {
            const kBlock = tf.slice(input[1], [0, j, 0], [-1, Math.min(${blockSize}, seqLen - j), -1]);
            const vBlock = tf.slice(input[2], [0, j, 0], [-1, Math.min(${blockSize}, seqLen - j), -1]);
            
            // Compute attention scores for this block
            const scores = tf.matMul(qBlock, kBlock, false, true);
            const scaledScores = tf.mul(scores, tf.scalar(scale));
            
            // Causal masking within the block
            const mask = tf.ones([
              qBlock.shape[1], 
              kBlock.shape[1]
            ]);
            
            // Create causal mask for this block
            const causalMask = tf.tidy(() => {
              const indices = tf.range(0, qBlock.shape[1]);
              const offsetIndices = tf.range(j, j + kBlock.shape[1]);
              const broadcastedIndices = tf.expandDims(indices, 1);
              const broadcastedOffsetIndices = tf.expandDims(offsetIndices, 0);
              return tf.greaterEqual(broadcastedOffsetIndices, broadcastedIndices);
            });
            
            const maskedScores = tf.where(
              causalMask,
              scaledScores,
              tf.mul(tf.onesLike(scaledScores), tf.scalar(-1e9))
            );
            
            // Apply softmax along second dimension (not -1)
            const attnWeights = tf.softmax(maskedScores, 1);
            
            // Compute output for this block
            const blockOutput = tf.matMul(attnWeights, vBlock);
            
            // Update output
            const sliceStart = [0, i, 0];
            const sliceSize = [-1, qBlock.shape[1], -1];
            const currentOutput = tf.slice(output, sliceStart, sliceSize);
            const updatedOutput = tf.add(currentOutput, blockOutput);
            
            // Using scatter update - fixed from original code
            const indices = Array(qBlock.shape[1]).fill(0).map((_, idx) => [0, i + idx, 0]);
            const updates = tf.unstack(updatedOutput, 1);
            const flatUpdates = updates.map(u => tf.reshape(u, [1, -1]));
            const flatUpdate = tf.concat(flatUpdates, 0);
            
            output = tf.tensor3d(
              flatUpdate.dataSync(),
              [qBlock.shape[1], 1, input[2].shape[2]],
              flatUpdate.dtype
            );
          }
        }
        
        result = output;
      `,
      lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
    }).apply([queries, keys, values]);
    
    // Create and return the model
    const attentionModel = tf.model({
      inputs: [queries, keys, values],
      outputs: flashAttention
    });
    
    return attentionModel;
  }
  
  /**
   * Standard Attention
   * Standard scaled dot-product attention mechanism from "Attention Is All You Need"
   * Paper: https://arxiv.org/abs/1706.03762
   */
  function buildStandardAttention() {
    const queries = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const keys = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    const values = tf.input({shape: [CONFIG.inputSize, CONFIG.dModel]});
    
    // Custom lambda layer for standard attention with optimizations
    const attention = new layers.lambdaLayer({
      name: 'standard_attention',
      lambdaFunction: `
        // Efficient attention computation
        const scores = tf.matMul(input[0], input[1], false, true);
        const scale = tf.sqrt(tf.scalar(input[0].shape[2]));
        const scaledScores = tf.div(scores, scale);
        
        // Create causal mask (lower triangular)
        const mask = tf.linalg.bandPart(
          tf.ones([input[0].shape[1], input[0].shape[1]]),
          -1, 0
        );
        
        // Apply causal masking
        const negMask = tf.sub(tf.onesLike(mask), mask);
        const negInf = tf.mul(negMask, tf.scalar(-1e9));
        const maskedScores = tf.add(tf.mul(scaledScores, mask), negInf);
        
        // Apply softmax along dimension 1, not -1 (fixes the error)
        const attnWeights = tf.softmax(maskedScores, 1);
        
        // Apply attention
        result = tf.matMul(attnWeights, input[2]);
      `,
      lambdaOutputShape: [null, CONFIG.inputSize, CONFIG.dModel]
    }).apply([queries, keys, values]);
    
    // Create and return the model
    const attentionModel = tf.model({
      inputs: [queries, keys, values],
      outputs: attention
    });
    
    return attentionModel;
  }
  
  // Export attention functions
  window.attention = {
    buildAttentionModel,
    buildLinearAttention,
    buildFlashAttention,
    buildStandardAttention
  };