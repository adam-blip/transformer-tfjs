/**
 * Text Generation
 * Text generation utilities using the trained transformer model.
 */

/**
 * Generate text using top-k and nucleus sampling
 * @param {string} prompt - The seed text to start generation from
 * @param {number} maxLength - Maximum number of tokens to generate
 */
async function generateText(prompt, maxLength = CONFIG.predictionLength) {
    await utils.log('Generating text...');
    
    // Get the tokenizer
    const tokenizer = window.gpt3TokenizerImport?.gpt3Encoder || window.gpt3Encoder;
    if (!tokenizer) {
      throw new Error('GPT-3 tokenizer not loaded properly');
    }
    
    // Start with the prompt as context
    let inputText = prompt.toLowerCase().trim();
    let startTokens = tokenizer.encode(inputText);
    startTokens = Array.from(startTokens).map(t => window.tokenizer.dictionary.indexOf(t));
    
    // Make sure we have the right context length
    if (startTokens.length > CONFIG.inputSize) {
      startTokens = startTokens.slice(-CONFIG.inputSize);
    } else if (startTokens.length < CONFIG.inputSize) {
      // Pad with zeros if needed
      startTokens = Array(CONFIG.inputSize - startTokens.length).fill(0).concat(startTokens);
    }
    
    // Set up the generation context
    let currentSequence = startTokens.slice();
    let generatedText = '';
    
    // Generate tokens one by one
    for (let i = 0; i < maxLength; i++) {
      // Get model prediction
      const inputTensor = tf.tensor2d([currentSequence], [1, CONFIG.inputSize]);
      const prediction = modelManager.model.predict(inputTensor);
      
      // Get probabilities for the next token (last position in sequence)
      const probabilities = prediction.slice([0, CONFIG.inputSize - 1, 0], [1, 1, window.tokenizer.vocabularySize]);
      const probs = await probabilities.reshape([window.tokenizer.vocabularySize]).array();
      
      // Apply temperature
      const logits = probs.map(p => Math.log(Math.max(p, 1e-10)) / CONFIG.temperature);
      
      // Apply softmax
      const expLogits = logits.map(l => Math.exp(l));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const temperedProbs = expLogits.map(e => e / sumExp);
      
      // Apply top-k sampling
      const topK = CONFIG.topK;
      const indices = temperedProbs.map((p, i) => ({ prob: p, index: i }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, topK);
      
      // Apply nucleus (top-p) sampling
      const topP = CONFIG.topP;
      let cumSum = 0;
      const filteredIndices = [];
      for (const item of indices) {
        filteredIndices.push(item);
        cumSum += item.prob;
        if (cumSum >= topP) break;
      }
      
      // Renormalize probabilities after filtering
      const totalFilteredProb = filteredIndices.reduce((sum, item) => sum + item.prob, 0);
      const normalizedProbs = filteredIndices.map(item => item.prob / totalFilteredProb);
      
      // Sample based on normalized probabilities
      const randomValue = Math.random();
      let cumulativeProb = 0;
      let sampledIndex = filteredIndices[0].index; // Default to highest probability
      
      for (let j = 0; j < normalizedProbs.length; j++) {
        cumulativeProb += normalizedProbs[j];
        if (randomValue <= cumulativeProb) {
          sampledIndex = filteredIndices[j].index;
          break;
        }
      }
      
      // Get the corresponding token
      const tokenValue = window.tokenizer.dictionary[sampledIndex];
      
      // Add to generated text
      try {
        const decodedToken = tokenizer.decode([tokenValue]);
        generatedText += decodedToken;
      } catch (e) {
        // Fallback if decode fails
        generatedText += ' ';
      }
      
      // Update context for next prediction
      currentSequence = currentSequence.slice(1).concat(sampledIndex);
      
      // Clean up tensors
      inputTensor.dispose();
      prediction.dispose();
    }
    
    return generatedText;
  }
  
  // Export text generation functions
  window.textGenerator = {
    generateText
  };