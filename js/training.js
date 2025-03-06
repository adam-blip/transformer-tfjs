/**
 * Training Functions
 * Handles training loops and evaluation.
 */

// Training state variables
let isTraining = false;
let stopRequested = false;
let currentEpoch = 0;

/**
 * Generate a training batch
 */
function generateBatch() {
  return tf.tidy(() => {
    const batchInputs = [];
    const batchTargets = [];
    
    for (let b = 0; b < CONFIG.batchSize; b++) {
      // Get a random slice of tokens
      const startIdx = utils.getRandomInt(tokenizer.tokens.length - (CONFIG.inputSize + 1));
      const inputTokens = tokenizer.tokens.slice(startIdx, startIdx + CONFIG.inputSize);
      const targetTokens = tokenizer.tokens.slice(startIdx + 1, startIdx + CONFIG.inputSize + 1);
      
      // Map tokens to dictionary indices
      const inputIndices = inputTokens.map(t => tokenizer.dictionary.indexOf(t));
      
      // Create one-hot encoded targets
      const targets = [];
      for (let i = 0; i < CONFIG.inputSize; i++) {
        const targetIdx = tokenizer.dictionary.indexOf(targetTokens[i]);
        const oneHot = new Array(tokenizer.vocabularySize).fill(0);
        oneHot[targetIdx] = 1;
        targets.push(oneHot);
      }
      
      batchInputs.push(inputIndices);
      batchTargets.push(targets);
    }
    
    // Create tensors
    const inputTensor = tf.tensor2d(batchInputs, [CONFIG.batchSize, CONFIG.inputSize]);
    const targetTensor = tf.tensor3d(batchTargets, [CONFIG.batchSize, CONFIG.inputSize, tokenizer.vocabularySize]);
    
    return {
      inputs: inputTensor,
      targets: targetTensor
    };
  });
}

/**
 * Training step with gradient accumulation
 */
async function trainStep(epoch) {
  const startTime = performance.now();
  let loss = 0;
  let accuracy = 0;
  let ppl = 0;
  
  // Gradient accumulation loop
  for (let i = 0; i < CONFIG.gradientAccumulation; i++) {
    const batch = generateBatch();
    
    // Forward and backward pass using tf.variableGrads
    const {value, grads} = tf.variableGrads(() => {
      const predictions = modelManager.model.apply(batch.inputs, {training: true});
      const batchLoss = tf.losses.softmaxCrossEntropy(batch.targets, predictions);
      const batchPpl = modelManager.perplexity(batch.targets, predictions);
      const batchAcc = tf.metrics.categoricalAccuracy(batch.targets, predictions);
      
      // Record metrics for logging
      loss += batchLoss.dataSync()[0] / CONFIG.gradientAccumulation;
      ppl += batchPpl.dataSync()[0] / CONFIG.gradientAccumulation;
      accuracy += batchAcc.mean().dataSync()[0] / CONFIG.gradientAccumulation;
      
      return batchLoss;
    });
    
    // Scale gradients by accumulation steps
    const scaledGrads = {};
    Object.keys(grads).forEach(key => {
      scaledGrads[key] = tf.div(grads[key], tf.scalar(CONFIG.gradientAccumulation));
    });
    
    // Apply gradients
    optimizer.applyGradients(scaledGrads);
    
    // Clean up
    Object.values(grads).forEach(grad => grad.dispose());
    batch.inputs.dispose();
    batch.targets.dispose();
  }
  
  // Calculate training time
  const endTime = performance.now();
  const timeMs = endTime - startTime;
  
  // Log progress
  if (epoch % 10 === 0) {
    await utils.log(
      `Epoch: ${epoch} | Loss: ${loss.toFixed(4)} | ` +
      `PPL: ${ppl.toFixed(2)} | Acc: ${(accuracy * 100).toFixed(1)}% | ` +
      `Time: ${timeMs.toFixed(0)}ms | LR: ${optimizer.learningRate.toFixed(6)}`
    );
  }
  
  // Update progress bar
  utils.updateProgressBar((epoch / CONFIG.epochs) * 100);
  
  // Memory cleanup
  if (epoch % CONFIG.memoryCleanupFrequency === 0) {
    await utils.cleanupMemory();
  }
  
  return { loss, accuracy, ppl };
}

/**
 * Main training loop
 */
async function train() {
  if (isTraining) return;
  isTraining = true;
  stopRequested = false;
  
  await utils.log('Starting training...');
  
  try {
    for (let epoch = currentEpoch; epoch < CONFIG.epochs && !stopRequested; epoch++) {
      currentEpoch = epoch;
      
      // Training step
      await trainStep(epoch);
      
      // Run multiple steps per epoch if configured
      for (let step = 1; step < CONFIG.stepsPerEpoch; step++) {
        await trainStep(epoch);
      }
      
      // Generate sample text periodically
      if (epoch % CONFIG.predictionEpochs === 0) {
        try {
          await utils.log('Saving model...');
          await modelManager.model.save('indexeddb://' + modelManager.modelSaveName);
          
          // Sample from the model
          const seedText = "The man";
          const generatedText = await textGenerator.generateText(seedText);
          await utils.log(`Sample [${epoch}]: ${seedText}${generatedText}`);
        } catch (err) {
          await utils.log(`Error in sampling: ${err.message}`);
        }
      }
      
      await utils.sleep(0); // Allow UI to refresh
    }
    
    await utils.log('Training complete!');
    
    // Final save
    await modelManager.model.save('indexeddb://' + modelManager.modelSaveName);
    
  } catch (err) {
    await utils.log(`Error during training: ${err.message}`);
    console.error(err);
  } finally {
    isTraining = false;
  }
}

/**
 * Stop training
 */
async function stopTraining() {
  stopRequested = true;
  await utils.log('Stopping training after current epoch completes...');
}

// Export training functions
window.trainer = {
  train,
  stopTraining,
  get isTraining() { return isTraining; },
  get currentEpoch() { return currentEpoch; }
};