/**
 * Main Application
 * Entry point and initialization for the transformer application.
 */

/**
 * Initialize the application
 */
async function initialize() {
    await utils.log('Starting initialization...');
    
    // Process and tokenize dataset
    await tokenizer.initializeTokenizer(DATASET);
    
    // Initialize optimizer with learning rate schedule
    await modelManager.initializeOptimizer();
    
    // Build attention model
    modelManager.attentionModel = await attention.buildAttentionModel();
    
    // Build the complete transformer model
    modelManager.model = await modelManager.buildModel();
    
    await utils.log('Initialization complete. Ready to train or generate text.');
    
    return modelManager.model;
  }
  
  /**
   * Initialize app UI and event handlers
   */
  async function initApp() {
    try {
      // Set up event listeners
      document.getElementById('train-button').addEventListener('click', trainer.train);
      document.getElementById('stop-button').addEventListener('click', trainer.stopTraining);
      document.getElementById('generate-button').addEventListener('click', async () => {
        const prompt = document.getElementById('prompt-input').value || 'The';
        const length = parseInt(document.getElementById('length-input').value) || CONFIG.predictionLength;
        const result = await textGenerator.generateText(prompt, length);
        await utils.log(`Generated: ${prompt}${result}`);
      });
      document.getElementById('download-button').addEventListener('click', async () => {
        try {
          await utils.log('Preparing model for download...');
          await modelManager.model.save('downloads://' + modelManager.modelSaveName);
          await utils.log('Model download initiated.');
        } catch (err) {
          await utils.log(`Error downloading model: ${err.message}`);
        }
      });
      
      // Setup configuration inputs
      const configInputs = document.querySelectorAll('.config-input');
      configInputs.forEach(input => {
        // Set initial values from CONFIG
        const configKey = input.dataset.config;
        if (configKey && CONFIG[configKey] !== undefined) {
          input.value = CONFIG[configKey];
        }
        
        // Add change listener
        input.addEventListener('change', () => {
          const key = input.dataset.config;
          if (key) {
            const value = input.type === 'number' ? Number(input.value) : input.value;
            CONFIG[key] = value;
            utils.log(`Updated ${key}: ${value}`);
          }
        });
      });
      
      // Initialize the model
      await initialize();
      
    } catch (err) {
      await utils.log(`Initialization error: ${err.message}`);
      console.error(err);
    }
  }
  
  /**
   * Document ready function
   */
  window.onload = async function() {
    try {
      await utils.log('Starting application...');
      
      // Check for TensorFlow.js
      await utils.log(`TensorFlow.js version: ${tf.version.tfjs}`);
      
      // Enable production mode for better performance
      tf.enableProdMode();
      
      // Try to use WebGL backend
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        await utils.log(`Using backend: ${tf.getBackend()}`);
        
        // Check for WebGL2
        const gl = document.createElement('canvas').getContext('webgl2');
        if (gl) {
          await utils.log('WebGL2 is supported (better performance)');
        } else {
          await utils.log('Using WebGL1 (limited performance)');
        }
      } catch (err) {
        await utils.log(`Backend initialization warning: ${err.message}`);
      }
      
      // Initialize the application
      await initApp();
      
    } catch (err) {
      await utils.log(`Startup error: ${err.message}`);
      console.error(err);
    }
  };