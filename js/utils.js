/**
 * Utility Functions
 * Common helper functions used throughout the application.
 */

// Simple sleep function for async operations
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  // Log message to the UI
  async function log(message) {
    const logElement = document.getElementById('log');
    logElement.innerHTML += message + '\n';
    logElement.scrollTop = logElement.scrollHeight;
    await sleep(1);
  }
  
  // Get random integer up to max
  function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
  }
  
  // Update progress bar in the UI
  function updateProgressBar(percent) {
    document.getElementById('progress-bar').style.width = percent + '%';
  }
  
  // Update memory usage info in the UI
  function updateMemoryInfo() {
    const memInfo = tf.memory();
    document.getElementById('memory-usage').textContent = 
      `${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`;
    document.getElementById('tensor-count').textContent = 
      memInfo.numTensors;
  }
  
  // Clean up unused memory and tensors
  async function cleanupMemory() {
    await log('Cleaning up memory...');
    if (tf.memory().numTensors > 1000) {
      await log(`WARNING: High tensor count: ${tf.memory().numTensors}`);
    }
    
    // Force garbage collection
    tf.tidy(() => {});
    
    // Update memory display
    updateMemoryInfo();
  }
  
  // Export utility functions
  window.utils = {
    sleep,
    log,
    getRandomInt,
    updateProgressBar,
    updateMemoryInfo,
    cleanupMemory
  };