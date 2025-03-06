/**
 * Tokenizer Utilities
 * Handles text tokenization and processing using the GPT-3 encoder.
 */

// Global token variables
let TOKENS = [];
let DICTIONARY = [];
let VOCABULARY_SIZE = 0;

/**
 * Initialize tokenizer with given dataset
 * @param {string} dataset - The raw text dataset to tokenize
 */
async function initializeTokenizer(dataset) {
  await utils.log('Initializing tokenizer...');
  
  // Process dataset
  await utils.log('Extracting and preprocessing text...');
  let processedData = dataset.substring(
    dataset.indexOf('PART I'), 
    dataset.indexOf('End of Project Gutenberg')
  );
  
  // Clean and normalize text
  processedData = processedData.toLowerCase()
    .replace(/[^a-z0-9.,!? \n]/g, '')
    .replace(/\.{3}/g, '.')
    .replace(/--/g, ', ')
    .replace(/;/g, ',')
    .replace(/:/g, '.')
    .replace(/\./g, ' .')
    .replace(/,/g, ' ,')
    .replace(/\?/g, ' ?')
    .replace(/!/g, ' !')
    .replace(/\s+/g, ' ');
  
  // Tokenize using the external GPT-3 encoder
  const tokenizer = window.gpt3TokenizerImport?.gpt3Encoder || window.gpt3Encoder;
  if (!tokenizer) {
    throw new Error('GPT-3 tokenizer not loaded properly. Please check the library import.');
  }
  
  // Use the encode function from the library
  TOKENS = tokenizer.encode(processedData);
  TOKENS = Array.from(TOKENS); // Ensure it's a proper array
  DICTIONARY = [...new Set(TOKENS)]; // Unique tokens
  VOCABULARY_SIZE = DICTIONARY.length;
  
  await utils.log(`Processed ${TOKENS.length} tokens with vocabulary size ${VOCABULARY_SIZE}`);
  
  return {
    tokens: TOKENS,
    dictionary: DICTIONARY,
    vocabularySize: VOCABULARY_SIZE
  };
}

// Export tokenizer functions and variables
window.tokenizer = {
  initializeTokenizer,
  get tokens() { return TOKENS; },
  get dictionary() { return DICTIONARY; },
  get vocabularySize() { return VOCABULARY_SIZE; }
};