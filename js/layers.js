/**
 * Custom Layer Definitions
 * Extensions to TensorFlow.js layers for transformer-specific functionality.
 */

/**
 * Lambda Layer
 * A custom layer for implementing complex operations
 */
class lambdaLayer extends tf.layers.Layer {
    constructor(config) {
      super(config);
      if (config.name === undefined) {
        config.name = ((+new Date) * Math.random()).toString(36);
      }
      this.name = config.name;
      this.lambdaFunction = config.lambdaFunction;
      this.lambdaOutputShape = config.lambdaOutputShape;
    }
    
    call(input) {
      return tf.tidy(() => {
        let result = null;
        eval(this.lambdaFunction);
        return result;
      });
    }
    
    computeOutputShape(inputShape) {
      if (this.lambdaOutputShape === undefined) {
        return inputShape[0];
      } else {
        return this.lambdaOutputShape;
      }
    }
    
    getConfig() {
      const config = super.getConfig();
      Object.assign(config, {
        lambdaFunction: this.lambdaFunction,
        lambdaOutputShape: this.lambdaOutputShape
      });
      return config;
    }
    
    static get className() {
      return 'lambdaLayer';
    }
  }
  
  // Register the custom layer with TensorFlow.js
  tf.serialization.registerClass(lambdaLayer);
  
  // Export layers
  window.layers = {
    lambdaLayer
  };