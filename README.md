# SingleLayerNetwork
This project implements a simple single-layer neural network to identify the language of a given text. It supports four languages: English, German, Polish, and Spanish.

 * **Input Representation**  
   Each text is converted into a 26-element vector, where each element represents the count of a letter (A–Z). The vector is then normalized.

 * **Network Structure**  
   - One perceptron per language  
   - Only one perceptron should activate for a given input  
   - Activation function can be either step or linear

- Supports flexible datasets (can adapt to a different number of languages)
- Can use linear or step activation
- Terminal input for real-time classification
- Optional display of misclassified test data
