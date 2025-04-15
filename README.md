# SingleLayerNetwork
This project implements a simple single-layer neural network to identify the language of a given text. It supports four languages: English, German, Polish, and Spanish.

The network is trained on a dataset of text samples using the letter frequencies of the 26-letter Latin alphabet (ignoring case, punctuation, diacritics, etc.). It learns to classify each text by activating the perceptron corresponding to its correct language.

---

## How It Works

1. **Input Representation**  
   Each text is converted into a 26-element vector, where each element represents the count of a letter (A–Z). The vector is then normalized.

2. **Network Structure**  
   - One perceptron per language  
   - Only one perceptron should activate for a given input  
   - Activation function can be either step or linear

3. **Training and Testing**  
   - Trained using `lang.train.csv`  
   - Tested using `lang.test.csv`  
   - Outputs test accuracy

4. **User Interface**  
   - User can input custom text via the terminal  
   - (Optional) Displays test samples that were misclassified

---

## Features

- Supports flexible datasets (can adapt to a different number of languages)
- Can use linear or step activation
- Terminal input for real-time classification
- Optional display of misclassified test data

---

## Running the Program

1. Make sure `lang.train.csv` and `lang.test.csv` are in the project folder.
2. Run the program using your Python interpreter.
3. Follow the prompts to classify text or view accuracy.

---

## Notes

- When reading the CSV files, use `split(",", 1)` to correctly handle texts containing commas.
- Linear activation is recommended to avoid multiple perceptrons activating for the same input.
