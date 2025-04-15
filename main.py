import math
import sys


class SingleLayerNetwork:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.languages = []
        self.weights = {}

    @staticmethod
    def load_file(filename):
        data = []
        try:
            with open(filename, 'r') as file:
                for line in file:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        label, text = parts
                        data.append((label.strip(), text.strip().lower()))
        except FileNotFoundError:
            print("file not exists")
        return data

    # FORMULA: |v| = sqrt(vi^2 * ....) -> gonna return v / |v|
    @staticmethod
    def normalize_vector(vector):
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude == 0:
            return vector[:]
        return [v / magnitude for v in vector]

    @staticmethod
    def dot_product(vector1, vector2):
        return sum(vector1[i] * vector2[i] for i in range(len(vector1)))

    def text_to_vector(self, text):
        vector = [0] * 26
        for char in text:
            if 'a' <= char <= 'z':
                index = ord(char) - ord('a')
                vector[index] += 1
        return self.normalize_vector(vector)

    def init_weights(self):
        self.weights = {}
        for lang in self.languages:
            self.weights[lang] = [0.0] * 26

    def train(self, train_data):
        self.languages = sorted(set(label for label, _ in train_data))
        self.init_weights()

        for epoch in range(self.epochs):
            for label, text in train_data:
                input_vector = self.text_to_vector(text)

                # compute outputs:
                outputs = {
                    lang: self.dot_product(self.weights[lang], input_vector)
                    for lang in self.languages
                }
                # update weights (this one is like how perceptron learns )
                # target == 1 is correct one this one tells if perceptron activates or not
                for lang in self.languages:
                    target = 1 if lang == label else 0

                    output = outputs[lang]
                    for i in range(26):
                        self.weights[lang][i] += (
                                self.learning_rate * (target - output) * input_vector[i]
                        )

    def predict(self, text):
        input_vector = self.text_to_vector(text)
        outputs = {lang: self.dot_product(self.weights[lang], input_vector)
                   for lang in self.languages}
        return max(outputs, key=outputs.get)

    # P = TP/ (TP + FP)
    # R = TP/ (TP + FN)
    # F1 = 2 * (P * R) / (P + R)
    def evaluate(self, test_data):
        TP = {lang: 0 for lang in self.languages}
        FP = {lang: 0 for lang in self.languages}
        FN = {lang: 0 for lang in self.languages}
        true_count = {lang: 0 for lang in self.languages}

        for true_label, text in test_data:
            predicted_label = self.predict(text)
            true_count[true_label] += 1

            if predicted_label == true_label:
                TP[true_label] += 1
            else:
                FP[true_label] += 1
                FN[true_label] += 1
        print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'TrueCount':>10}")
        print("-" * 55)

        total_correct = 0
        for lang in self.languages:
            total_correct += TP[lang]

            tp = TP[lang]
            fp = FP[lang]
            fn = FN[lang]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            print(f"{lang:<10} {precision:10.2f} {recall:10.2f} {f1:10.2f} {true_count[lang]:10}")

        print(f"Total Accuracy:{(total_correct / len(test_data)) * 100}%")


if __name__ == "__main__":
    args = sys.argv

    if len(sys.argv) < 2:
        print("main.py <train_file> <test_file>")
        sys.exit()

    s = SingleLayerNetwork(learning_rate=0.01, epochs=100)
    train_set = s.load_file(args[1])
    test_set = s.load_file(args[2])

    s.train(train_set)
    s.evaluate(test_set)
