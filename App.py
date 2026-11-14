"""
AI-Powered Iris Flower Predictor
Module 05 - AI Tools & Mini Project

An improved version of the Module 04 Iris classifier. It trains a machine
learning model and then lets the user type in flower measurements to get a
prediction with a confidence score.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]


def train_model():
    """Train a Random Forest on the Iris dataset and return it with its accuracy."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, iris.target_names, accuracy


def read_float(prompt):
    """Keep asking until the user enters a valid positive number."""
    while True:
        try:
            value = float(input(prompt))
            if value > 0:
                return value
            print("  Please enter a number greater than 0.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def predict_flower(model, species, measurements):
    """Return (species name, confidence %) for one flower."""
    probabilities = model.predict_proba([measurements])[0]
    best = probabilities.argmax()
    return species[best], probabilities[best] * 100


def main():
    print("=== AI-Powered Iris Flower Predictor ===")
    print("Training the model...")
    model, species, accuracy = train_model()
    print(f"Model ready! Test accuracy: {accuracy * 100:.1f}%\n")

    while True:
        print("Enter the flower measurements in cm:")
        measurements = [read_float(f"  {name}: ") for name in FEATURES]
        name, confidence = predict_flower(model, species, measurements)
        print(f"\nPrediction: Iris {name} ({confidence:.0f}% confident)\n")

        again = input("Predict another flower? (y/n): ").strip().lower()
        if again != "y":
            print("Thank you for using the predictor!")
            break
        print()


if __name__ == "__main__":
    main()
