from src.data_fetcher import data_fetch
from src.preprocess import X, y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def run_pipeline():
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Results
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

if __name__ == "__main__":
    data_fetch('AAPL')  # Ensure latest data
    run_pipeline()
