import src.data_fetcher
from src.data_fetcher import data_fetch
from src.preprocess import X, y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# from src.data_fetcher import data_fetch_amzn
# from src.preprocess import X1, y1


def run_pipeline_aapl():
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Results
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))


#
# def run_pipeline_amzn():
#
#     X1_train, X1_test, y1_train, y1_test = train_test_split(
#         X1,y1, test_size=0.3, random_state=42
#     )
#
#     model= RandomForestClassifier()
#     model.fit(X1_train, y1_train)
#
#     print("AMZN Train score: ", model.score(X1_train, y1_train))
#     print("AMZN Test score: ", model.score(X1_test, y1_test))


# for tests
if __name__ == "__main__":
    data_fetch('AAPL')  # Ensure latest data
    run_pipeline_aapl()
    #
    # data_fetch_amzn('AMZN')
    # run_pipeline_amzn()
    #
