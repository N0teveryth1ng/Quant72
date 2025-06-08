from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import X, y

# split train,test
X_train, X_test, y_train,y_test = train_test_split(
    X,y , test_size=0.3, random_state = 42
)

# model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# score tracker
print("train accs : ", model.score(X_train, y_train))
print("test accs : ", model.score(X_test, y_test))