from sklearn.svm import SVC
import pandas as pd
import joblib as jb

def main():
    df = pd.read_csv('trainData.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    model.fit(X, y)
    jb.dump(model, 'gestureModel.pkl')

main()