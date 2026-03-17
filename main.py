# ================= IMPORTS =================
import os
import cv2
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

# ================= LABELS =================
labels = ['Hand', 'Head', 'Leg']
path = "Dataset"

def getLabel(name):
    return labels.index(name)

print("Labels:", labels)

# ================= LOAD DATA =================
X = []
Y = []

if os.path.exists('model/X.txt.npy'):
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
else:
    for label in labels:
        folder = os.path.join(path, label)

        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))

                X.append(img)
                Y.append(getLabel(label))

    X = np.array(X)
    Y = np.array(Y)

    os.makedirs("model", exist_ok=True)
    np.save('model/X.txt', X)
    np.save('model/Y.txt', Y)

print("Total Images:", len(X))

# ================= PREPROCESS =================
X = X.astype('float32') / 255
Y_cat = to_categorical(Y)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
Y_cat = Y_cat[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2)

# ================= METRICS =================
accuracy = []
precision = []
recall = []
fscore = []

def calculateMetrics(name, pred, true):
    a = accuracy_score(true, pred) * 100
    p = precision_score(true, pred, average='macro') * 100
    r = recall_score(true, pred, average='macro') * 100
    f = f1_score(true, pred, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    print(f"\n{name} Results:")
    print("Accuracy:", a)
    print("Precision:", p)
    print("Recall:", r)
    print("F1 Score:", f)

    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title(name + " Confusion Matrix")
    plt.show()

# ================= CNN MODEL =================
cnn_model = Sequential()

cnn_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Conv2D(64, (3,3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Flatten())

cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(len(labels), activation='softmax'))

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ================= TRAIN =================
y_train_labels = np.argmax(y_train, axis=1)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

if not os.path.exists("model/cnn_weights.keras"):
    checkpoint = ModelCheckpoint("model/cnn_weights.keras", save_best_only=True)

    history = cnn_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        class_weight=class_weights
    )

    with open("model/history.pkl", "wb") as f:
        pickle.dump(history.history, f)
else:
    cnn_model.load_weights("model/cnn_weights.keras")

cnn_model.save("cnn_model.keras")

# ================= TEST CNN =================
pred = cnn_model.predict(X_test)
pred = np.argmax(pred, axis=1)
true = np.argmax(y_test, axis=1)

calculateMetrics("CNN", pred, true)

print("\nCNN Results:")
print("Accuracy:", accuracy_score(true, pred) * 100)

# ================= ML MODELS =================
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

y_train_flat = np.argmax(y_train, axis=1)
y_test_flat = np.argmax(y_test, axis=1)

# SVM
svm_model = svm.SVC()
svm_model.fit(X_train_flat, y_train_flat)
pred = svm_model.predict(X_test_flat)
calculateMetrics("SVM", pred, y_test_flat)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_flat, y_train_flat)
pred = dt_model.predict(X_test_flat)
calculateMetrics("Decision Tree", pred, y_test_flat)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_flat, y_train_flat)
pred = rf_model.predict(X_test_flat)
calculateMetrics("Random Forest", pred, y_test_flat)

# ================= GRAPH =================
df = pd.DataFrame({
    'Algorithm': ['CNN', 'SVM', 'DT', 'RF'],
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': fscore
})

df.set_index('Algorithm').plot(kind='bar', figsize=(8,4))
plt.title("Performance Comparison")
plt.show()

# ================= SEVERITY =================
def getSeverity(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,100,120])
    upper_red = np.array([15,255,255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    severity = "Unknown"

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 5000:
            severity = "Major"
        elif area > 1000:
            severity = "Moderate"
        else:
            severity = "Minor"

    return severity

# ================= RECOMMENDATION =================
def getRecommendation(label):
    with open(f"recommendation/{label}.txt", "r") as f:
        return f.read()

# ================= PREDICT =================
def predict(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (64,64))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    pred = cnn_model.predict(img_resized)
    idx = np.argmax(pred)

    label = labels[idx]
    confidence = np.max(pred) * 100

    severity = getSeverity(image_path)
    rec = getRecommendation(label)

    print("\n===== RESULT =====")
    print("Prediction:", label)
    print("Confidence:", confidence)
    print("Severity:", severity)
    print("Recommendation:\n", rec)

# ================= TEST =================
predict("testImages/1.jpg")
predict("testImages/4.jpg")
predict("testImages/10.jpg")