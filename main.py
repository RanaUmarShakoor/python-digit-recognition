from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

# Print the shape of the data
print(x.shape)

# Select a specific digit for visualization and prediction
digit_index = 9  # Index of the digit to be predicted
digit = x.iloc[digit_index].values
actual_label = y[digit_index]  # Actual label of the selected digit
# digit_image = digit.reshape(1, -1)


# Train/test split
x_train = x[:60000]
x_test = x[60000:70000]
y_train = y[:60000]
y_test = y[60000:70000]

# Convert y values to integers
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# Make prediction on the selected digit
predicted_label = knn.predict([digit])[0]


# Display results
plt.imshow(digit.reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
plt.axis('off')
plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
plt.show()

# Evaluate the classifier on the test set
accuracy = knn.score(x_test, y_test)
print(f"kNN Classifier Accuracy on Test Set: {accuracy:.2%}")
