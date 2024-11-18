# Python Digit Recognition Using MNIST and KNN  

This project implements a digit recognition system using the **MNIST Dataset** and the **K-Nearest Neighbors (KNN)** algorithm. It demonstrates how machine learning can be applied to classify handwritten digits (0–9) with high accuracy.  

## Features  
- **Dataset:** Utilizes the MNIST dataset, containing 70,000 grayscale images of handwritten digits.  
- **Algorithm:** Implements KNN for classification, using a `k` value of 3.  
- **Visualization:** Displays a sample digit along with its actual and predicted labels.  
- **Performance Evaluation:** Computes and displays the accuracy of the classifier on the test set.  

## Workflow  
1. **Load the Data:** Fetch the MNIST dataset from `sklearn.datasets`.  
2. **Data Preparation:** Split the data into training (60,000 samples) and test (10,000 samples) sets.  
3. **Train the Model:** Train a KNN classifier with the training set.  
4. **Make Predictions:** Predict the label for a selected digit and visualize the result.  
5. **Evaluate Performance:** Measure the accuracy of the model on the test set.  

## Code Highlights  
- Fetch the dataset using `fetch_openml`.  
- Preprocess and split the dataset for training and testing.  
- Use `KNeighborsClassifier` from `scikit-learn` for classification.  
- Visualize the selected digit and compare the actual label with the predicted label.  

### Example Output  
The program displays a digit image with the **actual** and **predicted labels** and outputs the accuracy of the classifier:  

```plaintext  
kNN Classifier Accuracy on Test Set: 96.88%  
```  

## Technologies Used  
- **Python** for scripting  
- **scikit-learn** for machine learning  
- **NumPy** for numerical operations  
- **matplotlib** for visualization  

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/python-digit-recognition.git  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the script:  
   ```bash  
   python digit_recognition.py  
   ```  

## Acknowledgments  
- **MNIST Dataset** by Yann LeCun et al.  
- **scikit-learn** for providing easy-to-use machine learning tools.  

Feel free to experiment with different values of `k` and improve the model! Contributions are welcome.  
