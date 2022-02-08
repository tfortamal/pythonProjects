import numpy as np  # to make numpy array
import pandas as pd  # to create data frames
from sklearn.preprocessing import StandardScaler  # to standardize the data
from sklearn.model_selection import train_test_split  # to split the data into training data and testing data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# loading the datasets as a pandas DataFrame
diabetes_dataset = pd.read_csv('https://raw.githubusercontent.com/tfortamal/pychunk/main/Diabetes%20Prediction/diabetes.csv')


# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
results = diabetes_dataset['Outcome']


# Data Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X.values)  # Here x.values will have only values without headers
# print("\n Printing standardized data \n")
# print(standardized_data)

# splitting the datasets into training data and labeled data
standardized_training_data, standardized_test_data, training_results, testing_results = train_test_split(
    standardized_data, results, test_size=0.2, stratify=results, random_state=2)
# print("\n \n printing the normal data, standardized data and the training data\n")
# print('the standardized data: ', standardized_data.shape)
# print('the standardized training data: ', standardized_training_data.shape)
# print('the standardized training data: ', standardized_test_data.shape)

# Training the model
# classifier = svm.SVC(kernel='linear')  # using support verto machine classifier and the model is linear
diabetes_model = RandomForestClassifier(criterion='gini', max_depth=4, min_samples_leaf=1, min_samples_split=2)  # using random forest classifier
# feeding training data into the model
diabetes_model.fit(standardized_training_data, training_results)

# evaluating the trained model
# finding the accuracy score on the training data
training_prediction = diabetes_model.predict(standardized_training_data)  # storing the prediction in the training_prediction variable
training_accuracy = accuracy_score(training_prediction, training_results)
print("\n\n\nThe accuracy score of training is: ", training_accuracy)
# finding hte accuracy score on the testing data
test_prediction = diabetes_model.predict(standardized_test_data)  # storing the prediction in the training_prediction variable
test_accuracy = accuracy_score(test_prediction, testing_results)
print("The accuracy score of testing is: ", test_accuracy)

# making a predictive system

# this is where we are going to input the medical information
# raw_data = input('Enter the Raw data in order: ')  # input in a single line
# input_data = raw_data.split(",")
input_data_types = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age', 'Outcome']
n = 8
j = 0
input_data = []
for i in range(n):
    input_data.append(float(input('Enter the {}: '.format(input_data_types[j]))))
    j += 1


# changing the input data to numpy array
# print(type(input_data))
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array
reshaped_input_data = input_data_as_numpy_array.reshape(1, -1)
# standardizing the input data
standardized_input_dara = scaler.transform(reshaped_input_data)
#  print("\n\n the standardized input data is: ", standardized_input_dara)

# loading the standardized input data intp the trained model
prediction = diabetes_model.predict(standardized_input_dara)
print('\n\n\nInput data in order: ', input_data)
print("making a prediction on the input data: ")
if prediction[0] == 0:
    print('The patient is no-diabetic. output: ', prediction)
else:
    print('The patient is diabetic. output: ', prediction)
