Requirements
To run this program, you'll need the following:

Python 3
Pandas
Scikit-learn
You can install these libraries using pip:

bash
Copy code
pip install pandas scikit-learn
Dataset
The program loads the dataset from the "Sales.csv" file. The dataset is assumed to have the following columns:

TV: Advertising spending on TV
Radio: Advertising spending on Radio
Sales: Sales figures
Data Preprocessing
The following data preprocessing steps are performed on the dataset:

Loading the dataset using Pandas.
Removing rows with missing values (NaN) using dropna.
Removing duplicate rows using drop_duplicates.
Splitting the Data
The dataset is split into training and testing sets using the train_test_split function from scikit-learn. 80% of the data is used for training, and 20% is reserved for testing.

Linear Regression Model
A linear regression model is used to predict sales based on advertising spending on TV and Radio. The scikit-learn library is used to create and train the model.

python
Copy code
model = LinearRegression()
model.fit(X_train, y_train)
Prediction and Evaluation
The model is used to make predictions on the test data, and the Mean Squared Error (MSE) is calculated to evaluate the model's performance.

python
Copy code
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
The MSE provides a measure of how well the model's predictions match the actual sales data. A lower MSE indicates a better fit of the model to the data.

Running the Program
You can run the program by executing the Python script. Make sure to have the "Sales.csv" dataset in the same directory as the script.

bash
Copy code
python your_script_name.py
The program will load the dataset, preprocess the data, train a linear regression model, make predictions, and display the Mean Squared Error.
