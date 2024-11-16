import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Suppress the numpy casting warning
np.seterr(invalid='ignore')

# Load the datasets
print("Loading the dataset...")
train_data = pd.read_csv('archive/train.csv')

# Define the target variable and features
target_variable_name = 'Dominant_Emotion'

feature_columns = [
    'User_ID',  # Ensure there are no spaces in the co2lumn names
    'Age',
    'Gender',
    'Platform',
    'Daily_Usage_Time (minutes)',
    'Posts_Per_Day',
    'Likes_Received_Per_Day',
    'Comments_Received_Per_Day',
    'Messages_Sent_Per_Day'
]

# Remove rows where the target variable is missing
print("Cleaning the data by removing rows with missing target variable...")
train_data.dropna(subset=[target_variable_name], inplace=True)

# Separate features into numeric and categorical
numeric_features = ['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 
                    'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 
                    'Messages_Sent_Per_Day']
categorical_features = ['Gender', 'Platform']

# Convert numeric columns to numeric type (coerce errors to NaN)
print("Converting numeric columns to numeric type...")
for col in numeric_features:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

# Create imputers for numeric and categorical features
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values for numeric features
print("Imputing missing values for numeric features...")
train_data[numeric_features] = numeric_imputer.fit_transform(train_data[numeric_features])

# Impute missing values for categorical features
print("Imputing missing values for categorical features...")
train_data[categorical_features] = categorical_imputer.fit_transform(train_data[categorical_features])

# Ensure the target variable is of consistent type
train_data[target_variable_name] = train_data[target_variable_name].astype(str)

# Split into features (X) and target variable (y) after imputation
X_train = train_data[feature_columns]  
y_train = train_data[target_variable_name]  

# One-hot encoding
print("Performing one-hot encoding on categorical features...")
X_train = pd.get_dummies(X_train, drop_first=True)

# Split the training data into train and validation sets for better evaluation
print("Splitting the data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize models
models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# Define hyperparameter grids
param_distributions = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],  # Removed extra space
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', None]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }
}

# Train and evaluate models
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    print(f"Training {model_name} model...")
    # Hyperparameter tuning using Random SearchCV
    random_search = RandomizedSearchCV(model, param_distributions[model_name], n_iter=50, cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model after tuning
    current_model = random_search.best_estimator_
    val_predictions = current_model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation Accuracy for {model_name}: {accuracy:.2f}")

    # Check if this is the best model so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = current_model

print(f"Best model is {best_model} with accuracy: {best_accuracy:.2f}")

# Save the best model
print("Saving the best model...")
joblib.dump(best_model, 'best_model.pkl')

# User input for prediction
print("Please provide the following information for emotion prediction:")
user_id = input("Enter User ID: ")
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
platform = input("Enter Platform (Whatsapp/Instagram...etc): ")
daily_usage_time = int(input("Enter Daily Usage Time (in minutes): "))
posts_per_day = int(input("Enter Posts Per Day: "))
likes_received_per_day = int(input("Enter Likes Received Per Day: "))
comments_received_per_day = int(input("Enter Comments Received Per Day: "))
messages_sent_per_day = int(input("Enter Messages Sent Per Day: "))

# Prepare user input for prediction
user_input = pd.DataFrame({
    'User_ID': [user_id],
    'Age': [age],
    'Gender': [gender],
    'Platform': [platform],
    'Daily_Usage_Time (minutes)': [daily_usage_time],
    'Posts_Per_Day': [posts_per_day],
    'Likes_Received_Per_Day': [likes_received_per_day],
    'Comments_Received_Per_Day': [comments_received_per_day],
    'Messages_Sent_Per_Day': [messages_sent_per_day]
})

# Preprocess user input
print("Preprocessing user input...")
user_input[numeric_features] = numeric_imputer.transform(user_input[numeric_features])
user_input[categorical_features] = categorical_imputer.transform(user_input[categorical_features])
user_input = pd.get_dummies(user_input, drop_first=True)

# Ensure the user input has the same columns as the training data
user_input = user_input.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
predicted_emotion = best_model.predict(user_input)
print(f"The predicted current mood or emotion is: {predicted_emotion[0]}")

# Plotting the results
print("Generating plot for predicted vs true emotions...")
true_counts = y_val.value_counts()
predicted_counts = pd.Series(predicted_emotion).value_counts()

plt.figure(figsize=(10, 5))
plt.plot(true_counts.index, true_counts.values, label='True Counts', marker='o')
plt.plot(predicted_counts.index, predicted_counts.values, label='Predicted Counts', marker='x')
plt.title('Predicted vs True Emotions')
plt.xlabel('Emotions')
plt.ylabel('Count')
plt.legend()
plt.show()