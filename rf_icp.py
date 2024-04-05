#%pip install pandas
#%pip install scikit-learn
#%pip install matplotlib
#%pip install seaborn

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
CSV_FILE = 'icp-raw.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

def load_and_prepare_data(csv_file):
    try:
        # Load the dataset into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Drop the 'KeyDecisionMaker' column if it exists
        if 'KeyDecisionMaker' in df.columns:
            df.drop('KeyDecisionMaker', axis=1, inplace=True)
        return df
    except FileNotFoundError:
        print(f"The file {csv_file} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"The file {csv_file} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

############### Applying multi-label encoding for ProductFeaturesUsed 
# Assuming 'ProductFeaturesUsed' contains strings of features separated by commas
# Split the 'ProductFeaturesUsed' column into a list of features
def encode_product_features(df):
    df['ProductFeaturesUsed'] = df['ProductFeaturesUsed'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    product_features_encoded = mlb.fit_transform(df['ProductFeaturesUsed'])
    product_features_df = pd.DataFrame(product_features_encoded, columns=mlb.classes_)
    df.drop('ProductFeaturesUsed', axis=1, inplace=True)
    df = pd.concat([df, product_features_df], axis=1)
    return df

############### Applying Binning and Label encodig for RecentFunding
def bin_and_encode_funding(df):
    label_encoder = LabelEncoder()
    # Convert 'RecentFunding' to numeric values (in millions)
    df['RecentFunding'] = df['RecentFunding'].apply(lambda x: 0 if x == 'No' else float(x.split()[0]))

    # Apply custom binning to 'RecentFunding' column
    funding_category_df = df['RecentFunding'].apply(bin_funding_helper)

    # Fit and transform the 'FundingCategory' column
    df['FundingCategory'] = label_encoder.fit_transform(funding_category_df)

    # Drop the original 'RecentFunding' column
    df = df.drop('RecentFunding', axis=1)

    return df

def bin_funding_helper(amount):
    if amount == 0:
        return 'No Funding'
    elif amount < 1:
        return 'Low Funding'
    elif 1 <= amount < 10:
        return 'Medium Funding'
    else:
        return 'High Funding'

def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['Industry', 'Title', 'Department', 'TechnologyStack', 'DecisionMaking', 'LeadershipChange']).astype(int)
    return df

def train_random_forest(X_train, y_train):
    # optimize hyper parameters
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE), 
                               param_grid=PARAM_GRID, 
                               cv=5, verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(**best_params, random_state=RANDOM_STATE)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    return rf_classifier


def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

############### Extract feature importance after the classifier has been trained
def extract_feature_importance(classifier, X_train):
    feature_importances = classifier.feature_importances_
    features = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    features.sort_values(by='Importance', ascending=False, inplace=True)

    # Calculate the 75th percentile of the feature importances
    threshold = np.percentile(feature_importances, 75)

    # Add a new column '75th_percentile' that adds a "*" if the feature's importance is above the threshold
    features['75th_percentile'] = features['Importance'].apply(lambda x: '*' if x >= threshold else '')

    return features

def plot_feature_importances(features):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features, hue='75th_percentile', dodge=False)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.legend(title='75th Percentile', loc='upper right')
    plt.tight_layout()
    plt.show()

# Customer Satisfaction Score
# To define the target variable for predicting the Ideal Customer Profile (ICP) for any business, we should consider a success metric that aligns with that business's goals and objectives. 
# We might consider a customer with a satisfaction score of 8 or above as successful. Hence, we define the target variable as 'is_successful'
def target_variable_is_successful(df):
    df['is_successful'] = df['CustomerSatisfactionScore'].apply(lambda x: 1 if x >= 8 else 0)
    
    # Prepare the feature set X by dropping the target variable and CustomerSatisfactionScore
    X = df.drop(['is_successful', 'CustomerSatisfactionScore'], axis=1)
    y = df['is_successful']

    return X,y

# High Engagement
# If engagement is crucial for your business, you could define success based on a combination of website visits and email opens. For example, a customer with more than 10 website visits and more than 5 email opens last month could be considered successful.
def target_variable_is_engaged(df):
    df['Is_Highly_Engaged'] = df.apply(lambda row: 1 if row['WebsiteVisits'] > 10 and row['EmailOpens'] > 5 else 0, axis=1)

    # Prepare the feature set X by dropping the target variable, WebsiteVisits, and EmailOpens
    X = df.drop(['Is_Highly_Engaged', 'WebsiteVisits', 'EmailOpens'], axis=1)

    # Prepare the target variable y
    y = df['Is_Highly_Engaged']

    return X,y

# Transaction-Based Success:
# If the number of past transactions is an indicator of success for your business, you could set a # # threshold for the number of transactions. For example, customers with more than 4 past transactions # might be considered successful.
def target_variable_is_transaction_success(df):
    df['Is_Transaction_Success'] = df['PastTransactions'].apply(lambda x: 1 if x > 4 else 0)

    # Prepare the feature set X by dropping the target variable and PastTransactions
    X = df.drop(['Is_Transaction_Success', 'PastTransactions'], axis=1)

    # Prepare the target variable y
    y = df['Is_Transaction_Success']

    return X,y

def main():
    df = load_and_prepare_data(CSV_FILE)
    df = encode_product_features(df)
    df = bin_and_encode_funding(df)
    df = one_hot_encode(df)

    X, y = target_variable_is_successful(df)
    # X, y = target_variable_is_engaged(df)
    # X, y = target_variable_is_transaction_success(df)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    rf_classifier = train_random_forest(X_train, y_train)
    evaluate_classifier(rf_classifier, X_test, y_test)

    feature_importances = extract_feature_importance(rf_classifier, X_train)
    #print(feature_importances)

    plot_feature_importances(feature_importances)

if __name__ == "__main__":
    main()

