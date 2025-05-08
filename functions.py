import os
import pickle
from glob import glob
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    ParameterGrid
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

import shap
from fastshap import KernelExplainer
from pathlib import Path


# Define preprocessing and model order (example values, adjust as needed)
preprocessing_order = ['No Preprocessing', 'PCA on MS', 'Masked Autoencoder', 'MI']
model_order = ['Logistic Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting', 'Support Vector Machine']
dataset_order = ["maccs", "smile"]


def create_output_directories(working_dir):
    """
    Create output directories for intermediate results, plots, and final results.

    Args:
        working_dir (str or Path): Base working directory path.

    Returns:
        dict: Dictionary containing paths to the created directories.
    """
    working_dir = Path(working_dir)

    dirs = {
        "intermediate_results_dir": working_dir / "intermediate_results",
        "plot_dir": working_dir / "plot",
        "results_directory": working_dir / "results"
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def full_fungi_data_pipeline(
    file_path,
    processed_data_save_path,
    plots_dir,
    nested_folds_save_path,
    final_data_dir
):
    # Step 1: Preprocess the data
    data_preprocessed = preprocess(file_path)
    data_preprocessed.to_csv(processed_data_save_path, index=False)

    # Step 2: Compute and print statistics
    stats_df = pd.DataFrame(
        [compute_statistics(data_preprocessed)], index=['Pre-processed Data']
    )
    print(stats_df)

    # Step 3: Unique samples analysis
    print("\nUnique samples per species in the whole set:")
    print(data_preprocessed.groupby("species")["sample_id"].nunique())

    print("\nUnique samples per drug in the whole set:")
    print(data_preprocessed.groupby("drug")["sample_id"].nunique())

    print("\nUnique samples per year in the whole set:")
    print(data_preprocessed.groupby("year")["sample_id"].nunique())

    # Step 4: Response distribution per species-drug pair
    response_counts = (
        data_preprocessed.groupby(["species", "drug", "response"])["sample_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={0: "num_samples_response_0", 1: "num_samples_response_1"})
    )
    response_counts["proportion_response_1"] = (
        response_counts["num_samples_response_1"] /
        (response_counts["num_samples_response_0"] + response_counts["num_samples_response_1"])
    )
    print("\nTop species-drug pairs by response proportion in the whole set:")
    print(response_counts.sort_values(by="proportion_response_1", ascending=False).head())

    # Step 5: Generate plots
    plot_data_distribution(
        data_preprocessed, 'species', 'Species',
        'Number of Unique Samples by Species and Response',
        save_path=os.path.join(plots_dir, 'species_distribution.png')
    )
    plot_data_distribution(
        data_preprocessed, 'drug', 'Drug',
        'Number of Unique Samples by Drug and Response',
        save_path=os.path.join(plots_dir, 'drug_distribution.png')
    )
    plot_data_distribution_per_species_drug(
        data_preprocessed, species_column='species', drug_column='drug',
        x_label='Species - Drug Pairs',
        save_path=os.path.join(plots_dir, 'species_drug_distribution.png')
    )

    # Step 6: Train-deployment split
    data_train, data_deploy = stratified_split(data_preprocessed)
    data_train.to_csv(os.path.join(final_data_dir, 'data_train.csv'), index=False)
    data_deploy.to_csv(os.path.join(final_data_dir, 'data_deployement.csv'), index=False)

    # Step 7: Statistics for splits
    print("\nTraining Data Statistics:")
    print(pd.DataFrame([compute_statistics(data_train)], index=['Training Data']))

    print("\nDeployment Data Statistics:")
    print(pd.DataFrame([compute_statistics(data_deploy)], index=['Deployment Data']))

    # Step 8: Nested Cross-Validation
    nested_folds = nested_cross_validation(
        data_train, sample_id_col='sample_id',
        n_splits_outer=3, n_splits_inner=3, random_state=42
    )

    with open(nested_folds_save_path, 'wb') as f:
        pickle.dump(nested_folds, f)
    print(f"Nested folds saved to {nested_folds_save_path}")

    # Step 9: Save fold data
    def save_pairs_to_csv(data, path, sample_id_col='sample_id', drug_col='drug'):
        data[[sample_id_col, drug_col]].to_csv(path, index=False)

    for outer_idx, (inner_folds, test_set) in enumerate(nested_folds):
        outer_dir = os.path.join(final_data_dir, f'outer_fold_{outer_idx+1}')
        os.makedirs(outer_dir, exist_ok=True)
        save_pairs_to_csv(test_set, os.path.join(outer_dir, 'test_set_pairs.csv'))

        inner_dir = os.path.join(outer_dir, 'inner_folds')
        os.makedirs(inner_dir, exist_ok=True)

        for inner_idx, (train_inner, val_inner) in enumerate(inner_folds):
            fold_dir = os.path.join(inner_dir, f'inner_fold_{inner_idx+1}')
            os.makedirs(fold_dir, exist_ok=True)
            save_pairs_to_csv(train_inner, os.path.join(fold_dir, 'inner_training_set_pairs.csv'))
            save_pairs_to_csv(val_inner, os.path.join(fold_dir, 'inner_validation_set_pairs.csv'))

    print("All fold data saved successfully.")
    
def preprocess(file_path):
    data = pd.read_csv(file_path)

    # Remove DRIAMS C
    DRIAMSC = data[data['dataset'] == 'C'] 
    data = data.drop(DRIAMSC.index)
    
    # Remove intrinsic resistance
    data = data[
        ~(
            ((data['species'] == 'candida krusei') & (data['drug'] == 'Fluconazole')) |
            ((data['species'] == 'aspergillus terreus') & (data['drug'] == 'Amphotericin B')) |
            ((data['species'] == 'aspergillus terreus') & (data['drug'] == 'Fluconazole B')) |
            ((data['species'] == 'aspergillus fumigatus') & (data['drug'] == 'Fluconazole'))
        )
    ]

    # Remove drug-species pairs with no EUCAST breakpoints
    species_to_remove = ['candida albicans','candida glabrata','candida krusei','candida parapsilosis','candida tropicalis']
    data = data[~((data['species'].isin(species_to_remove)) & (data['drug'] == 'Caspofungin'))]

    # Replace specified values in the 'species' column
    data['species'] = data['species'].replace({
        'candida albicans_(africana)': 'candida albicans',
        'mix!candida albicans': 'candida albicans',
        'mix!candida inconspicua': 'candida inconspicua',
        'mix!candida dubliniensis': 'candida dubliniensis',
        'mix!candida tropicalis': 'candida tropicalis'
    })
    
    # Remove species-drug pairs constantly sensitive or resistant
    grouped_data = data.groupby(['species', 'drug'])['response'].mean()
    groups_to_remove = grouped_data[(grouped_data == 0) | (grouped_data == 1)].index
    remaining_data = data[~data.set_index(['species', 'drug']).index.isin(groups_to_remove)].reset_index(drop=True)

    return remaining_data

def compute_statistics(data):
    # Total number of observations
    total_observations = len(data)
    
    # Number of species-drug combinations
    species_drug_combinations = data.groupby(['species', 'drug']).size().shape[0]
    
    # Number of unique pathogen (sample_id)
    unique_pathogens = data['sample_id'].nunique()
    
    # Number of unique species
    unique_species = data['species'].nunique()
    
    # Number of unique drug
    unique_drugs = data['drug'].nunique()
    
    # Proportion of resistant observations (assuming response=1 means resistant)
    proportion_resistant = (data['response'] == 1).mean()
    
    # Proportion of sensitive observations (assuming response=0 means sensitive)
    proportion_sensitive = (data['response'] == 0).mean()
    
    # Return the results as a dictionary
    return {
        'Total Observations': total_observations,
        'Species-Drug Combinations': species_drug_combinations,
        'Unique Pathogens (sample_id)': unique_pathogens,
        'Unique Species': unique_species,
        'Unique Drugs': unique_drugs,
        'Proportion Resistant': proportion_resistant,
        'Proportion Sensitive': proportion_sensitive
    }

# Plot and save data distribution
def plot_data_distribution(data, group_by_column, x_label, title, rotation=90, save_path=None, x_label_size=20):
    # Group the data by the specified column to count the unique sample_id
    unique_sample_counts = data.groupby(group_by_column)['sample_id'].nunique().reset_index(name='unique_sample_count')
    print(unique_sample_counts)

    # Format x-axis labels
    def format_label(label):
        label = str(label).capitalize()
        if group_by_column.lower() == 'species':
            label = r"$\it{" + label.replace(' ', r'\ ') + "}$"
        return label

    formatted_labels = [format_label(val) for val in unique_sample_counts[group_by_column]]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plotting the bars
    plt.bar(unique_sample_counts[group_by_column], unique_sample_counts['unique_sample_count'], color='#008B8B')

    # Adding labels and title
    plt.ylabel('Samples', fontsize=17)
    plt.xticks(ticks=range(len(formatted_labels)), labels=formatted_labels, rotation=rotation, fontsize=x_label_size)
    plt.tight_layout()

    # Save the plot if a save_path is provided, else display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()  # Close the plot to free memory




def plot_data_distribution_per_species_drug(data, species_column, drug_column, x_label, rotation=90, save_path=None, x_label_size=15):
    # Create a new column combining species and drug with capitalization
    data['species_drug'] = data[species_column].str.capitalize() + ' - ' + data[drug_column].str.capitalize()

    # Group the data by the new species-drug column and response to count unique sample_id
    unique_sample_counts = data.groupby(['species_drug', 'response'])['sample_id'].nunique().reset_index(name='unique_sample_count')
    print(unique_sample_counts)

    # Pivot the data for side-by-side bar plotting
    pivot_data = unique_sample_counts.pivot(index='species_drug', columns='response', values='unique_sample_count').fillna(0)

    # Format x-tick labels with italic species (before '-')
    def format_label(label):
        if '-' in label:
            before_dash, after_dash = label.split('-', 1)
            before_dash = before_dash.strip().replace(' ', r'\ ')
            after_dash = after_dash.strip()
            return r"$\it{" + before_dash + "}$" + " - " + after_dash
        return label

    formatted_labels = [format_label(label) for label in pivot_data.index]

    # Extract species names from index
    species_names = [label.split(' - ')[0].strip() for label in pivot_data.index]

    # Determine group spans for shaded boxes
    group_spans = []
    start_idx = 0
    current_species = species_names[0]

    for i, species in enumerate(species_names[1:], start=1):
        if species != current_species:
            group_spans.append((start_idx, i - 1, current_species))
            start_idx = i
            current_species = species
    group_spans.append((start_idx, len(species_names) - 1, current_species))

    # Set up plot
    bar_width = 0.35
    index = np.arange(len(pivot_data.index))

    plt.figure(figsize=(15, 8))

    # Draw shaded boxes for species groups
    gray_colors = ['#f0f0f0', '#d9d9d9']  # light gray and slightly darker gray
    for i, (start, end, _) in enumerate(group_spans):
        color = gray_colors[i % 2]  # Alternate between the two
        left = start - 0.35
        right = end + 1 - 0.35
        plt.axvspan(left, right, color=color, alpha=0.5, zorder=0)


    # Plot bars
    plt.bar(index, pivot_data[0], bar_width, label='Susceptible', color='#008B8B')
    plt.bar(index + bar_width, pivot_data[1], bar_width, label='Resistant', color='#FF8C00')

    # Set labels and ticks
    plt.ylabel('Samples')
    plt.xticks(index + bar_width / 2, formatted_labels, rotation=rotation, fontsize=x_label_size)
    plt.legend()
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def stratified_split(data, sample_id_col='sample_id', random_state=42):
    """
    Perform a stratified split of the dataset into 70% training and 30% testing.
    """
    # Create a stratification column based on species-drug combination
    data['stratify_group'] = data['species'] + "_" + data['drug']
    
    # Ensure no sample_id leaks across train and test
    unique_samples = data.drop_duplicates(subset=sample_id_col)[[sample_id_col, 'stratify_group']]
    
    # Perform stratified split on unique samples
    train_samples, test_samples = train_test_split(
        unique_samples,
        test_size=0.3,
        stratify=unique_samples['stratify_group'],
        random_state=random_state
    )
    
    # Get training and testing data
    train_data = data[data[sample_id_col].isin(train_samples[sample_id_col])]
    test_data = data[data[sample_id_col].isin(test_samples[sample_id_col])]
    
    return train_data, test_data


# Function for nested cross-validation
def nested_cross_validation(data, sample_id_col='sample_id', n_splits_outer=5, n_splits_inner=5, random_state=42):
    """
    Perform nested cross-validation.
    The outer loop splits data into training and test sets.
    The inner loop splits the outer training set into inner training and validation sets.
    """
    # Create a stratification column based on species-drug combination
    data['stratify_group'] = data['species'] + "_" + data['drug']

    # Ensure no sample_id leaks across train and test
    unique_samples = data.drop_duplicates(subset=sample_id_col)[[sample_id_col, 'stratify_group']]
    
    # Prepare cross-validation folds for outer loop
    skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    nested_folds = []

    for train_outer_index, test_outer_index in skf_outer.split(unique_samples[sample_id_col], unique_samples['stratify_group']):
        train_outer_sample_ids = unique_samples.iloc[train_outer_index][sample_id_col].tolist()
        test_outer_sample_ids = unique_samples.iloc[test_outer_index][sample_id_col].tolist()
        
        outer_train_data = data[data[sample_id_col].isin(train_outer_sample_ids)]
        test_data = data[data[sample_id_col].isin(test_outer_sample_ids)]
        
        # Prepare cross-validation folds for inner loop
        unique_inner_samples = outer_train_data.drop_duplicates(subset=sample_id_col)[[sample_id_col, 'stratify_group']]
        skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)

        inner_folds = []
        for train_inner_index, validation_inner_index in skf_inner.split(unique_inner_samples[sample_id_col], unique_inner_samples['stratify_group']):
            train_inner_sample_ids = unique_inner_samples.iloc[train_inner_index][sample_id_col].tolist()
            validation_inner_sample_ids = unique_inner_samples.iloc[validation_inner_index][sample_id_col].tolist()
            
            inner_train_data = outer_train_data[outer_train_data[sample_id_col].isin(train_inner_sample_ids)]
            validation_data = outer_train_data[outer_train_data[sample_id_col].isin(validation_inner_sample_ids)]
            
            inner_folds.append((inner_train_data, validation_data))
        
        nested_folds.append((inner_folds, test_data))

    return nested_folds

# Function to return the input data without preprocessing
def no_preprocessing(X_train, X_test):
    return X_train, X_test, None


def pca_MS(X_train, X_test, columns_prefix='X', n_components=0.95):
    # Select columns with the specified prefix
    columns = [col for col in X_train.columns if col.startswith(columns_prefix)]
    pca = PCA(n_components=n_components, random_state=42)
    
    # Apply PCA
    X_train_pca = pca.fit_transform(X_train[columns])
    X_test_pca = pca.transform(X_test[columns])
    
    # Determine the number of components based on explained variance
    num_components = X_train_pca.shape[1]
    
    # Replace original columns with PCA components
    X_train = X_train.drop(columns=columns)
    X_test = X_test.drop(columns=columns)
    X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f'pca_{i}' for i in range(num_components)])
    X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f'pca_{i}' for i in range(num_components)])
    X_train = pd.concat([X_train, X_train_pca_df], axis=1)
    X_test = pd.concat([X_test, X_test_pca_df], axis=1)
    
    # Get the loadings matrix (components)
    loadings = pd.DataFrame(pca.components_.T, index=columns, columns=[f'PC_{i}' for i in range(num_components)])
    
    return X_train, X_test, loadings


def mutual_information_feature_selection(X_train, y_train, X_test, columns_prefix='X', n_features=10):
    # Select columns with the specified prefix
    columns = [col for col in X_train.columns if col.startswith(columns_prefix)]
    
    # Calculate mutual information
    mi = mutual_info_classif(X_train[columns], y_train)
    mi_df = pd.DataFrame({'feature': columns, 'mi': mi})
    
    # Select top n_features based on mutual information using SelectKBest
    kbest = SelectKBest(score_func=mutual_info_classif, k=n_features)
    kbest.fit(X_train[columns], y_train)
    kbest_scores = kbest.scores_
    
    # Get top features based on kbest_scores
    top_features_indices = np.argsort(kbest_scores)[-n_features:][::-1]
    top_features = [columns[i] for i in top_features_indices]
    
    # Create new training and test sets with selected features
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    
    # Replace original columns with selected features
    X_train = X_train.drop(columns=columns)
    X_test = X_test.drop(columns=columns)
    X_train = pd.concat([X_train, X_train_selected], axis=1)
    X_test = pd.concat([X_test, X_test_selected], axis=1)
    
    return X_train, X_test, top_features
    
# Function to create masked data
def create_masked_data(df, num_copies=5, mask_ratio_range=(0.2, 0.5), seed=42):
    np.random.seed(seed)
    num_samples, num_features = df.shape
    masked_data, original_data = [], []

    for sample in df:
        for _ in range(num_copies):
            mask_ratio = np.random.uniform(*mask_ratio_range)
            num_masked_features = int(mask_ratio * num_features)
            mask_indices = np.random.choice(num_features, num_masked_features, replace=False)
            masked_sample = sample.copy()
            masked_sample[mask_indices] = 0
            masked_data.append(masked_sample)
            original_data.append(sample)

    return np.array(masked_data), np.array(original_data)
    
# Function to build an autoencoder
def build_autoencoder(input_dim, encoding_dim=512):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return autoencoder, encoder

# Function to preprocess data using a masked autoencoder on the MS
def masked_autoencoder_preprocessing(X_train, X_test, encoding_dim=512, num_copies=5, mask_ratio_range=(0.2, 0.5), epochs=50, batch_size=256, validation_split=0.2, seed=42):
    # Select the columns 'X1' to 'X6000'
    columns_to_transform = [f'X{i}' for i in range(1, 6001)]
    
    # Apply StandardScaler only on the selected columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[columns_to_transform] = scaler.fit_transform(X_train[columns_to_transform])
    X_test_scaled[columns_to_transform] = scaler.transform(X_test[columns_to_transform])
    
    # Create masked data only for the selected columns
    X_train_masked, X_train_original = create_masked_data(X_train_scaled[columns_to_transform].values, num_copies=num_copies, mask_ratio_range=mask_ratio_range, seed=seed)
    
    # Build and train the autoencoder
    input_dim = X_train_scaled[columns_to_transform].shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(X_train_masked, X_train_original, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split)
    
    # Encode the selected columns
    X_train_encoded = encoder.predict(X_train_scaled[columns_to_transform])
    X_test_encoded = encoder.predict(X_test_scaled[columns_to_transform])
    
    # Replace the original columns with the encoded features
    X_train_encoded_df = pd.DataFrame(X_train_encoded, index=X_train.index, columns=[f'encoded_{i}' for i in range(encoding_dim)])
    X_test_encoded_df = pd.DataFrame(X_test_encoded, index=X_test.index, columns=[f'encoded_{i}' for i in range(encoding_dim)])
    X_train = X_train.drop(columns=columns_to_transform).join(X_train_encoded_df)
    X_test = X_test.drop(columns=columns_to_transform).join(X_test_encoded_df)
    
    return X_train, X_test, scaler


def compute_metrics_per_drug_species(results_df):
    # Initialize an empty list to store the results
    metrics_results = []

    # Group the results by preprocessing, model, fold, species, and drug
    grouped_results = results_df.groupby(['preprocessing', 'model', 'fold', 'species', 'drug'])

    # Iterate over each group and compute the metrics
    for group_key, group_data in grouped_results:
        y_test = group_data['y_test']
        y_pred = group_data['y_pred']
        
        # Count the number of samples in the current group
        num_samples = len(group_data)
        
        # Calculate the ratio of true response 1 vs 0
        num_true_1 = sum(y_test)
        num_true_0 = num_samples - num_true_1
        ratio_1_to_0 = num_true_1 / num_true_0 if num_true_0 != 0 else float('inf')
        
        # Compute metrics only if ratio_1_to_0 is neither 0 nor Inf
        if ratio_1_to_0 > 0 and ratio_1_to_0 < float('inf'):
            # Compute the MCC for the current group
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Compute the balanced accuracy
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Compute the AUROC (Area Under the ROC Curve)
            try:
                auroc = roc_auc_score(y_test, y_pred)
            except ValueError:
                auroc = None  # Assign NA if only one class is present in y_test
            
            # Compute the AUPRC (Area Under the Precision-Recall Curve)
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                auprc = auc(recall, precision)
            except ValueError:
                auprc = None  # Assign NA if only one class is present in y_test

            # Store the result
            metrics_results.append({
                'preprocessing': group_key[0],
                'model': group_key[1],
                'fold': group_key[2],
                'species': group_key[3],
                'drug': group_key[4],
                'mcc': mcc,
                'balanced_accuracy': balanced_acc,
                'auroc': auroc,
                'auprc': auprc,
                'num_samples': num_samples,
                'ratio_1_to_0': ratio_1_to_0
            })

    # Convert the results to a DataFrame
    metrics_results_df = pd.DataFrame(metrics_results)
    
    return metrics_results_df

def run_nested_cv_pipeline(
    data_path,
    nested_folds_path,
    results_dir,
    feature_removals,
    preprocessing_options='default',
    model_grids='default',
    functions=None
):
    import os

    # Set default preprocessing_options if not provided
    if preprocessing_options == 'default':
        preprocessing_options = {
            'Mutual Info': {
                'func': functions.mutual_information_feature_selection,
                'params': {'n_features': [64, 128, 512]}
            },
            'Masked Autoencoder': {
                'func': functions.masked_autoencoder_preprocessing,
                'params': {
                    'encoding_dim': [512],
                    'num_copies': [5, 10],
                    'mask_ratio_range': [(0.2, 0.5)],
                    'epochs': [50],
                    'batch_size': [256]
                }
            },
            'PCA': {
                'func': functions.pca_MS,
                'params': {'n_components': [0.5, 0.75, 0.85, 0.95, 0.99]}
            },
            'None': {
                'func': functions.no_preprocessing,
                'params': {}
            }
        }

    # Set default model_grids if not provided
    if model_grids == 'default':
        model_grids = {
            'Logistic Regression': {
                'model': functions.LogisticRegression(),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear'],
                    'model__max_iter': [2000]
                }
            },
            'Random Forest': {
                'model': functions.RandomForestClassifier(),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20]
                }
            },
            'Support Vector Machine': {
                'model': functions.SVC(),
                'params': {
                    'model__probability': [True],
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf']
                }
            },
            'Neural Network': {
                'model': functions.MLPClassifier(),
                'params': {
                    'model__hidden_layer_sizes': [(100,), (50, 50)],
                    'model__activation': ['relu', 'tanh'],
                    'model__alpha': [0.0001, 0.001],
                    'model__max_iter': [1000]
                }
            },
            'Gradient Boosting': {
                'model': functions.GradientBoostingClassifier(),
                'params': {
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__n_estimators': [50, 100, 200]
                }
            }
        }


    # Load data and folds
    nested_folds = functions.pickle.load(open(nested_folds_path, 'rb'))
    print('Start Time:', functions.datetime.now().strftime('%H%M'))
    data=functions.pd.read_csv(data_path)
    for outer_fold_idx, (inner_folds, test_set_pairs) in enumerate(nested_folds):
        print(f"Outer Fold: {outer_fold_idx + 1}")
        test_set = functions.filter_data(data, test_set_pairs[['sample_id', 'drug']])
        X_test = test_set.drop(columns=feature_removals)
        y_test = test_set['response']
        species_test = test_set['species']
        drug_test = test_set['drug']

        outer_train_pairs = functions.pd.concat(
            [train for (train, _) in inner_folds], ignore_index=True)
        outer_train_set = functions.filter_data(data, outer_train_pairs[['sample_id', 'drug']])
        X_train = outer_train_set.drop(columns=feature_removals)
        y_train = functions.LabelEncoder().fit_transform(outer_train_set['response'])

        for preproc_name, preproc_config in preprocessing_options.items():
            preproc_func = preproc_config['func']
            preproc_grid = preproc_config['params']

            for model_name, model_grid in model_grids.items():
                print(f"Evaluating Preprocessing: {preproc_name}, Model: {model_name}")
                best_score = -float('inf')
                best_preproc_params = None
                best_model_params = None

                for preproc_params in functions.ParameterGrid(preproc_grid):
                    for model_params in functions.ParameterGrid(model_grid['params']):
                        scores = [
                            functions.process_fold(
                                train, val, data, feature_removals,
                                preproc_func, preproc_params, model_grid, model_params
                            )
                            for train, val in inner_folds
                        ]
                        avg_score = functions.pd.DataFrame(scores).mean().to_dict()
                        if avg_score['mcc'] > best_score:
                            best_score = avg_score['mcc']
                            best_preproc_params = preproc_params
                            best_model_params = model_params

                print(f"Best preprocessing params: {best_preproc_params}")
                print(f"Best model params: {best_model_params}")
                print(f"Best inner validation score: {best_score}")

                if preproc_func == functions.mutual_information_feature_selection:
                        X_train_proc, X_test_proc, _ = preproc_func(
                            X_train, y_train, X_test, columns_prefix='X', **best_preproc_params)
                else:
                    X_train_proc, X_test_proc, _ = preproc_func(
                        X_train, X_test, **best_preproc_params)

                pipeline = functions.Pipeline([
                    ('scaler', functions.StandardScaler()),
                    ('model', model_grid['model'])
                ])
                pipeline.set_params(**best_model_params)
                pipeline.fit(X_train_proc, y_train)
                y_pred = pipeline.predict_proba(X_test_proc)[:, 1]

                metrics = functions.compute_species_drug_metrics(
                    y_test.values, y_pred, species_test.values, drug_test.values
                )

                results = [{
                    'preprocessing': preproc_name,
                    'preproc_params': best_preproc_params,
                    'model': model_name,
                    'model_params': best_model_params,
                    'outer_fold': outer_fold_idx + 1,
                    'species': m['species'],
                    'drug': m['drug'],
                    'auprc': m['auprc'],
                    'mcc': m['mcc'],
                    'balanced_accuracy': m['balanced_accuracy']
                } for m in metrics]

                output_file = os.path.join(
                    results_dir, f"results_outer_fold_{outer_fold_idx + 1}_{preproc_name}_{model_name}.csv")

                results_df = functions.pd.DataFrame(results)
                write_mode = 'w' if not os.path.exists(output_file) else 'a'
                header = not os.path.exists(output_file)
                results_df.to_csv(output_file, mode=write_mode, header=header, index=False)
                print(f"Saved results to {output_file}")

    print('End Time:', functions.datetime.now().strftime('%H%M'))




def compute_avg_and_std_across_folds(data, n, metrics):
    # Initialize a list to store the results for each fold
    fold_results = []

    # Step 1: Group the data by model, preprocessing, and fold
    grouped = data.groupby(['model', 'preprocessing', 'fold'])
    
    # Step 2: Iterate through each group
    for name, group in grouped:
        # Prepare a dictionary to store the results for the current group
        result = {
            'model': name[0],
            'preprocessing': name[1],
            'fold': name[2]
        }
        
        # Step 3: Compute the average for each metric within the fold
        for metric in metrics:
            # Sort by the current metric in descending order and select the top n species-drug pairs
            top_n_group = group.sort_values(by=metric, ascending=False).head(n)
            # Compute the average for the current metric
            result[f'average_{metric}'] = top_n_group[metric].mean()
        
        # Store the result for this fold
        fold_results.append(result)
    
    # Step 4: Convert fold results into a DataFrame
    fold_results_df = pd.DataFrame(fold_results)
    
    # Step 5: Group by model and preprocessing to calculate mean and std across folds for each metric
    agg_dict = {f'average_{metric}': ['mean', 'std'] for metric in metrics}
    final_results = fold_results_df.groupby(['model', 'preprocessing']).agg(agg_dict).reset_index()

    # Flatten the MultiIndex columns
    final_results.columns = ['_'.join(col).strip('_') for col in final_results.columns.values]

    return final_results




# Aggregate metrics across species-drug pairs
def aggregate_metrics(metrics):
    """
    Aggregate metrics across all species-drug pairs.
    """
    if len(metrics) == 0:
        return {'auprc': None, 'mcc': None, 'balanced_accuracy': None}

    auprc = np.mean([m['auprc'] for m in metrics])
    mcc = np.mean([m['mcc'] for m in metrics])
    balanced_acc = np.mean([m['balanced_accuracy'] for m in metrics])
    return {'auprc': auprc, 'mcc': mcc, 'balanced_accuracy': balanced_acc}


def process_fold(inner_train_pairs, inner_validation_pairs, data, feature_removals, preproc_func, preproc_params, model_grid, model_params):
    """
    Process a single fold, train the model, and compute aggregated metrics for species-drug pairs.
    """
    # Prepare inner training and validation sets
    inner_train_set = filter_data(data, inner_train_pairs[['sample_id', 'drug']])
    inner_validation_set = filter_data(data, inner_validation_pairs[['sample_id', 'drug']])

    X_inner_train = inner_train_set.drop(columns=feature_removals)
    y_inner_train = LabelEncoder().fit_transform(inner_train_set['response'])

    X_inner_validation = inner_validation_set.drop(columns=feature_removals)
    y_inner_validation = inner_validation_set['response']
    species_validation = inner_validation_set['species']
    drug_validation = inner_validation_set['drug']

    # Preprocess data
    if preproc_func == mutual_information_feature_selection:
        X_inner_train_preprocessed, X_inner_validation_preprocessed, _ = preproc_func(
            X_inner_train, y_inner_train, X_inner_validation, columns_prefix='X', **preproc_params)
    else:
        X_inner_train_preprocessed, X_inner_validation_preprocessed, _ = preproc_func(
            X_inner_train, X_inner_validation, **preproc_params)

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_grid['model'])
    ])
    pipeline.set_params(**model_params)
    pipeline.fit(X_inner_train_preprocessed, y_inner_train)
    y_inner_validation_pred = pipeline.predict_proba(X_inner_validation_preprocessed)[:, 1]

    # Compute species-drug level metrics
    metrics = compute_species_drug_metrics(
        y_true=y_inner_validation.values,
        y_pred=y_inner_validation_pred,
        species=species_validation.values,
        drug=drug_validation.values
    )

    # Aggregate metrics for the fold
    aggregated_metrics = aggregate_metrics(metrics)

    # Return the aggregated metrics for this fold
    return aggregated_metrics

# Compute metrics at the species-drug level
def compute_species_drug_metrics(y_true, y_pred, species, drug):
    """
    Compute metrics (AUPRC, MCC, Balanced Accuracy) for each species-drug pair.
    """
    unique_combinations = set(zip(species, drug))
    metrics = []
    for sp, dr in unique_combinations:
        mask = (species == sp) & (drug == dr)
        y_true_sp_dr = y_true[mask]
        y_pred_sp_dr = y_pred[mask]

        # Skip if only one class is present (MCC and balanced accuracy won't be valid)
        if len(set(y_true_sp_dr)) < 2:
            continue

        # Compute metrics
        precision, recall, _ = precision_recall_curve(y_true_sp_dr, y_pred_sp_dr)
        auprc = auc(recall, precision)
        mcc = matthews_corrcoef(y_true_sp_dr, (y_pred_sp_dr > 0.5).astype(int))
        balanced_acc = balanced_accuracy_score(y_true_sp_dr, (y_pred_sp_dr > 0.5).astype(int))

        metrics.append({
            'species': sp,
            'drug': dr,
            'auprc': auprc,
            'mcc': mcc,
            'balanced_accuracy': balanced_acc
        })

    return metrics

# Function to load pairs
def load_pairs(file_path):
    return pd.read_csv(file_path)

# Function to filter data based on pairs
def filter_data(data, pairs):
    return data.merge(pairs, on=['sample_id', 'drug'])

# Function to compute the average for the top 10 species-drug pairs
def compute_top_10_averages(group):
    # Sort by the primary metric (e.g., 'auprc') and take the top 10
    top_10 = group.nlargest(10, 'mcc')
    # Compute averages of the metrics for the top 10 pairs
    return pd.Series({
        'average_auprc': top_10['auprc'].mean(),
        'std_auprc': top_10['auprc'].std(),
        'average_mcc': top_10['mcc'].mean(),
        'std_mcc': top_10['mcc'].std(),
        'average_balanced_accuracy': top_10['balanced_accuracy'].mean(),
        'std_balanced_accuracy': top_10['balanced_accuracy'].std()
     })

def summarize_and_plot_results(
    intermediate_results_dir,
    results_dir,
    plot_dir,
    functions,
    top_10_metric_file='top_10_average_scores_species_drug_level.csv',
    all_metric_file='average_scores_species_drug_level.csv',
    consolidated_file='final_results_species_drug_level.csv'
):
    import glob
    import os

    # Combine intermediate result CSVs
    all_result_files = glob.glob(f"{intermediate_results_dir}/results_*.csv")
    all_results = functions.pd.concat([functions.pd.read_csv(f) for f in all_result_files], ignore_index=True)

    # Save combined results
    final_results_path = os.path.join(results_dir, consolidated_file)
    all_results.to_csv(final_results_path, index=False)
    print(f"Consolidated results saved to {final_results_path}")

    # Compute average per outer fold
    avg_per_fold = all_results.groupby(
        ['outer_fold', 'preprocessing', 'model']
    ).agg(
        average_auprc=('auprc', 'mean'),
        average_mcc=('mcc', 'mean'),
        average_balanced_accuracy=('balanced_accuracy', 'mean'),
    ).reset_index()

    # Compute overall averages across folds
    avg_across_folds = avg_per_fold.groupby(
        ['preprocessing', 'model']
    ).agg(
        mean_auprc=('average_auprc', 'mean'),
        std_auprc=('average_auprc', 'std'),
        mean_mcc=('average_mcc', 'mean'),
        std_mcc=('average_mcc', 'std'),
        mean_balanced_accuracy=('average_balanced_accuracy', 'mean'),
        std_balanced_accuracy=('average_balanced_accuracy', 'std')
    ).reset_index()

    # Save overall averages
    all_metric_path = os.path.join(results_dir, all_metric_file)
    avg_across_folds.to_csv(all_metric_path, index=False)
    print(f"Averages across folds saved to {all_metric_path}")

    # Compute top 10 fold-wise averages
    top_10_avg_per_fold = all_results.groupby(
        ['outer_fold', 'preprocessing', 'model']
    ).apply(functions.compute_top_10_averages).reset_index()

    # Average across folds for top 10
    top_10_avg_across_folds = top_10_avg_per_fold.groupby(
        ['preprocessing', 'model']
    ).agg(
        mean_auprc=('average_auprc', 'mean'),
        std_auprc=('average_auprc', 'std'),
        mean_mcc=('average_mcc', 'mean'),
        std_mcc=('average_mcc', 'std'),
        mean_balanced_accuracy=('average_balanced_accuracy', 'mean'),
        std_balanced_accuracy=('average_balanced_accuracy', 'std')
    ).reset_index()

    # Save top 10 results
    top_10_path = os.path.join(results_dir, top_10_metric_file)
    top_10_avg_across_folds.to_csv(top_10_path, index=False)
    print(f"Top 10 average scores saved to {top_10_path}")

    # Preprocessing and model order for plots
    preprocessing_order = ["No dimensionality reduction", "PCA on MS", "Masked Autoencoder", "MI"]
    model_order = ['Logistic Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    metrics = ['mcc', 'auprc', 'balanced_accuracy']

    # Plotting top 10
    top_10_df = functions.pd.read_csv(top_10_path)
    top_10_df['preprocessing'] = top_10_df['preprocessing'].replace('No Preprocessing', 'No dimensionality reduction')
    top_10_pdf = os.path.join(plot_dir, "top_n_metrics_plots.pdf")
    with functions.PdfPages(top_10_pdf) as pdf:
        for metric in metrics:
            functions.plot_top_n_metric2(top_10_df, metric, pdf)
    print(f"Top 10 plots saved to {top_10_pdf}")

    # Plotting all models
    all_df = functions.pd.read_csv(all_metric_path)
    all_df['preprocessing'] = all_df['preprocessing'].replace('No Preprocessing', 'No dimensionality reduction')
    all_pdf = os.path.join(plot_dir, "top_all_metrics_plots.pdf")
    with functions.PdfPages(all_pdf) as pdf:
        for metric in metrics:
            functions.plot_top_n_metric2(all_df, metric, pdf)
    print(f"All model plots saved to {all_pdf}")


def run_and_evaluate_model_across_views(
    data_path,
    nested_folds_path,
    best_model,
    best_preproc_func,
    best_preproc_params,
    feature_removals,
    feature_removals_withoutDrugClass,
    min_samples,
    results_directory,
    functions
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(results_directory, exist_ok=True)

    # Load data and folds
    data = functions.pd.read_csv(data_path)
    with open(nested_folds_path, 'rb') as file:
        nested_folds = functions.pickle.load(file)

    def preprocess_and_train(X_train, y_train, X_test):
        if best_preproc_func == functions.mutual_information_feature_selection:
            X_train_pre, X_test_pre, _ = best_preproc_func(X_train, y_train, X_test, columns_prefix='X', **best_preproc_params)
        else:
            X_train_pre, X_test_pre, _ = best_preproc_func(X_train, X_test, **best_preproc_params)

        pipeline = functions.Pipeline([
            ('scaler', functions.StandardScaler()),
            ('model', best_model)
        ])
        pipeline.fit(X_train_pre, y_train)
        y_pred = pipeline.predict_proba(X_test_pre)[:, 1]
        return y_pred

    def save_results(results, filename):
        df = functions.pd.DataFrame(results)
        df.to_csv(os.path.join(results_directory, filename), index=False)
        print(f"Saved to {filename}")

    def prepare_X_y(dataset, features_to_drop, one_hot_drug_class=False):
        if one_hot_drug_class:
            drug_class_encoded = functions.pd.get_dummies(dataset['drug_class'], prefix='drug_class')
            dataset = dataset.drop(columns=features_to_drop + ['drug_class'])
            return functions.pd.concat([dataset, drug_class_encoded], axis=1)
        return dataset.drop(columns=features_to_drop)
   
    def run_by(granularity, filename, include_drug_class=False, by_species=False, by_drug=False):
        test_results = []
        for fold_idx, (inner_folds, test_pairs) in enumerate(nested_folds):
            print(f"Fold {fold_idx + 1}")
            test_data = functions.filter_data(data, test_pairs[['sample_id', 'drug']])
            outer_train_pairs = functions.pd.concat([tr for tr, _ in inner_folds], ignore_index=True)
            train_data = functions.filter_data(data, outer_train_pairs[['sample_id', 'drug']])

            test_groups = test_data.groupby(granularity) if granularity else [('all', test_data)]

            for key, test_group in test_groups:
                if by_species and len(train_data[train_data['species'] == key]) < min_samples:
                    continue
                if by_drug and len(train_data[train_data['drug'] == key]) < min_samples:
                    continue

                sub_train = train_data.copy()
                if by_species:
                    sub_train = sub_train[sub_train['species'] == key]
                if by_drug:
                    sub_train = sub_train[sub_train['drug'] == key]

                if len(sub_train['response'].unique()) < 2:
                    continue

                features = feature_removals_withoutDrugClass if include_drug_class else feature_removals
                X_test = prepare_X_y(test_group, features, include_drug_class)
                X_train = prepare_X_y(sub_train, features, include_drug_class)
                X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

                y_train = functions.LabelEncoder().fit_transform(sub_train['response'])
                y_test = test_group['response']
                species = test_group['species']
                drug = test_group['drug']

                y_pred = preprocess_and_train(X_train, y_train, X_test)

                metrics = functions.compute_species_drug_metrics(
                    y_true=y_test.values,
                    y_pred=y_pred,
                    species=species.values,
                    drug=drug.values
                )

                for m in metrics:
                    # Filtering by imbalance and resistant count
                    sp = m['species']
                    dr = m['drug']
                    mask = (test_group['species'] == sp) & (test_group['drug'] == dr)
                    subgroup = test_group[mask]

                    total = len(subgroup)
                    resistant_count = (subgroup['response'] == 1).sum()
                    susceptible_count = (subgroup['response'] == 0).sum()

                    if resistant_count < 2:
                        continue

                    resistance_frac = resistant_count / total
                    susceptible_frac = susceptible_count / total
                    if resistance_frac > 0.95 or susceptible_frac > 0.95:
                        continue

                    test_results.append({
                        'outer_fold': fold_idx + 1,
                        'species': sp,
                        'drug': dr,
                        'auprc': m['auprc'],
                        'mcc': m['mcc'],
                        'balanced_accuracy': m['balanced_accuracy']
                    })

        save_results(test_results, filename)

    # Run all 6 views
    run_by(granularity=None, filename='test_results_all.csv')
    run_by(granularity=None, filename='test_results_all_withDrugClass.csv', include_drug_class=True)
    run_by(granularity='species', filename='test_results_per_species.csv', by_species=True)
    run_by(granularity='species', filename='test_results_per_species_withDrugClass.csv', by_species=True, include_drug_class=True)
    run_by(granularity='drug', filename='test_results_per_drug.csv', by_drug=True)
    run_by(granularity=['species', 'drug'], filename='test_results_species_drug_pairs.csv')

    # Combine all result files
    def load_and_tag_results():
        file_types = {
            'species_drug': 'test_results_species_drug_pairs.csv',
            'drug': 'test_results_per_drug.csv',
            'species': 'test_results_per_species.csv',
            'species with drug class': 'test_results_per_species_withDrugClass.csv',
            'all': 'test_results_all.csv',
            'all with drug class': 'test_results_all_withDrugClass.csv'
        }

        dfs = []
        for t, fname in file_types.items():
            df = functions.pd.read_csv(os.path.join(results_directory, fname))
            df['type'] = t
            dfs.append(df)
        return functions.pd.concat(dfs, ignore_index=True)

    all_results = load_and_tag_results()

    # Filter only complete cases
    counts = all_results.groupby(['species', 'drug', 'outer_fold'])['type'].transform('nunique')
    all_types_count = all_results['type'].nunique()
    consistent_results = all_results[counts == all_types_count]

    # Compute stats
    perf_cols = ['mcc', 'auprc', 'balanced_accuracy']
    fold_stats = consistent_results.groupby(['outer_fold', 'type'])[perf_cols].mean().reset_index()
    summary = fold_stats.groupby('type')[perf_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()

    # Save summary
    summary_file = os.path.join(results_directory, 'summary_stats_per_type.csv')
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Plotting
    plot_dir = os.path.join(results_directory, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for metric in ['auprc', 'mcc', 'balanced_accuracy']:
        plt.figure(figsize=(10, 6))
        plt.bar(summary['type'], summary[f'{metric}_mean'], yerr=summary[f'{metric}_std'], capsize=5)
        plt.xlabel('Evaluation Type')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} by Evaluation Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{metric}_per_type.png'))
        plt.close()


def analyze_species_drug_performance(results_directory, functions, mcc_threshold=0.4):
    import os

    # Define file mapping
    file_map = {
        'species_drug': 'test_results_species_drug_pairs.csv',
        'drug': 'test_results_per_drug.csv',
        'species': 'test_results_per_species.csv',
        'species with drug class': 'test_results_per_species_withDrugClass.csv',
        'all': 'test_results_all.csv',
        'all with drug class': 'test_results_all_withDrugClass.csv'
    }

    # Load and tag results
    all_results = []
    for result_type, filename in file_map.items():
        df = functions.pd.read_csv(os.path.join(results_directory, filename))
        df['type'] = result_type
        all_results.append(df)
    all_results = functions.pd.concat(all_results, ignore_index=True)

    # Compute average and std performance per type/species/drug
    average_scores = all_results.groupby(['type', 'species', 'drug']).agg(
        avg_auprc=('auprc', 'mean'),
        avg_mcc=('mcc', 'mean'),
        avg_balanced_accuracy=('balanced_accuracy', 'mean'),
        std_auprc=('auprc', 'std'),
        std_mcc=('mcc', 'std'),
        std_balanced_accuracy=('balanced_accuracy', 'std'),
    ).reset_index()

    # Save average scores
    avg_path = os.path.join(results_directory, 'average_scores.csv')
    average_scores.to_csv(avg_path, index=False)
    print(f"Average scores saved to {avg_path}")

    # Filter species-drug pairs based on performance threshold
    filtered_results = average_scores.groupby(['species', 'drug']).filter(
        lambda g: any(g['avg_mcc'] >= mcc_threshold)
    )

    # Add label for plotting
    filtered_results['species_drug_pair'] = filtered_results['species'] + ' - ' + filtered_results['drug']

    # Plotting
    performance_measures = {
        'AUPRC': ('avg_auprc', 'std_auprc'),
        'MCC': ('avg_mcc', 'std_mcc'),
        'Balanced Accuracy': ('avg_balanced_accuracy', 'std_balanced_accuracy')
    }
    palette = ["#008B8B", "#FF8C00", "#006400", "#1E90FF", "#800080", "#FFD700"]
    types = filtered_results['type'].unique()
    species_drug_pairs = filtered_results['species_drug_pair'].unique()

    output_pdf_path = results_directory / "performancePerInputType_plots.pdf"
    
    with functions.PdfPages(output_pdf_path) as pdf:
        for measure, (mean_col, std_col) in performance_measures.items():
            fig, ax = functions.plt.subplots(figsize=(20, 10))
            n_types = len(types)
            n_pairs = len(species_drug_pairs)
            bar_width = 0.1
            index = functions.np.arange(n_pairs)

            for i, t in enumerate(types):
                subset = filtered_results[filtered_results['type'] == t]
                subset = subset.set_index('species_drug_pair').reindex(species_drug_pairs).reset_index()
                means = subset[mean_col]
                stds = subset[std_col]
                positions = index + i * bar_width - (n_types * bar_width) / 2 + bar_width / 2

                ax.bar(
                    positions,
                    means,
                    bar_width,
                    label=t,
                    color=palette[i % len(palette)],
                    yerr=stds,
                    capsize=4
                )

            ax.set_xlabel('Species - Drug Pair')
            ax.set_ylabel(measure)
            ax.set_title(f'{measure} by Species-Drug Pair and Model Type')
            ax.set_xticks(index)
            ax.set_xticklabels(species_drug_pairs, rotation=90, fontsize=9)
            ax.legend(title='Model Type')
            functions.plt.tight_layout()
            pdf.savefig(fig)
            functions.plt.close(fig)

    print(f"Plots saved to {output_pdf_path}")

    # Identify species-drug pairs passing threshold for each type
    identified_pairs = {}
    for metric_type in file_map:
        filtered_type = filtered_results[filtered_results['type'] == metric_type]
        passing = filtered_type[filtered_type['avg_mcc'] > mcc_threshold]
        pairs = passing['species_drug_pair'].unique()
        identified_pairs[metric_type] = pairs
        print(f"Species-Drug Pairs for Type '{metric_type}':\n{pairs}\n")

    # Get top 10 per type
    top_pairs = {}
    for metric_type in file_map:
        top = (
            filtered_results[filtered_results['type'] == metric_type]
            .sort_values(by="avg_mcc", ascending=False)
            .loc[:, ['species_drug_pair', 'avg_mcc']]
            .head(10)
        )
        top_pairs[metric_type] = top
        print(f"Top 10 Species-Drug Pairs for '{metric_type}':\n{top}\n")

    return {
        "average_scores": average_scores,
        "filtered_results": filtered_results,
        "identified_pairs": identified_pairs,
        "top_pairs": top_pairs
    }


def deploy_and_evaluate_model(
    data_path,
    deployment_path,
    results_directory,
    best_model,
    best_preproc_func,
    best_preproc_params,
    feature_removals,
    functions,
    min_samples=5
):
    import os
    from sklearn.metrics import (
        confusion_matrix, matthews_corrcoef,
        balanced_accuracy_score, average_precision_score
    )

    os.makedirs(results_directory, exist_ok=True)

    # Load datasets
    data = functions.pd.read_csv(data_path)
    deployment_set = functions.pd.read_csv(deployment_path)

    # Split data
    train_set = data[~data['sample_id'].isin(deployment_set['sample_id'])]
    deployment_filtered = data[data['sample_id'].isin(deployment_set['sample_id'])]

    # Extract features and labels
    X_train = train_set.drop(columns=feature_removals)
    y_train = train_set['response']
    # Only keep columns in feature_removals that are present in deployment_filtered (useful for external validation)
    columns_to_remove = [col for col in feature_removals if col in deployment_filtered.columns]
    # Drop those columns
    X_deploy = deployment_filtered.drop(columns=columns_to_remove)
    print(X_deploy.shape)
    y_deploy = deployment_filtered['response']
    species_deploy = deployment_filtered['species']
    drug_deploy = deployment_filtered['drug']

    # Encode labels
    y_train_encoded = functions.LabelEncoder().fit_transform(y_train)

    # Preprocessing
    if best_preproc_func == functions.mutual_information_feature_selection:
        X_train_pre, X_deploy_pre, _ = best_preproc_func(
            X_train, y_train_encoded, X_deploy, columns_prefix='X', **best_preproc_params
        )
    else:
        X_train_pre, X_deploy_pre, _ = best_preproc_func(
            X_train, X_deploy, **best_preproc_params
        )

    # Train pipeline
    pipeline = functions.Pipeline([
        ('scaler', functions.StandardScaler()),
        ('model', best_model)
    ])
    pipeline.fit(X_train_pre, y_train_encoded)

    # Predict
    y_pred_prob = pipeline.predict_proba(X_deploy_pre)[:, 1]
    y_pred = pipeline.predict(X_deploy_pre)

    # Store predictions
    sample_results = functions.pd.DataFrame({
        "sample_id": deployment_filtered["sample_id"],
        "species": species_deploy,
        "drug": drug_deploy,
        "response": y_deploy,
        "predicted_probability": y_pred_prob,
        "predicted_labels": y_pred
    })
    sample_results.to_csv(f"{results_directory}/sample_specific_results.csv", index=False)
    print("Sample-specific results exported.")

    # Evaluate per species-drug pair
    metrics = []
    for species in species_deploy.unique():
        for drug in deployment_filtered[deployment_filtered['species'] == species]['drug'].unique():
            mask = (species_deploy == species) & (drug_deploy == drug)
            y_true, y_prob, y_bin = y_deploy[mask], y_pred_prob[mask], y_pred[mask]

            if len(y_true) == 0 or len(functions.np.unique(y_true)) < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            vme = fn / (fn + tp) if tp > 0 else 'NA'
            me = fp / (tn + fp) if tn > 0 else 'NA'

            metrics.append({
                "species": species,
                "drug": drug,
                "auprc": average_precision_score(y_true, y_prob),
                "auroc": roc_auc_score(y_true, y_prob),
                "mcc": matthews_corrcoef(y_true, y_bin),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_bin),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "very_major_error": vme,
                "major_error": me,
                "imbalance": y_true.mean(),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "num_observations": len(y_true)
            })

    metrics_df = functions.pd.DataFrame(metrics)
    metrics_df.to_csv(f"{results_directory}/aggregated_metrics.csv", index=False)
    print("Aggregated metrics exported.")

    # Filter results for valid performance stats
    filtered_metrics = metrics_df[
        ((metrics_df['true_positives'] + metrics_df['false_negatives']) > min_samples) &
        (metrics_df['imbalance'].between(0.05, 0.95))
    ]

    # Plot MCC
    plot_df = filtered_metrics.sort_values(by='mcc', ascending=False)
    plot_df['Species-Drug Pair'] = plot_df['species'] + ' - ' + plot_df['drug']

    functions.plt.figure(figsize=(20, 10))
    functions.sns.barplot(
        x='mcc',
        y='Species-Drug Pair',
        data=plot_df,
        palette=["#008B8B"]
    )
    functions.plt.xlabel('MCC')
    functions.plt.ylabel('Species-Drug Pair')
    functions.plt.title('MCC per Species-Drug Pair')
    functions.plt.tight_layout()

    plot_path = f"{results_directory}/all_mcc_species_drug_plot.png"
    functions.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    functions.plt.show()
    print(f"Plot saved to {plot_path}")

    # Summary stats
    print("Mean MCC per species:")
    print(filtered_metrics.groupby("species")["mcc"].mean().sort_values(ascending=False).reset_index())

    print("\nMean MCC per drug:")
    print(metrics_df.groupby("drug")["mcc"].mean().sort_values(ascending=False).reset_index())




def deploy_and_evaluate_model_external(
    data_path,
    deployment_path,
    results_directory,
    best_model,
    best_preproc_func,
    best_preproc_params,
    feature_removals,
    functions,
    min_samples=5
):
    import os
    from sklearn.metrics import (
        confusion_matrix, matthews_corrcoef,
        balanced_accuracy_score, average_precision_score
    )

    os.makedirs(results_directory, exist_ok=True)

    # Load datasets
    train_set = functions.pd.read_csv(data_path)
    deployment_filtered = functions.pd.read_csv(deployment_path)

    # Extract features and labels
    X_train = train_set.drop(columns=feature_removals)
    y_train = train_set['response']
    # Only keep columns in feature_removals that are present in deployment_filtered (useful for external validation)
    columns_to_remove = [col for col in feature_removals if col in deployment_filtered.columns]
    # Drop those columns
    X_deploy = deployment_filtered.drop(columns=columns_to_remove)
    print(X_deploy.shape)
    y_deploy = deployment_filtered['response']
    species_deploy = deployment_filtered['species']
    drug_deploy = deployment_filtered['drug']

    # Encode labels
    y_train_encoded = functions.LabelEncoder().fit_transform(y_train)

    # Preprocessing
    if best_preproc_func == functions.mutual_information_feature_selection:
        X_train_pre, X_deploy_pre, _ = best_preproc_func(
            X_train, y_train_encoded, X_deploy, columns_prefix='X', **best_preproc_params
        )
    else:
        X_train_pre, X_deploy_pre, _ = best_preproc_func(
            X_train, X_deploy, **best_preproc_params
        )

    # Train pipeline
    pipeline = functions.Pipeline([
        ('scaler', functions.StandardScaler()),
        ('model', best_model)
    ])
    pipeline.fit(X_train_pre, y_train_encoded)

    # Predict
    y_pred_prob = pipeline.predict_proba(X_deploy_pre)[:, 1]
    y_pred = pipeline.predict(X_deploy_pre)

    # Store predictions
    sample_results = functions.pd.DataFrame({
        "sample_id": deployment_filtered["sample_id"],
        "species": species_deploy,
        "drug": drug_deploy,
        "response": y_deploy,
        "predicted_probability": y_pred_prob,
        "predicted_labels": y_pred
    })
    sample_results.to_csv(f"{results_directory}/sample_specific_results_external.csv", index=False)
    print("Sample-specific results exported.")

    # Evaluate per species-drug pair
    metrics = []
    for species in species_deploy.unique():
        for drug in deployment_filtered[deployment_filtered['species'] == species]['drug'].unique():
            mask = (species_deploy == species) & (drug_deploy == drug)
            y_true, y_prob, y_bin = y_deploy[mask], y_pred_prob[mask], y_pred[mask]

            if len(y_true) == 0 or len(functions.np.unique(y_true)) < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            vme = fn / (fn + tp) if tp > 0 else 'NA'
            me = fp / (tn + fp) if tn > 0 else 'NA'

            metrics.append({
                "species": species,
                "drug": drug,
                "auprc": average_precision_score(y_true, y_prob),
                "auroc": roc_auc_score(y_true, y_prob),
                "mcc": matthews_corrcoef(y_true, y_bin),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_bin),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "very_major_error": vme,
                "major_error": me,
                "imbalance": y_true.mean(),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "num_observations": len(y_true)
            })

    metrics_df = functions.pd.DataFrame(metrics)
    metrics_df.to_csv(f"{results_directory}/aggregated_metrics_external.csv", index=False)
    print("Aggregated metrics exported.")

    # Filter results for valid performance stats
    filtered_metrics = metrics_df[
        ((metrics_df['true_positives'] + metrics_df['false_negatives']) > min_samples) &
        (metrics_df['imbalance'].between(0.05, 0.95))
    ]

    # Plot MCC
    plot_df = filtered_metrics.sort_values(by='mcc', ascending=False)
    plot_df['Species-Drug Pair'] = plot_df['species'] + ' - ' + plot_df['drug']

    functions.plt.figure(figsize=(20, 10))
    functions.sns.barplot(
        x='mcc',
        y='Species-Drug Pair',
        data=plot_df,
        palette=["#008B8B"]
    )
    functions.plt.xlabel('MCC')
    functions.plt.ylabel('Species-Drug Pair')
    functions.plt.title('MCC per Species-Drug Pair')
    functions.plt.tight_layout()

    plot_path = f"{results_directory}/all_mcc_species_drug_plot_external.png"
    functions.plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    functions.plt.show()
    print(f"Plot saved to {plot_path}")

    # Summary stats
    print("Mean MCC per species:")
    print(filtered_metrics.groupby("species")["mcc"].mean().sort_values(ascending=False).reset_index())

    print("\nMean MCC per drug:")
    print(metrics_df.groupby("drug")["mcc"].mean().sort_values(ascending=False).reset_index())




def evaluate_clinical_guideline_effectiveness(
    data_path,
    results_directory,
    functions
):
    import pandas as pd
    import os
    from collections import Counter

    os.makedirs(results_directory, exist_ok=True)

    ##############################
    # Recreate Antibiotika Table #
    ##############################
    drug = ["Fluconazole", "Itraconazole", "Voriconazole", "Posaconazole", "Isavuconazole",
            "Caspofungin", "Anidulafungin", "Amphotericin B"]
    
    species = ["candida albicans", "candida dubliniensis", "candida glabrata", "candida guilliermondii",
               "candida krusei", "candida lusitaniae", "candida parapsilosis", "candida tropicalis",
               "cryptococcus sp.", "aspergillus fumigatus", "aspergillus terreus", "aspergillus flavus",
               "mucormycosis", "scedosporium apiospermum", "scedosporium prolificans", "fusarium sp.",
               "histoplasma capsulatum", "coccidioides immitis", "blastomyces dermatidis", "sporothrix schenckii"]

    ranking = functions.np.array([
        [1] + [2]*7,
        [1, 2, 2, 2, 2, 1, 1, 1],
        [3]*5 + [1]*3,
        [3, 1, 1, 1, 2, 2, 2, 1],
        [4, 4, 2, 2, 2, 1, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 3],
        [1] + [2]*6 + [1],
        [3]*5 + [1]*3,
        [1, 2, 2, 2, 2, 4, 4, 1],
        [4, 2, 1, 1, 1, 3, 3, 2],
        [4, 3, 1, 2, 1, 3, 3, 4],
        [4, 3, 1, 2, 1, 3, 3, 3],
        [4, 4, 4, 2, 2, 4, 4, 1],
        [4, 4, 2, 3, 2, 4, 4, 4],
        [4, 4, 3, 4, 4, 4, 4, 4],
        [4, 4, 3, 3, 3, 4, 4, 3],
        [3, 1, 2, 2, 10, 4, 4, 1],
        [1, 1, 2, 2, 10, 4, 4, 1],
        [3, 1, 2, 2, 10, 4, 4, 1],
        [3, 1, 3, 2, 10, 4, 4, 1]
    ])
    
    ranking_df = functions.pd.DataFrame(ranking, index=species, columns=drug)
    
    top_recommendations = [
        {"species": sp, "drug": dr}
        for sp in ranking_df.index
        for dr in ranking_df.columns
        if ranking_df.loc[sp, dr] == ranking_df.loc[sp, :].min()
    ]
    top_guidelines = functions.pd.DataFrame(top_recommendations)

    ##############################
    # Merge With Prediction Data #
    ##############################
    prediction_path=results_directory / "sample_specific_results.csv"
    GT = functions.pd.read_csv(prediction_path)
    full_MS = functions.pd.read_csv(data_path)
    full_MS = full_MS[["drug", "sample_id", "species", "response"]]
    full_MS = full_MS[full_MS["sample_id"].isin(GT["sample_id"].unique())]

    GT = functions.pd.merge(full_MS, GT, on=["drug", "sample_id", "species", "response"], how="left")
    merged = functions.pd.merge(top_guidelines, GT, on=["species", "drug"])

    ##############################
    # Summary Statistics         #
    ##############################
    print(f"Number of common drugs: {len(set(drug) & set(GT['drug'].unique()))}")
    print(f"Number of common species: {len(set(species) & set(GT['species'].unique()))}")
    print(f"Unique samples in validation set: {merged['sample_id'].nunique()}")

    # Wrong guideline predictions
    resistance = merged[merged["response"] == 1]
    wrong_counts = resistance.groupby("sample_id").size()
    print(f"Samples with at least one resistant response: {len(wrong_counts)}")

    # Alternative options
    resistant_sample_ids = resistance["sample_id"].unique()
    GT_wrong = GT[GT["sample_id"].isin(resistant_sample_ids)]
    alt_effective = GT_wrong.groupby("sample_id")["response"].apply(lambda x: (x == 0).any()).sum()
    print(f"Samples with an alternative effective drug: {alt_effective}")

    # Model usefulness
    detected = resistance[resistance["predicted_labels"] == 1]
    not_detected = resistance[resistance["predicted_labels"] == 0]
    no_prediction = resistance["predicted_labels"].isna().sum()

    print(f"Model correctly detected resistance: {len(detected)} / {len(resistance)} ({(len(detected)/len(resistance))*100:.2f}%)")
    print(f"Model missed resistance: {len(not_detected)} / {len(resistance)} ({(len(not_detected)/len(resistance))*100:.2f}%)")
    print(f"No prediction made: {no_prediction} / {len(resistance)} ({(no_prediction/len(resistance))*100:.2f}%)")

    # Model harm: predicting resistance when it's actually sensitivity
    sensitivity = merged[merged["response"] == 0]
    wrong_resistance = sensitivity[sensitivity["predicted_labels"] == 1]
    print(f"Model predicted resistance instead of sensitivity: {len(wrong_resistance)} / {len(sensitivity)} ({(len(wrong_resistance)/len(sensitivity))*100:.2f}%)")

    # Alternative top drug check
    alternative_found = sum(
        1 for sid in wrong_resistance["sample_id"].unique()
        if not merged[(merged["sample_id"] == sid) & (merged["predicted_labels"] != 1)].empty
    )
    print(f"Wrongly rejected top drug but another top drug predicted sensitive: {alternative_found}")

    ##############################
    # Observation Count Analysis #
    ##############################
    species_drug_interest = [
        ("candida parapsilosis", "Fluconazole"),
        ("candida tropicalis", "Anidulafungin"),
        ("candida albicans", "Fluconazole"),
        ("candida dubliniensis", "Fluconazole"),
        ("candida glabrata", "Anidulafungin"),
        ("candida glabrata", "Caspofungin"),
    ]
    
    full_data = pd.read_csv(data_path)
    counts = {
        f"{sp}-{dr}": len(full_data[(full_data["species"] == sp) & (full_data["drug"] == dr)])
        for sp, dr in species_drug_interest
    }
    counts_df = pd.DataFrame(counts.items(), columns=["Species-Drug Combination", "Observation Count"])
    print(counts_df)


def generate_shap_explanations_for_selected_cases(
    data_path,
    deployment_path,
    output_dir,
    functions,
    best_model,
    best_preproc_func,
    best_preproc_params,
    feature_removals,
    species_drug_filter,
    top_n_features=500
):
    import shap
    from fastshap import KernelExplainer
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = functions.pd.read_csv(data_path)
    deployment_set_ids = functions.pd.read_csv(deployment_path)
    train_set = data[~data['sample_id'].isin(deployment_set_ids['sample_id'])]
    deployment_set = data[data['sample_id'].isin(deployment_set_ids['sample_id'])]

    # Filter test set based on species-drug pairs
    conditions = functions.pd.Series(False, index=deployment_set.index)
    for sp, dr in species_drug_filter:
        conditions |= ((deployment_set['species'] == sp) & (deployment_set['drug'] == dr))
    test_set = deployment_set[conditions]
    print(f"Filtered test set shape: {test_set.shape}")

    # Prepare training and test sets
    X_train = train_set.drop(columns=feature_removals)
    y_train = functions.LabelEncoder().fit_transform(train_set['response'])

    X_test = test_set.drop(columns=feature_removals)
    y_test = functions.LabelEncoder().fit_transform(test_set['response'])

    # Apply PCA or mutual information
    if best_preproc_func == functions.mutual_information_feature_selection:
        X_train_pre, X_test_pre, _ = best_preproc_func(
            X_train, y_train, X_test, columns_prefix='X', **best_preproc_params
        )
    else:
        X_train_pre, X_test_pre, loadings = best_preproc_func(
            X_train, X_test, **best_preproc_params
        )

    # Feature selection: top PCA features
    overall_importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
    top_features = overall_importance.head(top_n_features).index.tolist()

    # Recombine top features with non-PCA (drug) features
    drug_columns = [col for col in X_train.columns if not col.startswith('X')]
    X_train_final = functions.pd.concat([X_train[top_features], X_train[drug_columns]], axis=1)
    X_test_final = functions.pd.concat([X_test[top_features], X_test[drug_columns]], axis=1)

    # Train model
    pipeline = functions.Pipeline([
        ('scaler', functions.StandardScaler()),
        ('model', best_model)
    ])
    pipeline.fit(X_train_final, y_train)

    # SHAP explanation
    def pipeline_predict(X):
        if not isinstance(X, functions.pd.DataFrame):
            X = functions.pd.DataFrame(X, columns=X_test_final.columns)
        X_scaled = pipeline.named_steps['scaler'].transform(X)
        return pipeline.named_steps['model'].predict_proba(X_scaled)

    explainer = shap.KernelExplainer(pipeline_predict, X_test_final)
    shap_values = explainer.shap_values(X_test_final)
    shap_values_class1 = shap_values[0]

    # Build output
    shap_df = functions.pd.DataFrame(shap_values_class1, columns=['shap_' + col for col in X_test_final.columns])
    pred_labels = pipeline.predict(X_test_final)
    output_df = test_set[['sample_id', 'response', 'drug', 'species']].copy()
    output_df['predicted_labels'] = pred_labels
    final_df = functions.pd.concat([output_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    # Save one CSV per species-drug pair
    for sp, dr in species_drug_filter:
        df_pair = final_df[(final_df['species'] == sp) & (final_df['drug'] == dr)]
        if df_pair.empty:
            continue
        fname = f"shap_{sp.replace(' ', '_')}_{dr.replace(' ', '_')}.csv"
        path = os.path.join(output_dir, fname)
        df_pair.to_csv(path, index=False)
        print(f"Exported SHAP values for {sp} & {dr} to {path}")


def plot_metric(df, metric, metric_std, metric_label, pdf):
    """
    Generalized function to plot a metric for multiple datasets.
    """
    # Reorder the DataFrame according to the desired order
    df['preprocessing'] = pd.Categorical(df['preprocessing'], categories=preprocessing_order, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)

    # Filter the DataFrame to maintain the order
    df = df.sort_values(by=['preprocessing', 'model'])

    # Get the list of unique datasets
    datasets = df['dataset'].unique()

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(10 * len(datasets), 6), sharey=True)

    # Ensure axes is iterable if only one dataset
    if len(datasets) == 1:
        axes = [axes]

    # Iterate over each dataset to create a subplot
    for ax, dataset in zip(axes, datasets):
        # Filter the DataFrame for the current dataset
        df_filtered = df[df['dataset'] == dataset]

        # Get the unique preprocessing methods in the desired order
        preprocessing_methods = df_filtered['preprocessing'].unique()

        # Set up the bar width and positions
        bar_width = 0.2
        index = np.arange(len(df_filtered['model'].unique()))  # X positions for models
        custom_palette = sns.color_palette("hsv", len(preprocessing_methods))  # Custom palette for preprocessing

        # Add grid behind the bars
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

        # Plotting each preprocessing method's bars
        for i, (preproc_method, color) in enumerate(zip(preprocessing_methods, custom_palette)):
            df_preproc = df_filtered[df_filtered['preprocessing'] == preproc_method]
            ax.bar(
                index + i * bar_width,
                df_preproc[metric],
                bar_width,
                label=preproc_method,
                color=color,
                yerr=df_preproc[metric_std],  # Error bars
                capsize=4  # Size of the error bar caps
            )

        # Adding labels and title for the subplot
        ax.set_title(f'Dataset: {dataset}')
        ax.set_xticks(index + bar_width * (len(preprocessing_methods) - 1) / 2)
        ax.set_xticklabels(df_filtered['model'].unique(), rotation=90, fontsize=12)
        ax.set_xlabel("Models")
        if ax == axes[0]:
            ax.set_ylabel(metric_label)

    # Adding a legend to the last subplot
    axes[-1].legend(title='Preprocessing', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Add space at the top for the title

    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)


# Define the custom color palette
custom_palette = ["#008B8B", "#FF8C00", "#006400", "#1E90FF"]

def plot_top_n_metric2(df, metric, pdf, group_by=None):
    """
    Plot top N metric values across models and preprocessing methods.

    Args:
        df (pd.DataFrame): DataFrame containing 'model', 'preprocessing', and metric columns.
        metric (str): Metric name (e.g., 'auprc').
        pdf (PdfPages): PDF object to save plots.
        group_by (str or None): Optional column name to group subplots by (e.g., 'view' or None).
    """
    metric_mean = f'mean_{metric}'
    metric_std = f'std_{metric}'
    
    # Handle optional grouping
    if group_by and group_by in df.columns:
        groups = df[group_by].unique()
    else:
        groups = [None]
        df[group_by] = "All Data"

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 6), sharey=True)

    # Ensure axes is iterable
    if len(groups) == 1:
        axes = [axes]

    for ax, group_value in zip(axes, groups):
        df_filtered = df[df[group_by] == group_value] if group_by else df

        models = df_filtered['model'].unique()
        preprocessing_methods = df_filtered['preprocessing'].unique()

        bar_width = 0.2
        index = np.arange(len(models))

        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

        for i, preprocessing_method in enumerate(preprocessing_methods):
            df_prep = df_filtered[df_filtered['preprocessing'] == preprocessing_method]
            ax.bar(
                index + i * bar_width,
                df_prep[metric_mean],
                bar_width,
                label=preprocessing_method,
                yerr=df_prep[metric_std],
                capsize=4
            )

        title = f'{group_by}: {group_value}' if group_by else 'All Data'
        ax.set_title(title)
        ax.set_xticks(index + bar_width * (len(preprocessing_methods) - 1) / 2)
        ax.set_xticklabels(models, rotation=90, fontsize=12)

        if ax == axes[0]:
            ax.set_ylabel(f'Average {metric.upper()}')

        if metric.lower() == 'auprc':
            ax.set_ylim(0, 1.0)

    axes[-1].legend(title='Preprocessing')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    pdf.savefig(fig)
    plt.close(fig)

# Function to plot the results
def plot_top_n_metric3(df, metric, pdf):
    metric_mean = f'mean_{metric}'
    metric_std = f'std_{metric}'
    
    # Reorder the DataFrame according to the desired order
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # Sort the DataFrame
    df = df.sort_values(by=['dataset', 'model'])
    
    # Get the list of unique datasets in the desired order
    datasets = df['dataset'].unique()
    
    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(20, 6), sharey=True)
    
    # If there's only one dataset, wrap the axes in a list
    if len(datasets) == 1:
        axes = [axes]
    
    # Iterate over each dataset to create a subplot
    for ax, dataset in zip(axes, datasets):
        # Filter the DataFrame for the current dataset
        df_filtered = df[df['dataset'] == dataset]
    
        # Get the unique preprocessing methods in the order they appear
        preprocessing_methods = df_filtered['preprocessing'].unique()
    
        # Set up the bar width and positions
        bar_width = 0.2
        index = np.arange(len(df_filtered['model'].unique()))  # X positions for models
    
        # Add grid behind the bars
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)
    
        # Plotting each preprocessing method's bars using the custom palette
        for i, (preprocessing_method, color) in enumerate(zip(preprocessing_methods, custom_palette)):
            df_prep = df_filtered[df_filtered['preprocessing'] == preprocessing_method]
            ax.bar(
                index + i * bar_width,
                df_prep[metric_mean],
                bar_width,
                label=preprocessing_method,
                color=color,
                yerr=df_prep[metric_std],  # Error bars
                capsize=4  # Size of the error bar caps
            )
    
        # Adding labels and title for the subplot
        ax.set_title(f'Dataset: {dataset}')
        ax.set_xticks(index + bar_width * (len(preprocessing_methods) - 1) / 2)
        ax.set_xticklabels(df_filtered['model'].unique(), rotation=90, fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel(f'Average {metric.upper()}')
    
        # Conditionally set y-axis limits if the metric is AUPRC
        if metric.lower() == 'auprc':
            ax.set_ylim(0, 1.0)
    
    # Adding a legend to the last subplot
    axes[-1].legend(title='Preprocessing')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Add space at the top for the title
    
    # Save the figure to the PDF
    pdf.savefig(fig)
    plt.close(fig)


