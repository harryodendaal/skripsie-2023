import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


def liwc_get_correlation_matrix(full):
    liwc_columns = [col for col in full.columns if "liwc" in col.lower()]
    liwc_df = full[liwc_columns + ["label"]]
    liwc_df.head()
    # grouped = liwc_df.groupby("label")

    correlation_matrix = liwc_df[liwc_columns + ["label"]].corr()

    return correlation_matrix


def liwc_print_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        xticklabels=["lex_liwc_WC"],
        yticklabels=correlation_matrix.index,
        cbar=False,
    )
    plt.title("Correlation Heatmap")

    plt.yticks(fontsize=8)
    plt.savefig("correlationmatrix.png")
    pass


def liwc_extract_features_correlation_matrix(correlation_matrix, full):
    n = 20

    positive_correlations = correlation_matrix["label"].sort_values(ascending=False)[
        1 : n + 1
    ]

    negative_correlations = correlation_matrix["label"].sort_values(ascending=True)[:n]

    # print("Top 20 most positively correlated features:")
    # print(positive_correlations)
    # print("\nTop 20 most negatively correlated features:")
    # print(negative_correlations)

    # Combine the selected correlated features
    combined_features = pd.concat(
        [positive_correlations, negative_correlations], axis=0
    )

    # Create a DataFrame with the combined selected features
    selected_features = full[combined_features.index]

    # Calculate VIF for each feature in the combined selected features DataFrame
    vif_data = pd.DataFrame()
    vif_data["feature"] = selected_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(selected_features.values, i)
        for i in range(len(selected_features.columns))
    ]

    # Display features with high VIF
    high_vif_features = vif_data[vif_data["VIF"] > 10]

    # print("Features with high VIF (potential multicollinearity):")
    # print(high_vif_features)

    # Drop features with high VIF
    reduced_features = selected_features.drop(columns=high_vif_features["feature"])

    # Use the 'reduced_features' DataFrame for further analysis and modeling
    return reduced_features


def get_all_liwc_features(X_train_full, X_test_full):
    # Drop the "Text" column to get all other columns
    X_train_liwc = X_train_full.drop(columns=["Text"])
    X_test_liwc = X_test_full.drop(columns=["Text"])
    return X_train_liwc, X_test_liwc
