# Customer Segmentation Using K-Means Clustering

An offline-friendly, beginner-oriented machine learning practical. Upload a customer CSV or explore the included synthetic data, select meaningful numeric attributes, run K-Means, inspect the groups, and download a copy of the data with cluster labels.

## Abstract

This project demonstrates unsupervised customer segmentation with K-Means. It prepares numerical features, fills missing values with medians, standardizes the selected columns, and groups similar customer rows. The Streamlit dashboard presents an elbow curve, a two-feature cluster plot, cluster sizes, profile summaries, and a downloadable CSV. It does not need an API key or internet connection to load the included sample.

## Problem statement

Businesses often need to understand different kinds of customers to plan suitable services or communications. A customer table may contain many rows but no known segment labels. This project explores whether K-Means can organize those customers into groups based on selected attributes.

## Objectives

- Build a local web interface for exploring and uploading customer data.
- Prepare and scale selected numerical features safely.
- Apply reproducible K-Means clustering and help explore a suitable K.
- Visualize, compare, and describe the resulting customer groups.
- Export a copy of the original data with its cluster label.

## Introduction to customer segmentation

Customer segmentation divides a customer population into groups with similar characteristics. A business might use those groups to investigate patterns or tailor a later strategy. The algorithm only finds patterns in the chosen data; a person must decide whether a pattern is useful and appropriate.

## What K-Means does

K-Means is an unsupervised learning algorithm. It groups data points by distance to a group center called a **centroid**. The user chooses **K**, the number of groups. The numeric labels (0, 1, …) are arbitrary identifiers and do not imply rank or value.

### K-Means, step by step

1. Choose K starting centroids.
2. Assign each customer to its nearest centroid in the selected feature space.
3. Move each centroid to the mean position of the customers assigned to it.
4. Repeat assignment and movement until the assignments stop changing substantially.
5. Return one cluster label per row.

K-Means can find different results for different K values, features, or data. This project fixes the random seed and uses multiple initializations so a run is repeatable for the same prepared data and settings.

## Why feature scaling is needed

Features may use different units. For example, income might be in thousands while age is measured in years. Since K-Means uses distances, a large-number feature could dominate just because of its units. `StandardScaler` centers each chosen feature around zero and scales it to roughly one standard deviation. It is fitted on the dataset currently being analysed. This makes feature units more comparable; it does not make every feature equally important in a business sense.

For selected numeric columns, blank, non-numeric, and infinite values are treated as missing and replaced with that column's median. Rows are retained. If an entire selected feature has no usable numeric values, the app asks the user to choose another feature.

## Elbow method and inertia

Inertia is the sum of squared distances between each customer and its assigned centroid. It generally falls as K increases, because more groups can fit the data more closely. The elbow method plots inertia across valid K values and looks for a bend where the improvement begins to slow. The bend is an estimate to help compare choices, not a guaranteed perfect answer; use the profiles and practical context too.

## Silhouette score

The silhouette score compares how close each customer is to its own group with how close it is to the nearest other group. Values nearer 1 often indicate more distinct groups; values near 0 indicate overlap, and negative values can mean some rows may fit another group better. The app shows the score only when it can calculate it. It is an **internal clustering-quality measure**, not classification accuracy: the dataset has no known correct segment labels.

## Dataset

`data/sample_customers.csv` contains 150 rows of **synthetic** customers. The values were generated locally with deterministic, repeating variations around five example customer patterns to illustrate varied age, annual-income, and spending-score combinations. They do not describe real people and are not an openly sourced real-world dataset. `CustomerID` is an identifier and must not be used as a clustering feature. The app excludes likely numeric ID fields from its initial feature selection, but always review the selected features yourself.

## Technologies

Python, Streamlit, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and Pytest.

## Project structure

```text
customer-segmentation/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_customers.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── visualization.py
└── tests/
    ├── test_preprocessing.py
    └── test_clustering.py
```

## System workflow

```text
Sample CSV or uploaded CSV
        ↓
Preview and inspect columns
        ↓
Select numeric customer features
        ↓
Median imputation → StandardScaler
        ↓
Elbow curve + K-Means + silhouette (when valid)
        ↓
Cluster plot, sizes, profiles → Download CSV
```

## Installation on Windows

Open PowerShell in the project folder. Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can leave the policy unchanged and invoke the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, activate the environment from Command Prompt with `.venv\Scripts\activate.bat`. Follow your organization's device policy if it manages script execution.

## Run the application

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`) to open in your browser. The included sample is loaded from disk and does not require internet access. Uploading a CSV also happens locally.

## Run automated tests

```powershell
python -m pytest
```

Tests check median imputation, numeric conversion, standardization, identifier defaults, invalid selections, successful and reproducible clustering, cluster labels, invalid K values, and safe elbow ranges. They use small in-memory examples and do not access the internet.

## How to use the application

1. Choose the built-in sample or upload a CSV in the sidebar.
2. Review the first rows, types, missing-value counts, and descriptive statistics.
3. Select numeric customer attributes. Do not select ID or row-number columns; those identify records rather than describe customer behaviour.
4. Choose K in the sidebar. Review the elbow chart after running clustering; compare a few plausible choices if needed.
5. Inspect the scatter plot (choose two axes), group sizes, silhouette score if available, and the calculated profiles.
6. Download the labelled CSV. It contains the original columns plus `Cluster`; it does not overwrite the uploaded data.

If the CSV cannot be parsed, contains no rows, or has no usable numerical features, the app displays a message explaining what to fix. Missing numerical values are median-imputed without dropping rows. K is limited by the number of rows and distinct selected-feature patterns.

## Illustrative interpretation

The following is only an example of how a student might explain a result; it is not a reported output from the included dataset: “One group has a higher average annual income and spending score than the other groups, so it could be described as higher-income, higher-spending in this analysis.” Always base your own interpretation on the profile table from your actual run. Cluster labels can switch between runs or K values and are not meaningful names by themselves.

## Limitations

- K-Means expects numeric features and is sensitive to feature selection, outliers, and K.
- It favors compact groups around means and may not fit irregularly shaped groups.
- Standardization addresses differences in units but does not remove outliers.
- Synthetic sample rows are for demonstration and must not be used for business decisions.
- A plot using two features cannot show all dimensions when clustering uses more than two.
- Cluster descriptions summarize averages; they do not explain causes or prove customer intent.
- Silhouette score and inertia are aids for comparison, not proof that the segments are useful.

## Future enhancements

- Add optional PCA projection for displaying high-dimensional results in two dimensions.
- Compare K-Means with other clustering methods.
- Add robust outlier handling and additional data-quality reports.
- Let users export cluster profiles and charts alongside the labelled CSV.
- Support thoughtful categorical-feature encoding with clear explanations.

## Conclusion

This practical connects data preparation, feature scaling, unsupervised learning, evaluation, visualization, and interpretation in a small local application. It provides a repeatable starting point for exploring customer patterns while keeping the user's original data available for download.

## Viva questions and simple answers

1. **What is machine learning?** It is a way for computers to learn patterns from data and use those patterns to make predictions or organize information.
2. **What is unsupervised learning?** Learning patterns from data that does not include target answers or labels.
3. **What is customer segmentation?** Dividing customers into groups with similar observed characteristics.
4. **What is K-Means?** An algorithm that assigns points to K groups around their nearest group centers.
5. **Why is K-Means unsupervised?** It finds groups without being given correct group labels.
6. **What is a centroid?** The mean position of the points in a cluster; it acts as that cluster's center.
7. **What does K mean?** The number of clusters requested from K-Means.
8. **How does K-Means work?** It assigns points to nearby centroids, recalculates centroid means, and repeats until the groups settle.
9. **Why do we scale features?** So a feature with larger measurement units does not dominate distance calculations just because its numbers are larger.
10. **What is StandardScaler?** A tool that centers a feature and scales it to roughly unit variance.
11. **What is inertia?** The sum of squared distances from customers to their assigned centroid; smaller values mean points sit closer to centers, but inertia always tends to decrease as K grows.
12. **What is the elbow method?** A plot-based way to look for a K after which extra clusters reduce inertia more slowly.
13. **What is a silhouette score?** A measure of how well rows fit their own cluster compared with nearby clusters.
14. **Why exclude CustomerID?** It is an arbitrary identifier, not a meaningful customer attribute; using it can create misleading distances.
15. **What are limitations of K-Means?** It needs K in advance, is affected by scaling and outliers, and works best for compact, roughly round groups.
16. **Why might results change when K changes?** Changing K changes how many centers are available and can split or combine groups.
17. **What is the difference between clustering and classification?** Clustering discovers groups without known labels; classification learns to predict known labels from examples.
