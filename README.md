# Heritage Housing Issues

Heritage Housing Issues is a predictive analytics dashboard built with Streamlit.

The project uses the Ames, Iowa housing dataset to investigate which house attributes are most associated with sale price and to predict sale prices for four inherited houses.

The dashboard was built for a fictional client, Lydia, who has inherited four houses in Ames, Iowa. Lydia does not know the Ames housing market and wants data-driven support before deciding how to value the inherited properties.

[View the live app](https://wymmij-heritage-housing-issues.herokuapp.com/)

[View the GitHub repository](https://github.com/wymmij/heritage-housing-issues)

---

## Dataset Content

The dataset is sourced from the Code Institute Heritage Housing Issues project dataset on Kaggle.

The dataset contains housing records from Ames, Iowa. Each record describes property attributes such as living area, basement area, garage size, kitchen quality, lot size, construction year, remodelling year, porch area, and sale price.

The project uses two source datasets:

* `house_prices_records.csv`: historical house sale records, including `SalePrice`;
* `inherited_houses.csv`: four inherited houses requiring sale price predictions.

The target variable for the machine learning task is `SalePrice`.

A full feature table is included below to explain the dataset variables and category codes.

| Variable      | Meaning                                     | Units / Values                     |
| ------------- | ------------------------------------------- | ---------------------------------- |
| 1stFlrSF      | First floor square feet                     | 334 - 4692                         |
| 2ndFlrSF      | Second floor square feet                    | 0 - 2065                           |
| BedroomAbvGr  | Bedrooms above grade                        | 0 - 8                              |
| BsmtExposure  | Basement exposure                           | Gd, Av, Mn, No, None               |
| BsmtFinType1  | Rating of basement finished area            | GLQ, ALQ, BLQ, Rec, LwQ, Unf, None |
| BsmtFinSF1    | Type 1 finished basement square feet        | 0 - 5644                           |
| BsmtUnfSF     | Unfinished basement square feet             | 0 - 2336                           |
| TotalBsmtSF   | Total basement square feet                  | 0 - 6110                           |
| GarageArea    | Garage size in square feet                  | 0 - 1418                           |
| GarageFinish  | Interior finish of garage                   | Fin, RFn, Unf, None                |
| GarageYrBlt   | Year garage was built                       | 1900 - 2010                        |
| GrLivArea     | Above-grade living area in square feet      | 334 - 5642                         |
| KitchenQual   | Kitchen quality                             | Ex, Gd, TA, Fa, Po                 |
| LotArea       | Lot size in square feet                     | 1300 - 215245                      |
| LotFrontage   | Linear feet of street connected to property | 21 - 313                           |
| MasVnrArea    | Masonry veneer area in square feet          | 0 - 1600                           |
| EnclosedPorch | Enclosed porch area in square feet          | 0 - 286                            |
| OpenPorchSF   | Open porch area in square feet              | 0 - 547                            |
| OverallCond   | Overall condition rating                    | 1 - 10                             |
| OverallQual   | Overall material and finish rating          | 1 - 10                             |
| WoodDeckSF    | Wood deck area in square feet               | 0 - 736                            |
| YearBuilt     | Original construction date                  | 1872 - 2010                        |
| YearRemodAdd  | Remodel date                                | 1950 - 2010                        |
| SalePrice     | Sale price                                  | 34900 - 755000                     |

---

## Business Requirements

The client has two business requirements.

### Business Requirement 1

The client wants to understand which house attributes are most strongly associated with sale price in Ames, Iowa.

This is answered through exploratory data analysis, correlation analysis, and dashboard visualisations showing the strongest relationships between house attributes and `SalePrice`.

### Business Requirement 2

The client wants to predict the sale price of four inherited houses and any other Ames house with similar attribute data.

This is answered through a supervised regression model embedded in the Streamlit dashboard.

---

## Project Hypothesis and Validation

### Hypothesis

Houses with higher overall quality `OverallQual`, larger above-ground living area `GrLivArea`, and newer construction dates `YearBuilt` tend to have higher sale prices.

### Validation Method

The hypothesis was validated through the Sale Price Study notebook and dashboard page.

The validation process used:

* Pearson correlation;
* Spearman correlation;
* box plots;
* scatter plots;
* feature importance from the final regression model.

### Hypothesis Result

The hypothesis was supported.

The strongest positive relationships with `SalePrice` included:

* overall quality `OverallQual`;
* above-ground living area `GrLivArea`;
* construction year `YearBuilt`;
* garage area `GarageArea`;
* total basement area `TotalBsmtSF`.

The analysis does not prove causation. It shows that these attributes are meaningfully associated with sale price in this dataset.

---

## Rationale: Mapping Business Requirements to Data Visualisations and ML Tasks

### Business Requirement 1: Sale Price Study

To answer the first business requirement, the project uses data analysis and visualisation.

The Sale Price Study investigates correlations between house attributes and `SalePrice`. It identifies the features most strongly associated with sale price and visualises the most important relationships.

Dashboard evidence:

* top correlated features table;
* bar chart of strongest Spearman correlations;
* box plot of `SalePrice` by `OverallQual`;
* scatter plot of `SalePrice` by `GrLivArea`;
* scatter plot of `SalePrice` by `YearBuilt`.

### Business Requirement 2: Sale Price Prediction

To answer the second business requirement, the project uses a supervised machine learning regression model.

The model predicts `SalePrice` from house attributes. The final fitted model is used in the dashboard to:

* predict sale prices for Lydia's four inherited houses;
* calculate the total predicted value of the inherited houses;
* allow the user to enter custom house attributes and generate a sale price prediction.

---

## ML Business Case

### Objective

The objective is to build a regression model that predicts house sale price from property attributes.

### ML Task

This is a supervised regression task.

### Target Variable

The target variable is:

* `SalePrice`

### Model Inputs

The model uses numerical and ordinal-encoded property attributes, including:

* quality and condition ratings;
* living area;
* basement area;
* garage area;
* lot area;
* porch and deck area;
* construction and remodelling age features.

### Model Output

The model outputs a predicted house sale price in US dollars.

### Success Metric

The main success metric is R² on the test set.

The model is considered successful if it achieves:

* test set R² >= 0.75

Additional metrics used for evaluation are:

* Mean Absolute Error;
* Root Mean Squared Error.

### Model Selection

Several regression models were compared:

* Linear Regression;
* Decision Tree Regressor;
* Random Forest Regressor;
* Extra Trees Regressor;
* Gradient Boosting Regressor.

Gradient Boosting was selected as the final model because it gave the strongest test-set performance and a better train/test balance than the more heavily overfitted tree ensemble alternatives.

The final Gradient Boosting model achieved:

* Train R²: 0.965
* Test R²: 0.880

The final model therefore met the success threshold of test R² >= 0.75.

---

## Dashboard Design

The Streamlit dashboard contains five pages.

### Page 1: Project Summary

This page introduces the project, the dataset, the client problem, and the two business requirements.

It helps the user understand the purpose of the dashboard before exploring the analysis or prediction tools.

### Page 2: Sale Price Study

This page answers Business Requirement 1.

It shows:

* the top features correlated with `SalePrice`;
* feature meanings for abbreviation-heavy variables;
* a bar chart of the strongest correlations;
* visualisations of key relationships between sale price and important house attributes.

### Page 3: Project Hypothesis

This page states the project hypothesis and explains how it was validated.

It shows the correlation evidence for the hypothesis features and states whether the analysis supports the hypothesis.

### Page 4: Predict Sale Price

This page answers Business Requirement 2.

It shows:

* predicted sale prices for Lydia's four inherited houses;
* the total predicted value of the inherited houses;
* a custom prediction form where users can enter house attributes and generate a sale price prediction.

The prediction form includes explanations for dataset abbreviations and category codes.

### Page 5: ML Performance

This page shows evidence for the final model's performance.

It includes:

* model comparison results;
* final model performance metrics;
* the final test-set R² score;
* actual-vs-predicted plots for train and test sets;
* feature importance table and chart.
