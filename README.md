The dataset df_student contains 10,000 entries and 6 columns, with no missing values across any column.
The dataset includes 5 numerical variables and 1 categorical variable.

Correlation between Numerical Variables:
'Previous Scores' shows a strong positive correlation (0.91) with 'Performance Index', suggesting that higher previous scores are highly indicative of a higher performance index.
'Hours Studied' also exhibits a significant positive correlation (0.37) with 'Performance Index'.
Other correlations between independent numerical variables are generally low to moderate.

Hours Studied, Previous Scores vs. Performance Index: Scatter plots reveal clear positive linear relationships between 'Performance Index' and 'Hours Studied' and 'Previous Scores', reinforcing the high correlation values. Students who study more or have higher previous scores tend to have higher performance indices.
Sleep Hours vs. Performance Index: A positive trend is visible, but with more spread, indicating that while more sleep generally correlates with better performance, it's not as strong a predictor as study hours or previous scores.

All three models (Linear Regression, Random Forest, and Gradient Boosting) performed exceptionally well in predicting the 'Performance Index', each achieving an R-squared score of 0.99. Linear Regression demonstrated the best performance with the lowest Mean Squared Error (MSE) of 4.08, followed by Gradient Boosting with an MSE of 4.36, and Random Forest with an MSE of 5.16.
Linear Regression achieved the lowest Mean Squared Error (MSE) of 4.08, indicating it is the most accurate model among the three for this specific dataset and problem, followed closely by Gradient Boosting (MSE: 4.36) and Random Forest (MSE: 5.16).
