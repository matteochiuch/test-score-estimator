The dataset df_student contains 10,000 entries and 6 columns, with no missing values across any column.
The dataset includes 5 numerical variables and 1 categorical variable.
<img width="686" height="447" alt="Immagine 2026-01-05 182331" src="https://github.com/user-attachments/assets/9c86d168-7111-4fa7-99fd-84f983337870" />

Correlation between Numerical Variables:
'Previous Scores' shows a strong positive correlation (0.91) with 'Performance Index', suggesting that higher previous scores are highly indicative of a higher performance index.
'Hours Studied' also exhibits a significant positive correlation (0.37) with 'Performance Index'.
Other correlations between independent numerical variables are generally low to moderate.
<img width="789" height="706" alt="image" src="https://github.com/user-attachments/assets/c366e10e-ca82-4af2-b6c4-e0bfe8d2bcf3" />


Hours Studied, Previous Scores vs. Performance Index: Scatter plots reveal clear positive linear relationships between 'Performance Index' and 'Hours Studied' and 'Previous Scores', reinforcing the high correlation values. Students who study more or have higher previous scores tend to have higher performance indices.
Sleep Hours vs. Performance Index: A positive trend is visible, but with more spread, indicating that while more sleep generally correlates with better performance, it's not as strong a predictor as study hours or previous scores.
<img width="1790" height="1189" alt="image" src="https://github.com/user-attachments/assets/10f9fcb3-be46-4199-bf5f-524e5ebd03e3" />


All three models (Linear Regression, Random Forest, and Gradient Boosting) performed exceptionally well in predicting the 'Performance Index', each achieving an R-squared score of 0.99. Linear Regression demonstrated the best performance with the lowest Mean Squared Error (MSE) of 4.08, followed by Gradient Boosting with an MSE of 4.36, and Random Forest with an MSE of 5.16.
<img width="281" height="114" alt="Immagine 2026-01-05 183110" src="https://github.com/user-attachments/assets/27f89e37-3d42-45b2-91c8-460a1a182cdb" />

Linear Regression achieved the lowest Mean Squared Error (MSE) of 4.08, indicating it is the most accurate model among the three for this specific dataset and problem, followed closely by Gradient Boosting (MSE: 4.36) and Random Forest (MSE: 5.16). However, since only 2 variables are strictly correlated with the output, I tried to train one more time the Linear Regression model to see if anything changes. As expected the model performed very good as well with an R-squared of 0.99 like the previous models and just a slightly bigger MSE of 5.24, indicating that we can simplify this model to 2 variables.

Finally, as an example, these are 10 predictions of the model compared to the actual values.

<img width="332" height="309" alt="Immagine 2026-01-05 183353" src="https://github.com/user-attachments/assets/c7b888dd-0528-4eaf-ba1b-0d298677e373" />

