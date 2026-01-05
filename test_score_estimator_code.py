import pandas as pd

#info on the dataset
df_student = pd.read_csv("StudentPerformance.csv")
display(df_student.head())

print("Prime 5 righe del DataFrame df_student:")
display(df_student.head())

print("\nInformazioni generali sul DataFrame df_student:")
df_student.info()

print("\nRiepilogo statistico descrittivo del DataFrame df_student:")
display(df_student.describe())

print("Conteggio dei valori mancanti per ogni colonna nel DataFrame df_student:")
df_student.isnull().sum()

#analysis of numerical variables
  import matplotlib.pyplot as plt

numeric_cols = df_student.select_dtypes(include=['number']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i + 1) # Adjust subplot grid based on number of numeric columns
    plt.hist(df_student[col], bins=20, edgecolor='black')
    plt.title(f'Distribuzione di {col}')
    plt.xlabel(col)
    plt.ylabel('Frequenza')
plt.tight_layout()
plt.show()

#analysis of categorical variable
categorical_cols = df_student.select_dtypes(include=['object']).columns

plt.figure(figsize=(10, 5))
for i, col in enumerate(categorical_cols):
    plt.subplot(1, len(categorical_cols), i + 1) # Adjust subplot grid dynamically
    df_student[col].value_counts().plot(kind='bar', edgecolor='black')
    plt.title(f'Distribuzione di {col}')
    plt.xlabel('Categoria')
    plt.ylabel('Frequenza')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#correlation matrix
import seaborn as sns

numeric_cols = df_student.select_dtypes(include=['number']).columns
correlation_matrix = df_student[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice di Correlazione tra Variabili Numeriche')
plt.show()

#performance analysis

# Scatter plots for numerical variables vs. Performance Index
numerical_features = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']

plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(data=df_student, x=col, y='Performance Index', hue='Extracurricular Activities', palette='viridis')
    plt.title(f'Performance Index vs. {col}')
    plt.xlabel(col)
    plt.ylabel('Performance Index')
plt.tight_layout()
plt.show()

# Box plot for categorical variable vs. Performance Index
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_student, x='Extracurricular Activities', y='Performance Index', hue='Extracurricular Activities', palette='viridis', legend=False)
plt.title('Performance Index vs. Extracurricular Activities')
plt.xlabel('Extracurricular Activities')
plt.ylabel('Performance Index')
plt.show()

#developing and selecting the best model

y = df_student['Performance Index']
X = df_student.drop('Performance Index', axis=1)

X = pd.get_dummies(X, columns=['Extracurricular Activities'], drop_first=True)

print("Prime 5 righe delle Features (X):")
display(X.head())
print("\nPrime 5 righe della Target (y):")
display(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dimensioni del set di training per le features (X_train):", X_train.shape)
print("Dimensioni del set di test per le features (X_test):", X_test.shape)
print("Dimensioni del set di training per la variabile target (y_train):", y_train.shape)
print("Dimensioni del set di test per la variabile target (y_test):", y_test.shape)

#train different models
#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

#random forest
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Mean Squared Error (MSE) per Random Forest: {mse_rf:.2f}")
print(f"R-squared (R2) per Random Forest: {r2_rf:.2f}")
print(f"R-squared (R2): {r2:.2f}")

#gradient boosting
from sklearn.ensemble import GradientBoostingRegressor

model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Mean Squared Error (MSE) per Gradient Boosting: {mse_gb:.2f}")
print(f"R-squared (R2) per Gradient Boosting: {r2_gb:.2f}")

#compare the models
model_performance = {
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [mse, mse_rf, mse_gb],
    'R2 Score': [r2, r2_rf, r2_gb]
}

performance_df = pd.DataFrame(model_performance)

print("Performance comparativa dei modelli:")
display(performance_df)

#show some predictions compared with actual results
comparison_df = pd.DataFrame({'Punteggio Effettivo': y_test, 'Punteggio Previsto': y_pred})

display(comparison_df.head(10))

#since high correlation with only 2 variables, i'm training the model dropping everithing but those 2 

X_selected = df_student[['Previous Scores', 'Hours Studied']]

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model_selected = LinearRegression()
model_selected.fit(X_train_selected, y_train_selected)
y_pred_selected = model_selected.predict(X_test_selected)
mse_selected = mean_squared_error(y_test_selected, y_pred_selected)
r2_selected = r2_score(y_test_selected, y_pred_selected)

print(f"Mean Squared Error (MSE) con feature selezionate: {mse_selected:.2f}")
print(f"R-squared (R2) con feature selezionate: {r2_selected:.2f}")
