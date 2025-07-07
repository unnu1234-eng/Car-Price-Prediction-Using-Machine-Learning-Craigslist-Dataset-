# 📦 Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🧼 Step 2: Load and Clean the Dataset
df = pd.read_csv("vehicles.csv")

# Keep relevant columns only
df = df[['price', 'year', 'manufacturer', 'model', 'condition', 'cylinders',
         'fuel', 'odometer', 'title_status', 'transmission', 'drive', 'type']]

# Drop missing target or odometer
df.dropna(subset=['price', 'odometer'], inplace=True)

# Filter outliers
df = df[(df['price'] > 500) & (df['price'] < 100000)]

# Drop remaining rows with missing values
df.dropna(inplace=True)

# 👁 Step 3: Exploratory Data Analysis (Visualizations)
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='odometer', y='price', data=df, alpha=0.5)
plt.title("Odometer vs Price")
plt.xlabel("Odometer (miles)")
plt.ylabel("Price ($)")
plt.show()

plt.figure(figsize=(10, 5))
df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Manufacturers by Average Price")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.show()

# 🔠 Step 4: Encode Categorical Columns
categorical_cols = ['manufacturer', 'model', 'condition', 'cylinders',
                    'fuel', 'title_status', 'transmission', 'drive', 'type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 🎯 Step 5: Feature Selection and Splitting
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 6: Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("📊 Linear Regression Results:")
print("R² Score:", r2_score(y_test, lr_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds)))
print()

# 🌳 Step 7: Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("📊 Random Forest Results:")
print("R² Score:", r2_score(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print()

# 📈 Step 8: Plot Feature Importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# 📊 Step 9: Plot Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=rf_preds)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Actual vs Predicted (Random Forest)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()