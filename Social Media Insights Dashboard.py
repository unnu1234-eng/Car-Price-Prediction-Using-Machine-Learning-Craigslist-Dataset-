# 📌 Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Set seaborn style for cleaner visuals
sns.set(style="whitegrid")

# 📌 Step 2: Load the dataset with encoding fix
df = pd.read_csv("instagram_data.csv", encoding='latin1')

# 📌 Step 3: Preview the data
print("Shape:", df.shape)
print(df.head())
print(df.info())

# 📌 Step 4: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 📌 Step 5: Drop rows with missing values (if any)
df.dropna(inplace=True)

# 📌 Step 6: Create engagement metric
df['engagement'] = df['likes'] + df['comments'] + df['saves']

# 📌 Step 7: Save cleaned data for Power BI
df.to_csv("cleaned_instagram_data.csv", index=False)
print("\nCleaned data exported as 'cleaned_instagram_data.csv'.")

# -------------------------------
# 📊 Step 8: Visualizations in Python
# -------------------------------

# 📌 Top 10 Posts by Likes
top_liked = df.sort_values(by='likes', ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x='likes', y='caption', data=top_liked, palette='magma')
plt.title("Top 10 Instagram Posts by Likes")
plt.xlabel("Likes")
plt.ylabel("Post Caption (Top 10)")
plt.tight_layout()
plt.show()

# 📌 Engagement vs Impressions Scatter Plot
plt.figure(figsize=(10,6))
sns.scatterplot(x='impressions', y='engagement', data=df, color='teal', alpha=0.7)
plt.title("Engagement vs Impressions")
plt.xlabel("Impressions")
plt.ylabel("Engagement (Likes + Comments + Saves)")
plt.tight_layout()
plt.show()

# 📌 Average Metrics Summary
avg_metrics = df[['likes', 'comments', 'shares', 'saves', 'follows', 'impressions']].mean()
print("\nAverage Metrics:\n", avg_metrics)

# 📌 Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[['impressions', 'likes', 'comments', 'saves', 'shares', 'follows', 'engagement']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
