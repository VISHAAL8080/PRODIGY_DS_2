import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Retail_Sales.csv")  # Make sure the CSV filename is correct

# ------------------------------
# 1. Clean Column Names
# ------------------------------
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces

# Rename column for convenience
df.rename(columns={'Transaction Date': 'Date'}, inplace=True)

# ------------------------------
# 2. Initial Data Check
# ------------------------------
print("Initial Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Stats:\n", df.describe(include='all'))

# ------------------------------
# 3. Data Cleaning
# ------------------------------

# Convert types
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')

# Fill missing values
df['Price Per Unit'].fillna(df['Price Per Unit'].median(), inplace=True)
df['Quantity'].fillna(df['Quantity'].median(), inplace=True)
df['Total Spent'].fillna(df['Total Spent'].median(), inplace=True)
df['Item'].fillna("Unknown Item", inplace=True)
df['Discount Applied'].fillna("False", inplace=True)

# Drop rows with invalid dates
df.dropna(subset=['Date'], inplace=True)

# Standardize categorical text columns
df['Category'] = df['Category'].str.title().str.strip()
df['Payment Method'] = df['Payment Method'].str.title().str.strip()
df['Location'] = df['Location'].str.title().str.strip()
df['Discount Applied'] = df['Discount Applied'].astype(str).str.title().str.strip()

# ------------------------------
# 4. Exploratory Data Analysis
# ------------------------------

# Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Sales over Time
df.groupby('Date')['Total Spent'].sum().plot(figsize=(12,5), title='Total Sales Over Time')
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# Sales by Category
df.groupby('Category')['Total Spent'].sum().sort_values().plot(kind='barh', title='Total Sales by Category')
plt.xlabel("Total Sales")
plt.show()

# Quantity by Payment Method
sns.boxplot(x='Payment Method', y='Quantity', data=df)
plt.title("Quantity Distribution by Payment Method")
plt.show()

# Discount Applied Pie Chart
df['Discount Applied'].value_counts().plot.pie(autopct='%1.1f%%', title='Discount Usage')
plt.ylabel("")
plt.show()

# ------------------------------
# 5. Key Insights
# ------------------------------
print("\n✅ Insights:")
print("- The majority of transactions were done via", df['Payment Method'].mode()[0])
print("- The most sold category is:", df.groupby('Category')['Total Spent'].sum().idxmax())
print("- Around", round((df['Discount Applied'].value_counts(normalize=True)['True'] * 100), 2), "% of transactions used discounts.")