import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\hari excel\\Candy_Sales.csv")
print(df)
print(df.head(10))
print(df.tail(10))
print(df.info())
print(df.describe())
#1. Line Chart – Sales Over Time
df['Month'] = df['Order Date'].dt.to_period('M')
sales_by_month = df.groupby('Month')['Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(sales_by_month['Month'].astype(str), sales_by_month['Sales'], color='g', marker='d', linewidth=2)
plt.title("Sales Over Time (Monthly)")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.grid(True)
#2.Bar Chart – Total Sales by Region
sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_region.values, y=sales_by_region.index, palette="viridis")
plt.title("Total Sales by Region")
plt.xlabel("Sales")
plt.ylabel("Region")
#3.Column Chart – Top 10 States by Units Sold
units_by_state = df.groupby('State/Province')['Units'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=units_by_state.values, y=units_by_state.index, palette="magma")
plt.title("Top 10 States by Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("State")
#4.Scatter Plot – Sales vs Gross Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Sales', y='Gross Profit', color="r")
plt.title("Sales vs Gross Profit")
plt.xlabel("Sales")
plt.ylabel("Gross Profit")
plt.grid(True)
#5.Boxplot – Outliers Detection
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='Sales', color='skyblue')
plt.title("Boxplot - Sales")
#6.Heatmap – Correlation Matrix
corr = df[['Sales', 'Units', 'Gross Profit', 'Cost']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
#7.Pair Plot – Numerical Feature Relationships
sns.pairplot(df[['Sales', 'Units', 'Gross Profit', 'Cost']], kind='scatter', diag_kind='hist', corner=True)
plt.suptitle("Pair Plot of Numerical Features", y=1.02)

plt.show()
