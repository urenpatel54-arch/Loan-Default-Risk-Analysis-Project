# Loan Default Risk Analysis Project

# Question 1
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Question 2
# Load Dataset
data = pd.read_csv("loan_applications.csv")
print(data[
    [
        "Customer_ID",
        "Age",
        "Income",
        "Loan_Amount",
        "Default_Status"
    ]].head(10))

# Question 3
# Dataset Information
print(data.info())
print("Dataset Shape =", data.shape)

# Question 4
# Mean Median Mode
print("Mean =", data["Income"].mean())
print("Median =", data["Income"].median())
print("Mode =", data["Income"].mode()[0])

# Question 5
# Variance and Standard Deviation
print("Variance =", data["Loan_Amount"].var())
print("Standard Deviation =", data["Loan_Amount"].std())

# Question 6
# Probability of Loan Default
probability = (
    data["Default_Status"].value_counts()["Yes"]
    / len(data)
)
print("Default Probability =", probability)

# Question 7
# Conditional Probability
low_credit = data[data["Credit_Score"] < 600]

conditional = (
    low_credit["Default_Status"].value_counts()["Yes"]
    / len(low_credit)
)
print("Conditional Probability =", conditional)

# Question 8
# Histogram
plt.hist(data["Credit_Score"])
plt.title("Credit Score Histogram")
plt.xlabel("Credit Score")
plt.ylabel("Frequency")
plt.show()

# Question 9
# Q-Q Plot
stats.probplot(
    data["Income"],
    dist="norm",
    plot=plt
)
plt.title("Q-Q Plot")
plt.show()

# Question 10
# Dot Product
vectors = data[
    [
        "Income",
        "Loan_Amount"
    ]
].head(2).values

dot_product = np.dot(
    vectors[0],
    vectors[1]
)

print("Dot Product =", dot_product)

# Question 11
# Norm
norm = np.linalg.norm(vectors[0])
print("Norm =", norm)

# Question 12
# Angle Between Vectors
value = dot_product / (
    np.linalg.norm(vectors[0])
    * np.linalg.norm(vectors[1])
)

angle = np.degrees(np.arccos(value))
print("Angle =", angle)

# Question 13
# Correlation Matrix

correlation = data[[
    "Age",
    "Income",
    "Loan_Amount",
    "Credit_Score",
    "Loan_Term"
]].corr()

print(correlation)

# Question 14
# Heatmap
plt.imshow(correlation)
plt.colorbar()
plt.title("Heatmap")
plt.show()

# Question 15
# Scatter Plot
plt.scatter(
    data["Income"],
    data["Loan_Amount"]
)

plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Scatter Plot")
plt.grid()
plt.show()

# Question 16
# Boxplot
plt.boxplot(data["Loan_Amount"])
plt.title("Boxplot")
plt.show()

# Question 17
# Pie Chart
status = data["Default_Status"].value_counts()

plt.pie(
    status,
    labels=status.index,
    autopct="%1.1f%%"
)

plt.title("Loan Default Status")
plt.show()

# Final Conclusion
print("Low credit score customers have high default risk")
print("Statistics helps banks identify risky customers")
print("Linear algebra helps represent customer data")