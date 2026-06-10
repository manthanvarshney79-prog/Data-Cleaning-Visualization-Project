import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../dataset/titanic.csv")

print("Missing Values:")
print(data.isnull().sum())

data = data.drop_duplicates()

data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

data["survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.savefig("../images/survival_count.png")
plt.close()

data["pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("../images/passenger_class.png")
plt.close()

data["sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("../images/gender_distribution.png")
plt.close()

print("Project Completed Successfully")
