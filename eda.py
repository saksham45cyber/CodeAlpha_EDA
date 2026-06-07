import pandas as pd 
df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())
print("\nshape:")
print(df.shape)
print("\ncolumns:")
print(df.columns)
print("\nInfo:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())
print("\nsurvival count:")
print(df['Survived'].value_counts())

print("\nmale/female count:")
print(df['Sex'].value_counts())
print("\nmale/female survival count:")
print(pd.crosstab(df['Sex'], df['Survived']))

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Sex', data=df)
plt.title('malevsfemale passenger')
plt.savefig('gender_count.png')
plt.show()

sns.countplot(x='Survived', data=df)

plt.title("Survival Count")

plt.savefig("survival_count.png")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(df['Age'].dropna(), bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig("age_distribution.png")

plt.show()

sns.countplot(x='Pclass', data=df)

plt.title("Passenger Class Distribution")

plt.savefig("passenger_class.png")

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(x='Sex', hue='Survived', data=df)

plt.title("Survival by Gender")

plt.savefig("survival_by_gender.png")

plt.show()

