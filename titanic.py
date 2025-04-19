import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print(df.head())

survival_rate = df.groupby("sex")["survived"].mean()
print("\nCinsiyete göre hayatta kalma oranı:\n",survival_rate)

sns.countplot(x = "sex", hue="survived", data=df)
plt.title("Cinsiyete GÖre Hayatta Kalma Dağılımı")
plt.legend(title="Survived", labels=["Hayır", "Evet"])
plt.show()

class_survival = df.groupby("pclass")["survived"].mean()
print("\nSınıfa göre hayatta kalma oranı:\n",class_survival)

sns.barplot(x=class_survival.index, y=class_survival.values)
plt.title("Sınıfa Göre Hayatta Kalma Oranı")
plt.xlabel("Yolcu Sınıfı")
plt.ylabel("Hayatta Kalma Oranı")
plt.show()