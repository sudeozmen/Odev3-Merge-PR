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

df['age_group'] = pd.cut(df['age'], bins=[0, 12, 20, 40, 60, 80],
                         labels=['0-12', '13-20', '21-40', '41-60', '61-80'])
clean_df = df[['age_group', 'sex', 'survived']].dropna()
grouped = clean_df.groupby(['age_group', 'sex', 'survived']).size().reset_index(name='count')
pivot = grouped.pivot_table(index=['age_group', 'sex'], columns='survived', values='count', fill_value=0)
pivot.columns = ['Not Survived', 'Survived']
pivot = pivot.reset_index()
fig, ax = plt.subplots(figsize=(10,6))

for sex in pivot['sex'].unique():
    sub_df = pivot[pivot['sex'] == sex]
    ax.bar(sub_df['age_group'], sub_df['Not Survived'], label=f"{sex} - Not Survived")
    ax.bar(sub_df['age_group'], sub_df['Survived'], bottom=sub_df['Not Survived'], label=f"{sex} - Survived")

ax.set_title("Yaş Grubu ve Cinsiyete Göre Hayatta Kalma Dağılımı")
ax.set_ylabel("Kişi Sayısı")
ax.set_xlabel("Yaş Grubu")
ax.legend()
plt.tight_layout()
plt.show()