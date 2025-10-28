
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ======== 1. LOAD DATA ========
DATA_CSV = '/users/buzhaoliu/developer/xai-eval-toolkit/bigfive.csv'
df = pd.read_csv(DATA_CSV)

# Drop incomplete rows just in case
df = df.dropna(subset=['UG_offer', 'UG_MAO', 'UG_120offer',
                       'Extraversion', 'Agreeableness', 'Conscientiousness',
                       'Neuroticism', 'Openness', 'priming'])

# ======== 2. CORRELATION HEATMAP ========
trait_cols = ['Extraversion','Agreeableness','Conscientiousness','Neuroticism','Openness']
behavior_cols = ['UG_offer','UG_MAO','UG_120offer']

corr = df[trait_cols + behavior_cols].corr().loc[trait_cols, behavior_cols]

plt.figure(figsize=(8,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Personality–Behavior Correlations")
plt.tight_layout()
plt.savefig("corr_heatmap.png", dpi=300)
plt.close()

print("\n=== Correlation Heatmap saved to corr_heatmap.png ===")
print(corr)

# ======== 3. PCA COOPERATION SCORE ========
scaler = StandardScaler()
X_behavior = df[behavior_cols]
X_scaled = scaler.fit_transform(X_behavior)
pca = PCA(n_components=1)
df['cooperation_score'] = pca.fit_transform(X_scaled)

print("\n=== PCA Cooperation Score ===")
print("Explained variance ratio:", pca.explained_variance_ratio_)

# ======== 4. REGRESSION MODELS ========

# 4.1 Main linear effects
formula_main = 'cooperation_score ~ Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness + priming'
model_main = smf.ols(formula_main, data=df).fit()
print("\n=== Main Effects Model ===")
print(model_main.summary())

# 4.2 Interaction terms (Personality × Priming)
formula_int = 'cooperation_score ~ priming * (Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness)'
model_int = smf.ols(formula_int, data=df).fit()
print("\n=== Interaction Model (Personality × Priming) ===")
print(model_int.summary())

# 4.3 Non-linear (quadratic) personality effects
for t in trait_cols:
    df[f'{t}_sq'] = df[t] ** 2

formula_poly = 'cooperation_score ~ ' + ' + '.join(trait_cols + [t + '_sq' for t in trait_cols])
model_poly = smf.ols(formula_poly, data=df).fit()
print("\n=== Non-linear (Quadratic) Model ===")
print(model_poly.summary())

# ======== 5. CLUSTERING ON BEHAVIOR ========
kmeans = KMeans(n_clusters=3, random_state=42)
df['behavior_cluster'] = kmeans.fit_predict(X_scaled)

trait_means = df.groupby('behavior_cluster')[trait_cols].mean()
behavior_means = df.groupby('behavior_cluster')[behavior_cols].mean()

print("\n=== Cluster Personality Means ===")
print(trait_means)
print("\n=== Cluster Behavior Means ===")
print(behavior_means)

# Visualize clusters in PCA space
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1] if X_scaled.shape[1]>1 else [0]*len(X_scaled),
                hue=df['behavior_cluster'], palette='Set2')
plt.title("Behavioral Clusters (from UG variables)")
plt.tight_layout()
plt.savefig("behavior_clusters.png", dpi=300)
plt.close()
print("\n=== Cluster plot saved to behavior_clusters.png ===")

# ======== 6. SAVE RESULTS ========
df.to_csv("bigfive_results.csv", index=False)
print("\nResults saved to bigfive_results.csv")
print("\nDone ✅")
