#!/usr/bin/env python3
"""
LLM Personality Simulation Study

Investigates whether GPT-4o-mini replicates human personality-behavior patterns,
specifically examining the relationship between Big Five Agreeableness traits
and generosity in Dictator Game outcomes.

Requirements:
    pip install openai pandas numpy scipy scikit-learn matplotlib seaborn

License: MIT
"""
from scipy import stats
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from openai import OpenAI
import os
import time
from datetime import datetime

# Config
OUTPUT_DIR = '' # Replace with your desired output directory
INPUT_CSV = '' # Replace with path to input CSV file containing human data

# Check API key
API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    print("ERROR: Set OPENAI_API_KEY environment variable")
    print("export OPENAI_API_KEY='your-key-here'")
    exit(1)

client = OpenAI(api_key=API_KEY)

# Load data with all variables
df = pd.read_csv(INPUT_CSV, index_col=0)
df = df[['DG_given', 'Agreeableness', 'Extraversion', 'Conscientiousness', 
         'Neuroticism', 'Openness', 'Sex', 'age', 'family_situation', 
         'economic_situation', 'Self.perceived.attractiveness', 
         'romantic.part_life', 'sexual.part_life']].dropna()


# Create prompt with full profile
def create_prompt(profile):
    return f"""Research study participant profile:
Age {int(profile['age'])}, {'Male' if profile['Sex']==1 else 'Female'}

Personality scores (1-7 scale):
Agreeableness (cooperation and compassion) score: {profile['Agreeableness']:.1f}
Extraversion (social engagement and energy) score : {profile['Extraversion']:.1f}  
Conscientiousness (organization and goal-orientation) score: {profile['Conscientiousness']:.1f}
Neuroticism (emotional sensitivity) score: {profile['Neuroticism']:.1f}
Openness (curiosity and imagination) score: {profile['Openness']:.1f}

THE GAME:
You have been given 400 tokens. You must decide how many tokens to give to another anonymous participant in this study. Whatever you give them, they receive. You keep the rest. 

Each token is worth approximately $0.10 USD. Your decision is completely private and anonymous. The other person has no say in the division.

Based on your personality, how many tokens (0-400) do you GIVE OUT to the other participant?

Note: It's okay to be extremely selfish or extremely generous, there's no right or wrong answer.

Respond with just a number between 0 to 400:"""

# Simulate
def simulate(profile):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a real human participating in a research study."},
            {"role": "user", "content": create_prompt(profile)}
        ],
        temperature=1.5,
        max_tokens=10
    )
    
    import re
    numbers = re.findall(r'\d+', response.choices[0].message.content.strip())
    return int(numbers[0]) if numbers else 200

# Run simulations
results = []
for idx, (i, row) in enumerate(df.iterrows(), 1):
    llm_tokens = simulate(row)
    results.append({
        'real_dg': row['DG_given'],
        'llm_dg': llm_tokens,
        'Agreeableness': row['Agreeableness']
    })
    print(f"{idx}/{len(df)}: Agree={row['Agreeableness']:.1f} → LLM={llm_tokens}, Human={int(row['DG_given'])}")


df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

# Question 1: Real data - Agreeableness effect
print("\n1. REAL HUMANS: Agreeableness → Generosity")
print("-"*80)
r_human, p_human = stats.pearsonr(df_results['Agreeableness'], df_results['real_dg'])
print(f"Correlation: r = {r_human:.3f}, p = {p_human:.4f}")

low_agree_human = df_results[df_results['Agreeableness'] < df_results['Agreeableness'].median()]['real_dg'].mean()
high_agree_human = df_results[df_results['Agreeableness'] >= df_results['Agreeableness'].median()]['real_dg'].mean()
print(f"Low Agreeableness: {low_agree_human:.1f} tokens")
print(f"High Agreeableness: {high_agree_human:.1f} tokens")
print(f"Difference: {high_agree_human - low_agree_human:.1f} tokens")

r_spear_human, p_spear_human = stats.spearmanr(df_results['Agreeableness'], df_results['real_dg'])
print(f"Monotonic correlation: r = {r_spear_human:.3f}, p = {p_spear_human:.4f}")


low_agree_human = df_results[df_results['Agreeableness'] < df_results['Agreeableness'].median()]['real_dg'].mean()
high_agree_human = df_results[df_results['Agreeableness'] >= df_results['Agreeableness'].median()]['real_dg'].mean()
print(f"Low Agreeableness: {low_agree_human:.1f} tokens")
print(f"High Agreeableness: {high_agree_human:.1f} tokens")
print(f"Difference: {high_agree_human - low_agree_human:.1f} tokens")

# Question 2: LLM simulation - Agreeableness effect  
print("\n2. LLM SIMULATION: Agreeableness → Generosity")
print("-"*80)
r_llm, p_llm = stats.pearsonr(df_results['Agreeableness'], df_results['llm_dg'])
print(f"Correlation: r = {r_llm:.3f}, p = {p_llm:.4f}")

low_agree_llm = df_results[df_results['Agreeableness'] < df_results['Agreeableness'].median()]['llm_dg'].mean()
high_agree_llm = df_results[df_results['Agreeableness'] >= df_results['Agreeableness'].median()]['llm_dg'].mean()
print(f"Low Agreeableness: {low_agree_llm:.1f} tokens")
print(f"High Agreeableness: {high_agree_llm:.1f} tokens")
print(f"Difference: {high_agree_llm - low_agree_llm:.1f} tokens")
# Non-linear correlation
r_spear_llm, p_spear_llm = stats.spearmanr(df_results['Agreeableness'], df_results['llm_dg'])
print(f"Monotonic correlation: r = {r_spear_llm:.3f}, p = {p_spear_llm:.4f}")

# Group comparison
low_agree_llm = df_results[df_results['Agreeableness'] < df_results['Agreeableness'].median()]['llm_dg'].mean()
high_agree_llm = df_results[df_results['Agreeableness'] >= df_results['Agreeableness'].median()]['llm_dg'].mean()
print(f"Low Agreeableness: {low_agree_llm:.1f} tokens")
print(f"High Agreeableness: {high_agree_llm:.1f} tokens")
print(f"Difference: {high_agree_llm - low_agree_llm:.1f} tokens")


# Question 3: Pattern Consistency Analysis
print("\n3. PATTERN CONSISTENCY: Do LLMs replicate human patterns?")
print("-"*80)

# Basic stats
print(f"Humans: Mean={df_results['real_dg'].mean():.1f}, SD={df_results['real_dg'].std():.1f}")
print(f"LLM:    Mean={df_results['llm_dg'].mean():.1f}, SD={df_results['llm_dg'].std():.1f}")

# Pattern 1: Correlation strength comparison
print(f"\nPattern Strength:")
print(f"  Human correlation: r={r_human:.3f}")
print(f"  LLM correlation:   r={r_llm:.3f}")
print(f"  Difference: {abs(r_llm - r_human):.3f}")

# Pattern 2: Regression Slopes (do they increase at same rate?)
from scipy.stats import linregress
slope_human, intercept_human, _, _, _ = linregress(df_results['Agreeableness'], df_results['real_dg'])
slope_llm, intercept_llm, _, _, _ = linregress(df_results['Agreeableness'], df_results['llm_dg'])
slope_ratio = slope_llm / slope_human if slope_human != 0 else float('inf')
print(f"\nSlope comparison:")
print(f"  Human: {slope_human:.2f} tokens per agreeableness point")
print(f"  LLM:   {slope_llm:.2f} tokens per agreeableness point")
print(f"  Ratio: {slope_ratio:.2f}x")
print(f"  {'✓ Similar rate of increase' if 0.7 < slope_ratio < 1.3 else '✗ Different rate'}")

# Pattern 3: Group differences
human_diff = high_agree_human - low_agree_human
llm_diff = high_agree_llm - low_agree_llm
print(f"\nGroup Differences (High - Low Agreeableness):")
print(f"  Human: {human_diff:.1f} tokens")
print(f"  LLM:   {llm_diff:.1f} tokens")
print(f"  {'✓ Similar pattern' if abs(human_diff - llm_diff) < 30 else '✗ Different magnitude'}")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Overlay scatter + regression lines
axes[0].scatter(df_results['Agreeableness'], df_results['real_dg'], 
                alpha=0.5, label='Human', color='blue')
axes[0].scatter(df_results['Agreeableness'], df_results['llm_dg'], 
                alpha=0.5, label='LLM', color='red')

# Add regression lines
x_range = np.linspace(df_results['Agreeableness'].min(), 
                      df_results['Agreeableness'].max(), 100)
axes[0].plot(x_range, slope_human * x_range + intercept_human, 
             'b-', linewidth=2, label=f'Human trend (r={r_human:.2f})')
axes[0].plot(x_range, slope_llm * x_range + intercept_llm, 
             'r-', linewidth=2, label=f'LLM trend (r={r_llm:.2f})')

axes[0].set_xlabel('Agreeableness')
axes[0].set_ylabel('Tokens Given')
axes[0].set_title('Pattern Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Group means comparison
groups = ['Low\nAgreeableness', 'High\nAgreeableness']
x_pos = np.arange(len(groups))
width = 0.35

axes[1].bar(x_pos - width/2, [low_agree_human, high_agree_human], 
            width, label='Human', color='blue', alpha=0.7)
axes[1].bar(x_pos + width/2, [low_agree_llm, high_agree_llm], 
            width, label='LLM', color='red', alpha=0.7)

axes[1].set_ylabel('Tokens Given')
axes[1].set_title('Group Differences')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(groups)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}analysis.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {OUTPUT_DIR}analysis.png")
print("="*80)