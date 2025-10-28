"""
simulate_xai_fungi_simple.py
=============================

"""
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelSettings:
    """Settings for OpenAI chat model."""
    model: str = "gpt-4o-mini"  # Can use any chat model!
    temperature: float = 1.0
    max_tokens: int = 50

CERTAINTY_ORDER = {
    "definitely_uncertain": 1,
    "moderately_uncertain": 2,
    "cannot_assess": 3,
    "moderately_certain": 4,
    "definitely_certain": 5
}

# ============================================================================
# PERSONA CONSTRUCTION
# ============================================================================

def build_persona(survey_row: pd.Series, group: str) -> str:

    # ============================================================================
    # 1. GROUP IDENTITY AND ROLE
    # ============================================================================
    
    group_descriptions = {
        "DE": """You are a domain expert in mycology.""",
        
        "IT": """You are a student with a data science and visualization background.""",
        
        "SSH": """You are a student from social sciences and humanities."""
    }
    
    persona_parts = [group_descriptions.get(group, "You are a research participant.")]
    
    # ============================================================================
    # 2. MUSHROOM KNOWLEDGE 
    # ============================================================================
    
    mushroom_knowledge = []
    
    # Knowledge level
    knowledge_col = "How would you rate your knowledge about wild edible and poisonous mushrooms?"
    if knowledge_col in survey_row.index:
        val = survey_row[knowledge_col]
        if pd.notna(val) and str(val).strip():
            mushroom_knowledge.append(f"Knowledge level: {val}")
    
    # Recognition methods - edible
    recog_edible_col = "How do you primarily recognize wild edible mushrooms?"
    if recog_edible_col in survey_row.index:
        val = survey_row[recog_edible_col]
        if pd.notna(val) and str(val).strip():
            mushroom_knowledge.append(f"How you recognize edible mushrooms: {val}")
    
    # Recognition methods - poisonous
    recog_poison_col = "How can you mainly recognize wild poisonous or inedible mushrooms?  "
    if recog_poison_col in survey_row.index:
        val = survey_row[recog_poison_col]
        if pd.notna(val) and str(val).strip():
            mushroom_knowledge.append(f"How you recognize poisonous mushrooms: {val}")
    
    # Source of knowledge
    knowledge_source_col = "Where does your knowledge about wild mushrooms come from?  "
    if knowledge_source_col in survey_row.index:
        val = survey_row[knowledge_source_col]
        if pd.notna(val) and str(val).strip():
            mushroom_knowledge.append(f"Your knowledge comes from: {val}")
    
    # Collection experience
    collect_col = "Do you ever collect wild mushrooms?"
    if collect_col in survey_row.index:
        val = survey_row[collect_col]
        if pd.notna(val) and str(val).strip():
            mushroom_knowledge.append(f"Mushroom collection experience: {val}")
    
    if mushroom_knowledge:
        persona_parts.append("\nYour Mushroom Knowledge:")
        persona_parts.extend([f"- {item}" for item in mushroom_knowledge])
    
    # ============================================================================
    # 3. DATA VISUALIZATION & ANALYSIS SKILLS (FOR IT/SSH)
    # ============================================================================
    
    if group in ["IT", "SSH"]:
        viz_skills = []
        
        # Data visualization skills
        viz_skills_col = "What are your skills in data visualization, such as creating charts, infographics, etc.? (Select up to two answers)."
        if viz_skills_col in survey_row.index:
            val = survey_row[viz_skills_col]
            if pd.notna(val) and str(val).strip():
                viz_skills.append(f"Visualization skills: {val}")
        
        # Data analysis skills
        analysis_col = "How do you rate your skills in data analysis?  "
        if analysis_col in survey_row.index:
            val = survey_row[analysis_col]
            if pd.notna(val) and str(val).strip():
                viz_skills.append(f"Data analysis skills: {val}")
        
        # Source of viz skills
        viz_source_col = "Where do you have your skills in data visualization from?  "
        if viz_source_col in survey_row.index:
            val = survey_row[viz_source_col]
            if pd.notna(val) and str(val).strip():
                viz_skills.append(f"Visualization training from: {val}")
        
        # Opinion on good visualizations
        viz_features_col = "What features should a well-made data visualization have in your opinion?"
        if viz_features_col in survey_row.index:
            val = survey_row[viz_features_col]
            if pd.notna(val) and str(val).strip():
                viz_skills.append(f"You believe good visualizations should: {val}")
        
        if viz_skills:
            persona_parts.append("\nYour Data & Visualization Background:")
            persona_parts.extend([f"- {item}" for item in viz_skills])
    
    # ============================================================================
    # 4. XAI INTERPRETATION ABILITIES (CRITICAL!)
    # ============================================================================
    
    xai_understanding = []
    
    # Understanding of feature importance (from SHAP-like question)
    shap_col = "Based on the chart, are you able to determine the features that most often appear on its right side and suggest a strong influence on classifying the mushroom as poisonous? Please respond and provide these features.  "
    if shap_col in survey_row.index:
        val = survey_row[shap_col]
        if pd.notna(val) and str(val).strip() and len(str(val)) > 10:
            xai_understanding.append(f"Your interpretation of feature importance: {val[:200]}")  # Truncate if too long
    
    # Understanding of counterfactuals
    counterfactual_col = "Looking at the following visualization, are you able to determine which changes in characteristics, according to the counterfactual analysis, have the greatest impact on changing the classification of a mushroom from poisonous to edible or vice versa? Please answer by describing these changes."
    if counterfactual_col in survey_row.index:
        val = survey_row[counterfactual_col]
        if pd.notna(val) and str(val).strip() and len(str(val)) > 10:
            xai_understanding.append(f"Your interpretation of counterfactual explanations: {val[:200]}")
    
    if xai_understanding:
        persona_parts.append("\nYour Understanding of AI Explanations:")
        persona_parts.extend([f"- {item}" for item in xai_understanding])
    
    # ============================================================================
    # 5. EDUCATIONAL BACKGROUND
    # ============================================================================
    
    education = []
    
    study_program_col = "Provide the name(s) of the study program(s) you are currently enrolled in:"
    if study_program_col in survey_row.index:
        val = survey_row[study_program_col]
        if pd.notna(val) and str(val).strip():
            education.append(f"Your current study program: {val}")
    
    if education:
        persona_parts.append("\nYour Educational Background:")
        persona_parts.extend([f"- {item}" for item in education])
    
    return "\n".join(persona_parts)

def get_xai_explanation() -> str:
    """
    Enhanced XAI explanation summary extracted from actual slides.
    Based on ORIGINAL_VISUALIZATIONS_EN.pdf pages 4-14.
    """
    
    explanation = """DATASET INFORMATION:
The AI model was trained on 61,069 mushroom specimens from 173 species (edible, non-edible, or poisonous).
All are cap mushrooms with stems and lamellar hymenophore. Some data is simulated based on real observations.
The XGBoost Gradient Boosting Classifier achieved 99.97% accuracy.

KEY FEATURES BY IMPORTANCE (SHAP Analysis - Slide 7):
The most influential features for predicting toxicity are:
1. stem_height_cm (HIGH value = more likely poisonous)
2. cap_diameter_cm (HIGH value = mixed effect, context-dependent)
3. stem_width_mm (HIGH value = more likely poisonous)
4. gill_attachment_decurrent (present = more likely edible)
5. cap_color_green (present = strongly indicates poisonous)
6. cap_color_yellow (present = strongly indicates poisonous)
7. does_bruise_or_bleed (bruises/bleeding present = slightly more edible; absent = more poisonous)
8. gill_attachment_none (present = indicates poisonous)

SHAP values > 0 (right side) indicate features that increase poisonous classification.
SHAP values < 0 (left side) indicate features that increase edible classification.

FEATURE INTERACTIONS:
- Thin stems (low stem_width_mm) with white color often indicate edibility
- Thick stems (high stem_width_mm > 16.56mm) strongly suggest poisonous
- Green or yellow cap colors are strong poison indicators
- Decurrent gill attachment is associated with edibility
- No bruising/bleeding combined with other features suggests poisonous

LIME LOCAL EXPLANATIONS (Slide 9):
For individual mushrooms, the model considers:
- stem_width_mm ≤ 5.20: strongly increases edible probability (+0.07)
- no bruising/bleeding: increases non-edible probability (+0.05)
- stem_width_mm in range 5.20-10.17: moderate edible indicator (+0.05)
- stem_height_cm > 7.74: slight poisonous indicator (+0.03)
- Thick stems (>16.56mm): strongly indicates poisonous (-0.07)

WATERFALL ANALYSIS (Slide 8):
For a specific poisonous mushroom example:
- Baseline prediction (average): 9.208
- stem_height_cm contributed: +0.78 (toward poisonous)
- stem_width_mm contributed: -0.22 (toward edible)
- cap_diameter_cm contributed: +0.18 (toward poisonous)
- Final prediction: 9.959 (confidently poisonous)

ANCHOR EXPLANATIONS (Slides 13-14):
If ALL of these conditions are true, the model predicts with high certainty:
- For EDIBLE: stem_width_mm > 10.17, stem_root = no_data, cap_surface = smooth, 
  stem_color = white, gill_attachment = sinuate, cap_diameter > 5.87, stem_height ≤ 7.74
  → Prediction: edible 97.2% of the time

- For POISONOUS: gill_attachment = adnexed, cap_color = green, cap_surface = no_data,
  cap_shape = flat, cap_diameter > 5.87
  → Prediction: poisonous 97.5% of the time

COUNTERFACTUAL ANALYSIS (Slide 10):
For a poisonous mushroom to be classified as edible, key changes needed:
- Change stem_height_cm from 1.43 to 23.5 (much taller stem)
- Other features like cap shape, color, and gill properties remain constant
This shows stem height is a critical decision boundary.

IMPORTANT CAVEATS:
- The model uses statistical patterns, not botanical rules
- High accuracy doesn't mean 100% reliability
- Some features have missing data (no_data) which affects predictions
- The model is a tool to assist identification, not a definitive answer"""

    return explanation


def format_mushroom(task_row: pd.Series) -> str:

    exclude = {"problem_id", "model_class", "model_probability"}
    
    features = []
    for col, val in task_row.items():
        if col in exclude or pd.isna(val):
            continue
        
        val_str = str(val).strip()
        if val_str.lower() not in {"", "nan", "n/a", "no_data", "missing_data"}:
            # Clean up column names
            col_clean = col.replace("_", " ")
            features.append(f"- {col_clean}: {val_str}")
    
    return "\n".join(features)



# ============================================================================
# SIMULATION
# ============================================================================

def simulate_decision(
    persona: str,
    explanation: str,
    mushroom: str,
    client: OpenAI,
    settings: ModelSettings
) -> str:
    """
    Simple decision simulation - just ask the LLM to decide.
    Returns: "edible" or "non-edible"
    """
    
    if client is None:
        return np.random.choice(["edible", "non-edible"])
    
    prompt = f"""{persona}

XAI Explanation:
{explanation}

Mushroom characteristics:
{mushroom}

Based on the AI explanation and your background, would you assess this mushroom as edible or non-edible?

Respond with ONLY ONE WORD: either "edible" or "non-edible" (no explanation needed)."""

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        answer = response.choices[0].message.content.strip().lower()
        
        if "non" in answer or "poison" in answer or "inedible" in answer:
            return "non-edible"
        elif "edible" in answer:
            return "edible"
        else:
            return np.random.choice(["edible", "non-edible"])
            
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2)
        return np.random.choice(["edible", "non-edible"])


def simulate_certainty(
    persona: str,
    explanation: str,
    mushroom: str,
    decision: str,
    client: OpenAI,
    settings: ModelSettings
) -> str:
    """
    Simple certainty simulation - ask LLM to choose confidence level.
    Returns: one of the 5 certainty levels
    """
    
    if client is None:
        return np.random.choice([
            "definitely_certain",
            "moderately_certain", 
            "cannot_assess",
            "moderately_uncertain",
            "definitely_uncertain"
        ])
    
    prompt = f"""{persona}

XAI Explanation:
{explanation}

Mushroom characteristics:
{mushroom}

Your decision: {decision}

How confident are you in this decision? Choose ONE of these options:
1. definitely certain
2. moderately certain
3. I can't assess
4. moderately uncertain
5. definitely uncertain

Respond with ONLY the number (1-5) or the exact phrase."""

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        answer = response.choices[0].message.content.strip().lower()
        
        if "1" in answer or "definitely certain" in answer:
            return "definitely_certain"
        elif "2" in answer or "moderately certain" in answer:
            return "moderately_certain"
        elif "3" in answer or "can't assess" in answer or "cannot assess" in answer:
            return "cannot_assess"
        elif "4" in answer or "moderately uncertain" in answer:
            return "moderately_uncertain"
        elif "5" in answer or "definitely uncertain" in answer:
            return "definitely_uncertain"
        else:
            return "moderately_certain"
            
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2)
        return "moderately_certain"


# ============================================================================
# EVALUATION WITH ORDINAL DISTANCE FOR CERTAINTY
# ============================================================================

def normalize_decision(s: str) -> str:
    """Normalize decision to 'edible' or 'non-edible'."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    if "non" in s or "poison" in s:
        return "non-edible"
    if "edible" in s:
        return "edible"
    return s


def normalize_certainty(s: str) -> str:
    """Normalize certainty to one of 5 standard levels."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    return s


def compute_kl_divergence(true_series: pd.Series, sim_series: pd.Series) -> float:
    """Compute KL divergence: KL(true || sim)."""
    true_counts = true_series.value_counts(normalize=True).sort_index()
    sim_counts = sim_series.value_counts(normalize=True).sort_index()
    
    all_values = sorted(set(true_counts.index) | set(sim_counts.index))
    
    epsilon = 1e-10
    true_dist = np.array([true_counts.get(v, 0) + epsilon for v in all_values])
    sim_dist = np.array([sim_counts.get(v, 0) + epsilon for v in all_values])
    
    true_dist = true_dist / true_dist.sum()
    sim_dist = sim_dist / sim_dist.sum()
    
    return np.sum(true_dist * np.log(true_dist / sim_dist))


def compute_js_divergence(true_series: pd.Series, sim_series: pd.Series) -> float:
    """Compute Jensen-Shannon divergence."""
    true_counts = true_series.value_counts(normalize=True).sort_index()
    sim_counts = sim_series.value_counts(normalize=True).sort_index()
    
    all_values = sorted(set(true_counts.index) | set(sim_counts.index))
    
    epsilon = 1e-10
    true_dist = np.array([true_counts.get(v, 0) + epsilon for v in all_values])
    sim_dist = np.array([sim_counts.get(v, 0) + epsilon for v in all_values])
    
    true_dist = true_dist / true_dist.sum()
    sim_dist = sim_dist / sim_dist.sum()
    
    m = (true_dist + sim_dist) / 2
    return 0.5 * np.sum(true_dist * np.log(true_dist / m)) + \
           0.5 * np.sum(sim_dist * np.log(sim_dist / m))


def compute_certainty_distance_metrics(cert_true: pd.Series, cert_sim: pd.Series) -> Dict:
    """
    Compute ordinal distance metrics for certainty.
    
    Returns:
        - mae: Mean Absolute Error in ordinal levels (0-4 scale)
        - rmse: Root Mean Squared Error in ordinal levels
        - spearman: Spearman rank correlation
        - within_1: Percentage within 1 level
        - exact_match: Percentage of exact matches
    """
    # Convert to ordinal
    true_ordinal = cert_true.map(CERTAINTY_ORDER)
    sim_ordinal = cert_sim.map(CERTAINTY_ORDER)
    
    # Remove any NaN (unmapped values)
    valid_mask = true_ordinal.notna() & sim_ordinal.notna()
    true_ordinal = true_ordinal[valid_mask]
    sim_ordinal = sim_ordinal[valid_mask]
    
    if len(true_ordinal) == 0:
        return {}
    
    # Compute metrics
    diff = np.abs(true_ordinal - sim_ordinal)
    
    metrics = {
        "mae": diff.mean(),  # Mean Absolute Error
        "rmse": np.sqrt((diff ** 2).mean()),  # Root Mean Squared Error
        "within_1": (diff <= 1).mean(),  # Within 1 level
        "within_2": (diff <= 2).mean(),  # Within 2 levels
        "exact_match": (diff == 0).mean()  # Exact match
    }
    
    # Spearman correlation (if enough variance)
    if len(true_ordinal.unique()) > 1 and len(sim_ordinal.unique()) > 1:
        corr, pval = spearmanr(true_ordinal, sim_ordinal)
        metrics["spearman"] = corr
        metrics["spearman_pval"] = pval
    
    return metrics


def evaluate_distributions(merged_df: pd.DataFrame) -> Dict:
    """
    Evaluate by comparing distributions.
    
    KEY FIX: Certainty is now evaluated by DISTANCE not just exact match.
    """
    
    # Normalize
    merged_df["dec_true"] = merged_df["participant_decision_en"].apply(normalize_decision)
    merged_df["dec_sim"] = merged_df["decision_sim"].apply(normalize_decision)
    merged_df["cert_true"] = merged_df["participant_certainty_en"].apply(normalize_certainty)
    merged_df["cert_sim"] = merged_df["certainty_sim"].apply(normalize_certainty)
    
    # Filter valid
    valid = merged_df[
        (merged_df["dec_true"] != "") & (merged_df["dec_sim"] != "")
    ].copy()
    
    if len(valid) == 0:
        return {"error": "No valid data"}
    
    # Extract group
    valid["group"] = valid["participant_id"].apply(lambda x: 
        next((p for p in str(x).split("_") if p in ["DE", "IT", "SSH"]), ""))
    
    metrics = {}
    
    # ========================================================================
    # OVERALL METRICS
    # ========================================================================
    
    metrics["n_total"] = len(valid)
    metrics["decision_accuracy"] = (valid["dec_true"] == valid["dec_sim"]).mean()
    metrics["decision_kl"] = compute_kl_divergence(valid["dec_true"], valid["dec_sim"])
    metrics["decision_js"] = compute_js_divergence(valid["dec_true"], valid["dec_sim"])
    
    # Certainty metrics (with ordinal distance)
    cert_valid = valid[(valid["cert_true"] != "") & (valid["cert_sim"] != "")]
    if len(cert_valid) > 0:
        metrics["certainty_kl"] = compute_kl_divergence(cert_valid["cert_true"], cert_valid["cert_sim"])
        metrics["certainty_js"] = compute_js_divergence(cert_valid["cert_true"], cert_valid["cert_sim"])
        
        # NEW: Ordinal distance metrics
        cert_distance = compute_certainty_distance_metrics(
            cert_valid["cert_true"], cert_valid["cert_sim"]
        )
        for k, v in cert_distance.items():
            metrics[f"certainty_{k}"] = v
    
    # ========================================================================
    # GROUP-LEVEL ANALYSIS
    # ========================================================================
    
    for group in ["DE", "IT", "SSH"]:
        group_data = valid[valid["group"] == group]
        
        if len(group_data) == 0:
            continue
        
        prefix = f"{group}_"
        metrics[f"{prefix}n"] = len(group_data)
        
        # Decision metrics
        metrics[f"{prefix}decision_accuracy"] = (
            group_data["dec_true"] == group_data["dec_sim"]
        ).mean()
        metrics[f"{prefix}decision_kl"] = compute_kl_divergence(
            group_data["dec_true"], group_data["dec_sim"]
        )
        metrics[f"{prefix}decision_js"] = compute_js_divergence(
            group_data["dec_true"], group_data["dec_sim"]
        )
        metrics[f"{prefix}decision_true_dist"] = group_data["dec_true"].value_counts(normalize=True).to_dict()
        metrics[f"{prefix}decision_sim_dist"] = group_data["dec_sim"].value_counts(normalize=True).to_dict()
        
        # Certainty metrics (with ordinal distance)
        group_cert = group_data[(group_data["cert_true"] != "") & (group_data["cert_sim"] != "")]
        
        if len(group_cert) > 0:
            metrics[f"{prefix}certainty_kl"] = compute_kl_divergence(
                group_cert["cert_true"], group_cert["cert_sim"]
            )
            metrics[f"{prefix}certainty_js"] = compute_js_divergence(
                group_cert["cert_true"], group_cert["cert_sim"]
            )
            metrics[f"{prefix}certainty_true_dist"] = group_cert["cert_true"].value_counts(normalize=True).to_dict()
            metrics[f"{prefix}certainty_sim_dist"] = group_cert["cert_sim"].value_counts(normalize=True).to_dict()
            
            # NEW: Ordinal distance metrics per group
            cert_distance = compute_certainty_distance_metrics(
                group_cert["cert_true"], group_cert["cert_sim"]
            )
            for k, v in cert_distance.items():
                metrics[f"{prefix}certainty_{k}"] = v
    
    # ========================================================================
    # PER-PROBLEM ANALYSIS
    # ========================================================================
    
    for prob_id in valid["problem_id"].unique():
        prob_data = valid[valid["problem_id"] == prob_id]
        prefix = f"problem_{prob_id}_"
        
        metrics[f"{prefix}n"] = len(prob_data)
        metrics[f"{prefix}decision_accuracy"] = (
            prob_data["dec_true"] == prob_data["dec_sim"]
        ).mean()
        metrics[f"{prefix}decision_kl"] = compute_kl_divergence(
            prob_data["dec_true"], prob_data["dec_sim"]
        )
        metrics[f"{prefix}decision_true_dist"] = prob_data["dec_true"].value_counts(normalize=True).to_dict()
        metrics[f"{prefix}decision_sim_dist"] = prob_data["dec_sim"].value_counts(normalize=True).to_dict()
    
    return metrics


def print_results(metrics: Dict):
    """Print evaluation results with ordinal distance metrics."""
    
    print("\n" + "="*70)
    print("SIMPLE SIMULATION - DISTRIBUTION COMPARISON")
    print("(Certainty evaluated by ORDINAL DISTANCE)")
    print("="*70)
    
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    # ========================================================================
    # OVERALL
    # ========================================================================
    
    print(f"\n📊 OVERALL (n={metrics['n_total']})")
    print("-" * 70)
    print(f"Decision Accuracy:    {metrics['decision_accuracy']:.2%}")
    print(f"Decision KL:          {metrics['decision_kl']:.4f}")
    print(f"Decision JS:          {metrics['decision_js']:.4f}")
    
    if "certainty_kl" in metrics:
        print(f"\nCertainty Performance:")
        print(f"  KL Divergence:      {metrics['certainty_kl']:.4f}")
        print(f"  JS Divergence:      {metrics['certainty_js']:.4f}")
        
        # NEW: Distance metrics
        if "certainty_mae" in metrics:
            print(f"\n  Ordinal Distance Metrics:")
            print(f"    MAE:              {metrics['certainty_mae']:.2f} levels")
            print(f"    RMSE:             {metrics['certainty_rmse']:.2f} levels")
            print(f"    Exact Match:      {metrics['certainty_exact_match']:.2%}")
            print(f"    Within 1 Level:   {metrics['certainty_within_1']:.2%}")
            print(f"    Within 2 Levels:  {metrics['certainty_within_2']:.2%}")
            
            if "certainty_spearman" in metrics:
                print(f"    Spearman Corr:    {metrics['certainty_spearman']:.3f}")
    
    # ========================================================================
    # GROUP-LEVEL
    # ========================================================================
    
    print(f"\n👥 GROUP-LEVEL (Distribution Comparison)")
    print("-" * 70)
    
    for group in ["DE", "IT", "SSH"]:
        if f"{group}_n" in metrics:
            print(f"\n{group} (n={metrics[f'{group}_n']}):")
            
            # Decision
            print(f"  Decision:")
            print(f"    Accuracy: {metrics[f'{group}_decision_accuracy']:.2%}")
            print(f"    KL Div:   {metrics[f'{group}_decision_kl']:.4f}")
            print(f"    JS Div:   {metrics[f'{group}_decision_js']:.4f}")
            print(f"    True:     {metrics[f'{group}_decision_true_dist']}")
            print(f"    Sim:      {metrics[f'{group}_decision_sim_dist']}")
            
            # Certainty (with distance metrics)
            if f"{group}_certainty_kl" in metrics:
                print(f"\n  Certainty:")
                print(f"    KL Div:           {metrics[f'{group}_certainty_kl']:.4f}")
                print(f"    JS Div:           {metrics[f'{group}_certainty_js']:.4f}")
                
                if f"{group}_certainty_mae" in metrics:
                    print(f"    MAE:              {metrics[f'{group}_certainty_mae']:.2f} levels")
                    print(f"    Exact Match:      {metrics[f'{group}_certainty_exact_match']:.2%}")
                    print(f"    Within 1 Level:   {metrics[f'{group}_certainty_within_1']:.2%}")
                    
                    if f"{group}_certainty_spearman" in metrics:
                        print(f"    Spearman Corr:    {metrics[f'{group}_certainty_spearman']:.3f}")
                
                print(f"    True:     {metrics[f'{group}_certainty_true_dist']}")
                print(f"    Sim:      {metrics[f'{group}_certainty_sim_dist']}")


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_group(pid: str) -> str:
    """Extract group (DE/IT/SSH) from participant_id."""
    if isinstance(pid, str):
        for part in pid.split("_"):
            if part in {"DE", "IT", "SSH"}:
                return part
    return ""


def load_data(gt_file: str, survey_file: str, tasks_file: str):
    """Load and filter data."""
    
    gt = pd.read_csv(gt_file, dtype=str)
    survey = pd.read_csv(survey_file, dtype=str)
    tasks = pd.read_csv(tasks_file, dtype=str)
    
    # Filter to ground truth participants only
    gt_pids = gt["participant_id"].dropna().unique()
    gt_problems = gt["problem_id"].dropna().unique()
    
    survey = survey[survey["participant_id"].isin(gt_pids)].copy()
    survey["group"] = survey["participant_id"].apply(extract_group)
    survey = survey[survey["group"] != ""].fillna("")
    
    tasks = tasks[tasks["problem_id"].isin(gt_problems)].copy()
    
    print(f"\nData loaded:")
    print(f"  Participants: {len(survey)} ({survey['group'].value_counts().to_dict()})")
    print(f"  Problems: {len(tasks)}")
    print(f"  Total simulations: {len(survey) * len(tasks)}")
    
    return gt, survey, tasks


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print("="*70)
    print("XAI-FUNGI SIMULATION")
    print("="*70)
    
    # Initialize client
    client = None
    if OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            print("✓ OpenAI client initialized")
    
    if client is None:
        print("⚠ Running without API (random simulation)")
    
    # Load data
    gt, survey, tasks = load_data(
        args.ground_truth_file,
        args.characteristics_file,
        args.tasks_file
    )
    
    # Get explanation
    explanation = get_xai_explanation()
    
    # Model settings
    settings = ModelSettings(
        model=args.model,
        temperature=args.temperature
    )
    
    print(f"\nModel: {settings.model}")
    print(f"Temperature: {settings.temperature}")
    
    # Run simulations
    print("\n🤖 Running simulations...")
    
    results = []
    
    for _, participant in tqdm(survey.iterrows(), total=len(survey), desc="Simulating"):
        persona = build_persona(participant, participant["group"])
        pid = participant["participant_id"]
        
        for _, task in tasks.iterrows():
            mushroom = format_mushroom(task)
            
            # Simulate decision
            decision = simulate_decision(
                persona, explanation, mushroom, client, settings
            )
            
            time.sleep(0.5)  # Rate limiting
            
            # Simulate certainty
            certainty = simulate_certainty(
                persona, explanation, mushroom, decision, client, settings
            )
            
            time.sleep(0.3)  # Rate limiting
            
            results.append({
                "problem_id": task["problem_id"],
                "participant_id": pid,
                "decision_sim": decision,
                "certainty_sim": certainty
            })
    
    # Save
    sim_df = pd.DataFrame(results)
    sim_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Saved: {args.output_file}")
    
    # Merge and evaluate
    merged = pd.merge(gt, sim_df, on=["problem_id", "participant_id"], how="inner")
    
    if len(merged) == 0:
        print("❌ No matching rows after merge")
        return
    
    print(f"✓ Merged: {len(merged)} rows")
    
    # Evaluate
    metrics = evaluate_distributions(merged)
    print_results(metrics)
    
    # Save merged
    if args.save_merged:
        merged_path = args.output_file.replace(".csv", "_merged.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n✓ Saved merged: {merged_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI-FUNGI Simple Simulation")
    
    parser.add_argument("--characteristics_file", required=True)
    parser.add_argument("--tasks_file", required=True)
    parser.add_argument("--ground_truth_file", required=True)
    parser.add_argument("--output_file", default="results_simple.csv")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_merged", action="store_true", default=True)
    
    args = parser.parse_args()
    main(args)