"""
simulate_xai_fungi_te_v2.py
==================================

"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelSettings:
    """Settings for OpenAI model queries."""
    model: str = "gpt-3.5-turbo-instruct"
    max_tokens: int = 1
    temperature: float = 1.0
    logprobs: int = 5
    echo: bool = False


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def create_decision_prompt_template() -> Tuple[str, str]:
    """Create paired prompts for edible vs non-edible decisions."""
    
    base_template = """Background: {background}

XAI Explanation Summary:
{explanation}

Mushroom Characteristics:
{characteristics}

Based on the AI explanation and your background expertise, you assess this mushroom as:"""

    prompt_edible = base_template + " edible"
    prompt_non_edible = base_template + " non-edible"
    
    return prompt_edible, prompt_non_edible


def create_certainty_prompt_template() -> Dict[str, str]:
    """Create prompts for each certainty level."""
    
    base = """Background: {background}

XAI Explanation Summary:
{explanation}

Mushroom Characteristics:
{characteristics}

Decision: {decision}

Your confidence in this assessment is:"""

    certainty_levels = {
        "definitely_certain": base + " definitely certain",
        "moderately_certain": base + " moderately certain",
        "cannot_assess": base + " I can't assess",
        "moderately_uncertain": base + " moderately uncertain",
        "definitely_uncertain": base + " definitely uncertain"
    }
    
    return certainty_levels


# ============================================================================
# PERSONA CONSTRUCTION
# ============================================================================

def build_persona_background(survey_row: pd.Series, group: str) -> str:
    """
    Build comprehensive persona from survey data.
    
    Following TE methodology: use extensive background to capture individual differences.
    The more context we provide, the better the simulation should match real behavior.
    """
    
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
    # 2. MUSHROOM KNOWLEDGE (CRITICAL FOR THIS TASK)
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
            education.append(f"Study program: {val}")
    
    if education:
        persona_parts.append("\nYour Educational Background:")
        persona_parts.extend([f"- {item}" for item in education])
    
    return "\n".join(persona_parts)

def extract_xai_explanation_summary() -> str:
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

def format_mushroom_characteristics(task_row: pd.Series) -> str:
    """Format mushroom features concisely."""
    
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
# SIMULATION WITH LOGPROBS
# ============================================================================

def get_completion_logprob(
    prompt: str,
    client: OpenAI,
    settings: ModelSettings
) -> Tuple[float, Optional[str]]:
    """Get log probability of the prompt's completion."""
    
    if client is None:
        return -1.0, None
    
    try:
        response = client.completions.create(
            model=settings.model,
            prompt=prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            logprobs=settings.logprobs,
            echo=settings.echo
        )
        
        if response.choices[0].logprobs and response.choices[0].logprobs.token_logprobs:
            logprob = response.choices[0].logprobs.token_logprobs[0]
            top_token = response.choices[0].logprobs.tokens[0]
            return logprob if logprob is not None else -10.0, top_token
        
        return -10.0, None
        
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2)
        return -10.0, None


def simulate_decision_with_logprobs(
    background: str,
    explanation: str,
    characteristics: str,
    client: OpenAI,
    settings: ModelSettings
) -> Dict[str, float]:
    """Simulate decision using probability comparison."""
    
    prompt_edible, prompt_non_edible = create_decision_prompt_template()
    
    prompt_edible_filled = prompt_edible.format(
        background=background,
        explanation=explanation,
        characteristics=characteristics
    )
    
    prompt_non_edible_filled = prompt_non_edible.format(
        background=background,
        explanation=explanation,
        characteristics=characteristics
    )
    
    logprob_edible, _ = get_completion_logprob(prompt_edible_filled, client, settings)
    time.sleep(0.5)
    logprob_non_edible, _ = get_completion_logprob(prompt_non_edible_filled, client, settings)
    
    prob_edible = np.exp(logprob_edible)
    prob_non_edible = np.exp(logprob_non_edible)
    
    prob_valid = prob_edible + prob_non_edible
    
    if prob_valid > 0:
        p_edible = prob_edible / prob_valid
        p_non_edible = prob_non_edible / prob_valid
    else:
        p_edible = 0.5
        p_non_edible = 0.5
    
    decision = "edible" if p_edible > p_non_edible else "non-edible"
    
    return {
        "p_edible": p_edible,
        "p_non_edible": p_non_edible,
        "p_valid": prob_valid,
        "decision": decision,
        "logprob_edible": logprob_edible,
        "logprob_non_edible": logprob_non_edible
    }

def simulate_certainty_with_logprobs(
    background: str,
    explanation: str,
    characteristics: str,
    decision: str,
    client: OpenAI,
    settings: ModelSettings
) -> Dict[str, float]:
    """Simulate certainty using categorical probability distribution over A–E levels."""
    
    certainty_options = {
        "A": "definitely certain",
        "B": "moderately certain",
        "C": "I can’t assess",
        "D": "moderately uncertain",
        "E": "definitely uncertain"
    }

    # Build a single prompt template like decision simulation style
    base_prompt = (
        "You are assessing how certain the persona feels about their decision.\n"
        "Options:\n"
        "A. definitely certain\n"
        "B. moderately certain\n"
        "C. I can’t assess\n"
        "D. moderately uncertain\n"
        "E. definitely uncertain\n\n"
        "Persona background: {background}\n"
        "Decision: {decision}\n"
        "Explanation: {explanation}\n"
        "Mushroom characteristics: {characteristics}\n\n"
        "Choose one option (A–E) that best describes the persona's certainty:"
    )

    prompt_filled = base_prompt.format(
        background=background,
        explanation=explanation,
        characteristics=characteristics,
        decision=decision
    )

    # Calculate logprob for each certainty option (A–E)
    logprobs = {}
    for key in certainty_options.keys():
        completion = f"{prompt_filled} {key}"
        logprob, _ = get_completion_logprob(completion, client, settings)
        logprobs[key] = logprob
        time.sleep(0.3)

    # Convert to normalized probability distribution
    scaled_logprobs = {k: v / 1.5 for k, v in logprobs.items()}  # smooths distribution
    probs = {k: np.exp(v) for k, v in scaled_logprobs.items()}
    total = sum(probs.values())
    if total > 0:
        normalized_probs = {k: v / total for k, v in probs.items()}
    else:
        normalized_probs = {k: 0.2 for k in probs.keys()}  # uniform fallback

    # Pick the most likely certainty option
    certainty_key = max(normalized_probs.items(), key=lambda x: x[1])[0]
    certainty_label = certainty_options[certainty_key]

    # Entropy = measure of certainty sharpness
    certainty_entropy = -sum(p * np.log(p + 1e-10) for p in normalized_probs.values())

    return {
        **{f"p_{k}": v for k, v in normalized_probs.items()},
        "certainty": certainty_label,
        "certainty_key": certainty_key,
        "certainty_entropy": certainty_entropy
    }


# ============================================================================
# EVALUATION
# ============================================================================

def normalize_decision(s: str) -> str:
    """Normalize decision strings."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    if "non" in s or "poison" in s:
        return "non-edible"
    if "edible" in s:
        return "edible"
    return s


def normalize_certainty(s: str) -> str:
    """Normalize certainty strings."""
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    
    mapping = {
        "definitely certain": "definitely_certain",
        "moderately certain": "moderately_certain",
        "i can't assess": "cannot_assess",
        "moderately uncertain": "moderately_uncertain",
        "definitely uncertain": "definitely_uncertain"
    }
    
    for key, val in mapping.items():
        if key in s:
            return val
    return s

def compute_distribution_similarity(
    true_data: pd.Series,
    sim_data: pd.Series,
    metric: str = "kl_divergence"
) -> float:
    """
    Compare two distributions using various metrics.
    
    Args:
        true_data: Series of true responses
        sim_data: Series of simulated responses
        metric: "kl_divergence", "js_divergence", or "chi_squared"
    
    Returns:
        Similarity score (lower is better for divergence metrics)
    """
    # Get value counts as distributions
    true_counts = true_data.value_counts(normalize=True).sort_index()
    sim_counts = sim_data.value_counts(normalize=True).sort_index()
    
    # Align indices (in case some values appear in one but not the other)
    all_values = sorted(set(true_counts.index) | set(sim_counts.index))
    
    true_dist = np.array([true_counts.get(v, 0) for v in all_values])
    sim_dist = np.array([sim_counts.get(v, 0) for v in all_values])
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    true_dist = true_dist + epsilon
    sim_dist = sim_dist + epsilon
    
    # Renormalize
    true_dist = true_dist / true_dist.sum()
    sim_dist = sim_dist / sim_dist.sum()
    
    if metric == "kl_divergence":
        # KL(true || sim): How much information is lost using sim instead of true
        return np.sum(true_dist * np.log(true_dist / sim_dist))
    
    elif metric == "js_divergence":
        # Jensen-Shannon divergence (symmetric, bounded 0-1)
        m = (true_dist + sim_dist) / 2
        return 0.5 * np.sum(true_dist * np.log(true_dist / m)) + \
               0.5 * np.sum(sim_dist * np.log(sim_dist / m))
    
    elif metric == "chi_squared":
        # Chi-squared test statistic
        return np.sum((true_dist - sim_dist)**2 / (true_dist + epsilon))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

def evaluate_with_probabilities(merged_df: pd.DataFrame) -> Dict:
    """
    Enhanced evaluation with DISTRIBUTION COMPARISON by group.
    
    This is the key insight: we compare whether the DISTRIBUTION of decisions
    and certainties matches for each group (DE, IT, SSH), not just overall accuracy.
    """
    
    print("\n🔍 Evaluating with distribution comparison...")
    
    # Verify required columns exist
    required_cols = ["participant_decision_en", "decision_sim", 
                     "participant_certainty_en", "certainty_sim"]
    missing = [col for col in required_cols if col not in merged_df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}
    
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
        return {"error": "No valid data after normalization"}
    
    print(f"✓ Valid responses: {len(valid)}/{len(merged_df)}")
    
    # Add group
    valid["group"] = valid["participant_id"].apply(lambda x: 
        next((p for p in str(x).split("_") if p in ["DE", "IT", "SSH"]), ""))
    
    metrics = {}
    
    # ========================================================================
    # OVERALL METRICS (for reference)
    # ========================================================================
    
    metrics["decision_accuracy"] = (valid["dec_true"] == valid["dec_sim"]).mean()
    
    cert_valid = valid[(valid["cert_true"] != "") & (valid["cert_sim"] != "")]
    if len(cert_valid) > 0:
        metrics["certainty_accuracy"] = (cert_valid["cert_true"] == cert_valid["cert_sim"]).mean()
    
    # Overall distribution similarity
    metrics["overall_decision_kl"] = compute_distribution_similarity(
        valid["dec_true"], valid["dec_sim"], "kl_divergence"
    )
    
    if len(cert_valid) > 0:
        metrics["overall_certainty_kl"] = compute_distribution_similarity(
            cert_valid["cert_true"], cert_valid["cert_sim"], "kl_divergence"
        )
    
    # ========================================================================
    # GROUP-LEVEL DISTRIBUTION COMPARISON (KEY CONTRIBUTION!)
    # ========================================================================
    
    print("\n📊 GROUP-LEVEL DISTRIBUTION COMPARISON:")
    print("="*70)
    
    for group in ["DE", "IT", "SSH"]:
        group_data = valid[valid["group"] == group]
        
        if len(group_data) == 0:
            continue
        
        print(f"\n{group} Group (n={len(group_data)}):")
        print("-" * 70)
        
        # Decision distribution
        true_dec_dist = group_data["dec_true"].value_counts(normalize=True).to_dict()
        sim_dec_dist = group_data["dec_sim"].value_counts(normalize=True).to_dict()
        
        print(f"  Decision Distribution:")
        print(f"    True:      {true_dec_dist}")
        print(f"    Simulated: {sim_dec_dist}")
        
        # Decision distribution similarity
        dec_kl = compute_distribution_similarity(
            group_data["dec_true"], group_data["dec_sim"], "kl_divergence"
        )
        dec_js = compute_distribution_similarity(
            group_data["dec_true"], group_data["dec_sim"], "js_divergence"
        )
        
        print(f"    KL Divergence: {dec_kl:.4f} (lower is better, 0 = perfect)")
        print(f"    JS Divergence: {dec_js:.4f} (lower is better, 0 = perfect)")
        
        metrics[f"{group}_decision_kl"] = dec_kl
        metrics[f"{group}_decision_js"] = dec_js
        metrics[f"{group}_decision_acc"] = (group_data["dec_true"] == group_data["dec_sim"]).mean()
        metrics[f"{group}_n"] = len(group_data)
        
        # Store distributions for later analysis
        metrics[f"{group}_true_decision_dist"] = true_dec_dist
        metrics[f"{group}_sim_decision_dist"] = sim_dec_dist
        
        # Certainty distribution
        group_cert = group_data[(group_data["cert_true"] != "") & (group_data["cert_sim"] != "")]
        
        if len(group_cert) > 0:
            true_cert_dist = group_cert["cert_true"].value_counts(normalize=True).to_dict()
            sim_cert_dist = group_cert["cert_sim"].value_counts(normalize=True).to_dict()
            
            print(f"\n  Certainty Distribution:")
            print(f"    True:      {true_cert_dist}")
            print(f"    Simulated: {sim_cert_dist}")
            
            cert_kl = compute_distribution_similarity(
                group_cert["cert_true"], group_cert["cert_sim"], "kl_divergence"
            )
            cert_js = compute_distribution_similarity(
                group_cert["cert_true"], group_cert["cert_sim"], "js_divergence"
            )
            
            print(f"    KL Divergence: {cert_kl:.4f}")
            print(f"    JS Divergence: {cert_js:.4f}")
            
            metrics[f"{group}_certainty_kl"] = cert_kl
            metrics[f"{group}_certainty_js"] = cert_js
            metrics[f"{group}_certainty_acc"] = (group_cert["cert_true"] == group_cert["cert_sim"]).mean()
            
            metrics[f"{group}_true_certainty_dist"] = true_cert_dist
            metrics[f"{group}_sim_certainty_dist"] = sim_cert_dist
        
        # Probability correlation (if available)
        if "p_edible_sim" in group_data.columns:
            group_data_copy = group_data.copy()
            group_data_copy["true_edible"] = (group_data_copy["dec_true"] == "edible").astype(float)
            corr_data = group_data_copy[["true_edible", "p_edible_sim"]].dropna()
            if len(corr_data) > 1:
                corr = corr_data.corr()
                if not corr.empty and len(corr) > 1:
                    metrics[f"{group}_prob_corr"] = corr.iloc[0, 1]
                    print(f"\n  Probability Correlation: {corr.iloc[0, 1]:.3f}")
    
    # ========================================================================
    # PER-PROBLEM ANALYSIS
    # ========================================================================
    
    problem_stats = {}
    for prob_id in valid["problem_id"].unique():
        prob_data = valid[valid["problem_id"] == prob_id]
        
        problem_stats[prob_id] = {
            "accuracy": (prob_data["dec_true"] == prob_data["dec_sim"]).mean(),
            "n": len(prob_data),
            "decision_kl": compute_distribution_similarity(
                prob_data["dec_true"], prob_data["dec_sim"], "kl_divergence"
            ),
            "true_dist": prob_data["dec_true"].value_counts(normalize=True).to_dict(),
            "sim_dist": prob_data["dec_sim"].value_counts(normalize=True).to_dict()
        }
    
    metrics["problem_stats"] = problem_stats
    
    print("\n" + "="*70)
    
    return metrics


def print_results(metrics: Dict):
    """Print comprehensive evaluation results with distribution comparison."""
    
    print("\n" + "="*70)
    print("TURING EXPERIMENT EVALUATION - DISTRIBUTION COMPARISON")
    print("="*70)
    
    if "error" in metrics:
        print(f"❌ Error: {metrics['error']}")
        return
    
    # ========================================================================
    # OVERALL PERFORMANCE
    # ========================================================================
    
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 70)
    print(f"Decision Accuracy:          {metrics.get('decision_accuracy', 0):.2%}")
    print(f"Decision KL Divergence:     {metrics.get('overall_decision_kl', 0):.4f}")
    
    if "certainty_accuracy" in metrics:
        print(f"Certainty Accuracy:         {metrics['certainty_accuracy']:.2%}")
        print(f"Certainty KL Divergence:    {metrics.get('overall_certainty_kl', 0):.4f}")
    
    # ========================================================================
    # GROUP-LEVEL COMPARISON (MAIN RESULT!)
    # ========================================================================
    
    print("\n👥 GROUP-LEVEL DISTRIBUTION SIMILARITY")
    print("-" * 70)
    print("(Lower KL/JS divergence = better match between true and simulated)")
    print()
    
    for group in ["DE", "IT", "SSH"]:
        if f"{group}_n" in metrics and metrics[f"{group}_n"] > 0:
            print(f"{group} (n={metrics[f'{group}_n']}):")
            
            # Decision metrics
            print(f"  Decision:")
            print(f"    Accuracy:      {metrics.get(f'{group}_decision_acc', 0):.2%}")
            print(f"    KL Divergence: {metrics.get(f'{group}_decision_kl', 0):.4f}")
            print(f"    JS Divergence: {metrics.get(f'{group}_decision_js', 0):.4f}")
            
            if f"{group}_prob_corr" in metrics:
                print(f"    Prob Corr:     {metrics[f'{group}_prob_corr']:.3f}")
            
            # Certainty metrics
            if f"{group}_certainty_kl" in metrics:
                print(f"  Certainty:")
                print(f"    Accuracy:      {metrics.get(f'{group}_certainty_acc', 0):.2%}")
                print(f"    KL Divergence: {metrics[f'{group}_certainty_kl']:.4f}")
                print(f"    JS Divergence: {metrics[f'{group}_certainty_js']:.4f}")
            
            print()
    
    # ========================================================================
    # INTERPRETATION GUIDE
    # ========================================================================
    
    print("📖 INTERPRETATION GUIDE:")
    print("-" * 70)
    print("KL Divergence:")
    print("  < 0.1  = Excellent match (distributions very similar)")
    print("  0.1-0.3 = Good match (minor differences)")
    print("  0.3-0.5 = Moderate match (noticeable differences)")
    print("  > 0.5  = Poor match (distributions quite different)")
    print()
    print("JS Divergence:")
    print("  < 0.05 = Excellent match")
    print("  0.05-0.1 = Good match")
    print("  0.1-0.2 = Moderate match")
    print("  > 0.2  = Poor match")
    
    # ========================================================================
    # PER-PROBLEM PERFORMANCE
    # ========================================================================
    
    if "problem_stats" in metrics:
        print("\n🍄 PER-PROBLEM PERFORMANCE")
        print("-" * 70)
        for prob_id, stats in metrics["problem_stats"].items():
            print(f"{prob_id}:")
            print(f"  Accuracy: {stats['accuracy']:.2%} (n={stats['n']})")
            print(f"  KL Divergence: {stats['decision_kl']:.4f}")
            print(f"  True dist: {stats['true_dist']}")
            print(f"  Sim dist:  {stats['sim_dist']}")
            print()
    
    print("=" * 70)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================



def extract_group(pid: str) -> str:
    """Extract group identifier from participant_id."""
    if isinstance(pid, str):
        for part in pid.split("_"):
            if part in {"DE", "IT", "SSH"}:
                return part
    return ""


def load_and_filter_data(
    ground_truth_file: str,
    characteristics_file: str,
    tasks_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data and filter participants to only those in ground truth.
    
    CRITICAL: Only simulates participants who have actual responses.
    """
    
    print("\n📁 Loading and filtering data...")
    
    # Load ground truth
    gt = pd.read_csv(ground_truth_file, dtype=str)
    print(f"  Ground truth: {len(gt)} responses")
    print(f"  GT columns: {gt.columns.tolist()}")
    
    # Get unique participants from ground truth
    gt_participants = gt["participant_id"].dropna().unique()
    print(f"  Unique participants in GT: {len(gt_participants)}")
    print(f"  Sample participant IDs: {list(gt_participants[:3])}")
    
    # Get unique problems from ground truth
    gt_problems = gt["problem_id"].dropna().unique()
    print(f"  Unique problems in GT: {len(gt_problems)}")
    print(f"  Problem IDs: {list(gt_problems)}")
    
    # Load and filter survey to ONLY participants in ground truth
    survey = pd.read_csv(characteristics_file, dtype=str)
    print(f"\n  Survey: {len(survey)} total participants")
    
    survey_filtered = survey[survey["participant_id"].isin(gt_participants)].copy()
    print(f"  Survey filtered: {len(survey_filtered)} participants (matched to GT)")
    
    # Add group information
    survey_filtered["group"] = survey_filtered["participant_id"].apply(extract_group)
    survey_filtered = survey_filtered[survey_filtered["group"] != ""].fillna("")
    
    print(f"  Survey with groups: {len(survey_filtered)} participants")
    print(f"  Groups: {survey_filtered['group'].value_counts().to_dict()}")
    
    # Load and filter tasks to ONLY problems in ground truth
    tasks = pd.read_csv(tasks_file, dtype=str)
    print(f"\n  Tasks: {len(tasks)} total problems")
    
    tasks_filtered = tasks[tasks["problem_id"].isin(gt_problems)].copy()
    print(f"  Tasks filtered: {len(tasks_filtered)} problems (matched to GT)")
    
    # Validation
    expected_simulations = len(survey_filtered) * len(tasks_filtered)
    print(f"\n✓ Will simulate: {expected_simulations} responses")
    print(f"  ({len(survey_filtered)} participants × {len(tasks_filtered)} problems)")
    
    if expected_simulations != len(gt):
        print(f"  ⚠ Note: Expected {expected_simulations} but GT has {len(gt)} responses")
        print(f"     This might be okay if not all participants answered all problems")
    
    return gt, survey_filtered, tasks_filtered


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print("="*70)
    print("XAI-FUNGI TURING EXPERIMENT v2 - LOGPROBS METHOD")
    print("="*70)
    
    # Initialize OpenAI client
    client = None
    if OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            print(f"✓ OpenAI client initialized")
        else:
            print("⚠ OPENAI_API_KEY not found - test mode")
    else:
        print("⚠ OpenAI library not installed - test mode")
    
    # Load and filter data (CRITICAL: only participants in ground truth)
    gt, survey, tasks = load_and_filter_data(
        args.ground_truth_file,
        args.characteristics_file,
        args.tasks_file
    )
    
    # Get XAI explanation summary (once, reused for all simulations)
    explanation = extract_xai_explanation_summary()
    
    # Model settings
    settings = ModelSettings(
        model=args.model,
        max_tokens=1,
        temperature=args.temperature,
        logprobs=5
    )
    
    # Run simulations
    print(f"\n🤖 Running simulations...")
    print(f"  Model: {settings.model}")
    print(f"  Temperature: {settings.temperature}")
    print(f"  Total simulations: {len(survey) * len(tasks)}")
    
    results = []
    
    for _, participant in tqdm(survey.iterrows(), total=len(survey), desc="Participants"):
        # Build persona from survey
        background = build_persona_background(participant, participant["group"])
        pid = participant["participant_id"]
        
        for _, task in tasks.iterrows():
            # Format task characteristics
            characteristics = format_mushroom_characteristics(task)
            
            # Simulate decision using logprobs
            decision_result = simulate_decision_with_logprobs(
                background=background,
                explanation=explanation,
                characteristics=characteristics,
                client=client,
                settings=settings
            )
            
            # Simulate certainty using logprobs
            certainty_result = simulate_certainty_with_logprobs(
                background=background,
                explanation=explanation,
                characteristics=characteristics,
                decision=decision_result["decision"],
                client=client,
                settings=settings
            )
            
            # Combine results
            result = {
                "problem_id": task["problem_id"],
                "participant_id": pid,
                "decision_sim": decision_result["decision"],
                "certainty_sim": certainty_result["certainty"],
                **{f"{k}_sim": v for k, v in decision_result.items()},
                **{f"{k}_sim": v for k, v in certainty_result.items()}
            }
            
            results.append(result)
    
    # Save simulated results
    sim_df = pd.DataFrame(results)
    sim_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Saved simulated results: {args.output_file}")
    
    # Merge with ground truth for evaluation
    print("\n📈 Merging with ground truth for evaluation...")
    merged = pd.merge(
        gt, 
        sim_df,
        on=["problem_id", "participant_id"],
        how="inner"
    )
    
    print(f"  Merged: {len(merged)} rows")
    
    if len(merged) == 0:
        print("❌ No matching rows after merge!")
        print("\nDebugging info:")
        print(f"  GT sample: {gt[['problem_id', 'participant_id']].head()}")
        print(f"  Sim sample: {sim_df[['problem_id', 'participant_id']].head()}")
        return
    
    # Evaluate
    metrics = evaluate_with_probabilities(merged)
    print_results(metrics)
    
    # Save merged data
    if args.save_merged:
        merged_path = args.output_file.replace(".csv", "_merged.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n✓ Saved merged results: {merged_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate XAI-FUNGI experiment using Turing Experiments logprobs method",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--characteristics_file",
        required=True,
        help="Survey responses CSV (SURVEY_en.csv)"
    )
    parser.add_argument(
        "--tasks_file",
        required=True,
        help="Mushroom problems CSV (PROBLEMS_en.csv)"
    )
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="Ground truth responses CSV (PROBLEMS_RESPONSES.csv)"
    )
    parser.add_argument(
        "--output_file",
        default="results_logprobs_final.csv",
        help="Output file for simulated results"
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo-instruct",
        help="OpenAI model (must support logprobs)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--save_merged",
        action="store_true",
        default=True,
        help="Save merged true+sim data"
    )
    
    args = parser.parse_args()
    main(args)