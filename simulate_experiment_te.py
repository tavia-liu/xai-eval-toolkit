"""
simulate_experiment_te.py
=========================

Simulates participant decisions in the XAI-FUNGI experiment.

Steps:
1. Render slide images from a PDF (PyMuPDF).
2. Use full survey responses (except IDs/emails) to build participant personas.
3. Combine task features into descriptions.
4. Prompt an OpenAI vision model to "role-play" as participants.
5. Save simulated responses and compare with ground truth.

Usage:
    export OPENAI_API_KEY='sk-...'
    python simulate_experiment_te.py \
      --characteristics_file SURVEY_en.csv \
      --tasks_file PROBLEMS_en.csv \
      --ground_truth_file PROBLEMS_RESPONSES.csv \
      --slides_pdf ORIGINAL_VISUALIZATIONS_EN.pdf \
      --output_file responses_te.csv \
      --model gpt-4o-mini \
      --max_slides 15
"""

import argparse, base64, os
from pathlib import Path
from typing import List, Tuple
import pandas as pd

try:
    import fitz
except ImportError:
    fitz = None
try:
    import openai
except ImportError:
    openai = None


# ----------------- Helpers -----------------

def extract_group(pid: str) -> str:
    if isinstance(pid, str):
        for part in pid.split("_"):
            if part in {"DE", "IT", "SSH"}:
                return part
    return ""


def load_slides(pdf_path: str, max_slides: int) -> List[str]:
    if fitz is None: return []
    doc, out = fitz.open(pdf_path), []
    for i in range(min(len(doc), max_slides)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
        out.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return out


def make_task_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {"problem_id","model_class","model_probability"}
    def row_desc(row):
        parts = [f"{c}: {v}" for c, v in row.items()
                 if c not in exclude and str(v).strip().lower() not in {"", "nan", "n/a"}]
        return "; ".join(parts)
    df = df.copy()
    df["description"] = df.apply(row_desc, axis=1)
    return df[["problem_id", "description"]]


def build_persona(row: pd.Series) -> str:
    """
    Build a participant persona in Q/A format from the full survey.
    Excludes identifiers and admin fields.
    """
    excluded = {
        "candidate_id", "Start time", "End time", "Email address",
        "participant_id", "comment", "participant_group"
    }
    qa_pairs = []
    for col, val in row.items():
        if col in excluded:
            continue
        val_str = str(val).strip()
        if val_str and val_str.lower() not in {"nan", "n/a"}:
            qa_pairs.append(f"Q: {col}\nA: {val_str}")
    return "\n\n".join(qa_pairs)


def normalise_decision(s) -> str:
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    if s.startswith("non") and "edible" in s: return "non-edible"
    if "edible" in s: return "edible"
    return s


# ----------------- Simulation -----------------

def simulate_response(name: str, traits: str, group: str, slides: List[str],
                      task_desc: str, model: str, temp: float) -> Tuple[str, str]:
    groups = {"DE": "a domain expert in mycology",
              "IT": "a data-visualisation student",
              "SSH": "a social sciences/humanities student"}
    prompt = (
    f"You are {name}, {groups.get(group, group)}.\n\n"
    "Here is your full pre-survey:\n"
    f"{traits}\n\n"
    "You have studied the explanation slides. "
    f"Now consider this mushroom: {task_desc}\n\n"
    "Decide if it is edible or non-edible, then pick certainty "
    "from: definitely certain, moderately certain, I can't assess, "
    "moderately uncertain, definitely uncertain.\n"
    "Format: decision | certainty"
)

    if openai and getattr(openai, "api_key", None):
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} for b64 in slides
        ]
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": content}],
                temperature=temp,
            )
            out = resp["choices"][0]["message"]["content"].strip()
            return (out.split("|",1)[0].strip(), out.split("|",1)[1].strip()) if "|" in out else (out, "I can't assess")
        except Exception:
            return "non-edible", "moderately certain"
    return "non-edible", "moderately certain"


# ----------------- Main -----------------

def main(args):
    slides = load_slides(args.slides_pdf, args.max_slides)
    gt = pd.read_csv(args.ground_truth_file, dtype=str)
    survey = pd.read_csv(args.characteristics_file, dtype=str)
    survey = survey[survey["participant_id"].isin(gt["participant_id"].dropna())].copy()
    survey["group"] = survey["participant_id"].apply(extract_group)
    survey = survey[survey["group"].astype(bool)].fillna("")
    tasks = make_task_descriptions(pd.read_csv(args.tasks_file, dtype=str))

    results = []
    for _, p in survey.iterrows():
        persona = build_persona(p)
        for _, t in tasks.iterrows():
            d, c = simulate_response(p["participant_id"], persona, p["group"],
                                     slides, t["description"], args.model, args.temperature)
            results.append((t["problem_id"], p["participant_id"], normalise_decision(d), c))

    resp = pd.DataFrame(results, columns=["problem_id","participant_id","participant_decision_en","participant_certainty_en"])
    resp.to_csv(args.output_file, index=False)

    merged = pd.merge(gt, resp, on=["problem_id","participant_id"], suffixes=("_true","_sim"))
    if len(merged):
        # normalise decisions
        merged["dec_true"] = merged["participant_decision_en_true"].apply(normalise_decision)
        merged["dec_sim"]  = merged["participant_decision_en_sim"].apply(normalise_decision)
        merged["cert_true"] = merged["participant_certainty_en_true"].str.lower().str.strip()
        merged["cert_sim"]  = merged["participant_certainty_en_sim"].str.lower().str.strip()

        # overall accuracy
        dec_acc  = (merged["dec_true"] == merged["dec_sim"]).mean()
        cert_acc = (merged["cert_true"] == merged["cert_sim"]).mean()
        print(f"Decision accuracy (overall): {dec_acc:.2%}")
        print(f"Certainty accuracy (overall): {cert_acc:.2%}")

        # 🔍 per-problem accuracy
        print("\nPer-problem accuracies:")
        grouped = merged.groupby("problem_id")
        for pid, group in grouped:
            d_acc = (group["dec_true"] == group["dec_sim"]).mean()
            c_acc = (group["cert_true"] == group["cert_sim"]).mean()
            print(f"  {pid}: decision={d_acc:.2%}, certainty={c_acc:.2%}")
    else:
        print("No overlapping participants for evaluation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulate the XAI-FUNGI experiment with full survey personas.")
    ap.add_argument("--characteristics_file", required=True)
    ap.add_argument("--tasks_file", required=True)
    ap.add_argument("--ground_truth_file", required=True)
    ap.add_argument("--slides_pdf", default="ORIGINAL_VISUALIZATIONS_EN.pdf")
    ap.add_argument("--output_file", default="responses_te.csv")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_slides", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    if openai: 
        key = os.getenv("OPENAI_API_KEY")
        if key: openai.api_key = key

    main(args)
