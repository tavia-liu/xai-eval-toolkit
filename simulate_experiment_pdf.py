"""

Key features:

1. **Automatic slide extraction:**  The script uses the
   ``pdfminer.six`` library to extract text from
   ``ORIGINAL_VISUALIZATIONS_EN.pdf``.  The entire extracted text is
   passed to the LLM as the explanation summary.  If you prefer to
   provide a curated summary, replace the ``extract_text`` call with
   your own string.
2. **No external summary file:**  Unlike earlier versions, this
   script does not require a ``--slides_summary_file`` argument.  It
   reads the PDF directly.
3. **LLM integration:**  The script expects an API key via the
   ``OPENAI_API_KEY`` environment variable.  If the key is missing,
   dummy answers are returned.  You can substitute your own LLM call
   inside ``simulate_response`` if desired.

Usage example:

```
export OPENAI_API_KEY=''
python simulate_experiment_pdf.py \
    --characteristics_file SURVEY_en.csv \
    --tasks_file PROBLEMS_en.csv \
    --ground_truth_file PROBLEMS_RESPONSES.csv \
    --output_file responses.csv
```

Before running, ensure that ``pdfminer.six`` is installed:
``pip install pdfminer.six``.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    # pdfminer.six must be installed in your environment to extract text
    from pdfminer.high_level import extract_text  # type: ignore
except ImportError:
    extract_text = None  # type: ignore[assignment]

try:
    import openai
except ImportError:
    openai = None


def extract_group(participant_id: str) -> str:
    """Return the group code (DE, IT, or SSH) from a participant_id."""
    if not isinstance(participant_id, str):
        return ""
    for part in participant_id.split("_"):
        if part in {"DE", "IT", "SSH"}:
            return part
    return ""


def assemble_task_descriptions(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine feature columns into a single semicolon‑separated description.

    By default all columns except ``problem_id`` are included in the
    description.  In the original XAI‑FUNGI experiment participants were
    shown the model’s predicted class (``model_class``) and its confidence
    (``model_probability``) alongside the mushroom’s features.  Including
    these columns in the description helps the LLM simulate the same
    decision context that humans experienced.
    """
    # Only ``problem_id`` is excluded from the description.  We keep
    # ``model_class`` and ``model_probability`` so the LLM knows what the
    # underlying AI predicted.
    exclude = {"problem_id"}
    descriptions = []
    for _, row in tasks_df.iterrows():
        parts = []
        for col in tasks_df.columns:
            if col in exclude:
                continue
            val = str(row[col])
            if val and val.lower() not in {"nan", "n/a"}:
                parts.append(f"{col}: {val}")
        descriptions.append("; ".join(parts))
    tasks_df = tasks_df.copy()
    tasks_df["description"] = descriptions
    return tasks_df[["problem_id", "description"]]


def simulate_response(
    participant_traits: str,
    group: str,
    slide_summary: str,
    task_description: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> Tuple[str, str]:
    """
    Simulate a single decision/certainty pair for one participant and task.

    The prompt describes the participant and their group, includes the
    explanation summary extracted from the PDF, and asks the LLM to
    classify the mushroom and rate certainty.  If the ``openai`` module
    or API key is unavailable, a stub response is returned.
    """
    group_desc_map = {
        "DE": "a domain expert in the field of mycology (DE)",
        "IT": "a student with a data science and visualisation background (IT)",
        "SSH": "a student from social sciences and humanities (SSH)",
    }
    group_desc = group_desc_map.get(group, group)
    # Compose a prompt that encourages the model to role‑play as the
    # participant.  We provide the participant’s traits, group and the
    # AI explanation (slide summary), and we include the AI model’s
    # prediction/probability in the task description.  The prompt
    # explicitly asks the LLM to mimic how this participant would decide
    # rather than simply classifying the mushroom on its own.
    prompt = (
        "You are taking part in a human‑computer study. You must role‑play "
        "as the participant described below and answer as they would.\n\n"
        f"Participant traits: {participant_traits}\n"
        f"Participant group: {group_desc}\n\n"
        f"Explanation (from the AI system):\n{slide_summary}\n\n"
        f"Task description (including the AI model's prediction): {task_description}\n\n"
        "As this participant, decide whether the mushroom is edible or non‑edible. "
        "Then select your certainty level from the following options: "
        "'definitely certain', 'moderately certain', 'I can't assess', "
        "'moderately uncertain', 'definitely uncertain'.\n"
        "Return your answer exactly in the format: decision | certainty"
    )
    # Call the LLM if available
    if openai is not None and getattr(openai, "api_key", None):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response["choices"][0]["message"]["content"].strip()
        if "|" in content:
            decision, certainty = [p.strip() for p in content.split("|", 1)]
        else:
            decision, certainty = content, "I can't assess"
    else:
        decision, certainty = "non-edible", "moderately certain"
    # Normalise decision
    dec = decision.lower().replace("eadible", "edible").replace("non-eadible", "non-edible").replace("non edible", "non-edible").strip()
    if dec.startswith("non"):
        dec = "non-edible"
    elif dec.startswith("edible"):
        dec = "edible"
    else:
        dec = decision
    return dec, certainty


def main(args: argparse.Namespace) -> None:
    # Check pdf extraction capability
    if extract_text is None:
        raise RuntimeError(
            "pdfminer.six is not installed. Please install it with `pip install pdfminer.six`"
        )
    # Extract slide summary from the PDF
    pdf_path = Path("ORIGINAL_VISUALIZATIONS_EN.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    print("Extracting explanation text from", pdf_path)
    slide_summary = extract_text(str(pdf_path))  # type: ignore[arg-type]

    # Load ground truth responses
    gt_df = pd.read_csv(args.ground_truth_file, dtype=str)
    participant_ids = gt_df["participant_id"].dropna().unique().tolist()

    # Load survey data and filter participants present in ground truth
    survey_df = pd.read_csv(args.characteristics_file, dtype=str)
    survey_df = survey_df[survey_df["participant_id"].isin(participant_ids)].copy()
    survey_df["group"] = survey_df["participant_id"].apply(extract_group)
    survey_df = survey_df[survey_df["group"].astype(bool)]

    # Select trait columns to provide to the LLM.  To ensure the model
    # sees the full personality of each participant we include all survey
    # questions except identifiers and metadata.  You can customise this
    # list if you wish to limit the prompt length.
    excluded_trait_cols = {
        "candidate_id", "Start time", "End time", "Email address",
        "participant_id", "comment", "participant_group"
    }
    trait_columns = [c for c in survey_df.columns if c not in excluded_trait_cols]
    survey_df[trait_columns] = survey_df[trait_columns].fillna("")

    # Load tasks and build descriptions
    tasks_df = pd.read_csv(args.tasks_file, dtype=str)
    tasks_df = assemble_task_descriptions(tasks_df)

    # Collect simulated responses
    results: List[Tuple[str, str, str, str]] = []
    for _, participant in survey_df.iterrows():
        traits = []
        for col in trait_columns:
            val = participant.get(col, "")
            if val:
                traits.append(f"{col}: {val}")
        trait_str = "; ".join(traits)
        group = participant["group"]
        pid = participant["participant_id"]
        for _, task in tasks_df.iterrows():
            dec, cert = simulate_response(trait_str, group, slide_summary, task["description"])
            results.append((task["problem_id"], pid, dec, cert))

    # Save to CSV
    resp_df = pd.DataFrame(results, columns=[
        "problem_id",
        "participant_id",
        "participant_decision_en",
        "participant_certaintity_en",
    ])
    resp_df.to_csv(args.output_file, index=False)

    # Evaluate
    # Normalize decision strings in both true and simulated data to handle typos
    def normalize_decision(s: str) -> str:
        """Normalize decision labels to 'edible' or 'non-edible'."""
        if not isinstance(s, str):
            return str(s)
        s = s.lower().strip()
        # common typos: eadible/non-eadible, missing hyphen, etc.
        s = s.replace("eadible", "edible")
        s = s.replace("non-eadible", "non-edible")
        s = s.replace("non edible", "non-edible")
        s = s.replace("non-ediblee", "non-edible")
        if s.startswith("non") and "edible" in s:
            return "non-edible"
        if "edible" in s:
            return "edible"
        return s

    merged = pd.merge(gt_df, resp_df, on=["problem_id", "participant_id"], how="inner", suffixes=("_true", "_sim"))
    total = len(merged)
    if total > 0:
        true_decisions = merged["participant_decision_en_true"].apply(normalize_decision)
        sim_decisions = merged["participant_decision_en_sim"].apply(normalize_decision)
        dec_acc = (true_decisions == sim_decisions).mean()
        true_certs = merged["participant_certaintity_en_true"].str.lower().str.strip()
        sim_certs = merged["participant_certaintity_en_sim"].str.lower().str.strip()
        cert_acc = (true_certs == sim_certs).mean()
    else:
        dec_acc = cert_acc = 0.0
    print(f"Decision accuracy: {dec_acc:.2%}")
    print(f"Certainty accuracy: {cert_acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate XAI‑FUNGI responses using PDF slides and an LLM")
    parser.add_argument("--characteristics_file", required=True, help="Path to SURVEY_en.csv")
    parser.add_argument("--tasks_file", required=True, help="Path to PROBLEMS_en.csv")
    parser.add_argument("--ground_truth_file", required=True, help="Path to PROBLEMS_RESPONSES.csv")
    parser.add_argument("--output_file", default="responses.csv", help="CSV file to save simulated responses")
    args = parser.parse_args()
    # Assign API key if available
    if openai is not None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
    main(args)