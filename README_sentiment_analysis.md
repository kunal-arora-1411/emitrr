# 💬 Medical Sentiment & Intent Analysis

A Jupyter Notebook for analyzing patient sentiment and intent in physician-patient conversations using transformer-based deep learning models.

## 📋 Overview

This notebook uses **DistilBERT** and **BART** to analyze patient statements from medical transcripts. It identifies emotional states (anxious vs. reassured) and communication intent (seeking reassurance, reporting symptoms, etc.) to help healthcare providers understand patient concerns better.

## ✨ Features

- **Sentiment Classification**: Detects if patients are anxious or reassured
- **Zero-Shot Intent Detection**: Identifies patient communication intent without training data
- **Patient-Specific Analysis**: Filters and analyzes only patient statements
- **Transformer Models**: Uses state-of-the-art pre-trained models
- **JSON Export**: Saves results in structured format for further analysis
- **Google Colab Ready**: Works seamlessly in cloud environments

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1. **Upload the notebook to Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Select `Sentment_Analysis.ipynb`

2. **Run all cells**
   - Click `Runtime` → `Run all`
   - Wait for dependencies to install (first run only)

3. **Modify the conversation**
   - Edit the `conversation` variable in Cell 2
   - Re-run from Cell 2 onwards

4. **Download results**
   - Results are saved to `analysis_results.json`
   - Download from the Files panel (left sidebar)

### Option 2: Local Jupyter Notebook

#### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- At least 4GB RAM (8GB recommended)

#### Installation

```bash
# Install dependencies
pip install transformers torch jupyter

# Launch Jupyter
jupyter notebook
```

#### Running the Notebook

1. Open `Sentment_Analysis.ipynb` in Jupyter
2. Run cells sequentially (Shift + Enter)
3. Modify the conversation in Cell 2
4. Results saved to `analysis_results.json` in the same directory

## 💻 Usage Guide

### Step-by-Step Walkthrough

#### Cell 1: Install Dependencies

```python
!pip install transformers torch
```

**What it does:** Installs the required libraries for transformer models.

**Note:** Use `!pip` in Jupyter/Colab. For terminal installation, use `pip` without `!`.

#### Cell 2: Input Your Conversation

```python
conversation = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""
```

**What to do:** 
- Replace the text between `"""` with your own transcript
- Keep the `Doctor:` and `Patient:` labels
- Maintain the triple quotes

#### Cell 3: Extract Patient Statements

```python
patient_lines = []
for line in conversation.split("\n"):
    if line.lower().startswith("patient:"):
        patient_lines.append(line.split(":", 1)[1].strip())
```

**What it does:** 
- Filters out only patient statements
- Removes "Patient:" prefix
- Cleans extra whitespace

**Output example:**
```
🩺 Extracted Patient Lines:
- I had a car accident. My neck and back hurt a lot for four weeks.
- Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
```

#### Cell 4: Load AI Models

```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english")
intent_classifier = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")

candidate_intents = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Expressing concern",
    "Reporting progress"
]
```

**What it does:**
- Loads pre-trained sentiment analysis model
- Loads zero-shot intent classification model
- Defines possible intent categories

**First run:** Downloads models (~500MB total). Subsequent runs are faster.

#### Cell 5: Analyze Sentiment & Intent

```python
results = []

for text in patient_lines:
    # Sentiment
    sentiment_raw = sentiment_analyzer(text)[0]['label']
    sentiment = "Anxious" if sentiment_raw == "NEGATIVE" else "Reassured"

    # Intent
    intent_result = intent_classifier(text, candidate_labels=candidate_intents)
    intent = intent_result['labels'][0]

    results.append({
        "Patient_Text": text,
        "Sentiment": sentiment,
        "Intent": intent
    })
```

**What it does:**
- Analyzes each patient statement
- Maps NEGATIVE → Anxious, POSITIVE → Reassured
- Predicts the most likely intent from candidates
- Stores results in structured format

#### Cell 6: Save Results

```python
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Results saved to analysis_results.json")
```

**What it does:** Exports results to a JSON file for further use.

## 📊 Example Output

### Input

```
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
```

### Output (analysis_results.json)

```json
[
  {
    "Patient_Text": "I had a car accident. My neck and back hurt a lot for four weeks.",
    "Sentiment": "Anxious",
    "Intent": "Reporting symptoms"
  },
  {
    "Patient_Text": "Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.",
    "Sentiment": "Reassured",
    "Intent": "Reporting progress"
  }
]
```

### Interpretation

**Statement 1:**
- **Sentiment**: Anxious - Patient is distressed about pain
- **Intent**: Reporting symptoms - Describing their condition

**Statement 2:**
- **Sentiment**: Reassured - Patient feels better about their recovery
- **Intent**: Reporting progress - Sharing treatment outcomes

## 🔧 Customization

### Adding New Intent Categories

Modify the `candidate_intents` list:

```python
candidate_intents = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Expressing concern",
    "Reporting progress",
    "Asking for advice",        # New
    "Expressing gratitude",     # New
    "Requesting clarification"  # New
]
```

The model will classify patient statements into the most likely category.

### Analyzing Doctor Statements

Modify the extraction logic:

```python
# Extract both doctor and patient statements
all_lines = []
for line in conversation.split("\n"):
    if line.lower().startswith("doctor:"):
        speaker = "Doctor"
        text = line.split(":", 1)[1].strip()
    elif line.lower().startswith("patient:"):
        speaker = "Patient"
        text = line.split(":", 1)[1].strip()
    else:
        continue
    
    all_lines.append({"speaker": speaker, "text": text})
```

### Fine-Grained Sentiment

Add more sentiment categories:

```python
# Use a different model with more labels
sentiment_analyzer = pipeline("text-classification", 
                             model="j-hartmann/emotion-english-distilroberta-base")

# This model outputs: anger, disgust, fear, joy, neutral, sadness, surprise
```

### Batch Processing Multiple Transcripts

```python
transcripts = [
    {"name": "Patient A", "conversation": "..."},
    {"name": "Patient B", "conversation": "..."},
    {"name": "Patient C", "conversation": "..."}
]

all_results = {}

for item in transcripts:
    patient_lines = extract_patient_lines(item["conversation"])
    results = analyze_sentiment_intent(patient_lines)
    all_results[item["name"]] = results

# Save all results
with open("batch_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

## 🎯 Use Cases

### 1. Clinical Communication Analysis

**Goal:** Understand patient emotional states during consultations.

**Application:**
- Identify patients who need more reassurance
- Flag anxious statements for follow-up
- Track emotional progress across multiple visits

### 2. Training Medical Students

**Goal:** Teach effective communication patterns.

**Application:**
- Analyze student-patient interactions
- Identify missed opportunities for reassurance
- Provide feedback on response to patient concerns

### 3. Telemedicine Quality Assurance

**Goal:** Monitor virtual consultation quality.

**Application:**
- Automated analysis of telehealth transcripts
- Identify patients with unresolved anxiety
- Quality metrics for provider communication

### 4. Mental Health Screening

**Goal:** Early detection of distress signals.

**Application:**
- Flag consistently anxious sentiment patterns
- Identify patients expressing concerning intent
- Trigger mental health referrals when appropriate

## 🤔 Understanding the Models

### DistilBERT for Sentiment Analysis

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

**Training:**
- Distilled version of BERT (40% smaller, 60% faster)
- Fine-tuned on Stanford Sentiment Treebank (SST-2)
- Binary classification: POSITIVE or NEGATIVE

**Advantages:**
- Fast inference (~10ms per sentence on CPU)
- Good accuracy for general sentiment
- Low memory footprint

**Limitations:**
- Binary sentiment only (no neutral option)
- Not specifically trained on medical text
- May miss medical-specific emotional nuances

### BART for Zero-Shot Intent Classification

**Model:** `facebook/bart-large-mnli`

**Training:**
- Pre-trained on natural language inference (NLI)
- Can classify into arbitrary categories without training
- Uses entailment scores for classification

**Advantages:**
- No training data required
- Flexible - add new intents anytime
- Works well on diverse text types

**Limitations:**
- Slower than specialized classifiers
- May require careful intent phrasing
- Confidence scores can be uncertain

### How Zero-Shot Classification Works

```python
# The model treats classification as an entailment task
# For text: "I'm worried about my back pain"
# Intent: "Expressing concern"

# Model checks: "This text implies expressing concern" → TRUE/FALSE
# Scores all intents and picks the highest
```

## 🔬 Fine-Tuning for Medical Context

### Why Fine-Tune?

Pre-trained models are trained on general text, not medical conversations. Fine-tuning improves:
- Medical terminology understanding
- Context-specific sentiment (e.g., "stable" is positive in medical settings)
- Domain-specific intent recognition

### Recommended Datasets

**For Sentiment Analysis:**
1. **Medical Transcriptions Dataset** (Kaggle)
   - ~5,000 medical transcripts
   - Various specialties
   - Requires manual sentiment labeling

2. **MIMIC-III Clinical Notes**
   - ICU patient records
   - Rich emotional context
   - Requires access approval

3. **Patient Reviews** (Drugs.com, Yelp Health)
   - Naturally labeled (star ratings)
   - Patient perspective
   - May not reflect in-person consultations

**For Intent Classification:**
1. **ChatDoctor Dataset**
   - Patient-doctor conversations
   - Various intent types
   - Good for training intent classifiers

2. **MedDialog**
   - English and Chinese medical dialogues
   - Labeled conversation acts
   - Research dataset

### Fine-Tuning Code Example

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load your labeled medical conversations
train_data = Dataset.from_dict({
    "text": ["I'm worried about my symptoms...", "The pain is getting better..."],
    "label": [0, 1]  # 0: Anxious, 1: Reassured
})

# Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training configuration
training_args = TrainingArguments(
    output_dir='./medical_sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()
```

## 📈 Performance Considerations

### Runtime

**First Run:**
- Model download: 2-5 minutes (one-time)
- Model loading: 10-20 seconds
- Analysis: ~1 second per statement

**Subsequent Runs:**
- Model loading: 5-10 seconds (cached)
- Analysis: ~1 second per statement

### Hardware Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 4GB
- Storage: 2GB (for models)

**Recommended:**
- CPU: Multi-core (faster batch processing)
- RAM: 8GB+
- GPU: Optional (3-5x speedup)

**Google Colab:**
- Free tier sufficient for small datasets
- GPU runtime available (faster)
- TPU not recommended for these models

### Optimization Tips

**1. Batch Processing**

```python
# Instead of processing one by one
for text in patient_lines:
    result = sentiment_analyzer(text)

# Process in batches
results = sentiment_analyzer(patient_lines, batch_size=8)
```

**2. Model Quantization**

```python
# Use smaller quantized models
from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU
    framework="pt",
    model_kwargs={"torchscript": True}  # Optimized inference
)
```

**3. GPU Acceleration**

```python
# Use GPU if available
import torch

device = 0 if torch.cuda.is_available() else -1

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)
```

## ⚠️ Limitations

### Current Limitations

1. **Binary Sentiment Only**
   - Only classifies as Anxious or Reassured
   - No neutral or mixed sentiment detection
   - May oversimplify complex emotional states

2. **English Only**
   - Models trained on English text
   - Performance degrades on other languages
   - No multilingual support

3. **No Context Memory**
   - Each statement analyzed independently
   - Doesn't consider conversation history
   - May miss sentiment shifts over time

4. **General Medical Focus**
   - Not specialized for specific medical domains
   - May miss field-specific terminology
   - Requires validation for specialized use

5. **Patient Speaker Only**
   - Current setup analyzes only patient statements
   - Doesn't analyze doctor empathy or communication
   - Misses conversational dynamics

### Handling Edge Cases

**Sarcasm or Irony:**
```
Patient: "Oh great, more pain. Just what I needed."
→ May classify as Reassured (due to "great")
```

**Solution:** Fine-tune on medical conversations with irony examples.

**Mixed Emotions:**
```
Patient: "I'm glad the pain is better, but I'm worried it might come back."
→ Returns single sentiment (Reassured or Anxious)
```

**Solution:** Use multi-label classification or sentence-level analysis.

**Short Statements:**
```
Patient: "Okay."
→ Insufficient context for accurate classification
```

**Solution:** Add minimum word count or combine with previous statement.

## 🚀 Advanced Features

### Multi-Turn Context Analysis

Track sentiment changes over conversation:

```python
sentiment_timeline = []

for i, text in enumerate(patient_lines):
    sentiment = analyze_sentiment(text)
    sentiment_timeline.append({
        "turn": i + 1,
        "text": text,
        "sentiment": sentiment
    })

# Identify sentiment shifts
for i in range(1, len(sentiment_timeline)):
    if sentiment_timeline[i]["sentiment"] != sentiment_timeline[i-1]["sentiment"]:
        print(f"Sentiment shift at turn {i}: {sentiment_timeline[i-1]['sentiment']} → {sentiment_timeline[i]['sentiment']}")
```

### Confidence Scores

Access model confidence:

```python
result = sentiment_analyzer(text, return_all_scores=True)
# Returns: [{'label': 'POSITIVE', 'score': 0.95}, {'label': 'NEGATIVE', 'score': 0.05}]

confidence = max(result[0]['score'], result[1]['score'])
if confidence < 0.7:
    print(f"Low confidence prediction: {confidence:.2f}")
```

### Intent Probability Distribution

See all intent probabilities:

```python
intent_result = intent_classifier(text, candidate_labels=candidate_intents)

print(f"Text: {text}")
for label, score in zip(intent_result['labels'], intent_result['scores']):
    print(f"  {label}: {score:.3f}")
```

### Emotion Detection (Beyond Sentiment)

Use emotion-specific models:

```python
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

emotions = emotion_classifier(text)
# Returns: anger, disgust, fear, joy, neutral, sadness, surprise
```

## 🧪 Testing & Validation

### Test Cases

```python
# Test 1: Clearly anxious statement
text1 = "I'm very worried about my condition and the pain is unbearable"
assert analyze_sentiment(text1) == "Anxious"

# Test 2: Clearly reassured statement
text2 = "I'm feeling much better now, the treatment really helped"
assert analyze_sentiment(text2) == "Reassured"

# Test 3: Intent detection
text3 = "Will my symptoms get worse over time?"
assert analyze_intent(text3) == "Seeking reassurance"

# Test 4: Progress reporting
text4 = "The physiotherapy sessions have reduced my pain significantly"
assert analyze_intent(text4) == "Reporting progress"
```

### Accuracy Validation

For clinical validation:

1. **Manual Annotation:** Have medical professionals label a test set
2. **Inter-Rater Reliability:** Measure agreement between annotators
3. **Model Comparison:** Compare model predictions to manual labels
4. **Error Analysis:** Identify common failure patterns

## 📚 Additional Resources

### Tutorials

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Sentiment Analysis Tutorial](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Zero-Shot Classification Guide](https://huggingface.co/tasks/zero-shot-classification)

### Research Papers

- **DistilBERT:** Sanh et al., "DistilBERT, a distilled version of BERT" (2019)
- **BART:** Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training" (2019)
- **Medical Sentiment Analysis:** Review papers on healthcare NLP

### Alternative Models

- **BioBERT:** Better for medical terminology
- **ClinicalBERT:** Specialized for clinical notes
- **RoBERTa:** Often better than BERT for sentiment
- **GPT-based models:** For more nuanced analysis

## 🔒 Privacy & Ethics

### Data Privacy

**Important Considerations:**
- Remove patient identifying information before analysis
- Comply with HIPAA regulations for US healthcare data
- Follow GDPR for European patients
- Obtain informed consent for AI analysis

**Data Anonymization:**
```python
import re

def anonymize_text(text):
    # Remove names (simplified)
    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
    # Remove dates
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '[DATE]', text)
    # Remove ages
    text = re.sub(r'\b\d{1,3} years? old\b', '[AGE] years old', text)
    return text
```

### Ethical Usage

**Appropriate Uses:**
- Supporting clinician decision-making
- Quality assurance and training
- Research with proper oversight
- Patient communication improvement

**Inappropriate Uses:**
- Replacing human clinical judgment
- Automated diagnosis or treatment decisions
- Patient discrimination based on sentiment
- Unsupervised deployment without validation

### Bias Considerations

AI models may have biases:
- **Cultural bias:** Different emotional expression across cultures
- **Language bias:** Trained primarily on Western English
- **Gender bias:** May interpret emotions differently by gender
- **Age bias:** Different communication patterns by age group

**Mitigation:**
- Validate on diverse patient populations
- Regular bias audits
- Human oversight for critical decisions
- Transparent reporting of limitations

## 📞 Troubleshooting

### Common Issues

**1. Model Download Fails**
```
Error: Connection timeout
```
**Solution:** Check internet connection or use offline model loading.

**2. Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or switch to CPU (`device=-1`).

**3. No Patient Lines Extracted**
```
Extracted Patient Lines: (empty)
```
**Solution:** Check that lines start with "Patient:" (case-insensitive).

**4. Import Error**
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution:** Run `pip install transformers torch` in terminal/cell.

**5. Slow Performance**
```
Analysis taking >1 minute per sentence
```
**Solution:** Use GPU or smaller model (distilbert instead of bert-base).

## 📄 License

This notebook is for educational and research purposes. Ensure compliance with healthcare regulations (HIPAA, GDPR) before clinical deployment.

## ⚕️ Medical Disclaimer

**Critical:** This tool provides supportive information only and should never replace professional medical judgment. All sentiment and intent predictions must be validated by qualified healthcare professionals.

---

**Questions?** Review the [use cases](#-use-cases) and [customization](#-customization) sections for guidance.
