# 🩺 Medical Transcript Summarization

An NLP-based tool for extracting medical entities and generating structured summaries from physician-patient conversations.

## 📋 Overview

This Python script uses **SciSpacy** and **YAKE** to automatically extract medical information from conversation transcripts and output structured JSON summaries. It identifies symptoms, diagnoses, treatments, prognosis, and relevant keywords.

## ✨ Features

- **Named Entity Recognition (NER)**: Extracts medical entities using SciSpacy
- **Entity Categorization**: Organizes information into Symptoms, Diagnosis, Treatment, and Prognosis
- **Keyword Extraction**: Identifies important medical phrases using YAKE algorithm
- **JSON Output**: Generates structured, machine-readable medical summaries
- **Duplicate Removal**: Automatically deduplicates extracted entities

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install spacy scispacy yake
```

### Step 2: Download SciSpacy Model

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
```

### Alternative Installation

Create a `requirements.txt` file:

```txt
spacy>=3.5.0
scispacy>=0.5.1
yake>=0.4.8
```

Install all at once:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

Run the script directly:

```bash
python medical_transcript.py
```

This will process the default example transcript and output the JSON summary.

### Using as a Module

Import and use the functions in your own code:

```python
from medical_transcript import summarize_to_json, extract_medical_entities, extract_keywords

# Your conversation transcript
transcript = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""

# Generate structured summary
patient_name = "Janet Jones"
result = summarize_to_json(patient_name, transcript)
print(result)
```

### Custom Entity Extraction

Extract only medical entities:

```python
from medical_transcript import extract_medical_entities

transcript = "Patient complains of severe headache and neck pain."
entities = extract_medical_entities(transcript)
print(entities)
```

**Output:**

```python
{
    "Symptoms": ["headache", "neck pain"],
    "Diagnosis": [],
    "Treatment": [],
    "Prognosis": []
}
```

### Keyword Extraction Only

Extract keywords from medical text:

```python
from medical_transcript import extract_keywords

text = "Patient underwent physiotherapy for whiplash injury after car accident."
keywords = extract_keywords(text, max_keywords=5)
print(keywords)
```

**Output:**

```python
["whiplash injury", "physiotherapy", "car accident", "patient", "underwent"]
```

## 📊 Example Output

### Input Transcript

```
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
```

### JSON Output

```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": [
    "neck hurt",
    "back hurt",
    "back pain"
  ],
  "Diagnosis": [],
  "Treatment": [
    "physiotherapy sessions"
  ],
  "Prognosis": [],
  "Keywords": [
    "physiotherapy sessions",
    "back pain",
    "occasional back pain",
    "car accident",
    "hurt"
  ]
}
```

## 🔧 How It Works

### 1. Medical Entity Extraction

The `extract_medical_entities()` function uses a rule-based approach:

**Symptoms Detection:**
- Keywords: "pain", "ache", "injury", "hurt", "discomfort", "headache"
- Example: "neck pain" → categorized as Symptom

**Diagnosis Detection:**
- Keywords: "diagnosed", "disease", "condition", "injury", "syndrome"
- Example: "whiplash injury" → categorized as Diagnosis

**Treatment Detection:**
- Keywords: "therapy", "treatment", "session", "medicine", "painkiller", "prescribed", "physiotherapy"
- Example: "physiotherapy sessions" → categorized as Treatment

**Prognosis Detection:**
- Keywords: "recovery", "improve", "heal", "better", "normal", "expected"
- Example: "expected recovery" → categorized as Prognosis

### 2. Keyword Extraction

Uses YAKE (Yet Another Keyword Extractor):
- Unsupervised algorithm
- Extracts single and multi-word keywords
- Ranks by relevance score
- Configurable number of keywords (default: 10)

### 3. Structured Output

Combines entity extraction and keywords into a JSON structure:
- Patient identification
- Categorized medical entities
- Important keywords
- Duplicate removal for cleaner output

## 📝 Function Reference

### `extract_medical_entities(text)`

Extracts medical entities from text.

**Parameters:**
- `text` (str): The conversation transcript or medical text

**Returns:**
- `dict`: Dictionary with keys "Symptoms", "Diagnosis", "Treatment", "Prognosis"

**Example:**

```python
entities = extract_medical_entities("Patient has severe headache")
# Returns: {"Symptoms": ["severe headache"], "Diagnosis": [], ...}
```

### `extract_keywords(text, max_keywords=10)`

Extracts important keywords from text.

**Parameters:**
- `text` (str): The text to extract keywords from
- `max_keywords` (int): Maximum number of keywords to extract (default: 10)

**Returns:**
- `list`: List of keyword strings

**Example:**

```python
keywords = extract_keywords("Patient underwent surgery", max_keywords=5)
# Returns: ["patient", "underwent", "surgery", ...]
```

### `summarize_to_json(patient_name, text)`

Generates a complete medical summary in JSON format.

**Parameters:**
- `patient_name` (str): Name of the patient
- `text` (str): The conversation transcript

**Returns:**
- `str`: JSON-formatted string with complete summary

**Example:**

```python
json_summary = summarize_to_json("John Doe", transcript)
print(json_summary)
```

## 🔍 Customization

### Modifying Entity Keywords

Edit the keyword lists in `extract_medical_entities()`:

```python
# Add custom symptom keywords
if any(k in t for k in ["pain", "ache", "injury", "hurt", "nausea", "fatigue"]):
    entities["Symptoms"].append(ent.text)

# Add custom treatment keywords
elif any(k in t for k in ["therapy", "treatment", "surgery", "medication"]):
    entities["Treatment"].append(ent.text)
```

### Adjusting Keyword Count

Change the number of extracted keywords:

```python
# Extract more keywords
keywords = extract_keywords(text, max_keywords=20)

# Extract fewer keywords
keywords = extract_keywords(text, max_keywords=5)
```

### Adding New Categories

Extend the entity dictionary:

```python
entities = {
    "Symptoms": [],
    "Diagnosis": [],
    "Treatment": [],
    "Prognosis": [],
    "Medications": [],  # New category
    "Tests": []         # New category
}
```

## 🤔 Handling Edge Cases

### Ambiguous or Missing Data

**Challenge:** Not all transcripts contain clear medical entities.

**Solution:**
- Returns empty lists for missing categories
- No assumptions made about absent information
- Keywords provide additional context when entities are unclear

**Example:**

```python
# Vague transcript
transcript = "Patient is doing fine."
result = summarize_to_json("Jane Doe", transcript)

# Output:
# {
#   "Patient_Name": "Jane Doe",
#   "Symptoms": [],
#   "Diagnosis": [],
#   "Treatment": [],
#   "Prognosis": ["fine"],  # Detected from keyword
#   "Keywords": ["patient", "fine", "doing"]
# }
```

### Context-Dependent Meanings

**Challenge:** Words like "fine" can be symptoms or prognosis depending on context.

**Current Approach:**
- Rule-based classification by keyword matching
- Entity context considered (surrounding words)

**Enhancement Ideas:**
- Add confidence scores for classifications
- Use contextual embeddings (BERT) for better understanding
- Implement relation extraction between entities

### Medical Abbreviations

**Current Limitation:** May not recognize medical abbreviations (e.g., "PT" for physiotherapy).

**Workarounds:**
- Expand abbreviations before processing
- Add abbreviations to keyword lists
- Use specialized medical NER models

## 🎯 Use Cases

### Clinical Documentation

Generate structured notes from voice-recorded consultations:

```python
# Process audio transcript
transcript = get_transcript_from_audio()
summary = summarize_to_json(patient.name, transcript)
save_to_ehr(summary)
```

### Medical Research

Extract entities from large corpora of medical conversations:

```python
# Batch processing
for transcript in medical_transcripts:
    entities = extract_medical_entities(transcript)
    analyze_symptom_patterns(entities["Symptoms"])
```

### Telemedicine

Real-time summarization during virtual consultations:

```python
# Live processing
conversation = capture_live_conversation()
summary = summarize_to_json(patient_name, conversation)
display_summary_to_physician(summary)
```

## ⚠️ Limitations

1. **Rule-based Extraction**: May miss complex medical terminology not in keyword lists
2. **No Temporal Information**: Doesn't extract dates, durations, or timelines
3. **No Dosage Extraction**: Medication amounts and frequencies not captured
4. **Single Speaker**: Doesn't distinguish between multiple doctors or patients
5. **English Only**: Currently supports only English text

## 🚀 Future Enhancements

### Short-term

- [ ] Add medication name and dosage extraction
- [ ] Extract temporal information (dates, durations)
- [ ] Implement confidence scores for entity classifications
- [ ] Support for medical abbreviations

### Long-term

- [ ] Train custom NER model on medical conversations
- [ ] Add relationship extraction between entities
- [ ] Multi-language support
- [ ] Integration with EHR systems
- [ ] Real-time processing for live transcription

## 🔬 Technical Details

### Why SciSpacy?

**Advantages:**
- Pre-trained on biomedical literature (PubMed, MIMIC-III)
- Recognizes medical entities better than general NER
- Lightweight and fast inference
- Easy integration with standard spaCy pipelines

**Alternatives:**
- **BioBERT**: Better accuracy but slower
- **ClinicalBERT**: Specialized for clinical notes
- **Custom NER**: Trained on specific medical domains

### Why YAKE?

**Advantages:**
- Unsupervised (no training data needed)
- Language-independent
- Fast and efficient
- Good performance on medical text

**Alternatives:**
- **KeyBERT**: Uses BERT embeddings for better semantic understanding
- **RAKE**: Rapid Automatic Keyword Extraction
- **TextRank**: Graph-based keyword extraction

## 📚 Model Information

### SciSpacy Model: en_core_sci_sm

- **Size**: Small (~15MB)
- **Training Data**: Biomedical literature
- **Entities**: Disease, chemical, gene, protein, cell type, etc.
- **Performance**: Balanced speed and accuracy

### Upgrading to Larger Models

For better accuracy, use larger SciSpacy models:

```bash
# Medium model (~100MB)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Large model (~800MB)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
```

Update the code:

```python
# Load larger model
nlp = spacy.load("en_core_sci_md")  # or en_core_sci_lg
```

## 🧪 Testing

### Sample Test Cases

```python
# Test 1: Clear symptoms
text1 = "Patient complains of severe headache and dizziness"
assert "headache" in extract_medical_entities(text1)["Symptoms"]

# Test 2: Multiple treatments
text2 = "Prescribed painkillers and recommended physiotherapy"
entities2 = extract_medical_entities(text2)
assert len(entities2["Treatment"]) >= 2

# Test 3: Empty input
text3 = ""
result3 = summarize_to_json("Test", text3)
assert result3 is not None
```
