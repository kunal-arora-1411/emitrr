import spacy
import json
import yake

# Load SciSpaCy model
nlp = spacy.load("en_core_sci_sm")

def extract_medical_entities(text):
    doc = nlp(text)
    entities = {
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": [],
        "Prognosis": []
    }

    # Rule-based mapping based on keywords and entity context
    for ent in doc.ents:
        t = ent.text.lower()
        if any(k in t for k in ["pain", "ache", "injury", "hurt", "discomfort", "headache"]):
            entities["Symptoms"].append(ent.text)
        elif any(k in t for k in ["diagnosed", "disease", "condition", "injury", "syndrome"]):
            entities["Diagnosis"].append(ent.text)
        elif any(k in t for k in ["therapy", "treatment", "session", "medicine", "painkiller", "prescribed", "physiotherapy"]):
            entities["Treatment"].append(ent.text)
        elif any(k in t for k in ["recovery", "improve", "heal", "better", "normal", "expected"]):
            entities["Prognosis"].append(ent.text)

    return entities


def extract_keywords(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return keywords


def summarize_to_json(patient_name, text):
    entities = extract_medical_entities(text)
    keywords = extract_keywords(text)

    summary = {
        "Patient_Name": patient_name,
        "Symptoms": list(set(entities["Symptoms"])),
        "Diagnosis": list(set(entities["Diagnosis"])),
        "Treatment": list(set(entities["Treatment"])),
        "Prognosis": list(set(entities["Prognosis"])),
        "Keywords": keywords
    }

    return json.dumps(summary, indent=2)


if __name__ == "__main__":
    transcript = """Doctor: How are you feeling today?
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain."""
    
    patient_name = "Janet Jones"
    result = summarize_to_json(patient_name, transcript)
    print(result)
