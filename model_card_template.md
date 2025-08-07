---
license: apache-2.0
tags:
- text-classification
- pytorch
- transformers
widget:
- text: "I'm so stupid, I can't do anything right"
  example_title: "CBT Trigger Example"
- text: "I had a great day at work today"
  example_title: "Normal Conversation"
---

# Binary Classifier for CBT Trigger Detection

This model is a fine-tuned DistilBERT model for binary classification to detect CBT (Cognitive Behavioral Therapy) triggering statements.

## Model Details

- **Base Model**: distilbert-base-uncased
- **Task**: Binary Text Classification
- **Labels**: 
  - LABEL_0: Normal conversation
  - LABEL_1: CBT trigger (self-defeating statement)

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="SaitejaJate/Binary_classifier")
result = classifier("I'm worthless and can't do anything right")
print(result)
```

## Training Data

The model was trained on a dataset of normal conversations and self-defeating statements that typically trigger CBT interventions.