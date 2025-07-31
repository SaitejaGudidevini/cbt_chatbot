from binary_classifier import CBTBinaryClassifier
classifier = CBTBinaryClassifier()
classifier.load_model('./cbt_classifier')
result = classifier.predict('I am happy cause I finished all of my tasks')
print(f"Prediction: {result['is_cbt_trigger']}, Confidence: {result['confidence']:.3f}")