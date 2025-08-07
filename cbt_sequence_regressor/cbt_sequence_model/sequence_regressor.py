
import joblib
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class CBTSequenceComplianceRegressor:
    """Combined model for CBT sequence compliance evaluation."""
    
    def __init__(self, model_dir: str):
        """Load all models and encoders."""
        self.model_dir = model_dir
        
        # Load models
        self.satisfaction_regressor = joblib.load(os.path.join(model_dir, 'satisfaction_regressor.joblib'))
        self.readiness_classifier = joblib.load(os.path.join(model_dir, 'readiness_classifier.joblib'))
        self.quality_classifier = joblib.load(os.path.join(model_dir, 'quality_classifier.joblib'))
        self.action_classifier = joblib.load(os.path.join(model_dir, 'action_classifier.joblib'))
        
        # Load encoders
        self.step_encoder = joblib.load(os.path.join(model_dir, 'step_encoder.joblib'))
        self.quality_encoder = joblib.load(os.path.join(model_dir, 'quality_encoder.joblib'))
        self.action_encoder = joblib.load(os.path.join(model_dir, 'action_encoder.joblib'))
        
        # Load model info
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            self.model_info = json.load(f)
        
        # Initialize text encoder
        self.text_encoder = SentenceTransformer(self.model_info['text_encoder_model'])
    
    def extract_features(self, model_question: str, user_response: str, 
                        conversation_context: str, trigger_statement: str, 
                        cbt_step: str) -> np.ndarray:
        """Extract features for a single example."""
        
        # Text embeddings
        question_emb = self.text_encoder.encode([model_question])
        response_emb = self.text_encoder.encode([user_response])
        context_emb = self.text_encoder.encode([conversation_context])
        trigger_emb = self.text_encoder.encode([trigger_statement])
        
        # Step encoding
        try:
            step_encoded = self.step_encoder.transform([cbt_step])
            step_onehot = np.eye(len(self.step_encoder.classes_))[step_encoded]
        except ValueError:
            # Unknown step, use zeros
            step_onehot = np.zeros((1, len(self.step_encoder.classes_)))
        
        # Response analysis
        response_length = len(user_response.split())
        contains_numbers = 1 if any(c.isdigit() for c in user_response) else 0
        
        emotion_words = ['sad', 'angry', 'frustrated', 'ashamed', 'happy', 'proud', 'confident', 
                        'anxious', 'worried', 'depressed', 'hopeless', 'excited', 'calm']
        emotion_count = sum(1 for word in emotion_words if word in user_response.lower())
        
        # Combine features
        features = np.concatenate([
            question_emb.flatten(),
            response_emb.flatten(),
            context_emb.flatten(),
            trigger_emb.flatten(),
            step_onehot.flatten(),
            [response_length, contains_numbers, emotion_count, 0]  # missing_count set to 0 for inference
        ]).reshape(1, -1)
        
        return features
    
    def predict(self, model_question: str, user_response: str, 
                conversation_context: str, trigger_statement: str, 
                cbt_step: str) -> Dict[str, Any]:
        """Make predictions for a single example."""
        
        # Extract features
        features = self.extract_features(
            model_question, user_response, conversation_context, 
            trigger_statement, cbt_step
        )
        
        # Make predictions
        satisfaction_score = float(self.satisfaction_regressor.predict(features)[0])
        ready_for_next = bool(self.readiness_classifier.predict(features)[0])
        quality_encoded = self.quality_classifier.predict(features)[0]
        action_encoded = self.action_classifier.predict(features)[0]
        
        # Decode categorical predictions
        response_quality = self.quality_encoder.inverse_transform([quality_encoded])[0]
        suggested_action = self.action_encoder.inverse_transform([action_encoded])[0]
        
        # Get prediction probabilities for confidence
        satisfaction_confidence = 1.0  # Regression doesn't have confidence
        ready_confidence = float(max(self.readiness_classifier.predict_proba(features)[0]))
        quality_confidence = float(max(self.quality_classifier.predict_proba(features)[0]))
        action_confidence = float(max(self.action_classifier.predict_proba(features)[0]))
        
        return {
            'satisfaction_score': satisfaction_score,
            'ready_for_next_step': ready_for_next,
            'response_quality': response_quality,
            'suggested_action': suggested_action,
            'confidence': {
                'satisfaction': satisfaction_confidence,
                'readiness': ready_confidence,
                'quality': quality_confidence,
                'action': action_confidence
            }
        }
