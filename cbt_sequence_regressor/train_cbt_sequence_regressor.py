import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CBTSequenceRegressorTrainer:
    """Train a multi-task model to evaluate CBT sequence compliance."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the trainer with text encoder."""
        logger.info("Initializing CBT Sequence Regressor Trainer")
        
        # Text encoder for embeddings
        self.text_encoder = SentenceTransformer(model_name)
        logger.info(f"Loaded text encoder: {model_name}")
        
        # Label encoders for categorical variables
        self.step_encoder = LabelEncoder()
        self.quality_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        
        # Models for different tasks
        self.satisfaction_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.readiness_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.quality_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        self.action_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        logger.info("Initialized all models")
    
    def load_training_data(self, json_file: str) -> List[Dict]:
        """Load and parse training data from JSON file."""
        logger.info(f"Loading training data from {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Debug: Print the structure
            logger.info(f"JSON keys: {list(data.keys())}")
            if 'examples' in data:
                logger.info(f"Number of examples: {len(data['examples'])}")
                if data['examples']:
                    logger.info(f"First example keys: {list(data['examples'][0].keys())}")
            
            # Extract examples and remove unnecessary fields (like 'id' and 'evaluation_notes')
            examples = []
            for i, example in enumerate(data['examples']):
                try:
                    processed_example = {
                        'cbt_step': example['cbt_step'],
                        'trigger_statement': example['trigger_statement'],
                        'conversation_context': example['conversation_context'],
                        'model_question': example['model_question'],
                        'user_response': example['user_response'],
                        'response_quality': example['response_quality'],
                        'satisfaction_score': float(example['satisfaction_score']),
                        'ready_for_next_step': bool(example['ready_for_next_step']),
                        'missing_elements': example['missing_elements'],
                        'suggested_action': example['suggested_action']
                        # Note: 'id' and 'evaluation_notes' are intentionally excluded
                    }
                    examples.append(processed_example)
                except KeyError as e:
                    logger.error(f"Missing key {e} in example {i}")
                    logger.error(f"Available keys: {list(example.keys())}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing example {i}: {e}")
                    raise
            
            logger.info(f"Successfully loaded {len(examples)} training examples")
            return examples
            
        except FileNotFoundError:
            logger.error(f"Training data file not found: {json_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {json_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def extract_features(self, examples: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract features from training examples."""
        logger.info("Extracting features from examples")
        
        # Text features
        questions = [ex['model_question'] for ex in examples]
        responses = [ex['user_response'] for ex in examples]
        contexts = [ex['conversation_context'] for ex in examples]
        triggers = [ex['trigger_statement'] for ex in examples]
        
        logger.info("Generating text embeddings...")
        question_embeddings = self.text_encoder.encode(questions, show_progress_bar=True)
        response_embeddings = self.text_encoder.encode(responses, show_progress_bar=True)
        context_embeddings = self.text_encoder.encode(contexts, show_progress_bar=True)
        trigger_embeddings = self.text_encoder.encode(triggers, show_progress_bar=True)
        
        # Categorical features
        steps = [ex['cbt_step'] for ex in examples]
        step_encoded = self.step_encoder.fit_transform(steps)
        step_onehot = np.eye(len(self.step_encoder.classes_))[step_encoded]
        
        # Response analysis features
        response_lengths = np.array([len(ex['user_response'].split()) for ex in examples])
        contains_numbers = np.array([1 if any(c.isdigit() for c in ex['user_response']) else 0 for ex in examples])
        
        # Emotion word detection
        emotion_words = ['sad', 'angry', 'frustrated', 'ashamed', 'happy', 'proud', 'confident', 
                        'anxious', 'worried', 'depressed', 'hopeless', 'excited', 'calm']
        emotion_counts = []
        for ex in examples:
            count = sum(1 for word in emotion_words if word in ex['user_response'].lower())
            emotion_counts.append(count)
        emotion_counts = np.array(emotion_counts)
        
        # Missing elements count
        missing_counts = np.array([len(ex['missing_elements']) for ex in examples])
        
        # Combine all features
        features = np.concatenate([
            question_embeddings,      # 384 dimensions
            response_embeddings,      # 384 dimensions
            context_embeddings,       # 384 dimensions
            trigger_embeddings,       # 384 dimensions
            step_onehot,             # 6 dimensions (for 6 CBT steps)
            response_lengths.reshape(-1, 1),  # 1 dimension
            contains_numbers.reshape(-1, 1),  # 1 dimension
            emotion_counts.reshape(-1, 1),    # 1 dimension
            missing_counts.reshape(-1, 1)     # 1 dimension
        ], axis=1)
        
        # Target variables
        targets = {
            'satisfaction_scores': np.array([ex['satisfaction_score'] for ex in examples]),
            'readiness': np.array([ex['ready_for_next_step'] for ex in examples]),
            'quality': self.quality_encoder.fit_transform([ex['response_quality'] for ex in examples]),
            'action': self.action_encoder.fit_transform([ex['suggested_action'] for ex in examples])
        }
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"CBT steps: {self.step_encoder.classes_}")
        logger.info(f"Response qualities: {self.quality_encoder.classes_}")
        logger.info(f"Suggested actions: {self.action_encoder.classes_}")
        
        return features, targets
    
    def train_models(self, features: np.ndarray, targets: Dict[str, np.ndarray], test_size: float = 0.2):
        """Train all models with train/test split."""
        logger.info("Training models with train/test split")
        
        # Split data
        X_train, X_test, y_sat_train, y_sat_test = train_test_split(
            features, targets['satisfaction_scores'], test_size=test_size, random_state=42
        )
        
        _, _, y_ready_train, y_ready_test = train_test_split(
            features, targets['readiness'], test_size=test_size, random_state=42
        )
        
        _, _, y_qual_train, y_qual_test = train_test_split(
            features, targets['quality'], test_size=test_size, random_state=42
        )
        
        _, _, y_action_train, y_action_test = train_test_split(
            features, targets['action'], test_size=test_size, random_state=42
        )
        
        # Train satisfaction score regressor
        logger.info("Training satisfaction score regressor...")
        self.satisfaction_regressor.fit(X_train, y_sat_train)
        sat_pred = self.satisfaction_regressor.predict(X_test)
        sat_r2 = r2_score(y_sat_test, sat_pred)
        sat_rmse = np.sqrt(mean_squared_error(y_sat_test, sat_pred))
        logger.info(f"Satisfaction Score - R²: {sat_r2:.4f}, RMSE: {sat_rmse:.4f}")
        
        # Train readiness classifier
        logger.info("Training readiness classifier...")
        self.readiness_classifier.fit(X_train, y_ready_train)
        ready_pred = self.readiness_classifier.predict(X_test)
        ready_acc = accuracy_score(y_ready_test, ready_pred)
        logger.info(f"Readiness Classifier - Accuracy: {ready_acc:.4f}")
        
        # Train quality classifier
        logger.info("Training quality classifier...")
        self.quality_classifier.fit(X_train, y_qual_train)
        qual_pred = self.quality_classifier.predict(X_test)
        qual_acc = accuracy_score(y_qual_test, qual_pred)
        logger.info(f"Quality Classifier - Accuracy: {qual_acc:.4f}")
        
        # Train action classifier
        logger.info("Training action classifier...")
        self.action_classifier.fit(X_train, y_action_train)
        action_pred = self.action_classifier.predict(X_test)
        action_acc = accuracy_score(y_action_test, action_pred)
        logger.info(f"Action Classifier - Accuracy: {action_acc:.4f}")
        
        # Detailed classification reports                                                                                   
        logger.info("\n=== DETAILED RESULTS ===")                                                                           
                                                                                                                            
        logger.info("\nReadiness Classification Report:")                                                                   
        logger.info(classification_report(y_ready_test, ready_pred, target_names=['Not Ready', 'Ready']))                   
                                                                                                                            
        logger.info("\nQuality Classification Report:")                                                                     
        logger.info(classification_report(y_qual_test, qual_pred, labels=range(len(self.quality_encoder.classes_)),         
        target_names=self.quality_encoder.classes_, zero_division=0))                                                               
                                                                                                                            
        logger.info("\nAction Classification Report:")                                                                      
        logger.info(classification_report(y_action_test, action_pred, labels=range(len(self.action_encoder.classes_)),      
        target_names=self.action_encoder.classes_, zero_division=0)) 
        
        return {
            'satisfaction_r2': sat_r2,
            'satisfaction_rmse': sat_rmse,
            'readiness_accuracy': ready_acc,
            'quality_accuracy': qual_acc,
            'action_accuracy': action_acc
        }
    
    def save_models(self, output_dir: str):
        """Save all trained models and encoders."""
        logger.info(f"Saving models to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.satisfaction_regressor, os.path.join(output_dir, 'satisfaction_regressor.joblib'))
        joblib.dump(self.readiness_classifier, os.path.join(output_dir, 'readiness_classifier.joblib'))
        joblib.dump(self.quality_classifier, os.path.join(output_dir, 'quality_classifier.joblib'))
        joblib.dump(self.action_classifier, os.path.join(output_dir, 'action_classifier.joblib'))
        
        # Save encoders
        joblib.dump(self.step_encoder, os.path.join(output_dir, 'step_encoder.joblib'))
        joblib.dump(self.quality_encoder, os.path.join(output_dir, 'quality_encoder.joblib'))
        joblib.dump(self.action_encoder, os.path.join(output_dir, 'action_encoder.joblib'))
        
        # Save text encoder info
        model_info = {
            'text_encoder_model': 'all-MiniLM-L6-v2',
            'feature_dimensions': {
                'question_embeddings': 384,
                'response_embeddings': 384,
                'context_embeddings': 384,
                'trigger_embeddings': 384,
                'step_onehot': len(self.step_encoder.classes_),
                'response_features': 4  # length, numbers, emotions, missing_count
            },
            'classes': {
                'steps': self.step_encoder.classes_.tolist(),
                'qualities': self.quality_encoder.classes_.tolist(),
                'actions': self.action_encoder.classes_.tolist()
            }
        }
        
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("All models and encoders saved successfully")
    
    def create_combined_model(self, output_dir: str):
        """Create a combined model class for easy inference."""
        
        combined_model_code = '''
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
'''
        
        # Save the combined model class
        with open(os.path.join(output_dir, 'sequence_regressor.py'), 'w') as f:
            f.write(combined_model_code)
        
        logger.info("Combined model class created")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CBT Sequence Compliance Regressor')
    parser.add_argument('--data', required=True, help='Path to training data JSON file')
    parser.add_argument('--output', default='./cbt_sequence_model', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CBTSequenceRegressorTrainer()
    
    # Load data
    examples = trainer.load_training_data(args.data)
    
    # Extract features
    features, targets = trainer.extract_features(examples)
    
    # Train models
    results = trainer.train_models(features, targets, test_size=args.test_size)
    
    # Save models
    trainer.save_models(args.output)
    
    # Create combined model
    trainer.create_combined_model(args.output)
    
    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info(f"Models saved to: {args.output}")
    logger.info("Final Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CBTSequenceRegressorTrainer:
    """Train a multi-task model to evaluate CBT sequence compliance."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the trainer with text encoder."""
        logger.info("Initializing CBT Sequence Regressor Trainer")
        
        # Text encoder for embeddings
        self.text_encoder = SentenceTransformer(model_name)
        logger.info(f"Loaded text encoder: {model_name}")
        
        # Label encoders for categorical variables
        self.step_encoder = LabelEncoder()
        self.quality_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        
        # Models for different tasks
        self.satisfaction_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.readiness_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.quality_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        self.action_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        logger.info("Initialized all models")
    
    def load_training_data(self, json_file: str) -> List[Dict]:
        """Load and parse training data from JSON file."""
        logger.info(f"Loading training data from {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Debug: Print the structure
            logger.info(f"JSON keys: {list(data.keys())}")
            if 'examples' in data:
                logger.info(f"Number of examples: {len(data['examples'])}")
                if data['examples']:
                    logger.info(f"First example keys: {list(data['examples'][0].keys())}")
            
            # Extract examples and remove unnecessary fields (like 'id' and 'evaluation_notes')
            examples = []
            for i, example in enumerate(data['examples']):
                try:
                    processed_example = {
                        'cbt_step': example['cbt_step'],
                        'trigger_statement': example['trigger_statement'],
                        'conversation_context': example['conversation_context'],
                        'model_question': example['model_question'],
                        'user_response': example['user_response'],
                        'response_quality': example['response_quality'],
                        'satisfaction_score': float(example['satisfaction_score']),
                        'ready_for_next_step': bool(example['ready_for_next_step']),
                        'missing_elements': example['missing_elements'],
                        'suggested_action': example['suggested_action']
                        # Note: 'id' and 'evaluation_notes' are intentionally excluded
                    }
                    examples.append(processed_example)
                except KeyError as e:
                    logger.error(f"Missing key {e} in example {i}")
                    logger.error(f"Available keys: {list(example.keys())}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing example {i}: {e}")
                    raise
            
            logger.info(f"Successfully loaded {len(examples)} training examples")
            return examples
            
        except FileNotFoundError:
            logger.error(f"Training data file not found: {json_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {json_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def extract_features(self, examples: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract features from training examples."""
        logger.info("Extracting features from examples")
        
        # Text features
        questions = [ex['model_question'] for ex in examples]
        responses = [ex['user_response'] for ex in examples]
        contexts = [ex['conversation_context'] for ex in examples]
        triggers = [ex['trigger_statement'] for ex in examples]
        
        logger.info("Generating text embeddings...")
        question_embeddings = self.text_encoder.encode(questions, show_progress_bar=True)
        response_embeddings = self.text_encoder.encode(responses, show_progress_bar=True)
        context_embeddings = self.text_encoder.encode(contexts, show_progress_bar=True)
        trigger_embeddings = self.text_encoder.encode(triggers, show_progress_bar=True)
        
        # Categorical features
        steps = [ex['cbt_step'] for ex in examples]
        step_encoded = self.step_encoder.fit_transform(steps)
        step_onehot = np.eye(len(self.step_encoder.classes_))[step_encoded]
        
        # Response analysis features
        response_lengths = np.array([len(ex['user_response'].split()) for ex in examples])
        contains_numbers = np.array([1 if any(c.isdigit() for c in ex['user_response']) else 0 for ex in examples])
        
        # Emotion word detection
        emotion_words = ['sad', 'angry', 'frustrated', 'ashamed', 'happy', 'proud', 'confident', 
                        'anxious', 'worried', 'depressed', 'hopeless', 'excited', 'calm']
        emotion_counts = []
        for ex in examples:
            count = sum(1 for word in emotion_words if word in ex['user_response'].lower())
            emotion_counts.append(count)
        emotion_counts = np.array(emotion_counts)
        
        # Missing elements count
        missing_counts = np.array([len(ex['missing_elements']) for ex in examples])
        
        # Combine all features
        features = np.concatenate([
            question_embeddings,      # 384 dimensions
            response_embeddings,      # 384 dimensions
            context_embeddings,       # 384 dimensions
            trigger_embeddings,       # 384 dimensions
            step_onehot,             # 6 dimensions (for 6 CBT steps)
            response_lengths.reshape(-1, 1),  # 1 dimension
            contains_numbers.reshape(-1, 1),  # 1 dimension
            emotion_counts.reshape(-1, 1),    # 1 dimension
            missing_counts.reshape(-1, 1)     # 1 dimension
        ], axis=1)
        
        # Target variables
        targets = {
            'satisfaction_scores': np.array([ex['satisfaction_score'] for ex in examples]),
            'readiness': np.array([ex['ready_for_next_step'] for ex in examples]),
            'quality': self.quality_encoder.fit_transform([ex['response_quality'] for ex in examples]),
            'action': self.action_encoder.fit_transform([ex['suggested_action'] for ex in examples])
        }
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"CBT steps: {self.step_encoder.classes_}")
        logger.info(f"Response qualities: {self.quality_encoder.classes_}")
        logger.info(f"Suggested actions: {self.action_encoder.classes_}")
        
        return features, targets
    
    def train_models(self, features: np.ndarray, targets: Dict[str, np.ndarray], test_size: float = 0.2):
        """Train all models with train/test split."""
        logger.info("Training models with train/test split")
        
        # Split data
        X_train, X_test, y_sat_train, y_sat_test = train_test_split(
            features, targets['satisfaction_scores'], test_size=test_size, random_state=42
        )
        
        _, _, y_ready_train, y_ready_test = train_test_split(
            features, targets['readiness'], test_size=test_size, random_state=42
        )
        
        _, _, y_qual_train, y_qual_test = train_test_split(
            features, targets['quality'], test_size=test_size, random_state=42
        )
        
        _, _, y_action_train, y_action_test = train_test_split(
            features, targets['action'], test_size=test_size, random_state=42
        )
        
        # Train satisfaction score regressor
        logger.info("Training satisfaction score regressor...")
        self.satisfaction_regressor.fit(X_train, y_sat_train)
        sat_pred = self.satisfaction_regressor.predict(X_test)
        sat_r2 = r2_score(y_sat_test, sat_pred)
        sat_rmse = np.sqrt(mean_squared_error(y_sat_test, sat_pred))
        logger.info(f"Satisfaction Score - R²: {sat_r2:.4f}, RMSE: {sat_rmse:.4f}")
        
        # Train readiness classifier
        logger.info("Training readiness classifier...")
        self.readiness_classifier.fit(X_train, y_ready_train)
        ready_pred = self.readiness_classifier.predict(X_test)
        ready_acc = accuracy_score(y_ready_test, ready_pred)
        logger.info(f"Readiness Classifier - Accuracy: {ready_acc:.4f}")
        
        # Train quality classifier
        logger.info("Training quality classifier...")
        self.quality_classifier.fit(X_train, y_qual_train)
        qual_pred = self.quality_classifier.predict(X_test)
        qual_acc = accuracy_score(y_qual_test, qual_pred)
        logger.info(f"Quality Classifier - Accuracy: {qual_acc:.4f}")
        
        # Train action classifier
        logger.info("Training action classifier...")
        self.action_classifier.fit(X_train, y_action_train)
        action_pred = self.action_classifier.predict(X_test)
        action_acc = accuracy_score(y_action_test, action_pred)
        logger.info(f"Action Classifier - Accuracy: {action_acc:.4f}")
        
        # Detailed classification reports                                                                                   
        logger.info("\n=== DETAILED RESULTS ===")                                                                           
                                                                                                                            
        logger.info("\nReadiness Classification Report:")                                                                   
        logger.info(classification_report(y_ready_test, ready_pred, target_names=['Not Ready', 'Ready']))                   
                                                                                                                            
        logger.info("\nQuality Classification Report:")                                                                     
        logger.info(classification_report(y_qual_test, qual_pred, labels=range(len(self.quality_encoder.classes_)),         
        target_names=self.quality_encoder.classes_, zero_division=0))                                                               
                                                                                                                            
        logger.info("\nAction Classification Report:")                                                                      
        logger.info(classification_report(y_action_test, action_pred, labels=range(len(self.action_encoder.classes_)),      
        target_names=self.action_encoder.classes_, zero_division=0)) 
        
        return {
            'satisfaction_r2': sat_r2,
            'satisfaction_rmse': sat_rmse,
            'readiness_accuracy': ready_acc,
            'quality_accuracy': qual_acc,
            'action_accuracy': action_acc
        }
    
    def save_models(self, output_dir: str):
        """Save all trained models and encoders."""
        logger.info(f"Saving models to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.satisfaction_regressor, os.path.join(output_dir, 'satisfaction_regressor.joblib'))
        joblib.dump(self.readiness_classifier, os.path.join(output_dir, 'readiness_classifier.joblib'))
        joblib.dump(self.quality_classifier, os.path.join(output_dir, 'quality_classifier.joblib'))
        joblib.dump(self.action_classifier, os.path.join(output_dir, 'action_classifier.joblib'))
        
        # Save encoders
        joblib.dump(self.step_encoder, os.path.join(output_dir, 'step_encoder.joblib'))
        joblib.dump(self.quality_encoder, os.path.join(output_dir, 'quality_encoder.joblib'))
        joblib.dump(self.action_encoder, os.path.join(output_dir, 'action_encoder.joblib'))
        
        # Save text encoder info
        model_info = {
            'text_encoder_model': 'all-MiniLM-L6-v2',
            'feature_dimensions': {
                'question_embeddings': 384,
                'response_embeddings': 384,
                'context_embeddings': 384,
                'trigger_embeddings': 384,
                'step_onehot': len(self.step_encoder.classes_),
                'response_features': 4  # length, numbers, emotions, missing_count
            },
            'classes': {
                'steps': self.step_encoder.classes_.tolist(),
                'qualities': self.quality_encoder.classes_.tolist(),
                'actions': self.action_encoder.classes_.tolist()
            }
        }
        
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("All models and encoders saved successfully")
    
    def create_combined_model(self, output_dir: str):
        """Create a combined model class for easy inference."""
        
        combined_model_code = '''
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
'''
        
        # Save the combined model class
        with open(os.path.join(output_dir, 'sequence_regressor.py'), 'w') as f:
            f.write(combined_model_code)
        
        logger.info("Combined model class created")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CBT Sequence Compliance Regressor')
    parser.add_argument('--data', required=True, help='Path to training data JSON file')
    parser.add_argument('--output', default='./cbt_sequence_model', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CBTSequenceRegressorTrainer()
    
    # Load data
    examples = trainer.load_training_data(args.data)
    
    # Extract features
    features, targets = trainer.extract_features(examples)
    
    # Train models
    results = trainer.train_models(features, targets, test_size=args.test_size)
    
    # Save models
    trainer.save_models(args.output)
    
    # Create combined model
    trainer.create_combined_model(args.output)
    
    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info(f"Models saved to: {args.output}")
    logger.info("Final Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
