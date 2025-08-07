import json
import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import joblib
from flask import Flask, request, jsonify
import uuid
from datetime import datetime

# Define cognitive distortion patterns
COGNITIVE_DISTORTIONS = {
    "perfectionism": ["perfect", "flawless", "mistake", "error", "failure"],
    "catastrophizing": ["disaster", "terrible", "horrible", "awful", "end of the world"],
    "black_and_white_thinking": ["always", "never", "completely", "totally", "all or nothing"],
    "mind_reading": ["think about me", "judging me", "think I'm", "perceive me"],
    "overgeneralization": ["everything", "nothing", "everyone", "no one", "always goes wrong"],
    "should_statements": ["should", "must", "have to", "ought to"],
    "emotional_reasoning": ["feel like a failure", "feel worthless", "feel inadequate"],
    "personalization": ["my fault", "blame myself", "responsible for", "caused this"]
}

class CBTEvaluatorSimple:
    def __init__(self, output_dir="cbt_evaluator_simple", max_features=5000, random_state=42):
        self.output_dir = output_dir
        self.max_features = max_features
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature extractor
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Initialize model
        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        )
    
    def identify_initial_problem(self, initial_input):
        """Identify the likely cognitive distortion from the initial input."""
        initial_input = initial_input.lower()
        
        # Count matches for each distortion type
        matches = defaultdict(int)
        for distortion, patterns in COGNITIVE_DISTORTIONS.items():
            for pattern in patterns:
                if pattern.lower() in initial_input:
                    matches[distortion] += 1
        
        # Return the distortion with the most matches, or "unspecified" if none
        if matches:
            return max(matches.items(), key=lambda x: x[1])[0]
        return "unspecified"
    
    def clean_therapist_response(self, content):
        """Clean the therapist response by removing reasoning tags."""
        # Remove reasoning tags
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        # Extract answer content
        content = re.sub(r'<answer>(.*?)</answer>', r'\1', content, flags=re.DOTALL)
        return content.strip()
    
    def format_conversation(self, turns):
        """Format conversation turns into a readable text format."""
        formatted_text = ""
        for turn in turns:
            speaker = "User" if turn["speaker"] == "patient" else "Assistant"
            content = turn["content"]
            
            # Clean the content for assistant responses
            if speaker == "Assistant":
                content = self.clean_therapist_response(content)
            
            formatted_text += f"{speaker}: {content}\n\n"
        
        return formatted_text.strip()
    
    def process_conversations(self, conversations_dir):
        """Process all conversation files into a dataset."""
        all_examples = []
        
        # Get all JSON files in the directory
        json_files = glob.glob(os.path.join(conversations_dir, "*.json"))
        print(f"Found {len(json_files)} conversation files")
        
        # Process each file with a progress bar
        for json_file in tqdm(json_files, desc="Processing conversations"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                # Extract basic information
                conversation_id = os.path.basename(json_file).replace('.json', '')
                
                # Check if the required fields exist
                if "metadata" not in conversation_data or "conversation" not in conversation_data or "progress_areas" not in conversation_data:
                    print(f"Warning: Missing required fields in {json_file}")
                    continue
                
                initial_input = conversation_data["metadata"]["initial_input"]
                total_turns = len(conversation_data["conversation"])
                
                # Extract reward scores
                try:
                    target_scores = {
                        "thought_identification": float(conversation_data["progress_areas"]["Thought Identification"]),
                        "emotion_exploration": float(conversation_data["progress_areas"]["Emotion Exploration"]),
                        "distortion_identification": float(conversation_data["progress_areas"]["Distortion Identification"]),
                        "evidence_gathering": float(conversation_data["progress_areas"]["Evidence Gathering"]),
                        "balanced_thinking": float(conversation_data["progress_areas"]["Balanced Thinking"])
                    }
                except (KeyError, ValueError) as e:
                    print(f"Warning: Issue with progress scores in {json_file}: {e}")
                    continue
                
                # Identify the initial problem/cognitive distortion
                initial_problem = self.identify_initial_problem(initial_input)
                
                # Create training examples at different conversation lengths
                for end_turn in range(2, total_turns + 1, 2):
                    # Extract conversation up to this point
                    conversation_so_far = conversation_data["conversation"][:end_turn]
                    
                    # Format the conversation text
                    formatted_text = self.format_conversation(conversation_so_far)
                    
                    # Calculate conversation progress
                    conversation_progress = round(end_turn / total_turns, 2)
                    
                    # Create the training example
                    example = {
                        "id": f"{conversation_id}_turn_{end_turn}",
                        "conversation_text": formatted_text,
                        "thought_identification": target_scores["thought_identification"],
                        "emotion_exploration": target_scores["emotion_exploration"],
                        "distortion_identification": target_scores["distortion_identification"],
                        "evidence_gathering": target_scores["evidence_gathering"],
                        "balanced_thinking": target_scores["balanced_thinking"],
                        "total_turns": total_turns,
                        "turns_included": end_turn,
                        "initial_problem": initial_problem,
                        "conversation_progress": conversation_progress
                    }
                    
                    all_examples.append(example)
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Generated {len(all_examples)} training examples from {len(json_files)} conversations")
        return pd.DataFrame(all_examples)
    
    def train(self, conversations_dir, test_size=0.2):
        """Train the model on conversation data."""
        # Process conversations
        df = self.process_conversations(conversations_dir)
        
        # Save the processed data
        df.to_csv(os.path.join(self.output_dir, "processed_data.csv"), index=False)
        
        # Split into features and targets
        X = df["conversation_text"]
        y = df[["thought_identification", "emotion_exploration", "distortion_identification", 
                "evidence_gathering", "balanced_thinking"]].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Fit the vectorizer
        print("Fitting vectorizer...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Train the model
        print("Training model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R²: {r2:.4f}")
        
        # Calculate metrics for each dimension
        dimension_names = ["thought_identification", "emotion_exploration", "distortion_identification", 
                          "evidence_gathering", "balanced_thinking"]
        
        dimension_metrics = {}
        for i, name in enumerate(dimension_names):
            dim_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            dim_r2 = r2_score(y_test[:, i], y_pred[:, i])
            dimension_metrics[name] = {"mse": dim_mse, "r2": dim_r2}
            print(f"  {name}: MSE = {dim_mse:.4f}, R² = {dim_r2:.4f}")
        
        # Save the model and vectorizer
        print(f"Saving model to {self.output_dir}...")
        joblib.dump(self.vectorizer, os.path.join(self.output_dir, "vectorizer.joblib"))
        joblib.dump(self.model, os.path.join(self.output_dir, "model.joblib"))
        
        # Save the metrics
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump({
                "overall": {"mse": mse, "r2": r2},
                "dimensions": dimension_metrics
            }, f, indent=2)
        
        # Visualize predictions vs actual values
        self.visualize_predictions(y_test, y_pred, dimension_names)
        
        return mse, r2, dimension_metrics
    
    def visualize_predictions(self, y_true, y_pred, dimension_names):
        """Visualize predictions vs actual values."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(dimension_names):
            ax = axes[i]
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            ax.plot([0, 1], [0, 1], 'r--')  # Diagonal line
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(name)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Add R² value
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes)
        
        # Remove the unused subplot
        fig.delaxes(axes[5])
        
        # Add overall title
        fig.suptitle("Predicted vs Actual Values", fontsize=16)
        fig.tight_layout()
        
        # Save the figure
        fig.savefig(os.path.join(self.output_dir, "predictions_vs_actual.png"))
        print(f"Visualization saved to {os.path.join(self.output_dir, 'predictions_vs_actual.png')}")
    
    def predict(self, conversation_text):
        """Predict CBT scores for a conversation."""
        # Transform the text
        X = self.vectorizer.transform([conversation_text])
        
        # Make predictions
        predictions = self.model.predict(X)[0]
        
        # Ensure predictions are in the range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        # Format the predictions
        dimension_names = ["thought_identification", "emotion_exploration", "distortion_identification", 
                          "evidence_gathering", "balanced_thinking"]
        
        return {name: float(pred) for name, pred in zip(dimension_names, predictions)}
    
    def count_turns(self, conversation_text):
        """Count the number of turns in the conversation."""
        # Count occurrences of "User:" and "Assistant:"
        user_turns = len(re.findall(r"User:", conversation_text))
        assistant_turns = len(re.findall(r"Assistant:", conversation_text))
        return user_turns + assistant_turns
    
    def identify_problem_from_text(self, conversation_text):
        """Identify the likely cognitive distortion from the conversation text."""
        # Extract the first user message
        match = re.search(r"User: (.*?)(?:\n\n|$)", conversation_text)
        if not match:
            return "unspecified"
        
        initial_input = match.group(1)
        return self.identify_initial_problem(initial_input)

# Create a Flask API for the model
def create_flask_app(model_dir="cbt_evaluator_simple"):
    app = Flask(__name__)
    
    # Load the model and vectorizer
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    # Create an evaluator instance
    evaluator = CBTEvaluatorSimple()
    evaluator.vectorizer = vectorizer
    evaluator.model = model
    
    @app.route('/evaluate', methods=['POST'])
    def evaluate_conversation():
        """Evaluate a conversation."""
        # Get conversation from request
        data = request.json
        conversation_text = data.get('conversation', '')
        
        if not conversation_text:
            return jsonify({"error": "No conversation provided"}), 400
        
        # Get predictions
        scores = evaluator.predict(conversation_text)
        
        # Count turns and identify initial problem
        total_turns = evaluator.count_turns(conversation_text)
        initial_problem = evaluator.identify_problem_from_text(conversation_text)
        
        # Create the final response
        result = {
            "id": f"eval_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "conversation_text": conversation_text,
            "target_scores": {
                "thought_identification": round(scores["thought_identification"], 2),
                "emotion_exploration": round(scores["emotion_exploration"], 2),
                "distortion_identification": round(scores["distortion_identification"], 2),
                "evidence_gathering": round(scores["evidence_gathering"], 2),
                "balanced_thinking": round(scores["balanced_thinking"], 2)
            },
            "metadata": {
                "total_turns": total_turns,
                "turns_included": total_turns,
                "initial_problem": initial_problem,
                "conversation_progress": 1.0  # Assuming full conversation
            }
        }
        
        return jsonify(result)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy"})
    
    return app

# Example usage
if __name__ == "__main__":
    # Check if we're training or serving
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Serve the model
        app = create_flask_app()
        app.run(debug=True, host='0.0.0.0', port=5009)
    else:
        # Train the model
        evaluator = CBTEvaluatorSimple()
        evaluator.train("/Users/saitejagudidevini/Documents/Dev/grpo_trainer/Evaluations/Gemini_QA_Testing/conversations")
        
        print("\nTo serve the model, run:")
        print("python cbt_evaluator_simple.py serve")
