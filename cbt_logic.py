"""
CBT Logic - Original Sophisticated Business Logic
From: conversation_manager.py + response_generator.py
"""

import json
import re
import logging
import sys
import os
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ConversationPhase(Enum):
    CHIT_CHAT = "chit_chat"
    CBT_REFACTORING = "cbt_refactoring"

class ConversationManager:
    """Tracks conversation state and detects CBT triggers with real-time progress evaluation."""
    
    def __init__(self, use_ml_classifier=True, classifier_model_path=None, classifier_threshold=None):
        self.use_ml_classifier = use_ml_classifier
        self.cbt_classifier = None
        self.cbt_evaluator = None
        self.classifier_threshold = classifier_threshold or float(os.getenv("CBT_CLASSIFIER_THRESHOLD", "0.5"))
        
        # Initialize ML classifier if requested
        if use_ml_classifier:
            try:
                # Add the BinaryClassifier directory to path
                binary_classifier_path = os.path.join(os.path.dirname(__file__), 'BinaryClassifier')
                
                # Check if BinaryClassifier directory exists, if not download from HF
                if not os.path.exists(binary_classifier_path):
                    logger.info("BinaryClassifier not found locally. Downloading from HuggingFace...")
                    os.makedirs(binary_classifier_path, exist_ok=True)
                    
                    # Download binary_classifier.py from HuggingFace
                    import urllib.request
                    classifier_url = "https://huggingface.co/SaitejaJate/Binary_classifier/resolve/main/binary_classifier.py"
                    classifier_file = os.path.join(binary_classifier_path, "binary_classifier.py")
                    
                    try:
                        urllib.request.urlretrieve(classifier_url, classifier_file)
                        logger.info(f"Downloaded binary_classifier.py to {classifier_file}")
                    except Exception as download_error:
                        logger.error(f"Failed to download binary_classifier.py: {download_error}")
                        raise
                    
                    # Create __init__.py to make it a proper Python package
                    init_file = os.path.join(binary_classifier_path, "__init__.py")
                    with open(init_file, "w") as f:
                        f.write("")
                
                if binary_classifier_path not in sys.path:
                    sys.path.append(binary_classifier_path)
                
                from BinaryClassifier.binary_classifier import CBTBinaryClassifier
                self.cbt_classifier = CBTBinaryClassifier()
                
                # Check if we should use HF Hub
                use_hf = os.getenv("USE_HF_MODEL", "false").lower() == "true"
                hf_model_id = os.getenv("HF_MODEL_ID", "SaitejaJate/Binary_classifier")
                
                if use_hf and classifier_model_path is None:
                    # Use HuggingFace model ID directly - let transformers handle the download
                    classifier_model_path = hf_model_id
                    logger.info(f"Will load model from HuggingFace: {hf_model_id}")
                elif classifier_model_path is None:
                    # Use default local path
                    classifier_model_path = os.path.join(binary_classifier_path, 'cbt_classifier')
                
                self.cbt_classifier.load_model(classifier_model_path)
                logger.info(f"ML classifier loaded successfully from: {classifier_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ML classifier: {e}. Falling back to regex.")
                self.use_ml_classifier = False
        
        # Initialize sequence regressor HF endpoint
        self.sequence_regressor_hf_model = os.getenv("SEQUENCE_REGRESSOR_HF_MODEL", "SaitejaJate/cbt_sequence_regressor")
        self.hf_api_token = os.getenv("HF_API_TOKEN", "")
        self.sequence_regressor_available = True  # Always available with HF endpoint

        # Initialize CBT evaluator HF endpoint for real-time progress tracking
        self.evaluator_hf_model = os.getenv("EVALUATOR_HF_MODEL", "SaitejaJate/Regression_Evaluation")
        self.cbt_evaluator_available = True  # Always available with HF endpoint
        logger.info(f"CBT evaluator will use HuggingFace model: {self.evaluator_hf_model}")
        
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset conversation to initial state."""
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.phase = ConversationPhase.CHIT_CHAT
        self.conversation_history = []
        self.cbt_trigger_detected = False
        self.trigger_statement = None
        
        # Reset progress tracking
        self.current_progress_scores = None
        self.progress_history = []
        self.last_evaluation_turn = 0
        
        # Reset system prompt compliance tracking
        self.current_compliance_scores = None
        self.compliance_history = []
        self.current_cbt_step = "initial_assessment"
        
        # Reset progress tracking
        self.current_progress_scores = None
        self.progress_history = []
        self.last_evaluation_turn = 0
        
        # Add flow tracking
        self.flow_transitions = []
        self.phase_start_time = datetime.now()
        self.message_count_in_phase = 0
    
    def _call_hf_inference_api(self, model_id: str, inputs: any, task: str = "text-classification") -> Optional[Dict]:
        """Call HuggingFace Inference API for a model."""
        try:
            headers = {}
            if self.hf_api_token:
                headers["Authorization"] = f"Bearer {self.hf_api_token}"
            
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            
            # Prepare the payload based on task type
            if task == "text-classification":
                payload = {"inputs": inputs}
            elif task == "feature-extraction":
                payload = {"inputs": inputs}
            else:
                payload = {"inputs": inputs}
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, retry after a moment
                logger.warning(f"Model {model_id} is loading, retrying...")
                import time
                time.sleep(5)
                response = requests.post(api_url, headers=headers, json=payload, timeout=20)
                if response.status_code == 200:
                    return response.json()
            
            logger.error(f"HF API error for {model_id}: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.error(f"Failed to call HF inference API for {model_id}: {e}")
            return None
    
    def _process_sequence_regressor_response(self, hf_response: any) -> Dict:
        """Process HuggingFace API response into compliance scores."""
        try:
            # Default compliance scores
            compliance_scores = {
                "satisfaction_score": 0.5,
                "ready_for_next_step": False,
                "response_quality": "moderate",
                "suggested_action": "continue_current_step"
            }
            
            if isinstance(hf_response, list) and len(hf_response) > 0:
                # Process classification results
                if isinstance(hf_response[0], dict):
                    # Extract scores from classification labels
                    for item in hf_response[0] if isinstance(hf_response[0], list) else [hf_response[0]]:
                        label = item.get('label', '').lower()
                        score = item.get('score', 0.5)
                        
                        if 'satisfaction' in label or 'positive' in label:
                            compliance_scores['satisfaction_score'] = score
                        elif 'ready' in label or 'complete' in label:
                            compliance_scores['ready_for_next_step'] = score > 0.6
                        elif 'quality' in label:
                            if score > 0.7:
                                compliance_scores['response_quality'] = 'high'
                            elif score > 0.4:
                                compliance_scores['response_quality'] = 'moderate'
                            else:
                                compliance_scores['response_quality'] = 'low'
            
            return compliance_scores
        except Exception as e:
            logger.error(f"Error processing sequence regressor response: {e}")
            return {
                "satisfaction_score": 0.5,
                "ready_for_next_step": False,
                "response_quality": "unknown",
                "suggested_action": "continue_current_step"
            }
    
    def _process_evaluator_response(self, hf_response: any) -> Dict:
        """Process HuggingFace Regression Evaluation API response into progress scores."""
        try:
            # Default progress scores matching CBT evaluator dimensions
            progress_scores = {
                "problem_identification": 0.5,
                "cognitive_restructuring": 0.5,
                "solution_generation": 0.5,
                "therapeutic_rapport": 0.5,
                "overall_progress": 0.5
            }
            
            if isinstance(hf_response, list) and len(hf_response) > 0:
                # Process regression/classification results
                if isinstance(hf_response[0], dict):
                    # Handle classification-style response
                    items = hf_response if isinstance(hf_response[0], list) else hf_response
                    for item in items:
                        label = item.get('label', '').lower()
                        score = item.get('score', 0.5)
                        
                        # Map labels to CBT dimensions
                        if 'problem' in label or 'identification' in label:
                            progress_scores['problem_identification'] = score
                        elif 'cognitive' in label or 'restructuring' in label:
                            progress_scores['cognitive_restructuring'] = score
                        elif 'solution' in label or 'generation' in label:
                            progress_scores['solution_generation'] = score
                        elif 'rapport' in label or 'therapeutic' in label:
                            progress_scores['therapeutic_rapport'] = score
                        elif 'overall' in label or 'progress' in label:
                            progress_scores['overall_progress'] = score
                elif isinstance(hf_response[0], (float, int)):
                    # Handle regression-style response (single score)
                    overall_score = float(hf_response[0])
                    # Distribute the overall score across dimensions
                    progress_scores = {
                        "problem_identification": overall_score * 0.9,
                        "cognitive_restructuring": overall_score * 0.85,
                        "solution_generation": overall_score * 0.8,
                        "therapeutic_rapport": overall_score * 0.95,
                        "overall_progress": overall_score
                    }
            
            # Ensure scores are in valid range [0, 1]
            for key in progress_scores:
                progress_scores[key] = max(0.0, min(1.0, progress_scores[key]))
            
            return progress_scores
        except Exception as e:
            logger.error(f"Error processing evaluator response: {e}")
            # Return default neutral scores
            return {
                "problem_identification": 0.5,
                "cognitive_restructuring": 0.5,
                "solution_generation": 0.5,
                "therapeutic_rapport": 0.5,
                "overall_progress": 0.5
            }
    
    def detect_self_defeating_statement(self, user_input: str) -> bool:
        """Detect if user input contains self-defeating statements using ML classifier."""
        logger.info(f"Checking for self-defeating patterns in: '{user_input}'")
        
        # Use ML classifier if available
        if self.use_ml_classifier and self.cbt_classifier:
            try:
                result = self.cbt_classifier.predict(user_input, threshold=self.classifier_threshold)
                is_trigger = result['is_cbt_trigger']
                confidence = result['confidence']
                
                logger.info(f"ML classifier result: {is_trigger} (confidence: {confidence:.3f}, threshold: {self.classifier_threshold})")
                return is_trigger
                
            except Exception as e:
                logger.error(f"ML classifier failed: {e}. Falling back to regex.")
                # Fall through to regex method
        
        # Fallback to regex patterns (only if ML classifier fails)
        logger.warning("Using regex fallback for CBT trigger detection")
        self_defeating_patterns = [
            r"\bi'?m\s+(so\s+)?(stupid|dumb|worthless|useless|failure|terrible|awful|horrible)",
            r"\bi\s+(always|never)\s+",
            r"\bi\s+can'?t\s+do\s+anything\s+right",
            r"\bi'?m\s+not\s+good\s+enough",
            r"\bi'?ll\s+never\s+",
            r"\beverything\s+i\s+do\s+is\s+wrong",
            r"\bi\s+mess\s+everything\s+up",
            r"\bi'?m\s+a\s+complete\s+",
            r"\bi\s+hate\s+myself",
            r"\bi'?m\s+the\s+worst",
            r"nobody\s+likes\s+me",
            r"\bi\s+ruin\s+everything"
        ]
        
        user_input_lower = user_input.lower()
        for pattern in self_defeating_patterns:
            if re.search(pattern, user_input_lower):
                logger.info(f"Regex fallback: Self-defeating pattern matched: {pattern}")
                return True
        
        logger.info("No self-defeating patterns found (regex fallback)")
        return False
    
    def add_message(self, speaker: str, content: str, model_question: str = None):
        """Add a message to conversation history and evaluate compliance if it's a model response."""
        self.conversation_history.append({
            "speaker": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase.value
        })
        
        # Track user messages for flow analysis
        if speaker == "user":
            self.message_count_in_phase += 1
            
            # If we have a model question and this is a user response in CBT phase, evaluate compliance
            if model_question and self.phase == ConversationPhase.CBT_REFACTORING:
                self.evaluate_system_prompt_compliance(model_question, content)
    
    def set_cbt_phase(self, trigger_statement: str = None):
        """Set conversation to CBT phase and track the transition."""
        # Calculate time spent in previous phase
        time_in_previous_phase = (datetime.now() - self.phase_start_time).total_seconds()
        
        # Record the transition
        transition = {
            "conversation_id": self.conversation_id,
            "from_phase": self.phase.value,
            "to_phase": ConversationPhase.CBT_REFACTORING.value,
            "trigger_statement": trigger_statement,
            "timestamp": datetime.now().isoformat(),
            "user_message_count": self.message_count_in_phase,
            "time_in_previous_phase": time_in_previous_phase
        }
        self.flow_transitions.append(transition)
        
        # Update phase
        self.phase = ConversationPhase.CBT_REFACTORING
        self.cbt_trigger_detected = True
        if trigger_statement:
            self.trigger_statement = trigger_statement
        
        # Reset tracking for new phase
        self.phase_start_time = datetime.now()
        self.message_count_in_phase = 0
        
        # Reset progress tracking for new CBT session
        self.current_progress_scores = None
        self.progress_history = []
        self.last_evaluation_turn = len(self.conversation_history)
        
        # Reset compliance tracking for new CBT session
        self.current_compliance_scores = None
        self.compliance_history = []
        self.current_cbt_step = "problem_identification"
        
        # Reset progress tracking for new CBT session
        self.current_progress_scores = None
        self.progress_history = []
        self.last_evaluation_turn = len(self.conversation_history)
        
        # Log the transition
        logger.info(f"FLOW TRANSITION: {self.conversation_id} -> CBT mode. Trigger: '{trigger_statement}'")
    
    def set_chit_chat_phase(self):
        """Set conversation back to chit-chat phase."""
        self.phase = ConversationPhase.CHIT_CHAT
        self.cbt_trigger_detected = False
        
        # Reset phase tracking
        self.phase_start_time = datetime.now()
        self.message_count_in_phase = 0
        
        # Finalize progress tracking
        if self.current_progress_scores:
            logger.info(f"CBT session completed. Final scores: {self.current_progress_scores}")
        
        # Finalize compliance tracking
        if self.current_compliance_scores:
            logger.info(f"CBT session completed. Final compliance scores: {self.current_compliance_scores}")
        
        # Finalize progress tracking
        if self.current_progress_scores:
            logger.info(f"CBT session completed. Final scores: {self.current_progress_scores}")
    
    def _format_conversation_for_evaluation(self) -> str:
        """Format conversation history for CBT evaluation."""
        if not self.conversation_history:
            return ""
        
        formatted_text = ""
        for message in self.conversation_history:
            speaker = "User" if message["speaker"] == "user" else "Assistant"
            content = message["content"]
            formatted_text += f"{speaker}: {content}\n\n"
        
        return formatted_text.strip()
    
    def evaluate_system_prompt_compliance(self, model_question: str, user_response: str) -> Optional[Dict]:
        """Evaluate how well the model is following the system prompt using the sequence regressor API."""
        if not self.sequence_regressor_available or self.phase != ConversationPhase.CBT_REFACTORING:
            return None
        
        try:
            # Get conversation context
            conversation_context = self._format_conversation_for_evaluation()
            
            # Use the trigger statement if available, otherwise use a default
            trigger_statement = self.trigger_statement or "I'm feeling overwhelmed"
            
            # Prepare API request
            payload = {
                "model_question": model_question,
                "user_response": user_response,
                "conversation_context": conversation_context,
                "trigger_statement": trigger_statement,
                "cbt_step": self.current_cbt_step
            }
            
            # Call the HuggingFace sequence regressor endpoint
            # Format input for the model
            input_text = f"Model: {model_question}\nUser: {user_response}\nContext: {conversation_context[:500]}\nTrigger: {trigger_statement}\nStep: {self.current_cbt_step}"
            
            result = self._call_hf_inference_api(
                self.sequence_regressor_hf_model,
                input_text,
                "text-classification"
            )
            
            if not result:
                logger.error("Failed to get response from HF sequence regressor")
                return None
            
            # Process the HF API response into compliance scores
            compliance_scores = self._process_sequence_regressor_response(result)
            
            # Update current compliance scores
            self.current_compliance_scores = compliance_scores
            
            # Add to compliance history
            compliance_entry = {
                "timestamp": datetime.now().isoformat(),
                "turn_count": len(self.conversation_history),
                "model_question": model_question,
                "user_response": user_response,
                "cbt_step": self.current_cbt_step,
                "compliance_scores": compliance_scores.copy()
            }
            self.compliance_history.append(compliance_entry)
            
            logger.info(f"System Prompt Compliance Evaluation:")
            logger.info(f"  Satisfaction Score: {compliance_scores['satisfaction_score']:.3f}")
            logger.info(f"  Ready for Next Step: {compliance_scores['ready_for_next_step']}")
            logger.info(f"  Response Quality: {compliance_scores['response_quality']}")
            logger.info(f"  Suggested Action: {compliance_scores['suggested_action']}")
            
            # Update CBT step based on readiness
            if compliance_scores['ready_for_next_step']:
                self._advance_cbt_step()
            
            return compliance_scores
            
        except Exception as e:
            logger.error(f"Error evaluating system prompt compliance: {e}")
            return self.current_compliance_scores
    
    def _advance_cbt_step(self):
        """Advance to the next CBT step based on current progress."""
        step_progression = [
            "initial_assessment",
            "problem_identification", 
            "thought_exploration",
            "cognitive_restructuring",
            "behavioral_planning",
            "skill_practice",
            "progress_review"
        ]
        
        try:
            current_index = step_progression.index(self.current_cbt_step)
            if current_index < len(step_progression) - 1:
                self.current_cbt_step = step_progression[current_index + 1]
                logger.info(f"Advanced to CBT step: {self.current_cbt_step}")
        except ValueError:
            logger.warning(f"Unknown CBT step: {self.current_cbt_step}")
    
    def evaluate_current_conversation(self) -> Optional[Dict]:
        """Evaluate current conversation progress using the CBT evaluator HF endpoint."""
        if not self.cbt_evaluator_available or self.phase != ConversationPhase.CBT_REFACTORING:
            return None
        
        # Only evaluate if we have enough new content (at least 2 new turns)
        current_turn_count = len(self.conversation_history)
        if current_turn_count - self.last_evaluation_turn < 2:
            return self.current_progress_scores
        
        try:
            # Format conversation for evaluation
            conversation_text = self._format_conversation_for_evaluation()
            if not conversation_text:
                return None
            
            # Call HuggingFace Regression Evaluation model
            result = self._call_hf_inference_api(
                self.evaluator_hf_model,
                conversation_text,
                "text-classification"
            )
            
            if not result:
                logger.warning("Failed to get evaluation from HF evaluator")
                return self.current_progress_scores
            
            # Process the response into progress scores
            scores = self._process_evaluator_response(result)
            
            # Update current scores
            self.current_progress_scores = scores
            self.last_evaluation_turn = current_turn_count
            
            # Add to progress history
            progress_entry = {
                "timestamp": datetime.now().isoformat(),
                "turn_count": current_turn_count,
                "scores": scores.copy(),
                "conversation_progress": min(current_turn_count / 20, 1.0)  # Assume 20 turns max
            }
            self.progress_history.append(progress_entry)
            
            logger.info(f"CBT Progress Evaluation - Turn {current_turn_count}:")
            for dimension, score in scores.items():
                logger.info(f"  {dimension}: {score:.3f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluating conversation progress: {e}")
            return self.current_progress_scores

    def get_progress_summary(self) -> Dict:
        """Get a summary of CBT progress."""
        if not self.current_progress_scores:
            return {"status": "no_progress_data"}
        
        # Calculate overall progress
        avg_score = sum(self.current_progress_scores.values()) / len(self.current_progress_scores)
        
        # Identify strongest and weakest areas
        sorted_scores = sorted(self.current_progress_scores.items(), key=lambda x: x[1])
        weakest_area = sorted_scores[0]
        strongest_area = sorted_scores[-1]
        
        # Calculate improvement if we have history
        improvement = {}
        if len(self.progress_history) >= 2:
            first_scores = self.progress_history[0]["scores"]
            latest_scores = self.current_progress_scores
            for dimension in first_scores:
                improvement[dimension] = latest_scores[dimension] - first_scores[dimension]
        
        return {
            "status": "active",
            "overall_progress": round(avg_score, 3),
            "current_scores": self.current_progress_scores,
            "strongest_area": {"dimension": strongest_area[0], "score": round(strongest_area[1], 3)},
            "weakest_area": {"dimension": weakest_area[0], "score": round(weakest_area[1], 3)},
            "improvement": improvement,
            "total_evaluations": len(self.progress_history),
            "conversation_turns": len(self.conversation_history)
        }

    def get_compliance_summary(self) -> Dict:
        """Get a summary of system prompt compliance."""
        if not self.current_compliance_scores:
            return {"status": "no_compliance_data"}
        
        # Calculate average confidence across all predictions
        confidence_scores = self.current_compliance_scores.get('confidence', {})
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
        
        # Calculate compliance trend if we have history
        compliance_trend = {}
        if len(self.compliance_history) >= 2:
            recent_scores = [entry['compliance_scores'] for entry in self.compliance_history[-3:]]
            if len(recent_scores) >= 2:
                first_satisfaction = recent_scores[0]['satisfaction_score']
                latest_satisfaction = recent_scores[-1]['satisfaction_score']
                compliance_trend['satisfaction_trend'] = latest_satisfaction - first_satisfaction
        
        return {
            "status": "active",
            "current_cbt_step": self.current_cbt_step,
            "satisfaction_score": round(self.current_compliance_scores['satisfaction_score'], 3),
            "ready_for_next_step": self.current_compliance_scores['ready_for_next_step'],
            "response_quality": self.current_compliance_scores['response_quality'],
            "suggested_action": self.current_compliance_scores['suggested_action'],
            "average_confidence": round(avg_confidence, 3),
            "compliance_trend": compliance_trend,
            "total_evaluations": len(self.compliance_history),
            "conversation_turns": len(self.conversation_history)
        }
    
    def get_conversation_state(self) -> Dict:
        """Get current conversation state as JSON-serializable dict."""
        state = {
            "conversation_id": self.conversation_id,
            "phase": self.phase.value,
            "conversation_history": self.conversation_history,
            "cbt_trigger_detected": self.cbt_trigger_detected,
            "trigger_statement": self.trigger_statement
        }
        
        # Add compliance tracking data if available
        if self.current_compliance_scores:
            state["current_compliance_scores"] = self.current_compliance_scores
            state["compliance_summary"] = self.get_compliance_summary()
            state["current_cbt_step"] = self.current_cbt_step
        
        # Add progress tracking data if available
        if self.current_progress_scores:
            state["current_progress_scores"] = self.current_progress_scores
            state["progress_summary"] = self.get_progress_summary()
        
        return state
    
    def get_flow_data(self) -> Dict:
        """Get flow tracking data for this conversation."""
        return {
            "conversation_id": self.conversation_id,
            "transitions": self.flow_transitions,
            "current_phase": self.phase.value,
            "total_messages": len(self.conversation_history),
            "cbt_triggered": self.cbt_trigger_detected,
            "trigger_statement": self.trigger_statement
        }
    
    def load_conversation_state(self, state: Dict):
        """Load conversation state from dict."""
        self.conversation_id = state.get("conversation_id", self.conversation_id)
        self.phase = ConversationPhase(state.get("phase", ConversationPhase.CHIT_CHAT.value))
        self.conversation_history = state.get("conversation_history", [])
        self.cbt_trigger_detected = state.get("cbt_trigger_detected", False)
        self.trigger_statement = state.get("trigger_statement", None)
        
        # Load compliance tracking data if available
        self.current_compliance_scores = state.get("current_compliance_scores", None)
        self.compliance_history = state.get("compliance_history", [])
        self.current_cbt_step = state.get("current_cbt_step", "initial_assessment")
        
        # Load progress tracking data if available
        self.current_progress_scores = state.get("current_progress_scores", None)
        self.progress_history = state.get("progress_history", [])
        self.last_evaluation_turn = state.get("last_evaluation_turn", 0)


class ResponseGenerator:
    """Response generator that can use either HuggingFace or Ollama."""
    
    def __init__(self, hf_endpoint_url: str = None, hf_api_token: str = None, 
                 use_ollama: bool = False, ollama_model: str = "qwen2.5:14b-instruct"):
        
        self.use_ollama = use_ollama
        
        if use_ollama:
            from ollama_client import OllamaClient
            self.llm_client = OllamaClient(model_name=ollama_model)
            logger.info(f"Using Ollama for LLM inference with model: {ollama_model}")
        else:
            from llm_client import HuggingFaceInferenceClient
            self.llm_client = HuggingFaceInferenceClient(hf_endpoint_url, hf_api_token)
            logger.info("Using HuggingFace for LLM inference")

    def generate_response(self, user_input: str, conversation_manager: ConversationManager) -> str:
        """Generate response using LLM with conversation context and compliance-based adaptation."""
        try:
            # Check for CBT trigger using our trained classifier
            cbt_triggered = self.check_for_cbt_trigger(user_input, conversation_manager)
            
            # Build conversation messages
            messages = self._build_conversation_messages(conversation_manager, user_input)
            
            # Create base conversation context
            context = {
                "phase": conversation_manager.phase.value,
                "cbt_trigger_detected": conversation_manager.cbt_trigger_detected,
                "trigger_statement": conversation_manager.trigger_statement
            }
            
            # ENHANCED: Add compliance-based context adjustments
            if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
                context = self._enhance_context_with_compliance(context, conversation_manager)
            
            # Generate response using LLM client
            response = self.llm_client.generate_response(messages, context)
            
            # Perform real-time CBT progress evaluation if in CBT phase
            if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
                progress_scores = conversation_manager.evaluate_current_conversation()
                if progress_scores:
                    logger.info(f"Real-time CBT progress: {progress_scores}")
                    # Add progress context for potential future use
                    context["current_progress_scores"] = progress_scores
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(conversation_manager)
    
    def _build_conversation_messages(self, conversation_manager: ConversationManager, user_input: str) -> List[Dict]:
        """Build conversation messages for LLM client."""
        messages = []
        
        # Get last 10 messages to avoid token limits
        recent_messages = conversation_manager.conversation_history[-10:]
        
        for message in recent_messages:
            role = "user" if message["speaker"] == "user" else "assistant"
            messages.append({
                "role": role,
                "content": message["content"]
            })
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def check_for_cbt_trigger(self, user_input: str, conversation_manager: ConversationManager) -> bool:
        """Check if user input triggers CBT mode using our trained classifier."""
        if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
            return False  # Already in CBT mode
        
        # Use the conversation manager's ML classifier
        is_trigger = conversation_manager.detect_self_defeating_statement(user_input)
        
        if is_trigger:
            logger.info(f"CBT trigger detected: '{user_input}'")
            conversation_manager.set_cbt_phase(user_input)
        
        return is_trigger
    
    def _enhance_context_with_compliance(self, context: Dict, conversation_manager: ConversationManager) -> Dict:
        """Enhance context with compliance-based guidance for adaptive questioning."""
        
        # Get current compliance scores
        compliance = conversation_manager.current_compliance_scores
        
        if not compliance:
            # First interaction in CBT mode - use standard approach
            context["cbt_guidance"] = "Follow the CBT question sequence. Start with step 1.0 (emotional state)."
            context["current_cbt_step"] = conversation_manager.current_cbt_step
            return context
        
        # Extract compliance metrics
        satisfaction_score = compliance['satisfaction_score']
        ready_for_next_step = compliance['ready_for_next_step']
        response_quality = compliance['response_quality']
        suggested_action = compliance['suggested_action']
        
        logger.info(f"Adapting response based on compliance: satisfaction={satisfaction_score:.3f}, ready={ready_for_next_step}, action={suggested_action}")
        
        # Build adaptive guidance based on compliance scores
        if not ready_for_next_step and satisfaction_score < 0.5:
            # User is struggling - need to rephrase/simplify
            context["cbt_guidance"] = self._get_adaptive_guidance(suggested_action, satisfaction_score, response_quality)
            context["adaptation_needed"] = True
            context["previous_satisfaction"] = satisfaction_score
            
        elif ready_for_next_step and satisfaction_score > 0.7:
            # User is engaged and ready - proceed to next step
            context["cbt_guidance"] = f"User is ready to advance. Move to the next CBT step in the sequence."
            context["adaptation_needed"] = False
            
        else:
            # Moderate engagement - continue current approach but be supportive
            context["cbt_guidance"] = f"Continue current CBT step with supportive approach. User showing moderate engagement."
            context["adaptation_needed"] = False
        
        # Add current step and compliance context
        context["current_cbt_step"] = conversation_manager.current_cbt_step
        context["compliance_scores"] = compliance
        
        return context

    def _get_adaptive_guidance(self, suggested_action: str, satisfaction_score: float, response_quality: str) -> str:
        """Generate specific guidance based on suggested action and scores."""
        
        base_guidance = f"ADAPTATION REQUIRED (satisfaction: {satisfaction_score:.2f}, quality: {response_quality})\n"
        
        if suggested_action == "rephrase_question":
            return base_guidance + """
        The user didn't understand or engage well with the previous question. Please:
        - Rephrase the question in simpler, more accessible language
        - Break complex questions into smaller parts
        - Use concrete examples or analogies
        - Show more empathy and understanding
        - Avoid therapeutic jargon
        Example: Instead of "What cognitive distortions do you notice?" try "When you think about that situation, what specific thoughts go through your mind?"
        """
            
        elif suggested_action == "provide_support":
            return base_guidance + """
        The user needs more emotional support before proceeding. Please:
        - Validate their feelings and experience
        - Offer encouragement and reassurance
        - Acknowledge how difficult this might be
        - Build rapport before asking probing questions
        - Use phrases like "That sounds really tough" or "I can understand why you'd feel that way"
        """
            
        elif suggested_action == "slow_down":
            return base_guidance + """
        The user seems overwhelmed. Please:
        - Slow down the pace significantly
        - Ask only one simple question at a time
        - Provide more explanation of what you're doing and why
        - Check in with how they're feeling
        - Offer breaks or pauses if needed
        """
            
        elif suggested_action == "continue_exploration":
            return base_guidance + """
        Continue exploring the current topic more deeply. Please:
        - Ask follow-up questions about what they've shared
        - Dig deeper into specific details
        - Help them explore their thoughts and feelings more thoroughly
        - Don't rush to the next CBT step yet
        """
            
        elif suggested_action == "redirect_gently":
            return base_guidance + """
        The user may be avoiding or deflecting. Please:
        - Gently acknowledge their response
        - Redirect back to the CBT question with understanding
        - Explain why this exploration is helpful
        - Make it feel safe to engage with difficult topics
        """
            
        else:
            return base_guidance + f"Suggested action: {suggested_action}. Adapt your approach accordingly."

    def _get_fallback_response(self, conversation_manager: ConversationManager) -> str:
        """Provide fallback response if LLM fails."""
        if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
            return "I'm here to help you work through this. Could you tell me more about what you're experiencing?"
        else:
            return "I'm here to chat with you. How can I help you today?"