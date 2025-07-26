"""
CBT Engine - Wraps existing sophisticated CBT components
Uses your existing conversation_manager, response_generator, and ML models
"""

import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root for imports
from config.deployment_config import config
project_root = config.project_root
sys.path.append(str(project_root))

# Import existing CBT components
from conversation_manager import ConversationManager, ConversationPhase
from response_generator import ResponseGenerator
from models import StartConversationResponse, SendMessageResponse

# Import existing context engineering components
from core.enhanced_conversation_manager import EnhancedConversationManager
from core.enhanced_response_generator import EnhancedResponseGenerator

logger = logging.getLogger(__name__)

class CBTEngine:
    """
    CBT Engine that wraps your existing sophisticated CBT system
    """
    
    def __init__(self):
        logger.info("🧠 Initializing CBT Engine with existing components")
        
        # Use existing config for paths
        self.classifier_model_path = str(config.get_model_path('classifier'))
        self.sequence_regressor_path = str(config.get_model_path('sequence_regressor'))
        self.evaluator_model_path = str(config.get_model_path('evaluator'))
        
        # Statistics
        self.stats = {
            "conversations_created": 0,
            "messages_processed": 0,
            "cbt_triggers_detected": 0,
            "crisis_interventions": 0
        }
        
        logger.info("✅ CBT Engine initialized with existing sophisticated components")
    
    def create_conversation(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create new CBT conversation using existing components"""
        
        try:
            # Create enhanced conversation manager (uses your existing components)
            enhanced_manager = EnhancedConversationManager(
                use_ml_classifier=True,
                classifier_model_path=self.classifier_model_path,
                knowledge_graph_path=config.get_api_config('knowledge_graph_path')
            )
            
            # Start conversation using existing logic
            conversation_data = enhanced_manager.start_conversation_sync(user_id=user_id)
            
            self.stats["conversations_created"] += 1
            
            logger.info(f"Created conversation using existing CBT components")
            
            return {
                "conversation_id": conversation_data["conversation_id"],
                "enhanced_manager": enhanced_manager,
                "initial_response": conversation_data.get("initial_response", 
                    "Hello! I'm here to support you. How are you feeling today?"),
                "phase": enhanced_manager.conversation_manager.phase.value,
                "conversation_data": conversation_data
            }
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            # Fallback to basic conversation manager
            basic_manager = ConversationManager(use_ml_classifier=True)
            self.stats["conversations_created"] += 1
            
            return {
                "conversation_id": basic_manager.conversation_id,
                "enhanced_manager": None,
                "basic_manager": basic_manager,
                "initial_response": "Hello! I'm here to support you. How are you feeling today?",
                "phase": basic_manager.phase.value,
                "conversation_data": {"conversation_id": basic_manager.conversation_id}
            }
    
    def process_message(self, message: str, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process message using existing CBT logic"""
        
        try:
            # Get the enhanced manager
            enhanced_manager = conversation_data.get("enhanced_manager")
            basic_manager = conversation_data.get("basic_manager")
            
            if enhanced_manager:
                # Use sophisticated enhanced conversation manager
                response_data = enhanced_manager.process_message_sync(message)
                
                # Extract CBT information
                cbt_info = {
                    "trigger_detected": enhanced_manager.conversation_manager.cbt_trigger_detected,
                    "current_phase": enhanced_manager.conversation_manager.phase.value,
                    "progress_scores": getattr(enhanced_manager.conversation_manager, 'current_progress_scores', {}),
                    "conversation_history_length": len(enhanced_manager.conversation_manager.conversation_history)
                }
                
                # Update stats
                if enhanced_manager.conversation_manager.cbt_trigger_detected:
                    self.stats["cbt_triggers_detected"] += 1
                
            elif basic_manager:
                # Fallback to basic manager
                basic_manager.add_message("user", message)
                
                # Generate response using existing response generator
                response_generator = ResponseGenerator(
                    use_ollama=True,
                    ollama_model=config.get_api_config('ollama_model'),
                    ollama_base_url=config.get_api_config('ollama_base_url')
                )
                
                response = response_generator.generate_response(message, basic_manager)
                basic_manager.add_message("assistant", response)
                
                response_data = {
                    "response": response,
                    "conversation_id": basic_manager.conversation_id
                }
                
                cbt_info = {
                    "trigger_detected": basic_manager.cbt_trigger_detected,
                    "current_phase": basic_manager.phase.value,
                    "progress_scores": getattr(basic_manager, 'current_progress_scores', {}),
                    "conversation_history_length": len(basic_manager.conversation_history)
                }
            
            else:
                raise ValueError("No conversation manager available")
            
            self.stats["messages_processed"] += 1
            
            return {
                "response": response_data["response"],
                "phase": cbt_info["current_phase"],
                "cbt_triggered": cbt_info["trigger_detected"],
                "cbt_info": cbt_info,
                "conversation_id": response_data["conversation_id"]
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I'm here to listen and support you. Could you tell me more about what's on your mind?",
                "phase": "chit_chat",
                "cbt_triggered": False,
                "cbt_info": {},
                "conversation_id": conversation_data.get("conversation_id", "unknown")
            }
    
    def get_phase(self, conversation_data: Dict[str, Any]) -> str:
        """Get current conversation phase"""
        enhanced_manager = conversation_data.get("enhanced_manager")
        basic_manager = conversation_data.get("basic_manager")
        
        if enhanced_manager:
            return enhanced_manager.conversation_manager.phase.value
        elif basic_manager:
            return basic_manager.phase.value
        else:
            return "chit_chat"
    
    def get_message_count(self, conversation_data: Dict[str, Any]) -> int:
        """Get message count"""
        enhanced_manager = conversation_data.get("enhanced_manager")
        basic_manager = conversation_data.get("basic_manager")
        
        if enhanced_manager:
            return len(enhanced_manager.conversation_manager.conversation_history)
        elif basic_manager:
            return len(basic_manager.conversation_history)
        else:
            return 0
    
    def cleanup_conversation(self, conversation_data: Dict[str, Any]):
        """Cleanup conversation resources"""
        # Any cleanup needed for existing components
        logger.info("Cleaned up conversation resources")
    
    def get_status(self) -> str:
        """Get engine status"""
        return "operational"
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get CBT analytics"""
        return self.stats.copy()