"""
Context Logic - All Context Engineering from core/ files
From: core/enhanced_conversation_manager.py + core/enhanced_response_generator.py + core/context_manager.py
"""

import os
import sys
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import utilities and CBT logic
from utils import config
from cbt_logic import ConversationManager, ConversationPhase, ResponseGenerator
from knowledge_graph import CBTKnowledgeGraph

logger = logging.getLogger(__name__)

# ========================================
# CONTEXT MANAGER DATA STRUCTURES (from core/context_manager.py)
# ========================================

class MemoryType(Enum):
    SESSION = "session"           
    EPISODIC = "episodic"        
    SEMANTIC = "semantic"        
    WORKING = "working"          
    PROCEDURAL = "procedural"    

class ConstraintType(Enum):
    TOKEN_BUDGET = "token_budget"
    SAFETY_RULES = "safety_rules"
    CBT_PROTOCOL = "cbt_protocol"
    USER_PREFERENCES = "user_preferences"
    THERAPEUTIC_BOUNDARIES = "therapeutic_boundaries"

@dataclass
class ContextWindow:
    """Represents the current context window with all relevant information"""
    system_prompt: str
    conversation_history: List[Dict]
    cbt_context: Dict
    user_profile: Dict
    therapeutic_tools: List[str]
    safety_context: Dict
    memory_snippets: List[Dict]
    constraints: Dict
    token_count: int
    created_at: datetime

@dataclass
class MemoryEntry:
    """Represents a memory entry with metadata"""
    content: str
    memory_type: MemoryType
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str]
    metadata: Dict

# ========================================
# CONTEXT MANAGER (from core/context_manager.py)
# ========================================

class ContextManager:
    """
    Advanced Context Engineering Manager for CBT Conversations
    """
    
    def __init__(self, knowledge_graph_path: str = "memory_store", max_token_budget: int = 8192):
        self.knowledge_graph_path = knowledge_graph_path
        self.max_token_budget = max_token_budget
        
        # Initialize memory stores
        self.session_memory = {}
        self.episodic_memory = {}
        self.semantic_memory = {}
        self.working_memory = {}
        self.procedural_memory = {}
        
        # Initialize constraints
        self.active_constraints = {}
        
        # Initialize tools registry
        self.available_tools = {
            "cbt_techniques": [
                "cognitive_restructuring",
                "thought_challenging",
                "behavioral_activation",
                "exposure_therapy",
                "mindfulness_grounding"
            ],
            "crisis_resources": [
                "crisis_hotline_numbers",
                "emergency_protocols",
                "safety_planning",
                "immediate_coping_strategies"
            ],
            "therapeutic_exercises": [
                "mood_tracking",
                "thought_records",
                "behavioral_experiments",
                "gratitude_journaling"
            ]
        }
        
        # Load existing memory if available
        self._load_memory_store()
        
        # Initialize base system prompt
        self.base_system_prompt = self._load_base_system_prompt()
        
        logger.info("Context Manager initialized with context engineering capabilities")
    
    def build_context_window(self, 
                           user_input: str, 
                           conversation_id: str,
                           current_cbt_state: Dict) -> ContextWindow:
        """Build a comprehensive context window using context engineering principles"""
        
        # 1. Retrieve and update working memory
        self._update_working_memory(user_input, conversation_id, current_cbt_state)
        
        # 2. Get active constraints
        constraints = self._get_active_constraints(conversation_id)
        
        # 3. Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(
            user_input, conversation_id, constraints["token_budget"]
        )
        
        # 4. Build adaptive system prompt
        system_prompt = self._build_adaptive_system_prompt(
            current_cbt_state, relevant_memories, constraints
        )
        
        # 5. Format conversation history with constraints
        conversation_history = self._format_conversation_history(
            conversation_id, constraints["token_budget"]
        )
        
        # 6. Get user profile and safety context
        user_profile = self._get_user_profile(conversation_id)
        safety_context = self._get_safety_context(user_input, user_profile)
        
        # 7. Select relevant tools
        therapeutic_tools = self._select_relevant_tools(
            current_cbt_state, user_profile, safety_context
        )
        
        # 8. Calculate token usage
        token_count = self._estimate_token_count(
            system_prompt, conversation_history, relevant_memories
        )
        
        # 9. Create context window
        context_window = ContextWindow(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            cbt_context=current_cbt_state,
            user_profile=user_profile,
            therapeutic_tools=therapeutic_tools,
            safety_context=safety_context,
            memory_snippets=relevant_memories,
            constraints=constraints,
            token_count=token_count,
            created_at=datetime.now()
        )
        
        logger.info(f"Built context window: {token_count} tokens, {len(relevant_memories)} memories")
        return context_window
    
    def _build_adaptive_system_prompt(self, 
                                    cbt_state: Dict, 
                                    memories: List[Dict], 
                                    constraints: Dict) -> str:
        """Build adaptive system prompt using context engineering"""
        
        # Base system prompt
        prompt_parts = [self.base_system_prompt]
        
        # Add CBT state context
        if cbt_state.get("phase") == "cbt_refactoring":
            prompt_parts.append(f"""
CURRENT CBT CONTEXT:
- Phase: CBT Refactoring Active
- Current Step: {cbt_state.get('current_step', 'assessment')}
- Trigger Statement: "{cbt_state.get('trigger_statement', 'N/A')}"
- Progress: {cbt_state.get('progress_scores', {})}
""")
        
        # Add relevant memories
        if memories:
            memory_context = "RELEVANT CONTEXT FROM MEMORY:\n"
            for memory in memories[:3]:  # Top 3 most relevant
                memory_context += f"- {memory.get('content', '')[:100]}...\n"
            prompt_parts.append(memory_context)
        
        # Add constraint-based guidance
        if constraints.get("safety_rules", {}).get("enhanced_monitoring"):
            prompt_parts.append("""
SAFETY ALERT: Enhanced monitoring active due to risk indicators.
- Prioritize safety and wellbeing
- Be extra attentive to crisis signals
- Offer immediate support resources if needed
""")
        
        # Add available tools context
        available_tools = self._get_available_tools_context(cbt_state)
        if available_tools:
            prompt_parts.append(f"AVAILABLE THERAPEUTIC TOOLS:\n{available_tools}")
        
        # Add user preferences
        user_prefs = constraints.get("user_preferences", {})
        if user_prefs:
            prefs_text = f"USER PREFERENCES: {json.dumps(user_prefs, indent=2)}"
            prompt_parts.append(prefs_text)
        
        return "\n\n".join(prompt_parts)
    
    def _load_base_system_prompt(self) -> str:
        """Load the base system prompt for CBT conversations"""
        return """You are a sophisticated CBT (Cognitive Behavioral Therapy) assistant with context engineering capabilities.

Your primary purpose is to engage in natural conversation and provide structured CBT interventions when appropriate.

CORE PRINCIPLES:
1. Context-Aware Responses: Always consider the full context including memories, user profile, and conversation history
2. Memory-Informed Decisions: Use relevant memories to inform your responses and maintain therapeutic continuity
3. Constraint-Respectful Processing: Operate within given constraints while maximizing therapeutic value
4. Tool-Integrated Interventions: Seamlessly incorporate available therapeutic tools and resources
5. Adaptive System Behavior: Adjust your approach based on user feedback and therapeutic progress

CONVERSATION PHASES:
- Phase 1: Chit-Chat - Natural conversation building rapport
- Phase 2: CBT Refactoring - Structured therapeutic intervention

CONTEXT ENGINEERING INTEGRATION:
- Use provided memories to maintain therapeutic continuity
- Respect token constraints while maximizing context value
- Integrate available tools naturally into conversations
- Adapt based on user profile and preferences
- Follow safety protocols based on risk assessment

Remember: You are not just processing the current input, but maintaining and evolving a rich therapeutic relationship through sophisticated context management."""

    # All the helper methods with existing implementations
    def _update_working_memory(self, user_input: str, conversation_id: str, cbt_state: Dict):
        """Update working memory with current processing context"""
        self.working_memory[conversation_id] = {
            "current_user_input": user_input,
            "cbt_state": cbt_state,
            "processing_timestamp": datetime.now(),
            "input_analysis": {
                "length": len(user_input),
                "sentiment": self._analyze_sentiment(user_input),
                "crisis_indicators": self._detect_crisis_indicators(user_input),
                "cbt_relevance": self._assess_cbt_relevance(user_input)
            }
        }
    
    def _get_active_constraints(self, conversation_id: str) -> Dict:
        """Get active constraints for context window construction"""
        constraints = {
            "token_budget": self.max_token_budget,
            "safety_rules": self._get_safety_constraints(),
            "cbt_protocol": self._get_cbt_protocol_constraints(),
            "user_preferences": self._get_user_preferences(conversation_id),
            "therapeutic_boundaries": self._get_therapeutic_boundaries()
        }
        
        if conversation_id in self.working_memory:
            working_mem = self.working_memory[conversation_id]
            
            if working_mem["input_analysis"]["crisis_indicators"]:
                constraints["token_budget"] = min(self.max_token_budget, 6144)
            
            if working_mem["input_analysis"]["sentiment"] == "very_negative":
                constraints["safety_rules"]["enhanced_monitoring"] = True
        
        return constraints
    
    def _retrieve_relevant_memories(self, user_input: str, conversation_id: str, token_budget: int) -> List[Dict]:
        """Retrieve relevant memories using context engineering principles"""
        relevant_memories = []
        available_tokens = token_budget * 0.2
        
        episodic_memories = self._search_episodic_memory(user_input, conversation_id)
        semantic_memories = self._search_semantic_memory(user_input)
        procedural_memories = self._search_procedural_memory(user_input)
        
        all_memories = episodic_memories + semantic_memories + procedural_memories
        ranked_memories = self._rank_memories_by_relevance(all_memories, user_input)
        
        current_tokens = 0
        for memory in ranked_memories:
            memory_tokens = self._estimate_token_count(memory.get("content", ""))
            if current_tokens + memory_tokens <= available_tokens:
                relevant_memories.append(memory)
                current_tokens += memory_tokens
            else:
                break
        
        return relevant_memories
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for context assessment"""
        negative_words = ["terrible", "awful", "horrible", "hate", "failure", "useless", "worthless"]
        positive_words = ["good", "great", "happy", "excellent", "wonderful", "amazing"]
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if negative_count > positive_count * 2:
            return "very_negative"
        elif negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"
    
    def _detect_crisis_indicators(self, text: str) -> bool:
        """Detect crisis indicators in user input"""
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "hurt myself", 
            "not worth living", "better off dead", "no point in living"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def _assess_cbt_relevance(self, text: str) -> float:
        """Assess CBT relevance of user input"""
        cbt_keywords = [
            "feel", "think", "believe", "always", "never", "should", "must",
            "terrible", "awful", "failure", "stupid", "worthless", "hopeless"
        ]
        text_lower = text.lower()
        matches = sum(1 for keyword in cbt_keywords if keyword in text_lower)
        return min(matches / 5.0, 1.0)
    
    def _estimate_token_count(self, *texts) -> int:
        """Rough estimation of token count for text"""
        total_chars = sum(len(str(text)) for text in texts if text)
        return int(total_chars / 3.5)
    
    def _load_memory_store(self):
        """Load memory store from disk"""
        if os.path.exists(self.knowledge_graph_path):
            try:
                with open(self.knowledge_graph_path, 'rb') as f:
                    memory_data = pickle.load(f)
                    self.episodic_memory = memory_data.get('episodic', {})
                    self.semantic_memory = memory_data.get('semantic', {})
                    self.procedural_memory = memory_data.get('procedural', {})
                logger.info(f"Memory store loaded successfully from {self.knowledge_graph_path}")
            except Exception as e:
                logger.error(f"Error loading memory store: {e}")
        else:
            logger.info("No existing memory store found - starting fresh")
    
    def _save_memory_store(self):
        """Save memory store to disk"""
        try:
            memory_data = {
                'episodic': self.episodic_memory,
                'semantic': self.semantic_memory,
                'procedural': self.procedural_memory
            }
            os.makedirs(os.path.dirname(self.knowledge_graph_path), exist_ok=True)
            with open(self.knowledge_graph_path, 'wb') as f:
                pickle.dump(memory_data, f)
            logger.info(f"Memory store saved successfully to {self.knowledge_graph_path}")
        except Exception as e:
            logger.error(f"Error saving memory store: {e}")
    
    # Simplified helper methods (existing implementations)
    def _search_episodic_memory(self, query: str, conversation_id: str) -> List[Dict]:
        return []
    def _search_semantic_memory(self, query: str) -> List[Dict]:
        return []
    def _search_procedural_memory(self, query: str) -> List[Dict]:
        return []
    def _rank_memories_by_relevance(self, memories: List[Dict], query: str) -> List[Dict]:
        return memories
    def _get_safety_constraints(self) -> Dict:
        return {"enhanced_monitoring": False}
    def _get_cbt_protocol_constraints(self) -> Dict:
        return {"strict_sequence": True}
    def _get_user_preferences(self, conversation_id: str) -> Dict:
        return {}
    def _get_therapeutic_boundaries(self) -> Dict:
        return {"no_advice": True, "maintain_boundaries": True}
    def _get_user_profile(self, conversation_id: str) -> Dict:
        return {}
    def _get_safety_context(self, user_input: str, user_profile: Dict) -> Dict:
        return {"crisis_risk": "low"}
    def _select_relevant_tools(self, cbt_state: Dict, user_profile: Dict, safety_context: Dict) -> List[str]:
        return ["cognitive_restructuring", "thought_challenging"]
    def _get_available_tools_context(self, cbt_state: Dict) -> str:
        return "- Cognitive Restructuring\n- Thought Challenging"
    def _format_conversation_history(self, conversation_id: str, token_budget: int) -> List[Dict]:
        return []

# ========================================
# ENHANCED CONVERSATION MANAGER (from core/enhanced_conversation_manager.py)
# ========================================

class EnhancedConversationManager:
    """Enhanced conversation manager with AWS deployment support"""
    
    def __init__(self, 
                 classifier_model_path: str = None,
                 use_ml_classifier: bool = True,
                 knowledge_graph_path: str = None,
                 db_session=None):
        
        logger.info("🧠 Initializing Enhanced Conversation Manager (AWS Ready)")
        
        # Use configuration-based paths
        self.classifier_model_path = classifier_model_path or str(config.get_model_path('classifier'))
        self.knowledge_graph_path = knowledge_graph_path or config.get_api_config('knowledge_graph_path')
        self.db_session = db_session
        
        # Initialize components
        self._initialize_components(use_ml_classifier)
        
        logger.info("✅ Enhanced Conversation Manager initialized successfully")
    
    def _initialize_components(self, use_ml_classifier: bool):
        """Initialize all components with AWS-ready configuration"""
        try:
            # Initialize knowledge graph - Simplified for existing code only
            self.knowledge_graph = CBTKnowledgeGraph(storage_path=self.knowledge_graph_path)
            
            # Initialize active conversations tracking
            self.active_conversations = {}
            
            logger.info("✅ Components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def start_conversation(self, user_id: str = None, channel_info: Dict = None) -> Dict:
        """Start a new conversation with knowledge graph enhancement."""
        logger.info(f"Starting enhanced conversation for user: {user_id}")
        
        try:
            # Create original ConversationManager with full ML pipeline
            self.conversation_manager = ConversationManager(
                use_ml_classifier=True,
                classifier_model_path=self.classifier_model_path
            )
            
            conversation_id = self.conversation_manager.conversation_id
            logger.info(f"Original CBT conversation manager created: {conversation_id}")
            
            # Create user-specific knowledge graph if db_session is available
            if self.db_session and user_id:
                self.knowledge_graph = CBTKnowledgeGraph(
                    storage_path=None,  # No file storage when using DB
                    db_session=self.db_session,
                    user_id=user_id
                )
                logger.info(f"Created user-specific knowledge graph for user: {user_id}")
            else:
                # Fallback to file-based storage
                self.knowledge_graph = CBTKnowledgeGraph(storage_path=self.knowledge_graph_path)
            
            # Initialize context engineering session data
            context_session_data = {
                "session_id": conversation_id,
                "user_id": user_id,
                "start_time": datetime.now().isoformat(),
                "channel_info": channel_info,
                "ml_components_status": {
                    "cbt_classifier_loaded": self.conversation_manager.cbt_classifier is not None,
                    "sequence_regressor_available": self.conversation_manager.sequence_regressor_available,
                    "cbt_evaluator_loaded": self.conversation_manager.cbt_evaluator is not None
                }
            }
            
            # Store enhanced conversation data
            self.active_conversations[conversation_id] = {
                "conversation_manager": self.conversation_manager,
                "context_session": context_session_data,
                "user_id": user_id,
                "channel_info": channel_info,
                "created_at": datetime.now()
            }
            
            logger.info(f"Enhanced conversation started successfully: {conversation_id}")
            
            return {
                "conversation_id": conversation_id,
                "context_session": context_session_data,
                "user_context": self.knowledge_graph.get_user_context(user_id or conversation_id),
                "ml_components_status": context_session_data["ml_components_status"]
            }
            
        except Exception as e:
            logger.error(f"Error starting enhanced conversation: {e}")
            raise
    
    async def process_message_with_context(self, user_input: str, conversation_id: str, llm_client) -> Dict:
        """Process message through original CBT pipeline enhanced with knowledge graph."""
        logger.info(f"Processing message with context for conversation: {conversation_id}")
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Get conversation data
        conv_data = self.active_conversations[conversation_id]
        original_manager = conv_data["conversation_manager"]
        user_id = conv_data.get("user_id", conversation_id)
        
        # Store current phase for comparison
        previous_phase = original_manager.phase
        
        try:
            # STEP 1: Get last assistant message for compliance evaluation
            last_assistant_message = None
            for msg in reversed(original_manager.conversation_history):
                if msg["speaker"] == "assistant":
                    last_assistant_message = msg["content"]
                    break
            
            # STEP 2: Add user message to original conversation manager
            original_manager.add_message("user", user_input, model_question=last_assistant_message)
            logger.info(f"Original CBT: User message added, phase: {original_manager.phase.value}")
            
            # STEP 3: Check if CBT trigger was detected
            cbt_triggered = (previous_phase == ConversationPhase.CHIT_CHAT and 
                           original_manager.phase == ConversationPhase.CBT_REFACTORING)
            
            # STEP 4 & 5: Smart memory collection
            entities = {}
            user_context = ""

            if original_manager.phase == ConversationPhase.CBT_REFACTORING:
                # CBT Mode: Read existing memory + Store new memory
                logger.info(f"CBT Mode: Reading and storing memory for user {user_id}")
                entities = self.knowledge_graph.extract_entities(user_input, cbt_triggered)
                self.knowledge_graph.store_conversation_turn(user_id, entities, cbt_triggered)
                user_context = self.knowledge_graph.get_user_context(user_id, entities)
                
            elif cbt_triggered:
                # CBT Just Triggered: Store the trigger message
                logger.info(f"CBT Triggered: Storing trigger message for user {user_id}")
                entities = self.knowledge_graph.extract_entities(user_input, cbt_triggered)
                self.knowledge_graph.store_conversation_turn(user_id, entities, cbt_triggered)
                user_context = self.knowledge_graph.get_user_context(user_id, entities)
                
            else:
                # Chit-Chat Mode: Only read existing memory
                logger.info(f"Chit-chat Mode: Reading existing memory for user {user_id}")
                user_context = self.knowledge_graph.get_user_context(user_id)
            
            if cbt_triggered:
                logger.info(f"CBT trigger detected by original ML classifier: '{original_manager.trigger_statement}'")
            
            # STEP 7: Generate response using enhanced response generator
            if hasattr(llm_client, 'generate_response_with_knowledge_graph'):
                response = llm_client.generate_response_with_knowledge_graph(
                    user_input=user_input,
                    conversation_manager=original_manager,
                    knowledge_graph_context=user_context,
                    entities=entities,
                    conversation_id=conversation_id
                )
            else:
                response = llm_client.generate_response(user_input, original_manager)
            
            # STEP 8: Add assistant response
            original_manager.add_message("assistant", response)
            logger.info(f"Original CBT: Assistant response added")
            
            # STEP 9: Build response with enhanced data
            enhanced_response = {
                "conversation_id": conversation_id,
                "response": response,
                "phase": original_manager.phase.value,
                "cbt_trigger_detected": cbt_triggered,
                "trigger_statement": original_manager.trigger_statement,
                "current_progress_scores": original_manager.current_progress_scores,
                "current_compliance_scores": original_manager.current_compliance_scores,
                "extracted_entities": entities,
                "user_context": user_context,
                "knowledge_graph_insights": self.knowledge_graph.get_user_insights(user_id)
            }
            
            logger.info(f"Enhanced response prepared for conversation: {conversation_id}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing message with context: {e}")
            response = llm_client.generate_response(user_input, original_manager)
            original_manager.add_message("assistant", response)
            
            return {
                "conversation_id": conversation_id,
                "response": response,
                "phase": original_manager.phase.value,
                "error": str(e),
                "fallback_mode": True
            }
    
    async def get_enhanced_conversation_state(self, conversation_id: str) -> Dict:
        """Get enhanced conversation state combining original CBT data with context engineering."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conv_data = self.active_conversations[conversation_id]
        original_manager = conv_data["conversation_manager"]
        
        # Get original conversation state
        original_state = original_manager.get_conversation_state()
        
        # Add context engineering data
        enhanced_state = {
            **original_state,
            "context_engineering": {
                "session_id": conv_data["context_session"]["session_id"],
                "enhanced_features_active": True
            }
        }
        
        return enhanced_state
    
    async def reset_conversation(self, conversation_id: str) -> Dict:
        """Reset conversation while preserving context engineering setup."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conv_data = self.active_conversations[conversation_id]
        user_id = conv_data["user_id"]
        channel_info = conv_data["channel_info"]
        
        # Start new enhanced conversation
        reset_data = await self.start_conversation(user_id=user_id, channel_info=channel_info)
        
        # Update conversation tracking
        new_conversation_id = reset_data["conversation_id"]
        self.active_conversations[new_conversation_id] = self.active_conversations.pop(conversation_id)
        self.active_conversations[new_conversation_id]["conversation_id"] = new_conversation_id
        
        return {
            **reset_data,
            "initial_response": "Hello! I'm here to chat with you about anything on your mind. How's your day been so far?"
        }

# ========================================
# ENHANCED RESPONSE GENERATOR (from core/enhanced_response_generator.py)
# ========================================

class EnhancedResponseGenerator:
    """Enhanced response generator with AWS deployment support"""
    
    def __init__(self,
                 use_ollama: bool = True,
                 ollama_model: str = None,
                 ollama_base_url: str = None,
                 context_manager=None):
        
        logger.info("🎯 Initializing Enhanced Response Generator (AWS Ready)")
        
        # Use configuration-based settings
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model or config.get_api_config('ollama_model')
        self.ollama_base_url = ollama_base_url or config.get_api_config('ollama_base_url')
        
        # Initialize components
        self._initialize_components(context_manager)
        
        logger.info("✅ Enhanced Response Generator initialized successfully")
    
    def _initialize_components(self, context_manager):
        """Initialize all components with AWS-ready configuration"""
        try:
            # Initialize original response generator
            self.original_generator = ResponseGenerator(
                use_ollama=self.use_ollama,
                ollama_model=self.ollama_model
            )
            
            # Store context manager
            self.context_manager = context_manager
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def generate_response(self, user_input: str, conversation_manager: ConversationManager) -> str:
        """Generate response using original CBT logic."""
        logger.info("Generating response using original CBT logic")
        
        try:
            response = self.original_generator.generate_response(user_input, conversation_manager)
            logger.info("Original CBT response generated successfully")
            return response
        except Exception as e:
            logger.error(f"Error in original response generation: {e}")
            return self.original_generator._get_fallback_response(conversation_manager)
    
    def generate_response_with_knowledge_graph(self, 
                                             user_input: str, 
                                             conversation_manager: ConversationManager,
                                             knowledge_graph_context: str = None,
                                             entities: Dict = None,
                                             conversation_id: str = None) -> str:
        """Generate response using the knowledge graph context."""
        logger.info("Generating response with knowledge graph context")
        
        try:
            # STEP 1: Check for CBT trigger using original ML classifier
            cbt_triggered = self.original_generator.check_for_cbt_trigger(user_input, conversation_manager)
            logger.info(f"Original CBT trigger check: {cbt_triggered}")
            
            # STEP 2: Build enhanced conversation messages
            messages = self._build_messages_with_kg_context(
                conversation_manager, 
                user_input, 
                knowledge_graph_context,
                entities
            )
            
            # STEP 3: Create enhanced conversation context
            context = self._create_context_with_kg(
                conversation_manager, 
                knowledge_graph_context,
                entities,
                conversation_id
            )
            
            # STEP 4: Generate response using original LLM client
            if self.use_ollama and hasattr(self.original_generator.llm_client, 'generate_response'):
                response = self.original_generator.llm_client.generate_response(messages, context)
            else:
                response = self.original_generator.llm_client.generate_response(messages, context)
            
            # STEP 5: Perform original real-time CBT progress evaluation
            if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
                progress_scores = conversation_manager.evaluate_current_conversation()
                if progress_scores:
                    logger.info(f"Original CBT progress evaluation: {progress_scores}")
            
            logger.info("Knowledge graph enhanced response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in knowledge graph response generation: {e}")
            return self.generate_response(user_input, conversation_manager)
    
    def _build_messages_with_kg_context(self, 
                                  conversation_manager: ConversationManager, 
                                  user_input: str,
                                  kg_context: str = None,
                                  entities: Dict = None) -> List[Dict]:
        """Build conversation messages with knowledge graph context."""
        
        messages = []
        
        # Get last 10 messages (same as original)
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
    
    def _create_context_with_kg(self, 
                              conversation_manager: ConversationManager,
                              kg_context: str = None,
                              entities: Dict = None,
                              conversation_id: str = None) -> Dict:
        """Create conversation context with knowledge graph data."""
        
        # Build original context
        context = {
            "phase": conversation_manager.phase.value,
            "trigger_statement": conversation_manager.trigger_statement,
            "conversation_history": conversation_manager.conversation_history,
            "current_progress_scores": conversation_manager.current_progress_scores,
            "entities": entities or {}
        }
        
        # Add CBT-specific context
        if conversation_manager.phase == ConversationPhase.CBT_REFACTORING:
            compliance = conversation_manager.current_compliance_scores
            if compliance:
                satisfaction_score = compliance['satisfaction_score']
                ready_for_next_step = compliance['ready_for_next_step']
                
                if not ready_for_next_step and satisfaction_score < 0.5:
                    context["cbt_guidance"] = "Adaptation required: rephrase question, provide support, or slow down."
                    context["adaptation_needed"] = True
                elif ready_for_next_step and satisfaction_score > 0.7:
                    context["cbt_guidance"] = "User is ready to advance. Move to the next CBT step."
                    context["adaptation_needed"] = False
                else:
                    context["cbt_guidance"] = "Continue current CBT step with supportive approach."
                    context["adaptation_needed"] = False
                
                context["current_cbt_step"] = getattr(conversation_manager, 'current_cbt_step', 'unknown')
                context["compliance_scores"] = compliance
            else:
                context["cbt_guidance"] = "Follow the CBT question sequence. Start with step 1.0 (emotional state)."
                context["current_cbt_step"] = getattr(conversation_manager, 'current_cbt_step', 'initial_assessment')
        
        # Add knowledge graph context
        if kg_context:
            context["knowledge_graph_context"] = kg_context
            logger.info(f"Added memory context to system prompt: {kg_context[:100]}...")
        
        # Add conversation tracking
        if conversation_id:
            context["conversation_tracking"] = {
                "conversation_id": conversation_id,
                "enhanced_features_active": True,
                "memory_integration": True
            }
        
        return context

# ========================================

# KNOWLEDGE GRAPH (using enhanced MCP-style implementation)
# ========================================
# SimpleKnowledgeGraph has been replaced by CBTKnowledgeGraph from knowledge_graph.py
# The CBTKnowledgeGraph provides:
# - MCP-style entity-relationship extraction
# - LLM-based intelligent entity detection
# - Rich contextual relationship mapping
# - Enhanced user insights and memory persistence
