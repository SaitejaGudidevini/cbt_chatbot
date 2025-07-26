"""
Context Engineering Engine
Handles context strategy selection, memory management, and response enhancement
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils import MemoryDatabase, calculate_similarity, format_timestamp

logger = logging.getLogger(__name__)

@dataclass
class ContextSession:
    """Context session data container"""
    conversation_id: str
    strategy: str = "auto"
    current_strategy: str = "minimal"
    memories: List[Dict] = None
    tools: List[str] = None
    safety_level: str = "standard"
    token_budget: int = 200
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.memories is None:
            self.memories = []
        if self.tools is None:
            self.tools = []
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class Memory:
    """Memory entry structure"""
    id: str
    conversation_id: str
    content: str
    memory_type: str  # conversation, insight, pattern, crisis
    importance: float
    timestamp: datetime
    context_tags: List[str] = None
    
    def __post_init__(self):
        if self.context_tags is None:
            self.context_tags = []

class ContextEngine:
    """
    Context Engineering Engine - manages context strategies and memory
    """
    
    def __init__(self, memory_db_path: str = "context_memory.db"):
        
        logger.info("🔧 Initializing Context Engine")
        
        # Initialize memory database
        self.memory_db = MemoryDatabase(memory_db_path)
        
        # Context strategies
        self.strategies = {
            "minimal": self._minimal_strategy,
            "rich": self._rich_strategy, 
            "crisis": self._crisis_strategy,
            "auto": self._auto_strategy
        }
        
        # Active sessions
        self.active_sessions = {}
        
        # Context engineering statistics
        self.stats = {
            "sessions_created": 0,
            "memories_stored": 0,
            "strategies_applied": defaultdict(int),
            "context_enhancements": 0,
            "crisis_interventions": 0
        }
        
        # Strategy configuration
        self.strategy_config = {
            "minimal": {"token_budget": 100, "memory_limit": 2, "tools": ["basic_response"]},
            "rich": {"token_budget": 300, "memory_limit": 5, "tools": ["cbt_techniques", "progress_tracking"]},
            "crisis": {"token_budget": 200, "memory_limit": 3, "tools": ["crisis_intervention", "safety_resources"]},
        }
        
        logger.info("✅ Context Engine initialized successfully")
    
    def create_session(self, conversation_id: str) -> ContextSession:
        """Create a new context session"""
        
        session = ContextSession(
            conversation_id=conversation_id,
            strategy="auto",
            current_strategy="minimal"
        )
        
        self.active_sessions[conversation_id] = session
        self.stats["sessions_created"] += 1
        
        logger.info(f"Created context session: {conversation_id}")
        return session
    
    def update_context(self, session: ContextSession, user_input: str, cbt_response: Dict[str, Any]):
        """Update context based on conversation state"""
        
        # Determine appropriate strategy
        new_strategy = self._determine_strategy(user_input, cbt_response, session)
        
        if new_strategy != session.current_strategy:
            logger.info(f"Strategy changed: {session.current_strategy} -> {new_strategy}")
            session.current_strategy = new_strategy
            self.stats["strategies_applied"][new_strategy] += 1
        
        # Apply the strategy
        strategy_config = self.strategies[session.current_strategy](session, user_input, cbt_response)
        
        # Update session with strategy configuration
        session.tools = strategy_config["tools"]
        session.token_budget = strategy_config["token_budget"]
        session.safety_level = strategy_config["safety_level"]
        
        # Retrieve relevant memories
        session.memories = self._retrieve_relevant_memories(
            conversation_id=session.conversation_id,
            query=user_input,
            limit=strategy_config["memory_limit"]
        )
        
        session.last_updated = datetime.now()
        
        logger.info(f"Context updated for {session.conversation_id}: strategy={session.current_strategy}, memories={len(session.memories)}")
    
    def enhance_response(self, response: str, session: ContextSession, cbt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance response with context information"""
        
        enhanced_response = response
        safety_info = {"level": session.safety_level}
        
        # Apply strategy-specific enhancements
        if session.current_strategy == "crisis":
            enhanced_response, safety_info = self._apply_crisis_enhancement(response, session)
            self.stats["crisis_interventions"] += 1
            
        elif session.current_strategy == "rich":
            enhanced_response = self._apply_rich_enhancement(response, session, cbt_data)
            
        elif session.current_strategy == "minimal":
            enhanced_response = self._apply_minimal_enhancement(response, session)
        
        self.stats["context_enhancements"] += 1
        
        # Prepare context info for response
        context_info = {
            "strategy": session.current_strategy,
            "memories_used": len(session.memories),
            "tools_available": len(session.tools),
            "safety_level": session.safety_level,
            "token_budget": session.token_budget
        }
        
        return {
            "response": enhanced_response,
            "context_info": context_info,
            "safety_info": safety_info
        }
    
    def store_interaction(self, conversation_id: str, user_message: str, assistant_response: str):
        """Store conversation interaction in memory"""
        
        # Store user message
        user_memory = Memory(
            id=f"{conversation_id}_{datetime.now().timestamp()}_user",
            conversation_id=conversation_id,
            content=f"User: {user_message}",
            memory_type="conversation",
            importance=0.7,
            timestamp=datetime.now(),
            context_tags=self._extract_context_tags(user_message)
        )
        
        # Store assistant response
        assistant_memory = Memory(
            id=f"{conversation_id}_{datetime.now().timestamp()}_assistant",
            conversation_id=conversation_id,
            content=f"Assistant: {assistant_response}",
            memory_type="conversation", 
            importance=0.6,
            timestamp=datetime.now(),
            context_tags=self._extract_context_tags(assistant_response)
        )
        
        # Store in database
        self.memory_db.store_memory(user_memory)
        self.memory_db.store_memory(assistant_memory)
        
        self.stats["memories_stored"] += 2
        
        # Check for insights or patterns
        self._detect_and_store_insights(conversation_id, user_message, assistant_response)
    
    def _determine_strategy(self, user_input: str, cbt_response: Dict[str, Any], session: ContextSession) -> str:
        """Determine the appropriate context strategy"""
        
        # Crisis detection has highest priority
        if cbt_response.get("crisis_detected", False) or self._detect_crisis_keywords(user_input):
            return "crisis"
        
        # CBT-triggered conversations use rich strategy
        if cbt_response.get("cbt_triggered", False) or cbt_response.get("phase") == "CBT_REFACTORING":
            return "rich"
        
        # Default to minimal for casual conversation
        return "minimal"
    
    def _minimal_strategy(self, session: ContextSession, user_input: str, cbt_response: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal context strategy for casual conversation"""
        return {
            "tools": ["basic_response"],
            "token_budget": 100,
            "memory_limit": 2,
            "safety_level": "standard"
        }
    
    def _rich_strategy(self, session: ContextSession, user_input: str, cbt_response: Dict[str, Any]) -> Dict[str, Any]:
        """Rich context strategy for CBT sessions"""
        return {
            "tools": ["cbt_techniques", "progress_tracking", "therapeutic_questions", "insight_generation"],
            "token_budget": 300,
            "memory_limit": 5,
            "safety_level": "therapeutic"
        }
    
    def _crisis_strategy(self, session: ContextSession, user_input: str, cbt_response: Dict[str, Any]) -> Dict[str, Any]:
        """Crisis context strategy for safety situations"""
        return {
            "tools": ["crisis_intervention", "safety_assessment", "resource_links", "immediate_support"],
            "token_budget": 200,
            "memory_limit": 3,
            "safety_level": "crisis"
        }
    
    def _auto_strategy(self, session: ContextSession, user_input: str, cbt_response: Dict[str, Any]) -> Dict[str, Any]:
        """Auto strategy - determines best strategy automatically"""
        determined_strategy = self._determine_strategy(user_input, cbt_response, session)
        return self.strategies[determined_strategy](session, user_input, cbt_response)
    
    def _retrieve_relevant_memories(self, conversation_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for context"""
        
        memories = self.memory_db.search_memories(
            conversation_id=conversation_id,
            query=query,
            limit=limit
        )
        
        # Convert to dict format for session
        return [
            {
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
                "tags": memory.context_tags
            }
            for memory in memories
        ]
    
    def _apply_crisis_enhancement(self, response: str, session: ContextSession) -> tuple[str, Dict[str, Any]]:
        """Apply crisis-specific enhancements"""
        
        crisis_resources = (
            "\n\n🆘 **Immediate Support Resources:**\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• National Suicide Prevention Lifeline: 988\n"
            "• Emergency Services: 911\n"
            "• Crisis Chat: suicidepreventionlifeline.org"
        )
        
        # Add crisis resources if not already present
        if "988" not in response and "crisis" not in response.lower():
            enhanced_response = response + crisis_resources
        else:
            enhanced_response = response
        
        safety_info = {
            "level": "crisis",
            "resources_provided": True,
            "requires_escalation": True,
            "safety_check_needed": True
        }
        
        return enhanced_response, safety_info
    
    def _apply_rich_enhancement(self, response: str, session: ContextSession, cbt_data: Dict[str, Any]) -> str:
        """Apply rich context enhancements for CBT sessions"""
        
        enhanced_response = response
        
        # Add CBT technique suggestions if appropriate
        if "anxiety" in response.lower() and "breathing" not in response.lower():
            enhanced_response += "\n\n💡 *Would you like to try a breathing exercise to help manage these anxious feelings?*"
        
        # Add progress acknowledgment if available
        cbt_info = cbt_data.get("cbt_info", {})
        progress = cbt_info.get("cbt_progress", {})
        
        if progress.get("status") == "good_progress":
            enhanced_response += "\n\n✨ *I notice you're making good progress in our conversation. Keep up the great work!*"
        
        # Reference relevant memories if available
        relevant_memories = [m for m in session.memories if m.get("importance", 0) > 0.8]
        if relevant_memories:
            enhanced_response += "\n\n🧠 *Based on what we've discussed before, this connects to some patterns we've identified.*"
        
        return enhanced_response
    
    def _apply_minimal_enhancement(self, response: str, session: ContextSession) -> str:
        """Apply minimal enhancements for casual conversation"""
        
        # Very light enhancement - just ensure warmth and engagement
        if len(response) < 50:  # Very short responses
            enhanced_response = response + " I'm here to listen and support you."
        else:
            enhanced_response = response
        
        return enhanced_response
    
    def _detect_crisis_keywords(self, text: str) -> bool:
        """Detect crisis keywords in text"""
        crisis_keywords = [
            "kill myself", "end my life", "suicide", "suicidal",
            "hurt myself", "harm myself", "want to die", "better off dead"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def _extract_context_tags(self, text: str) -> List[str]:
        """Extract context tags from text"""
        tags = []
        
        # Emotion tags
        emotion_keywords = {
            "anxiety": ["anxious", "worried", "nervous", "panic"],
            "depression": ["sad", "depressed", "hopeless", "empty"],
            "anger": ["angry", "frustrated", "mad", "irritated"],
            "positive": ["happy", "good", "better", "great", "excellent"]
        }
        
        text_lower = text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(emotion)
        
        # CBT technique tags
        cbt_keywords = {
            "cognitive": ["thoughts", "thinking", "beliefs", "assumptions"],
            "behavioral": ["behavior", "actions", "habits", "patterns"],
            "mindfulness": ["mindful", "present", "awareness", "meditation"]
        }
        
        for technique, keywords in cbt_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(technique)
        
        return tags
    
    def _detect_and_store_insights(self, conversation_id: str, user_message: str, assistant_response: str):
        """Detect and store therapeutic insights"""
        
        # Look for insight indicators in user messages
        insight_indicators = [
            "i realize", "i understand", "makes sense", "i see that",
            "i never thought", "that helps", "good point", "i get it"
        ]
        
        user_lower = user_message.lower()
        
        for indicator in insight_indicators:
            if indicator in user_lower:
                # Store as insight memory
                insight_memory = Memory(
                    id=f"{conversation_id}_insight_{datetime.now().timestamp()}",
                    conversation_id=conversation_id,
                    content=f"INSIGHT: {user_message}",
                    memory_type="insight",
                    importance=0.9,
                    timestamp=datetime.now(),
                    context_tags=["insight", "breakthrough"]
                )
                
                self.memory_db.store_memory(insight_memory)
                logger.info(f"Stored therapeutic insight for conversation {conversation_id}")
                break
    
    # ========================================
    # Public Interface Methods
    # ========================================
    
    def get_conversation_memories(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a conversation"""
        memories = self.memory_db.get_conversation_memories(conversation_id)
        
        return [
            {
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "timestamp": format_timestamp(memory.timestamp),
                "tags": memory.context_tags
            }
            for memory in memories
        ]
    
    def get_current_strategy(self, session: ContextSession) -> str:
        """Get current strategy for session"""
        return session.current_strategy
    
    def set_strategy(self, session: ContextSession, strategy: str):
        """Manually set strategy for session"""
        if strategy in self.strategies:
            session.strategy = strategy
            session.current_strategy = strategy
            logger.info(f"Manually set strategy to {strategy} for {session.conversation_id}")
    
    def get_session_info(self, session: ContextSession) -> Dict[str, Any]:
        """Get session information"""
        return {
            "strategy": session.current_strategy,
            "memories": len(session.memories),
            "tools": len(session.tools),
            "safety_level": session.safety_level,
            "token_budget": session.token_budget,
            "last_updated": format_timestamp(session.last_updated)
        }
    
    def cleanup_session(self, session: ContextSession):
        """Cleanup session resources"""
        conversation_id = session.conversation_id
        if conversation_id in self.active_sessions:
            del self.active_sessions[conversation_id]
        logger.info(f"Cleaned up context session: {conversation_id}")
    
    def cleanup_old_memories(self, days: int = 30) -> int:
        """Cleanup old memories"""
        return self.memory_db.cleanup_old_memories(days)
    
    # ========================================
    # Status and Analytics Methods  
    # ========================================
    
    def get_status(self) -> str:
        """Get simple status"""
        return "operational"
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed engine status"""
        return {
            "status": "operational",
            "active_sessions": len(self.active_sessions),
            "memory_db_status": "connected",
            "strategies_available": list(self.strategies.keys()),
            "statistics": self.stats.copy()
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get context engine analytics"""
        return {
            "sessions_created": self.stats["sessions_created"],
            "memories_stored": self.stats["memories_stored"],
            "strategies_applied": dict(self.stats["strategies_applied"]),
            "context_enhancements": self.stats["context_enhancements"],
            "crisis_interventions": self.stats["crisis_interventions"],
            "active_sessions": len(self.active_sessions)
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self.memory_db.get_stats() 