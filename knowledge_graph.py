"""
CBT Knowledge Graph System - MCP Style
Entity-Relationship approach for capturing user context
"""

import json
import requests
import re
import os
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CBTKnowledgeGraph:
    """
    MCP-Style Knowledge Graph for CBT conversations
    Stores entities, relations, and observations like real MCP system
    """
    
    def __init__(self, storage_path: str = "cbt_knowledge_graph.json", db_session=None, user_id=None):
        """Initialize the knowledge graph with MCP-style structure
        
        Args:
            storage_path: Path for file-based storage (used if db_session is None)
            db_session: Database session for PostgreSQL storage
            user_id: User ID for database storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.db_session = db_session
        self.user_id = user_id
        self.user = None
        
        # MCP-style storage
        self.entities = {}  # {entity_name: entity_data}
        self.relations = []  # [{from, to, relationType, timestamp}]
        
        # Load existing memory
        self._load_memory()
        
        mode = "database" if db_session else "file"
        logger.info(f"CBT Knowledge Graph initialized with MCP-style entity system (mode: {mode})")
    
    def extract_entities_no_training(self, user_input: str, cbt_triggered: bool) -> Dict[str, Any]:
        """
        Extract entities and relations using LLM analysis (MCP-style)
        Keeps same method name for compatibility but uses new approach
        """
        logger.info(f"🧠 Extracting entities from: '{user_input}'")
        
        analysis_prompt = f"""Extract entities and relationships from this user message for a therapy knowledge graph.

User message: "{user_input}"

Extract:
1. ENTITIES: Key people, concepts, events, or situations mentioned
2. RELATIONS: How these entities relate to each other
3. OBSERVATIONS: Specific facts about each entity from this message

Respond with ONLY valid JSON:
{{
  "entities": [
    {{
      "name": "User_Self",
      "entityType": "person",
      "observations": ["specific observation from message"]
    }},
    {{
      "name": "concept_or_person_mentioned", 
      "entityType": "concept/person/event/situation",
      "observations": ["what user said about this"]
    }}
  ],
  "relations": [
    {{
      "from": "User_Self",
      "to": "other_entity",
      "relationType": "feels_about/struggles_with/experiences/etc"
    }}
  ]
}}

Important: 
- Always include "User_Self" as main entity
- Capture the actual context, not just emotions
- Use clear relationship descriptions
- Include specific observations from what user said"""

        try:
            # Use Ollama for analysis with environment variable URL
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            headers = {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"  # Skip ngrok warning page
            }
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "qwen2.5:14b-instruct",
                    "prompt": analysis_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                response_text = response.json()["response"]
                logger.info(f"🤖 LLM analysis received: {len(response_text)} characters")
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Log extracted entities
                    entities = analysis.get("entities", [])
                    relations = analysis.get("relations", [])
                    
                    logger.info(f"📊 Extracted: {len(entities)} entities, {len(relations)} relations")
                    for entity in entities:
                        logger.info(f"   🏷️ {entity['name']} ({entity['entityType']}): {len(entity['observations'])} observations")
                    
                    # Convert to expected format for compatibility
                    return {
                        "entities": entities,
                        "relations": relations,
                        "timestamp": datetime.now().isoformat(),
                        "raw_input": user_input,
                        "extraction_method": "llm_mcp_style",
                        "cbt_triggered": cbt_triggered,
                        
                        # Legacy compatibility fields
                        "emotions": self._extract_emotions_from_entities(entities),
                        "trigger_sources": self._extract_triggers_from_entities(entities),
                        "emotional_intensity": self._assess_intensity_from_observations(entities),
                        "context_category": "entity_based"
                    }
                    
        except Exception as e:
            logger.error(f"❌ LLM entity extraction failed: {e}")
        
        # Fallback: basic entity extraction
        return self._basic_entity_extraction(user_input, cbt_triggered)
    
    def extract_entities(self, user_input: str, cbt_triggered: bool) -> Dict[str, Any]:
        """Alias for extract_entities_no_training for compatibility"""
        return self.extract_entities_no_training(user_input, cbt_triggered)
    
    def _extract_emotions_from_entities(self, entities: List[Dict]) -> List[str]:
        """Extract emotion-like concepts from entities for legacy compatibility"""
        emotions = []
        for entity in entities:
            entity_name = entity.get("name", "").lower()
            observations = entity.get("observations", [])
            
            # Look for emotional content in entity names and observations
            emotional_indicators = ["feeling", "emotion", "mood", "sad", "happy", "angry", "anxious", "stressed"]
            
            if any(indicator in entity_name for indicator in emotional_indicators):
                emotions.append(entity_name)
            
            for obs in observations:
                obs_lower = obs.lower()
                if any(indicator in obs_lower for indicator in emotional_indicators):
                    # Extract emotional content from observation
                    emotional_content = [word for word in obs_lower.split() if word in emotional_indicators]
                    emotions.extend(emotional_content)
        
        return list(set(emotions))[:5]  # Limit to 5 unique emotions
    
    def _extract_triggers_from_entities(self, entities: List[Dict]) -> List[str]:
        """Extract trigger-like concepts from entities for legacy compatibility"""
        triggers = []
        for entity in entities:
            if entity.get("entityType") in ["concept", "situation", "event"]:
                triggers.append(entity["name"])
        
        return triggers[:5]  # Limit to 5 triggers
    
    def _assess_intensity_from_observations(self, entities: List[Dict]) -> str:
        """Assess emotional intensity from entity observations"""
        intensity_indicators = {
            "high": ["very", "extremely", "really", "so", "completely", "totally", "overwhelming"],
            "medium": ["quite", "somewhat", "pretty", "fairly"],
            "low": ["a bit", "slightly", "little"]
        }
        
        all_observations = []
        for entity in entities:
            all_observations.extend(entity.get("observations", []))
        
        all_text = " ".join(all_observations).lower()
        
        for level, indicators in intensity_indicators.items():
            if any(indicator in all_text for indicator in indicators):
                return level
        
        return "medium"  # default
    
    def _basic_entity_extraction(self, user_input: str, cbt_triggered: bool) -> Dict[str, Any]:
        """Fallback entity extraction if LLM fails"""
        logger.warning("🔄 Using fallback entity extraction")
        
        # Create basic User_Self entity
        user_entity = {
            "name": "User_Self",
            "entityType": "person", 
            "observations": [f"Said: '{user_input}'"]
        }
        
        entities = [user_entity]
        relations = []
        
        # Basic emotional content detection
        emotional_words = ["sad", "happy", "angry", "anxious", "stressed", "worried", "frustrated"]
        detected_emotions = [word for word in emotional_words if word in user_input.lower()]
        
        if detected_emotions:
            emotion_entity = {
                "name": f"Emotion_{detected_emotions[0].title()}",
                "entityType": "concept",
                "observations": [f"User is experiencing {detected_emotions[0]}"]
            }
            entities.append(emotion_entity)
            
            relations.append({
                "from": "User_Self",
                "to": f"Emotion_{detected_emotions[0].title()}",
                "relationType": "experiences"
            })
        
        return {
            "entities": entities,
            "relations": relations,
            "timestamp": datetime.now().isoformat(),
            "raw_input": user_input,
            "extraction_method": "fallback_basic",
            "cbt_triggered": cbt_triggered,
            
            # Legacy compatibility
            "emotions": detected_emotions,
            "trigger_sources": ["general_situation"],
            "emotional_intensity": "medium",
            "context_category": "fallback"
        }
    
    def store_conversation_turn(self, user_id: str, entities_data: Dict[str, Any], cbt_triggered: bool):
        """Store conversation turn with entities and relations (MCP-style)"""
        logger.info(f"💾 Storing MCP-style entities for user: {user_id}")
        
        entities = entities_data.get("entities", [])
        relations = entities_data.get("relations", [])
        
        # Store entities
        for entity in entities:
            entity_name = entity["name"]
            
            # Replace User_Self with actual user ID
            if entity_name == "User_Self":
                entity_name = f"User_{user_id}"
            
            if entity_name not in self.entities:
                self.entities[entity_name] = {
                    "name": entity_name,
                    "entityType": entity["entityType"],
                    "observations": entity["observations"].copy(),
                    "first_seen": datetime.now().isoformat(),
                    "conversation_count": 1
                }
                logger.info(f"🆕 New entity: {entity_name} ({entity['entityType']})")
            else:
                # Add new observations
                existing_obs = set(self.entities[entity_name]["observations"])
                new_obs = [obs for obs in entity["observations"] if obs not in existing_obs]
                
                if new_obs:
                    self.entities[entity_name]["observations"].extend(new_obs)
                    logger.info(f"📝 Updated {entity_name}: +{len(new_obs)} observations")
                
                self.entities[entity_name]["conversation_count"] += 1
        
        # Store relations  
        for relation in relations:
            # Replace User_Self with actual user ID
            from_entity = relation["from"]
            to_entity = relation["to"]
            
            if from_entity == "User_Self":
                from_entity = f"User_{user_id}"
            if to_entity == "User_Self":
                to_entity = f"User_{user_id}"
            
            # Check if relation already exists
            relation_exists = any(
                r["from"] == from_entity and 
                r["to"] == to_entity and 
                r["relationType"] == relation["relationType"]
                for r in self.relations
            )
            
            if not relation_exists:
                new_relation = {
                    "from": from_entity,
                    "to": to_entity,
                    "relationType": relation["relationType"],
                    "timestamp": datetime.now().isoformat()
                }
                self.relations.append(new_relation)
                logger.info(f"🔗 New relation: {from_entity} → {relation['relationType']} → {to_entity}")
        
        # Save to file with enhanced logging
        self._save_memory()
    
    def get_user_context(self, user_id: str, current_entities: Dict[str, Any] = None) -> str:
        """Generate contextual information from entity relationships"""
        user_entity_name = f"User_{user_id}"
        
        if user_entity_name not in self.entities:
            return "This is our first conversation. I'm here to help you work through whatever you're experiencing."
        
        # Get user entity and related information
        user_entity = self.entities[user_entity_name]
        user_relations = [r for r in self.relations if r["from"] == user_entity_name or r["to"] == user_entity_name]
        
        context_parts = []
        
        # Recent observations about the user
        recent_observations = user_entity["observations"][-3:]  # Last 3 observations
        if recent_observations:
            context_parts.append(f"From our conversations, I understand that you {', and '.join(recent_observations).lower()}.")
        
        # Key relationships
        if user_relations:
            relationship_summaries = []
            for relation in user_relations[-3:]:  # Recent relations
                if relation["from"] == user_entity_name:
                    related_entity = relation["to"].replace("_", " ").lower()
                    relationship_type = relation["relationType"].replace("_", " ")
                    relationship_summaries.append(f"you {relationship_type} {related_entity}")
                else:
                    related_entity = relation["from"].replace("_", " ").lower()
                    relationship_type = relation["relationType"].replace("_", " ")
                    relationship_summaries.append(f"{related_entity} {relationship_type} you")
            
            if relationship_summaries:
                context_parts.append(f"I also notice that {', and '.join(relationship_summaries)}.")
        
        return " ".join(context_parts) if context_parts else "I'm here to continue supporting you."
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights from entity relationships"""
        user_entity_name = f"User_{user_id}"
        
        if user_entity_name not in self.entities:
            return {"status": "no_data"}
        
        user_entity = self.entities[user_entity_name]
        user_relations = [r for r in self.relations if r["from"] == user_entity_name or r["to"] == user_entity_name]
        
        # Find related entities
        related_entities = []
        for relation in user_relations:
            other_entity = relation["to"] if relation["from"] == user_entity_name else relation["from"]
            if other_entity != user_entity_name:
                related_entities.append({
                    "entity": other_entity,
                    "relationship": relation["relationType"],
                    "entity_type": self.entities.get(other_entity, {}).get("entityType", "unknown")
                })
        
        return {
            "total_conversations": user_entity.get("conversation_count", 0),
            "total_observations": len(user_entity.get("observations", [])),
            "related_entities": related_entities[:5],  # Top 5 related entities
            "entity_types_involved": list(set([re["entity_type"] for re in related_entities])),
            "relationship_patterns": list(set([re["relationship"] for re in related_entities]))
        }
    
    def _save_memory(self):
        """Save knowledge graph to disk or database with detailed logging"""
        if self.db_session and self.user_id:
            self._save_to_db()
        else:
            self._save_to_file()
    
    def _save_to_db(self):
        """Save knowledge graph to database"""
        try:
            # Import here to avoid circular imports
            from utils import DBUser
            
            if not self.user:
                self.user = self.db_session.query(DBUser).filter_by(id=self.user_id).first()
            
            if self.user:
                # Prepare knowledge graph data
                kg_data = {
                    "entities": self.entities,
                    "relations": self.relations,
                    "metadata": {
                        "last_saved": datetime.now().isoformat(),
                        "total_entities": len(self.entities),
                        "total_relations": len(self.relations),
                        "system_type": "mcp_style_cbt_kg_db"
                    }
                }
                
                # Update user's knowledge graph
                self.user.knowledge_graph = kg_data
                self.db_session.commit()
                
                # Enhanced logging
                entity_types = {}
                for entity in self.entities.values():
                    entity_type = entity.get("entityType", "unknown")
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                logger.info(f"🧠 KNOWLEDGE GRAPH SAVED TO DATABASE:")
                logger.info(f"   👤 User: {self.user_id}")
                logger.info(f"   🏷️ Entities: {len(self.entities)} ({entity_types})")
                logger.info(f"   🔗 Relations: {len(self.relations)}")
                logger.info(f"   ⏰ Saved: {datetime.now().strftime('%H:%M:%S')}")
                
            else:
                logger.error(f"❌ User {self.user_id} not found in database")
                
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge graph to DB: {e}")
            if self.db_session:
                self.db_session.rollback()
    
    def _save_to_file(self):
        """Save knowledge graph to file"""
        try:
            if not self.storage_path:
                logger.warning("No storage path configured for file saving")
                return
                
            memory_data = {
                "entities": self.entities,
                "relations": self.relations,
                "metadata": {
                    "last_saved": datetime.now().isoformat(),
                    "total_entities": len(self.entities),
                    "total_relations": len(self.relations),
                    "system_type": "mcp_style_cbt_kg"
                }
            }
            
            # Get file stats
            file_existed = self.storage_path.exists()
            old_size = self.storage_path.stat().st_size if file_existed else 0
            
            # Save the file
            with open(self.storage_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            # Get new stats
            new_size = self.storage_path.stat().st_size
            
            # Enhanced logging
            entity_types = {}
            for entity in self.entities.values():
                entity_type = entity.get("entityType", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            logger.info(f"🧠 KNOWLEDGE GRAPH UPDATED (MCP-Style):")
            logger.info(f"   📁 File: {self.storage_path}")
            logger.info(f"   📊 Size: {old_size} → {new_size} bytes ({new_size - old_size:+d})")
            logger.info(f"   🏷️ Entities: {len(self.entities)} ({entity_types})")
            logger.info(f"   🔗 Relations: {len(self.relations)}")
            logger.info(f"   ⏰ Saved: {datetime.now().strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge graph to file: {e}")
    
    def _load_memory(self):
        """Load knowledge graph from disk or database"""
        if self.db_session and self.user_id:
            self._load_from_db()
        else:
            self._load_from_file()
    
    def _load_from_db(self):
        """Load knowledge graph from database"""
        try:
            # Import here to avoid circular imports
            from utils import DBUser
            
            # Get user from database
            self.user = self.db_session.query(DBUser).filter_by(id=self.user_id).first()
            
            if self.user and self.user.knowledge_graph:
                kg_data = self.user.knowledge_graph
                self.entities = kg_data.get("entities", {})
                self.relations = kg_data.get("relations", [])
                logger.info(f"📂 Loaded knowledge graph from DB: {len(self.entities)} entities, {len(self.relations)} relations")
            else:
                logger.info("🆕 No existing knowledge graph in DB - starting fresh")
                self.entities = {}
                self.relations = []
                
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph from DB: {e}")
            self.entities = {}
            self.relations = []
    
    def _load_from_file(self):
        """Load knowledge graph from file"""
        try:
            if self.storage_path and self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    memory_data = json.load(f)
                
                self.entities = memory_data.get("entities", {})
                self.relations = memory_data.get("relations", [])
                
                logger.info(f"📂 Loaded knowledge graph from file: {len(self.entities)} entities, {len(self.relations)} relations")
            else:
                logger.info("🆕 Starting fresh knowledge graph")
                
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph from file: {e}")
            self.entities = {}
            self.relations = []