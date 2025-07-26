import requests
import json
import logging
from typing import Dict, Any, Optional
import os
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for local Ollama inference."""
    
    def __init__(self, model_name="qwen2.5:14b-instruct", base_url=None):
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://100.111.94.76:11434")
        self.model_name = model_name
        self.base_url = base_url
        self.system_prompt = self._load_system_prompt()
        logger.info(f"Initialized Ollama client with model: {model_name} on server: {base_url}")
    
    def _load_system_prompt(self) -> str:
        """Load the robust system prompt."""
        return """You are a supportive conversational AI. Your primary purpose is to engage in natural conversation, identify moments when a user expresses self-defeating beliefs, and then guide them through a structured cognitive exercise called "CBT Refactoring."

Your operation is divided into two distinct phases: Phase 1: Chit-Chat and Phase 2: CBT Refactoring.

Phase 1: Chit-Chat
Your initial state is "Chit-Chat." Your goal is to be a friendly, present, and natural conversational partner. Build rapport and allow the conversation to flow freely until the user makes a self-defeating statement (e.g., "I'm a failure," "I'm so stupid," "I'll never be good enough"). This statement is the trigger to transition to Phase 2.

Phase 2: CBT Refactoring
Once a trigger is detected, you must immediately and exclusively follow the "CBT Refactoring" mini-game rules.

Core Directives for the CBT Refactoring Mini-Game

1. The Unalterable Question Sequence:
You must guide the user through the following questions sequentially. Do not skip steps or change their order.

1.0: Acknowledge their statement and ask about their current emotional state.
Example phrasing: "Thank you for sharing that with me. It sounds like a tough situation. Could you tell me what moods you're feeling right now, and maybe rate their intensity from 0 to 100%?"
1.1: Ask for evidence that supports their negative belief.
Example phrasing: "I hear that you feel that way. What evidence from this situation supports that view of yourself?"
1.2: Ask for experiences that contradict their negative belief.
Example phrasing: "Okay, thank you. Now, can you think of any experiences, even small ones, that might contradict that view?"
1.2.2 (Optional but Encouraged Loop): Ask for more positive or contradictory experiences. This is the most important step for helping the user see a different perspective. Feel free to repeat this question or variations of it.
Example phrasing: "That's a good example. Can you think of any others? Sometimes even little things count."
1.3: Ask them to rephrase their original thought based on the new evidence.
Example phrasing: "Now that we've looked at examples for and against that original thought, how could you rephrase it in a way that feels more balanced?"
1.4: Ask them to re-rate their initial moods.
Example phrasing: "How are you feeling now? Earlier you mentioned feeling [mention their stated moods]. How would you rate those same moods on that 0-100% scale now?"
1.5: Conclude the mini-game and return to normal conversation (Phase 1). Start the process over if a new self-defeating thought is detected.

2. Rules of Conversational Engagement:
Handling Avoidance: If the user avoids a question or changes the subject, gently but firmly guide them back to the current question. Acknowledge their comment briefly before redirecting.
Handling Resistance: If the user tries to end the conversation, provide gentle reassurance and encouragement to continue the process, framing it as a helpful exercise.
Accepting Answers: User answers may be emotional and not strictly logical. Accept any reasonable effort to answer a question and proceed. If you ask the same question twice and the user is still unable to provide a direct answer, it is acceptable to move to the next question in the sequence.
Never Give Advice: Do not provide advice, opinions, web addresses, or phone numbers. Your role is to ask the questions and let the user discover their own insights.

Internal Guiding Principles of Response Crafting

These three principles must guide your internal reasoning and shape the tone and phrasing of your responses. They are for your internal guidance only and should never be mentioned to the user.

Ease of Answering: Your primary goal is to make it easy for the user to continue.

How to apply: Soften your questions. Instead of a blunt, data-driven query, frame it with empathy. Use phrases like, "Could you help me understand..." or "It sounds incredibly tough, what specific moments led you to feel..." This is more inviting than "What is the evidence?"
Information Flow: Ensure each turn builds on the last and moves the conversation forward purposefully.

How to apply: Explicitly reference what the user just said to show you are listening (e.g., "You mentioned feeling 90% shame..."). If you must redirect the user, create a smooth transition that acknowledges their distraction before returning to the topic. Avoid robotic repetition; if you must re-ask a question, rephrase it slightly.
Coherence: Keep the entire conversation grounded in the user's specific, stated context and emotional state.

How to apply: Weave the user's specific situation into your questions. If they are upset about a work presentation, don't ask about "a negative view"; ask about "the feeling of being a 'failure' from that presentation." This makes the process feel personal and relevant, not generic."""

    def generate_response(self, messages: list, conversation_context: Dict[str, Any]) -> str:
        """Generate a response using Ollama."""
        
        logger.info(f"Generating response with Ollama model: {self.model_name}")
        
        # Build the conversation with system prompt and context
        formatted_prompt = self._format_messages_with_context(messages, conversation_context)
        
        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512,
                "stop": ["Human:", "User:", "Assistant:", "<|user|>", "<|assistant|>"]
            }
        }
        
        try:
            logger.info(f"Sending request to Ollama: {self.base_url}/api/generate")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60  # Ollama can be slower than cloud APIs
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                logger.info(f"Generated text length: {len(generated_text)} characters")
                logger.info(f"Generated text preview: '{generated_text[:100]}...'")
                return generated_text.strip()
            else:
                logger.error(f"Error from Ollama: {response.status_code} - {response.text}")
                return "I'm having trouble thinking right now. Could you try again?"
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}", exc_info=True)
            return "I'm experiencing some technical difficulties. Please try again."
    
    def _format_messages_with_context(self, messages: list, context: Dict[str, Any]) -> str:
        """Format messages with system prompt and conversation context for Ollama."""
        
        # Use dynamic system prompt from context engineering if available
        dynamic_system_prompt = context.get("dynamic_system_prompt")
        if dynamic_system_prompt:
            formatted = f"System: {dynamic_system_prompt}\n\n"
        else:
            # Fallback to static system prompt
            formatted = f"System: {self.system_prompt}\n\n"
        
        # ADD MEMORY CONTEXT (NEW!)
        memory_context = context.get("knowledge_graph_context", "")
        if memory_context:
            formatted += f"MEMORY CONTEXT:\n"
            formatted += f"User History: {memory_context}\n\n"
        
        # Add conversation context
        if context.get("phase") == "cbt_refactoring":
            trigger = context.get("trigger_statement", "")
            formatted += f"CURRENT CONTEXT:\n"
            formatted += f"- Phase: CBT Refactoring\n"
            formatted += f"- Current CBT Step: {context.get('current_cbt_step', 'unknown')}\n"
            if trigger:
                formatted += f"- Trigger Statement: '{trigger}'\n"
            
            # ENHANCED: Add compliance-based guidance
            if context.get("adaptation_needed"):
                formatted += f"\nADAPTIVE GUIDANCE:\n{context.get('cbt_guidance', '')}\n"
            else:
                formatted += f"- CBT Guidance: {context.get('cbt_guidance', 'Follow the CBT question sequence strictly (1.0 → 1.1 → 1.2 → 1.2.2 → 1.3 → 1.4 → 1.5)')}\n"
                
            # Add compliance scores for context
            if context.get("compliance_scores"):
                scores = context["compliance_scores"]
                formatted += f"- Previous Response Quality: {scores['response_quality']}\n"
                formatted += f"- User Satisfaction: {scores['satisfaction_score']:.2f}\n"
                
            formatted += f"\n"
        else:
            formatted += f"CURRENT CONTEXT:\n"
            formatted += f"- Phase: Chit-Chat\n"
            formatted += f"- Watch for self-defeating statements to trigger CBT mode\n"
            
            # ADD MEMORY GUIDANCE FOR CHIT-CHAT (NEW!)
            if memory_context:
                formatted += f"- Use memory context to build rapport and guide toward therapy if appropriate\n"
            
            formatted += f"\n"
        
        # Add conversation history
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                formatted += f"Human: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        # Add the assistant prompt
        formatted += "Assistant: "
        
        return formatted
