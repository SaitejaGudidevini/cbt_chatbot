import requests
import json
import logging
from typing import Dict, Any, Optional
import os
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OLLAMA = "ollama"
    GROQ = "groq"

class OllamaClient:
    """Client for both local Ollama and cloud Groq inference."""
    
    def __init__(self, model_name="qwen2.5:14b-instruct", base_url=None, provider=None):
        # Determine provider
        if provider is None:
            # Check environment variable for provider preference
            use_groq = os.getenv("USE_GROQ", "false").lower() == "true"
            self.provider = LLMProvider.GROQ if use_groq else LLMProvider.OLLAMA
        else:
            self.provider = LLMProvider(provider.lower()) if isinstance(provider, str) else provider
        
        # Initialize based on provider
        if self.provider == LLMProvider.GROQ:
            # Groq configuration
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
            logger.info(f"Initialized Groq client with model: {self.groq_model}")
        else:
            # Ollama configuration (existing logic)
            if base_url is None:
                env_url = os.getenv("OLLAMA_BASE_URL")
                logger.info(f"OLLAMA_BASE_URL from env: {env_url}")
                base_url = env_url if env_url else "http://100.111.94.76:11434"
            self.model_name = model_name
            self.base_url = base_url
            logger.info(f"Initialized Ollama client with model: {model_name} on server: {base_url}")
        
        self.system_prompt = self._load_system_prompt()
    
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
        """Generate a response using either Ollama or Groq."""
        
        if self.provider == LLMProvider.GROQ:
            return self._generate_with_groq(messages, conversation_context)
        else:
            return self._generate_with_ollama(messages, conversation_context)
    
    def _generate_with_ollama(self, messages: list, conversation_context: Dict[str, Any]) -> str:
        """Generate a response using Ollama (existing logic preserved)."""
        
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
            headers = {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true"  # Skip ngrok warning page
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
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
    
    def _generate_with_groq(self, messages: list, conversation_context: Dict[str, Any]) -> str:
        """Generate a response using Groq API."""
        
        logger.info(f"Generating response with Groq model: {self.groq_model}")
        
        # Format messages for Groq (uses OpenAI-compatible format)
        formatted_messages = self._format_messages_for_groq(messages, conversation_context)
        
        payload = {
            "model": self.groq_model,
            "messages": formatted_messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "stream": False
        }
        
        try:
            logger.info(f"Sending request to Groq API")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            response = requests.post(
                self.groq_url,
                json=payload,
                headers=headers,
                timeout=30  # Groq is faster than Ollama
            )
            
            logger.info(f"Groq response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                
                logger.info(f"Generated text length: {len(generated_text)} characters")
                logger.info(f"Tokens used: {result.get('usage', {})}")
                return generated_text.strip()
            else:
                logger.error(f"Error from Groq: {response.status_code} - {response.text}")
                return "I'm having trouble thinking right now. Could you try again?"
                
        except Exception as e:
            logger.error(f"Error calling Groq: {e}", exc_info=True)
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
    
    def _format_messages_for_groq(self, messages: list, context: Dict[str, Any]) -> list:
        """Format messages for Groq API (OpenAI-compatible format)."""
        
        formatted_messages = []
        
        # Build system message with context (same logic as Ollama, just different format)
        system_content = ""
        
        # Use dynamic system prompt from context engineering if available
        dynamic_system_prompt = context.get("dynamic_system_prompt")
        if dynamic_system_prompt:
            system_content = dynamic_system_prompt
        else:
            system_content = self.system_prompt
        
        # Add memory context
        memory_context = context.get("knowledge_graph_context", "")
        if memory_context:
            system_content += f"\n\nMEMORY CONTEXT:\nUser History: {memory_context}"
        
        # Add conversation context
        if context.get("phase") == "cbt_refactoring":
            trigger = context.get("trigger_statement", "")
            system_content += f"\n\nCURRENT CONTEXT:"
            system_content += f"\n- Phase: CBT Refactoring"
            system_content += f"\n- Current CBT Step: {context.get('current_cbt_step', 'unknown')}"
            if trigger:
                system_content += f"\n- Trigger Statement: '{trigger}'"
            
            if context.get("adaptation_needed"):
                system_content += f"\n\nADAPTIVE GUIDANCE:\n{context.get('cbt_guidance', '')}"
            else:
                system_content += f"\n- CBT Guidance: {context.get('cbt_guidance', 'Follow the CBT question sequence strictly')}"
                
            if context.get("compliance_scores"):
                scores = context["compliance_scores"]
                system_content += f"\n- Previous Response Quality: {scores['response_quality']}"
                system_content += f"\n- User Satisfaction: {scores['satisfaction_score']:.2f}"
        else:
            system_content += f"\n\nCURRENT CONTEXT:"
            system_content += f"\n- Phase: Chit-Chat"
            system_content += f"\n- Watch for self-defeating statements to trigger CBT mode"
            if memory_context:
                system_content += f"\n- Use memory context to build rapport and guide toward therapy if appropriate"
        
        # Add system message
        formatted_messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map roles to Groq format
            if role in ["user", "human"]:
                formatted_messages.append({"role": "user", "content": content})
            elif role in ["assistant", "ai"]:
                formatted_messages.append({"role": "assistant", "content": content})
        
        return formatted_messages
