import os                                                                                                       
import json                                                                                                     
import re                                                                                                       
from sentence_transformers import SentenceTransformer                                                           
import torch                                                                                                    
import sys                                                                                                      
                                                                                                                
# Add the parent directory to the path so we can import the evaluator                                           
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                                    
from gemini_qa_evaluator_api import SingleResponseCBTFlowEvaluator                                              
                                                                                                                
# Initialize the evaluator                                                                                      
evaluator = SingleResponseCBTFlowEvaluator()                                                                    
                                                                                                                
# Directory containing the conversation JSON files                                                              
conversations_dir = "/Users/saitejagudidevini/Documents/Dev/grpo_trainer/Evaluations/Gemini_QA_Testing/conversationsfordeepseek"                                                      
# Process each JSON file in the directory                                                                       
for filename in os.listdir(conversations_dir):                                                                  
    if filename.endswith(".json"):                                                                              
        file_path = os.path.join(conversations_dir, filename)                                                   
                                                                                                                
        # Load the conversation                                                                                 
        with open(file_path, 'r') as f:                                                                         
            conversation_data = json.load(f)                                                                    
                                                                                                                
        # Reset the evaluator for this conversation                                                             
        evaluator = SingleResponseCBTFlowEvaluator()                                                            
                                                                                                                
        # Process each therapist response                                                                       
        for turn in conversation_data["conversation"]:                                                          
            if turn["speaker"] == "therapist":                                                                  
                # Evaluate the therapist's response                                                             
                scores = evaluator.evaluate_response(turn["content"])                                           
                evaluator.update_conversation_context(scores)                                                   
                                                                                                                
        # Get the final progress areas                                                                          
        progress = evaluator.conversation_context["progress"]                                                   
                                                                                                                
        # Format the progress areas as requested                                                                
        formatted_progress = {                                                                                  
            "Balanced Thinking": f"{progress['balanced_thinking']:.2f}",                                        
            "Distortion Identification": f"{progress['distortion_identification']:.2f}",                        
            "Emotion Exploration": f"{progress['emotion_exploration']:.2f}",                                    
            "Evidence Gathering": f"{progress['evidence_gathering']:.2f}",                                      
            "Thought Identification": f"{progress['thought_identification']:.2f}"                               
        }                                                                                                       
                                                                                                                
        # Add the progress areas to the conversation data                                                       
        conversation_data["progress_areas"] = formatted_progress                                                
                                                                                                                
        # Write the updated conversation data back to the file                                                  
        with open(file_path, 'w') as f:                                                                         
            json.dump(conversation_data, f, indent=2)                                                           
                                                                                                                
        print(f"Processed {filename}")                                                                          
        print(f"Progress areas: {formatted_progress}")  