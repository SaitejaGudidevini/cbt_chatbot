import json                                                                                                                                           
import os                                                                                                                                             
import glob                                                                                                                                           
import re                                                                                                                                             
from collections import defaultdict                                                                                                                   
import pandas as pd                                                                                                                                   
from tqdm import tqdm     

# Define cognitive distortion patterns to identify initial problems                                                                                   
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

def identify_initial_problem(initial_input):                                                                                                          
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

def clean_therapist_response(content):                                                                                                                
    """Clean the therapist response by removing reasoning tags."""                                                                                    
    # Remove reasoning tags                                                                                                                           
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)                                                                     
    # Extract answer content                                                                                                                          
    content = re.sub(r'<answer>(.*?)</answer>', r'\1', content, flags=re.DOTALL)                                                                      
    return content.strip()                                                                                                                            

def format_conversation(turns):                                                                                                                       
    """Format conversation turns into a readable text format."""                                                                                      
    formatted_text = ""                                                                                                                               
    for turn in turns:                                                                                                                                
        speaker = "User" if turn["speaker"] == "patient" else "Assistant"                                                                             
        content = turn["content"]                                                                                                                     
                                                                                                                                                      
        # Clean the content for assistant responses                                                                                                   
        if speaker == "Assistant":                                                                                                                    
            content = clean_therapist_response(content)                                                                                               
                                                                                                                                                      
        formatted_text += f"{speaker}: {content}\n\n"                                                                                                 
                                                                                                                                                      
    return formatted_text.strip()  

def transform_conversation_to_training_examples(json_file_path):                                                                                      
    """Transform a conversation JSON file into multiple training examples."""                                                                         
    try:                                                                                                                                              
        with open(json_file_path, 'r', encoding='utf-8') as f:                                                                                        
            conversation_data = json.load(f)                                                                                                          
                                                                                                                                                      
        # Extract basic information                                                                                                                   
        conversation_id = os.path.basename(json_file_path).replace('.json', '')                                                                       
                                                                                                                                                      
        # Check if the required fields exist                                                                                                          
        if "metadata" not in conversation_data or "conversation" not in conversation_data or "progress_areas" not in conversation_data:               
            print(f"Warning: Missing required fields in {json_file_path}")                                                                            
            return []                                                                                                                                 
                                                                                                                                                      
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
            print(f"Warning: Issue with progress scores in {json_file_path}: {e}")                                                                    
            return []                                                                                                                                 
                                                                                                                                                      
        # Identify the initial problem/cognitive distortion                                                                                           
        initial_problem = identify_initial_problem(initial_input)                                                                                     
                                                                                                                                                      
        # Create training examples at different conversation lengths                                                                                  
        training_examples = []                                                                                                                        
                                                                                                                                                      
        # Process the conversation at different points (every 2 turns)                                                                                
        for end_turn in range(2, total_turns + 1, 2):                                                                                                 
            # Extract conversation up to this point                                                                                                   
            conversation_so_far = conversation_data["conversation"][:end_turn]                                                                        
                                                                                                                                                      
            # Format the conversation text                                                                                                            
            formatted_text = format_conversation(conversation_so_far)                                                                                 
                                                                                                                                                      
            # Calculate conversation progress                                                                                                         
            conversation_progress = round(end_turn / total_turns, 2)                                                                                  
                                                                                                                                                      
            # Create the training example                                                                                                             
            example = {                                                                                                                               
                "id": f"{conversation_id}_turn_{end_turn}",                                                                                           
                "conversation_text": formatted_text,                                                                                                  
                "target_scores": target_scores,                                                                                                       
                "metadata": {                                                                                                                         
                    "total_turns": total_turns,                                                                                                       
                    "turns_included": end_turn,                                                                                                       
                    "initial_problem": initial_problem,                                                                                               
                    "conversation_progress": conversation_progress                                                                                    
                }                                                                                                                                     
            }                                                                                                                                         
                                                                                                                                                      
            training_examples.append(example)                                                                                                         
                                                                                                                                                      
        return training_examples                                                                                                                      
                                                                                                                                                      
    except Exception as e:                                                                                                                            
        print(f"Error processing {json_file_path}: {e}")                                                                                              
        return []            

def process_all_conversations(conversations_dir, output_file):                                                                                        
    """Process all conversation files and save as a single JSON file."""                                                                              
    all_examples = []                                                                                                                                 
                                                                                                                                                      
    # Get all JSON files in the directory                                                                                                             
    json_files = glob.glob(os.path.join(conversations_dir, "*.json"))                                                                                 
    print(f"Found {len(json_files)} conversation files")                                                                                              
                                                                                                                                                      
    # Process each file with a progress bar                                                                                                           
    for json_file in tqdm(json_files, desc="Processing conversations"):                                                                               
        examples = transform_conversation_to_training_examples(json_file)                                                                             
        all_examples.extend(examples)                                                                                                                 
                                                                                                                                                      
    print(f"Generated {len(all_examples)} training examples from {len(json_files)} conversations")                                                    
                                                                                                                                                      
    # Save all examples to a single file                                                                                                              
    with open(output_file, 'w', encoding='utf-8') as f:                                                                                               
        json.dump(all_examples, f, indent=2)                                                                                                          
                                                                                                                                                      
    print(f"Saved training examples to {output_file}")                                                                                                
                                                                                                                                                      
    # Also save a flattened version for easier inspection                                                                                             
    flattened_examples = []                                                                                                                           
    for example in all_examples:                                                                                                                      
        flat_example = {                                                                                                                              
            "id": example["id"],                                                                                                                      
            "conversation_text": example["conversation_text"],                                                                                        
            "thought_identification": example["target_scores"]["thought_identification"],                                                             
            "emotion_exploration": example["target_scores"]["emotion_exploration"],                                                                   
            "distortion_identification": example["target_scores"]["distortion_identification"],                                                       
            "evidence_gathering": example["target_scores"]["evidence_gathering"],                                                                     
            "balanced_thinking": example["target_scores"]["balanced_thinking"],                                                                       
            "total_turns": example["metadata"]["total_turns"],                                                                                        
            "turns_included": example["metadata"]["turns_included"],                                                                                  
            "initial_problem": example["metadata"]["initial_problem"],                                                                                
            "conversation_progress": example["metadata"]["conversation_progress"]                                                                     
        }                                                                                                                                             
        flattened_examples.append(flat_example)                                                                                                       
                                                                                                                                                      
    # Save as CSV for easy viewing                                                                                                                    
    df = pd.DataFrame(flattened_examples)                                                                                                             
    csv_output = output_file.replace('.json', '.csv')                                                                                                 
    df.to_csv(csv_output, index=False)                                                                                                                
    print(f"Saved flattened examples to {csv_output} for easy inspection")                                                                            
                                                                                                                                                      
    return all_examples             


# Example usage                                                                                                                                       
if __name__ == "__main__":                                                                                                                            
    # Set your conversations directory and output file                                                                                                
    conversations_dir = "/Users/saitejagudidevini/Documents/Dev/grpo_trainer/Evaluations/Gemini_QA_Testing/conversations"                                                                                 
    output_file = "cbt_evaluator_training_data.json"                                                                                                  
                                                                                                                                                      
    # Process all conversations                                                                                                                       
    examples = process_all_conversations(conversations_dir, output_file)                                                                              
                                                                                                                                                      
    # Print some statistics                                                                                                                           
    if examples:                                                                                                                                      
        print("\nDataset Statistics:")                                                                                                                
        print(f"Total examples: {len(examples)}")                                                                                                     
                                                                                                                                                      
        # Count examples by initial problem                                                                                                           
        problem_counts = {}                                                                                                                           
        for example in examples:                                                                                                                      
            problem = example["metadata"]["initial_problem"]                                                                                          
            problem_counts[problem] = problem_counts.get(problem, 0) + 1                                                                              
                                                                                                                                                      
        print("\nExamples by initial problem:")                                                                                                       
        for problem, count in sorted(problem_counts.items(), key=lambda x: x[1], reverse=True):                                                       
            print(f"  {problem}: {count}")                                                                                                            
                                                                                                                                                      
        # Show distribution of conversation lengths                                                                                                   
        turn_counts = {}                                                                                                                              
        for example in examples:                                                                                                                      
            turns = example["metadata"]["turns_included"]                                                                                             
            turn_counts[turns] = turn_counts.get(turns, 0) + 1                                                                                        
                                                                                                                                                      
        print("\nExamples by conversation length (turns):")                                                                                           
        for turns, count in sorted(turn_counts.items()):                                                                                              
            print(f"  {turns} turns: {count}")      