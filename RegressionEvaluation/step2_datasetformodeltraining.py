import json                                                                                                                           
import torch                                                                                                                          
import numpy as np                                                                                                                    
from sklearn.model_selection import train_test_split                                                                                  
from transformers import AutoTokenizer                                                                                                
from datasets import Dataset, DatasetDict                                                                                             
import pandas as pd                                                                                                                   
from tqdm import tqdm                                                                                                                 
import os                                                                                                                             
                                                                                                                                      
def prepare_cbt_datasets(input_file, model_name="roberta-base", test_size=0.15, val_size=0.15, max_length=512, seed=42):              
    """                                                                                                                               
    Prepare datasets for training a CBT evaluator model.                                                                              
                                                                                                                                      
    Args:                                                                                                                             
        input_file: Path to the JSON file containing the training examples                                                            
        model_name: Name of the pre-trained model to use                                                                              
        test_size: Fraction of data to use for testing                                                                                
        val_size: Fraction of data to use for validation                                                                              
        max_length: Maximum sequence length for tokenization                                                                          
        seed: Random seed for reproducibility                                                                                         
                                                                                                                                      
    Returns:                                                                                                                          
        A DatasetDict containing train, validation, and test datasets                                                                 
    """                                                                                                                               
    print(f"Loading data from {input_file}...")                                                                                       
                                                                                                                                      
    # Load the data                                                                                                                   
    with open(input_file, 'r', encoding='utf-8') as f:                                                                                
        examples = json.load(f)                                                                                                       
                                                                                                                                      
    print(f"Loaded {len(examples)} examples")                                                                                         
                                                                                                                                      
    # Flatten the examples for easier processing                                                                                      
    flattened_examples = []                                                                                                           
    for example in examples:                                                                                                          
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
                                                                                                                                      
    # Convert to DataFrame for easier manipulation                                                                                    
    df = pd.DataFrame(flattened_examples)                                                                                             
                                                                                                                                      
    # Print dataset statistics                                                                                                        
    print("\nDataset Statistics:")                                                                                                    
    print(f"Number of examples: {len(df)}")                                                                                           
    print(f"Number of unique conversations: {df['id'].str.split('_turn_').str[0].nunique()}")                                         
    print(f"Average conversation length: {df['conversation_text'].str.len().mean():.1f} characters")                                  
                                                                                                                                      
    print("\nTarget Score Distributions:")                                                                                            
    for col in ["thought_identification", "emotion_exploration", "distortion_identification",                                         
                "evidence_gathering", "balanced_thinking"]:                                                                           
        print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")       
                                                                                                                                      
    # Split by conversation ID to prevent data leakage                                                                                
    # Extract the base conversation ID (without the turn number)                                                                      
    df['conversation_base_id'] = df['id'].str.split('_turn_').str[0]                                                                  
                                                                                                                                      
    # Get unique conversation IDs                                                                                                     
    unique_conv_ids = df['conversation_base_id'].unique()                                                                             
                                                                                                                                      
    # Split conversation IDs into train, validation, and test sets                                                                    
    train_ids, temp_ids = train_test_split(                                                                                           
        unique_conv_ids,                                                                                                              
        test_size=test_size + val_size,                                                                                               
        random_state=seed                                                                                                             
    )                                                                                                                                 
                                                                                                                                      
    val_ids, test_ids = train_test_split(                                                                                             
        temp_ids,                                                                                                                     
        test_size=test_size / (test_size + val_size),                                                                                 
        random_state=seed                                                                                                             
    )                                                                                                                                 
                                                                                                                                      
    # Create train, validation, and test dataframes                                                                                   
    train_df = df[df['conversation_base_id'].isin(train_ids)]                                                                         
    val_df = df[df['conversation_base_id'].isin(val_ids)]                                                                             
    test_df = df[df['conversation_base_id'].isin(test_ids)]                                                                           
                                                                                                                                      
    print(f"\nSplit sizes:")                                                                                                          
    print(f"  Train: {len(train_df)} examples from {len(train_ids)} conversations")                                                   
    print(f"  Validation: {len(val_df)} examples from {len(val_ids)} conversations")                                                  
    print(f"  Test: {len(test_df)} examples from {len(test_ids)} conversations")                                                      
                                                                                                                                      
    # Convert to Hugging Face datasets                                                                                                
    train_dataset = Dataset.from_pandas(train_df)                                                                                     
    val_dataset = Dataset.from_pandas(val_df)                                                                                         
    test_dataset = Dataset.from_pandas(test_df)                                                                                       
                                                                                                                                      
    # Create a DatasetDict                                                                                                            
    dataset_dict = DatasetDict({                                                                                                      
        'train': train_dataset,                                                                                                       
        'validation': val_dataset,                                                                                                    
        'test': test_dataset                                                                                                          
    })                                                                                                                                
                                                                                                                                      
    # Load tokenizer                                                                                                                  
    print(f"\nLoading tokenizer for {model_name}...")                                                                                 
    tokenizer = AutoTokenizer.from_pretrained(model_name)                                                                             
                                                                                                                                      
    # Define tokenization function                                                                                                    
    def tokenize_function(examples):                                                                                                  
        return tokenizer(                                                                                                             
            examples["conversation_text"],                                                                                            
            padding="max_length",                                                                                                     
            truncation=True,                                                                                                          
            max_length=max_length                                                                                                     
        )                                                                                                                             
                                                                                                                                      
    # Tokenize the datasets                                                                                                           
    print("Tokenizing datasets...")                                                                                                   
    tokenized_datasets = dataset_dict.map(                                                                                            
        tokenize_function,                                                                                                            
        batched=True,                                                                                                                 
        desc="Tokenizing",                                                                                                            
        remove_columns=['conversation_base_id']  # Remove the helper column we added                                                  
    )                                                                                                                                 
                                                                                                                                      
    # Save the processed datasets                                                                                                     
    output_dir = "cbt_evaluator_data"                                                                                                 
    os.makedirs(output_dir, exist_ok=True)                                                                                            
                                                                                                                                      
    print(f"\nSaving processed datasets to {output_dir}...")                                                                          
    tokenized_datasets.save_to_disk(output_dir)                                                                                       
                                                                                                                                      
    # Also save the tokenizer for consistency                                                                                         
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))                                                                  
                                                                                                                                      
    print("Dataset preparation complete!")                                                                                            
    return tokenized_datasets                                                                                                         
                                                                                                                                      
# Example usage                                                                                                                       
if __name__ == "__main__":                                                                                                            
    # Set your input file (output from Step 1)                                                                                        
    input_file = "cbt_evaluator_training_data.json"                                                                                   
                                                                                                                                      
    datasets = prepare_cbt_datasets(                                                                                                               
        input_file=input_file,                                                                                                                     
        model_name="roberta-base",  # You can change this to another model if preferred                                                            
        test_size=0.15,                                                                                                                            
        val_size=0.15,                                                                                                                             
        max_length=512,                                                                                                                            
        seed=42                                                                                                                                    
    )                                                                                                                                              
                                                                                                                                                   
    # Print dataset sizes                                                                                                                          
    print("\nFinal dataset sizes:")                                                                                                                
    for split, dataset in datasets.items():                                                                                                        
        print(f"  {split}: {len(dataset)} examples")                                                                                               
                                                                                                                                                   
    # Print example from training set                                                                                                              
    print("\nExample from training set:")                                                                                                          
    example = datasets["train"][0]                                                                                                                 
    print(f"ID: {example['id']}")                                                                                                                  
    print(f"Input IDs shape: {len(example['input_ids'])}")                                                                                         
    print(f"Target scores:")                                                                                                                       
    print(f"  Thought identification: {example['thought_identification']}")                                                                        
    print(f"  Emotion exploration: {example['emotion_exploration']}")                                                                              
    print(f"  Distortion identification: {example['distortion_identification']}")                                                                  
    print(f"  Evidence gathering: {example['evidence_gathering']}")                                                                                
    print(f"  Balanced thinking: {example['balanced_thinking']}")                                                                                  
                                                                         