import pandas as pd                                                                                                                
import numpy as np                                                                                                                 
from sklearn.model_selection import train_test_split                                                                               
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix                                                
from transformers import (                                                                                                         
    AutoTokenizer, AutoModelForSequenceClassification,                                                                             
    TrainingArguments, Trainer, DataCollatorWithPadding                                                                            
)                                                                                                                                  
import torch                                                                                                                       
from datasets import Dataset                                                                                                       
import logging                                                                                                                     
import os                                                                                                                          
                                                                                                                                   
logger = logging.getLogger(__name__)                                                                                               
                                                                                                                                   
class CBTBinaryClassifier:                                                                                                         
    """Binary classifier to distinguish normal conversation from CBT-triggering statements."""                                     
                                                                                                                                   
    def __init__(self, model_name="distilbert-base-uncased"):                                                                      
        # Use a lightweight model that's good for your laptop                                                                      
        self.model_name = model_name                                                                                               
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)                                                                 
        self.model = None                                                                                                          
        self.trainer = None                                                                                                        
        self.inference_pipeline = None                                                                                             
                                                                                                                                   
        # Add padding token if it doesn't exist                                                                                    
        if self.tokenizer.pad_token is None:                                                                                       
            self.tokenizer.pad_token = self.tokenizer.eos_token                                                                    
                                                                                                                                   
    def prepare_data(self, normal_csv_path, cbt_csv_path, text_column="text"):                                                     
        """Load and prepare training data from CSV files"""                                                                        
                                                                                                                                   
        logger.info(f"Loading normal conversations from {normal_csv_path}")                                                        
        normal_df = pd.read_csv(normal_csv_path)                                                                                   
        normal_df['label'] = 0  # Normal conversation = 0                                                                          
        normal_df['text'] = normal_df[text_column]                                                                                 
                                                                                                                                   
        logger.info(f"Loading CBT conversations from {cbt_csv_path}")                                                              
        cbt_df = pd.read_csv(cbt_csv_path)                                                                                         
        cbt_df['label'] = 1  # CBT trigger = 1                                                                                     
        cbt_df['text'] = cbt_df[text_column]                                                                                       
                                                                                                                                   
        # Combine datasets                                                                                                         
        combined_df = pd.concat([                                                                                                  
            normal_df[['text', 'label']],                                                                                          
            cbt_df[['text', 'label']]                                                                                              
        ], ignore_index=True)                                                                                                      
                                                                                                                                   
        # Shuffle the data                                                                                                         
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)                                           
                                                                                                                                   
        logger.info(f"Total examples: {len(combined_df)}")                                                                         
        logger.info(f"Normal conversations: {len(normal_df)}")                                                                     
        logger.info(f"CBT triggers: {len(cbt_df)}")                                                                                
                                                                                                                                   
        return combined_df                                                                                                         
                                                                                                                                   
    def tokenize_data(self, df, max_length=128):                                                                                   
        """Tokenize the text data"""                                                                                               
                                                                                                                                   
        def tokenize_function(examples):                                                                                           
            return self.tokenizer(                                                                                                 
                examples['text'],                                                                                                  
                truncation=True,                                                                                                   
                padding='max_length',                                                                                              
                max_length=max_length,                                                                                             
                return_tensors=None                                                                                                
            )                                                                                                                      
                                                                                                                                   
        # Convert to HuggingFace Dataset                                                                                           
        dataset = Dataset.from_pandas(df)                                                                                          
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text'])                                                           
                                                                                                                                   
        return tokenized_dataset                                                                                                   
                                                                                                                                   
    def split_data(self, dataset, test_size=0.2, val_size=0.1):                                                                    
        """Split data into train/validation/test sets"""                                                                           
                                                                                                                                   
        # First split: train + val vs test                                                                                         
        train_val, test = dataset.train_test_split(                                                                                
            test_size=test_size,                                                                                                   
            seed=42                                                                                                                
        ).values()                                                                                                                 
                                                                                                                                   
        # Second split: train vs validation                                                                                        
        val_ratio = val_size / (1 - test_size)                                                                                     
        train, val = train_val.train_test_split(                                                                                   
            test_size=val_ratio,                                                                                                   
            seed=42                                                                                                                
        ).values()                                                                                                                 
                                                                                                                                   
        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")                                                    
        return train, val, test                                                                                                    
                                                                                                                                   
    def train_model(self, train_dataset, val_dataset, output_dir="./cbt_classifier"):                                              
        """Train the binary classifier with laptop-friendly settings"""                                                            
                                                                                                                                   
        # Create output directory                                                                                                  
        os.makedirs(output_dir, exist_ok=True)                                                                                     
                                                                                                                                   
        # Initialize model                                                                                                         
        self.model = AutoModelForSequenceClassification.from_pretrained(                                                           
            self.model_name,                                                                                                       
            num_labels=2                                                                                                           
        )                                                                                                                          
                                                                                                                                   
        # Create data collator for dynamic padding                                                                                 
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)                                                          
                                                                                                                                   
        # Laptop-friendly training arguments                                                                                       
        training_args = TrainingArguments(                                                                                         
            output_dir=output_dir,                                                                                                 
            num_train_epochs=2,  # Reduced epochs                                                                                  
            per_device_train_batch_size=8,  # Smaller batch size                                                                   
            per_device_eval_batch_size=8,                                                                                          
            gradient_accumulation_steps=2,  # Simulate larger batch size                                                           
            warmup_steps=100,  # Reduced warmup                                                                                    
            weight_decay=0.01,                                                                                                     
            logging_dir=f'{output_dir}/logs',                                                                                      
            logging_steps=50,                                                                                                      
            eval_strategy="steps",                                                                                                 
            eval_steps=200,                                                                                                        
            save_strategy="steps",                                                                                                 
            save_steps=200,                                                                                                        
            load_best_model_at_end=True,                                                                                           
            metric_for_best_model="eval_accuracy",                                                                                 
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available                                                
            dataloader_num_workers=0,  # Reduce CPU usage                                                                          
            remove_unused_columns=True,                                                                                           
        )                                                                                                                          
                                                                                                                                   
        # Metrics function                                                                                                         
        def compute_metrics(eval_pred):                                                                                            
            predictions, labels = eval_pred                                                                                        
            predictions = np.argmax(predictions, axis=1)                                                                           
            return {                                                                                                               
                'accuracy': accuracy_score(labels, predictions),                                                                   
            }                                                                                                                      
                                                                                                                                   
        # Initialize trainer                                                                                                       
        self.trainer = Trainer(                                                                                                    
            model=self.model,                                                                                                      
            args=training_args,                                                                                                    
            train_dataset=train_dataset,                                                                                           
            eval_dataset=val_dataset,                                                                                              
            compute_metrics=compute_metrics,                                                                                       
            data_collator=data_collator,                                                                                           
        )                                                                                                                          
                                                                                                                                   
        # Train the model                                                                                                          
        logger.info("Starting training...")                                                                                        
        self.trainer.train()                                                                                                       
                                                                                                                                   
        # Save the model                                                                                                           
        self.trainer.save_model()                                                                                                  
        self.tokenizer.save_pretrained(output_dir)                                                                                 
                                                                                                                                   
        logger.info(f"Model saved to {output_dir}")                                                                                
                                                                                                                                   
    def evaluate_model(self, test_dataset):                                                                                        
        """Evaluate the trained model"""                                                                                           
                                                                                                                                   
        if self.trainer is None:                                                                                                   
            raise ValueError("Model not trained yet!")                                                                             
                                                                                                                                   
        # Get predictions                                                                                                          
        predictions = self.trainer.predict(test_dataset)                                                                           
        y_pred = np.argmax(predictions.predictions, axis=1)                                                                        
        y_true = predictions.label_ids                                                                                             
                                                                                                                                   
        # Print results                                                                                                            
        print("\n=== Evaluation Results ===")                                                                                      
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")                                                                   
        print("\nClassification Report:")                                                                                          
        print(classification_report(y_true, y_pred,                                                                                
                                  target_names=['Normal', 'CBT Trigger']))                                                         
        print("\nConfusion Matrix:")                                                                                               
        print(confusion_matrix(y_true, y_pred))                                                                                    
                                                                                                                                   
        return y_true, y_pred                                                                                                      
                                                                                                                                   
    def load_model(self, model_path="./cbt_classifier"):                                                                           
        """Load a pre-trained model for inference"""                                                                               
                                                                                                                                   
        from transformers import pipeline                                                                                          
                                                                                                                                   
        self.inference_pipeline = pipeline(                                                                                        
            "text-classification",                                                                                                 
            model=model_path,                                                                                                      
            tokenizer=model_path,                                                                                                  
            return_all_scores=True                                                                                                 
        )                                                                                                                          
                                                                                                                                   
        logger.info(f"Model loaded from {model_path}")                                                                             
                                                                                                                                   
    def predict(self, text, threshold=0.7):                                                                                        
        """Predict if text is CBT-triggering"""                                                                                    
                                                                                                                                   
        if self.inference_pipeline is None:                                                                                        
            raise ValueError("Model not loaded! Call load_model() first.")                                                         
                                                                                                                                   
        result = self.inference_pipeline(text)                                                                                     
                                                                                                                                   
        # Extract confidence for CBT trigger class (LABEL_1)                                                                       
        cbt_confidence = next(                                                                                                     
            score['score'] for score in result[0]                                                                                  
            if score['label'] == 'LABEL_1'                                                                                         
        )                                                                                                                          
                                                                                                                                   
        return {                                                                                                                   
            'is_cbt_trigger': cbt_confidence > threshold,                                                                          
            'confidence': cbt_confidence,                                                                                          
            'threshold': threshold                                                                                                 
        }                                                                                                                          
                                                                                                                                   
    def batch_predict(self, texts, threshold=0.7):                                                                                 
        """Predict for multiple texts"""                                                                                           
                                                                                                                                   
        if self.inference_pipeline is None:                                                                                        
            raise ValueError("Model not loaded! Call load_model() first.")                                                         
                                                                                                                                   
        results = []                                                                                                               
        for text in texts:                                                                                                         
            result = self.predict(text, threshold)                                                                                 
            results.append(result)                                                                                                 
                                                                                                                                   
        return results        