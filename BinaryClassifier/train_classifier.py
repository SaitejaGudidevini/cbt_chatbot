"""                                                                                                                                
Training script for CBT binary classifier.                                                                                         
Run this script to train the model on your CSV data.                                                                               
"""                                                                                                                                
                                                                                                                                   
import argparse                                                                                                                    
import logging                                                                                                                     
from binary_classifier import CBTBinaryClassifier                                                                                  
                                                                                                                                   
# Setup logging                                                                                                                    
logging.basicConfig(                                                                                                               
    level=logging.INFO,                                                                                                            
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'                                                                  
)                                                                                                                                  
                                                                                                                                   
def main():                                                                                                                        
    parser = argparse.ArgumentParser(description='Train CBT Binary Classifier')                                                    
    parser.add_argument('--normal_csv', required=True,                                                                             
                       help='Path to CSV file with normal conversations')                                                          
    parser.add_argument('--cbt_csv', required=True,                                                                                
                       help='Path to CSV file with CBT conversations')                                                             
    parser.add_argument('--text_column', default='text',                                                                           
                       help='Name of the text column in CSV files')                                                                
    parser.add_argument('--output_dir', default='./cbt_classifier',                                                                
                       help='Directory to save the trained model')                                                                 
    parser.add_argument('--model_name', default='distilbert-base-uncased',                                                         
                       help='Pre-trained model to use (distilbert-base-uncased recommended for laptops)')                          
                                                                                                                                   
    args = parser.parse_args()                                                                                                     
                                                                                                                                   
    # Initialize classifier                                                                                                        
    classifier = CBTBinaryClassifier(model_name=args.model_name)                                                                   
                                                                                                                                   
    # Prepare data                                                                                                                 
    print("Preparing data...")                                                                                                     
    df = classifier.prepare_data(                                                                                                  
        normal_csv_path=args.normal_csv,                                                                                           
        cbt_csv_path=args.cbt_csv,                                                                                                 
        text_column=args.text_column                                                                                               
    )                                                                                                                              
                                                                                                                                   
    # Tokenize data                                                                                                                
    print("Tokenizing data...")                                                                                                    
    dataset = classifier.tokenize_data(df)                                                                                         
                                                                                                                                   
    # Split data                                                                                                                   
    print("Splitting data...")                                                                                                     
    train_dataset, val_dataset, test_dataset = classifier.split_data(dataset)                                                      
                                                                                                                                   
    # Train model                                                                                                                  
    print("Training model...")                                                                                                     
    print("Note: Training optimized for laptop performance (smaller batches, fewer epochs)")                                       
    classifier.train_model(train_dataset, val_dataset, output_dir=args.output_dir)                                                 
                                                                                                                                   
    # Evaluate model                                                                                                               
    print("Evaluating model...")                                                                                                   
    classifier.evaluate_model(test_dataset)                                                                                        
                                                                                                                                   
    print(f"\nTraining complete! Model saved to {args.output_dir}")                                                                
    print("\nTo use the model for inference:")                                                                                     
    print(f"from binary_classifier import CBTBinaryClassifier")                                                                    
    print(f"classifier = CBTBinaryClassifier()")                                                                                   
    print(f"classifier.load_model('{args.output_dir}')")                                                                           
    print(f"result = classifier.predict('Your text here')")                                                                        
                                                                                                                                   
if __name__ == "__main__":                                                                                                         
    main()                  