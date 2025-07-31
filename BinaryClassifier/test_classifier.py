"""                                                                                                                                
Test script for the trained CBT binary classifier.                                                                                 
"""                                                                                                                                
                                                                                                                                   
import argparse                                                                                                                    
from binary_classifier import CBTBinaryClassifier                                                                                  
                                                                                                                                   
def main():                                                                                                                        
    parser = argparse.ArgumentParser(description='Test CBT Binary Classifier')                                                     
    parser.add_argument('--model_path', default='./cbt_classifier',                                                                
                       help='Path to the trained model')                                                                           
    parser.add_argument('--threshold', type=float, default=0.7,                                                                    
                       help='Confidence threshold for CBT trigger detection')                                                      
                                                                                                                                   
    args = parser.parse_args()                                                                                                     
                                                                                                                                   
    # Load the trained model                                                                                                       
    classifier = CBTBinaryClassifier()                                                                                             
    classifier.load_model(args.model_path)                                                                                         
                                                                                                                                   
    # Test examples                                                                                                                
    test_texts = [                                                                                                                 
        # Normal conversation examples                                                                                             
        "How was your weekend?",                                                                                                   
        "Nice weather today!",                                                                                                     
        "Did you see that movie last night?",                                                                                      
        "I had a great lunch at that new restaurant",                                                                              
        "What are your plans for tonight?",                                                                                        
                                                                                                                                   
        # CBT trigger examples                                                                                                     
        "I'm such a failure at everything",                                                                                        
        "I always mess things up",                                                                                                 
        "Everyone probably thinks I'm stupid",                                                                                     
        "I'm not good enough for this job",                                                                                        
        "I'll never be successful",                                                                                                
        "It's all my fault that this happened"                                                                                     
    ]                                                                                                                              
                                                                                                                                   
    print(f"Testing classifier with threshold: {args.threshold}")                                                                  
    print("=" * 60)                                                                                                                
                                                                                                                                   
    for text in test_texts:                                                                                                        
        result = classifier.predict(text, threshold=args.threshold)                                                                
                                                                                                                                   
        status = "🚨 CBT TRIGGER" if result['is_cbt_trigger'] else "✅ NORMAL"                                                     
        confidence = result['confidence']                                                                                          
                                                                                                                                   
        print(f"{status} (confidence: {confidence:.3f})")                                                                          
        print(f"Text: '{text}'")                                                                                                   
        print("-" * 60)                                                                                                            
                                                                                                                                   
    # Interactive testing                                                                                                          
    print("\nInteractive testing (type 'quit' to exit):")                                                                          
    while True:                                                                                                                    
        user_input = input("\nEnter text to classify: ").strip()                                                                   
                                                                                                                                   
        if user_input.lower() in ['quit', 'exit', 'q']:                                                                            
            break                                                                                                                  
                                                                                                                                   
        if not user_input:                                                                                                         
            continue                                                                                                               
                                                                                                                                   
        result = classifier.predict(user_input, threshold=args.threshold)                                                          
                                                                                                                                   
        status = "🚨 CBT TRIGGER" if result['is_cbt_trigger'] else "✅ NORMAL"                                                     
        confidence = result['confidence']                                                                                          
                                                                                                                                   
        print(f"{status} (confidence: {confidence:.3f})")                                                                          
                                                                                                                                   
if __name__ == "__main__":                                                                                                         
    main()                       