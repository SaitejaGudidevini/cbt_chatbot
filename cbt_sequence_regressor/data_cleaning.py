import json                                                                                                                 
import argparse                                                                                                             
import os                                                                                                                   
                                                                                                                            
def remove_id_from_json(input_file: str, output_file: str = None):                                                          
    """Remove 'id' field from all examples in the JSON training data."""                                                    
                                                                                                                            
    # Set output file name if not provided                                                                                  
    if output_file is None:                                                                                                 
        name, ext = os.path.splitext(input_file)                                                                            
        output_file = f"{name}_cleaned{ext}"                                                                                
                                                                                                                            
    print(f"Loading data from: {input_file}")                                                                               
                                                                                                                            
    try:                                                                                                                    
        # Load the JSON data                                                                                                
        with open(input_file, 'r', encoding='utf-8') as f:                                                                  
            data = json.load(f)                                                                                             
                                                                                                                            
        print(f"Found {len(data.get('examples', []))} examples")                                                            
                                                                                                                            
        # Remove 'id' field from each example                                                                               
        cleaned_examples = []                                                                                               
        for i, example in enumerate(data.get('examples', [])):                                                              
            cleaned_example = {k: v for k, v in example.items() if k != 'id'}                                               
            cleaned_examples.append(cleaned_example)                                                                        
                                                                                                                            
            # Show progress                                                                                                 
            if (i + 1) % 50 == 0:                                                                                           
                print(f"Processed {i + 1} examples...")                                                                     
                                                                                                                            
        # Update the data                                                                                                   
        data['examples'] = cleaned_examples                                                                                 
                                                                                                                            
        # Save the cleaned data                                                                                             
        with open(output_file, 'w', encoding='utf-8') as f:                                                                 
            json.dump(data, f, indent=2, ensure_ascii=False)                                                                
                                                                                                                            
        print(f"✅ Cleaned data saved to: {output_file}")                                                                   
        print(f"✅ Removed 'id' field from {len(cleaned_examples)} examples")                                               
                                                                                                                            
        # Show sample of cleaned data                                                                                       
        if cleaned_examples:                                                                                                
            print(f"\nSample cleaned example keys: {list(cleaned_examples[0].keys())}")                                     
                                                                                                                            
        return output_file                                                                                                  
                                                                                                                            
    except FileNotFoundError:                                                                                               
        print(f"❌ Error: File '{input_file}' not found")                                                                   
        return None                                                                                                         
    except json.JSONDecodeError as e:                                                                                       
        print(f"❌ Error: Invalid JSON in file '{input_file}': {e}")                                                        
        return None                                                                                                         
    except Exception as e:                                                                                                  
        print(f"❌ Error: {e}")                                                                                             
        return None                                                                                                         
                                                                                                                            
def main():                                                                                                                 
    parser = argparse.ArgumentParser(description='Remove ID field from CBT training data JSON')                             
    parser.add_argument('input_file', help='Input JSON file path')                                                          
    parser.add_argument('--output', '-o', help='Output JSON file path (optional)')                                          
                                                                                                                            
    args = parser.parse_args()                                                                                              
                                                                                                                            
    # Clean the data                                                                                                        
    output_file = remove_id_from_json(args.input_file, args.output)                                                         
                                                                                                                            
    if output_file:                                                                                                         
        print(f"\n🎉 Success! You can now use the cleaned file for training:")                                              
        print(f"python train_cbt_sequence_regressor.py --data {output_file} --output ./cbt_sequence_model")                 
                                                                                                                            
if __name__ == "__main__":                                                                                                  
    main()