import sys                                                                                                                  
sys.path.append('./cbt_sequence_model')                                                                                     
                                                                                                                            
from cbt_sequence_model.sequence_regressor import CBTSequenceComplianceRegressor                                                               
                                                                                                                            
# Test the trained model                                                                                                    
model = CBTSequenceComplianceRegressor('./cbt_sequence_model')                                                              
                                                                                                                            
# Test prediction                                                                                                           
result = model.predict(                                                                                                     
    model_question="How is your bike?",                                             
    user_response="I feel terrible and sad",                                                                                
    conversation_context="User is discussing work stress",                                                                  
    trigger_statement="I'm a failure",                                                                                      
    cbt_step="1.0"                                                                                                          
)                                                                                                                           
                                                                                                                            
print("Prediction result:", result)  