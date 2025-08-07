import os                                                                                                                           
import torch                                                                                                                        
import numpy as np                                                                                                                  
from transformers import (                                                                                                          
    AutoModelForSequenceClassification,                                                                                             
    AutoTokenizer,                                                                                                                  
    TrainingArguments,                                                                                                              
    Trainer,                                                                                                                        
    EarlyStoppingCallback                                                                                                           
)                                                                                                                                   
from datasets import load_from_disk                                                                                                 
import evaluate                                                                                                                     
from sklearn.metrics import mean_squared_error, r2_score                                                                            
import matplotlib.pyplot as plt                                                                                                     
import pandas as pd                                                                                                                 
from tqdm import tqdm     