"""
Security Analysis Pipeline - Integration Guide
===============================================

Use this code to integrate the security analysis pipeline into your projects.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import json

class SecurityAnalysisPipeline:
    """Complete security analysis pipeline"""
    
    def __init__(self, clf_model_path, ner_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.tokenizer_clf = AutoTokenizer.from_pretrained(clf_model_path)
        self.model_clf = AutoModelForSequenceClassification.from_pretrained(clf_model_path)
        self.model_clf.to(self.device).eval()
        
        self.tokenizer_ner = AutoTokenizer.from_pretrained(ner_model_path)
        self.model_ner = AutoModelForTokenClassification.from_pretrained(ner_model_path)
        self.model_ner.to(self.device).eval()
        
        with open(f'{ner_model_path}/label_mappings.json', 'r') as f:
            self.id2label = json.load(f)['id2label']
    
    def analyze_log(self, log_text):
        """Analyze a single log entry"""
        # 1. Detect security event
        inputs = self.tokenizer_clf(log_text, return_tensors='pt', 
                                     truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model_clf(**inputs)
            prob = F.softmax(outputs.logits, dim=-1)
            is_security = torch.argmax(prob, dim=-1).item() == 1
        
        if not is_security:
            return {'is_security': False, 'log': log_text}
        
        # 2. Extract vulnerabilities
        inputs = self.tokenizer_ner(log_text, return_tensors='pt',
                                     truncation=True, padding=True,
                                     return_offsets_mapping=True).to(self.device)
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        with torch.no_grad():
            outputs = self.model_ner(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        # Extract entities
        tokens = self.tokenizer_ner.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        current_entity = None
        
        for idx, (token, pred) in enumerate(zip(tokens, predictions)):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            label = self.id2label[str(pred.item())]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token.replace('Ġ', ' ').strip(),
                    'type': label[2:]
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += token.replace('Ġ', ' ')
        
        if current_entity:
            entities.append(current_entity)
        
        return {
            'is_security': True,
            'log': log_text,
            'entities': entities
        }

# Usage Example
pipeline = SecurityAnalysisPipeline(
    clf_model_path='./security-detector-lora-final',
    ner_model_path='./vulnerability-extractor-lora-final'
)

# Analyze logs
logs = [
    "Failed password for root from 192.168.1.100",
    "systemd[1]: Started service"
]

for log in logs:
    result = pipeline.analyze_log(log)
    if result['is_security']:
        print(f"🚨 Security Event: {log}")
        print(f"   Entities: {result['entities']}")
    else:
        print(f"✅ Normal: {log}")
