
# ============================================================================
# Using Security Analysis Models from Hugging Face
# ============================================================================

# STEP 1: Install required packages
# ----------------------------------
pip install transformers torch

# STEP 2: Load Security Event Detector
# -------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from Hugging Face
tokenizer_clf = AutoTokenizer.from_pretrained("Swapnanil09/security-event-detector")
model_clf = AutoModelForSequenceClassification.from_pretrained("Swapnanil09/security-event-detector")

# Analyze a log
log = "Failed password for root from 192.168.1.100 port 22 ssh2"
inputs = tokenizer_clf(log, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model_clf(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    is_security = prediction.item() == 1
    confidence = probabilities[0][prediction.item()].item()

print(f"Security Event: {is_security}")
print(f"Confidence: {confidence:.2%}")

# STEP 3: Load Vulnerability Extractor
# -------------------------------------
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load from Hugging Face
tokenizer_ner = AutoTokenizer.from_pretrained("Swapnanil09/vulnerability-extractor")
model_ner = AutoModelForTokenClassification.from_pretrained("Swapnanil09/vulnerability-extractor")

# Extract vulnerabilities
log = "Apache 2.4.49 path traversal attack attempt detected"
inputs = tokenizer_ner(log, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model_ner(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Decode entities
tokens = tokenizer_ner.convert_ids_to_tokens(inputs['input_ids'][0])
entities = []
current_entity = None

for idx, (token, pred) in enumerate(zip(tokens, predictions[0])):
    if token in ['<s>', '</s>', '<pad>']:
        continue
    
    label = model_ner.config.id2label[pred.item()]
    
    if label.startswith('B-'):
        if current_entity:
            entities.append(current_entity)
        current_entity = {'text': token.replace('Ġ', ' ').strip(), 'type': label[2:]}
    elif label.startswith('I-') and current_entity:
        current_entity['text'] += token.replace('Ġ', ' ')

if current_entity:
    entities.append(current_entity)

print(f"Extracted entities: {entities}")

# STEP 4: Complete Pipeline (Both Models Together)
# -------------------------------------------------
class SecurityAnalysisPipeline:
    def __init__(self):
        # Load both models from Hugging Face
        self.tokenizer_clf = AutoTokenizer.from_pretrained("Swapnanil09/security-event-detector")
        self.model_clf = AutoModelForSequenceClassification.from_pretrained("Swapnanil09/security-event-detector")
        
        self.tokenizer_ner = AutoTokenizer.from_pretrained("Swapnanil09/vulnerability-extractor")
        self.model_ner = AutoModelForTokenClassification.from_pretrained("Swapnanil09/vulnerability-extractor")
        
        self.model_clf.eval()
        self.model_ner.eval()
    
    def analyze(self, log_text):
        # Step 1: Detect security event
        inputs = self.tokenizer_clf(log_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model_clf(**inputs)
            prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
            is_security = torch.argmax(prob, dim=-1).item() == 1
        
        if not is_security:
            return {'is_security': False, 'log': log_text}
        
        # Step 2: Extract vulnerabilities
        inputs = self.tokenizer_ner(log_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model_ner(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        tokens = self.tokenizer_ner.convert_ids_to_tokens(inputs['input_ids'][0])
        entities = []
        current_entity = None
        
        for token, pred in zip(tokens, predictions[0]):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            label = self.model_ner.config.id2label[pred.item()]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'text': token.replace('Ġ', ' ').strip(), 'type': label[2:]}
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += token.replace('Ġ', ' ')
        
        if current_entity:
            entities.append(current_entity)
        
        return {
            'is_security': True,
            'log': log_text,
            'entities': entities
        }

# Use the pipeline
pipeline = SecurityAnalysisPipeline()

# Analyze logs
test_logs = [
    "Failed password for root from 192.168.1.100 port 22 ssh2",
    "Apache 2.4.49 path traversal attack attempt detected",
    "systemd[1]: Started service successfully"
]

for log in test_logs:
    result = pipeline.analyze(log)
    if result['is_security']:
        print(f"🚨 SECURITY EVENT: {log}")
        print(f"   Entities: {result['entities']}")
    else:
        print(f"✅ Normal: {log}")

# ============================================================================
# Model URLs
# ============================================================================

Security Event Detector: https://huggingface.co/Swapnanil09/security-event-detector
Vulnerability Extractor: https://huggingface.co/Swapnanil09/vulnerability-extractor

