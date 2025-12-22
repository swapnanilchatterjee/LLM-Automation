
# 🔒 Security Analysis Models - Now on Hugging Face!

I've published two AI models for security log analysis:

## 🚨 Security Event Detector
Detects security-relevant events in system logs
🔗 https://huggingface.co/Swapnanil09/security-event-detector

## 🔍 Vulnerability Extractor  
Extracts vulnerability indicators (software, version, errors, exploits)
🔗 https://huggingface.co/Swapnanil09/vulnerability-extractor

## 💡 Quick Start
```python
pip install transformers torch

from transformers import pipeline
classifier = pipeline("text-classification", model="Swapnanil09/security-event-detector")
result = classifier("Failed password for root from 192.168.1.100")
print(result)
```

Built with CodeBERT + LoRA | MIT License | Ready for production
