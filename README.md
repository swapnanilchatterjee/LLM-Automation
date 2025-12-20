# Cloud Log Classifier using CodeBERT

This project implements a cloud platform log classifier using a fine-tuned CodeBERT model. It can classify logs from **AWS**, **Azure**, and **GCP** with high accuracy.

The model was fine-tuned on a dataset of simulated cloud logs using `microsoft/codebert-base-mlm` as the base model.

## 📂 Project Structure

```
.
├── Cloud_Classifier_using_codebert.ipynb  # Jupyter Notebook containing training and evaluation code
├── cloud-log-classifier-final/            # Saved model directory (generated after training)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── cloud-log-classifier-final.zip         # Zipped model for distribution
└── README.md                              # This file
```

## 🚀 Usage

You can use the fine-tuned model in your own projects using the `CloudLogClassifier` class.

### Prerequisites

```bash
pip install torch transformers scikit-learn numpy
```

### Python Inference Code

Save the following code as `classifier.py` or use it directly in your python scripts. Ensure you have the `cloud-log-classifier-final` folder (unzipped) in your working directory.

```python
import torch
import json
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class CloudLogClassifier:
    """
    Reusable classifier for cloud platform detection from logs.
    """

    def __init__(self, model_path):
        """
        Load the fine-tuned model and tokenizer
        
        Args:
            model_path (str): Path to the directory containing the saved model files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            with open(f'{model_path}/label_mapping.json', 'r') as f:
                mappings = json.load(f)
                # Convert keys back to integers for the dictionary
                self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def predict(self, log_text):
        """
        Predict cloud platform from log text

        Args:
            log_text (str): Log text to classify

        Returns:
            dict: Prediction results with label and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            log_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'platform': self.id2label[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                self.id2label[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }

# Usage Example
if __name__ == "__main__":
    # Path to your unzipped model folder
    model_path = './cloud-log-classifier-final'
    
    try:
        classifier = CloudLogClassifier(model_path)
        
        test_logs = [
            "[    3.6936] ena 0000:00:05.0: Elastic Network Adapter (ENA)",
            "AzureLinuxAgent: INFO Starting Azure Linux Agent",
            "google_guest_agent INFO GCE Agent running"
        ]

        print("Predictions:")
        for log in test_logs:
            result = classifier.predict(log)
            print(f"\nLog: {log}")
            print(f"Predicted Platform: {result['platform'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            
    except Exception as e:
        print(f"Error: {e}")
```

## 📊 Model Performance

The model achieves high accuracy on the test set, effectively distinguishing between different cloud provider log formats (AWS, Azure, GCP).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~98% |
| **Precision** | >95% |
| **Recall** | >95% |
| **F1-Score** | >95% |

## 🛠️ Training

The model was trained using the `Trainer` API from HuggingFace Transformers.

- **Base Model**: microsoft/codebert-base-mlm
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: Default (5e-5)
