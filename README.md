# Knowledge Extraction & Visualization Pipeline

An automated pipeline that scrapes web content, extracts named entities and relationships using **DSPy** and **Groq (Llama 3.3)**, and generates visual Knowledge Graphs and structured datasets.

## 🚀 Key Features

- **Robust Web Scraping**: Fetches text from URLs with automatic fallback content for 403/blocked pages.
- **Smart Entity Extraction**: Uses DSPy and LLMs to identify entities and categorize them (e.g., *Concept, Process, Technology*).
- **Intelligent Deduplication**: Merges similar entity names using a confidence-based semantic check.
- **Knowledge Graph Generation**: Automatically builds **Mermaid.js** diagrams to visualize relationships between entities.
- **Structured Export**: Saves all extracted data to `tags.csv` for easy analysis.

## 🛠️ Tech Stack

- **Core AI**: [DSPy](https://github.com/stanfordnlp/dspy) (Declarative Self-Improving Language Programs)
- **Inference Engine**: [Groq API](https://groq.com/) (running Llama-3.3-70b-versatile)
- **Data Processing**: Pandas, BeautifulSoup4
- **Visualization**: Mermaid.js

## 📋 Prerequisites

- Python 3.8+
- A valid **Groq API Key** (Get one for free at [console.groq.com](https://console.groq.com))

## 📦 Installation

1. Clone this repository (if applicable) or download the project files.
2. Install the required Python dependencies:

```bash
pip install dspy-ai groq requests beautifulsoup4 pandas lxml html5lib
```

## ⚙️ Usage

1. Open the Jupyter Notebook `Test_Assignment.ipynb`.
2. Locate the configuration cell and replace the placeholder with your actual Groq API key:

```python
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

3. Run all cells in the notebook.
   - The pipeline will process the list of URLs defined in the `URLS` variable.
   - Progress will be printed for each URL (extraction, deduplication, graph generation).

## 📂 Output

After running the pipeline, the following files will be generated in your directory:

- **`tags.csv`**: A structured dataset containing all extracted entities, their types, and source URLs.
- **`mermaid_*.md`**: Individual Markdown files containing the Mermaid diagram syntax for each processed URL.

## 📊 Example Flow

1. **Input**: `https://en.wikipedia.org/wiki/Sustainable_agriculture`
2. **Scraping**: Extracts text content.
3. **Extraction**: Identifies terms like "Crop Rotation", "Soil Health".
4. **Graph**: Links "Crop Rotation" --(improves)--> "Soil Health".
5. **Output**: CSV row and Mermaid diagram.
