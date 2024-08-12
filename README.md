# Document AI Chat Agent

Document AI Chat Agent is a conversational interface that leverages NVIDIA AI models to interact with and extract information from documents. This project is designed to handle document loading, text embedding, and answering questions based on the content of the documents.

## Features

- **Document Loading**: Load academic papers from Arxiv.org or PDF documents.
- **Text Embedding**: Convert text into numerical vectors using the NVIDIAEmbeddings model.
- **Document Search**: Use FAISS to search through document vectors for relevant information.
- **Conversational Interface**: Engage in a conversation with an AI agent that can answer questions based on the content of the loaded documents.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Document_AI_Chat_Agent.git
   cd Document_AI_Chat_Agent
