# My_Ultra

A comprehensive document processing system that uses Large Language Models (LLMs) to extract structured information from various document types, with support for multiple processing strategies and providers.

## 🚀 Features

- **Multi-Strategy Processing**: Direct file, text-first, image-first, and hybrid processing strategies
- **Multi-Provider Support**: Google Gemini, OpenAI GPT, Claude, DeepSeek, HuggingFace, and Ollama
- **Parallel Processing**: Efficient batch processing with configurable worker pools
- **Real-time Monitoring**: Live dashboard for processing status and results
- **Benchmark Validation**: Compare extracted data against reference datasets
- **Cost Tracking**: Monitor token usage and processing costs
- **Flexible Configuration**: Easy-to-configure processing parameters

## 📁 Project Structure

```
My_Ultra/
├── My_Ultra_Parallel/          # Main processing engine
│   ├── config/                 # Configuration files
│   ├── llm_client/            # LLM provider clients
│   ├── llm_strategies/        # Processing strategies
│   ├── processors/            # Core processing logic
│   ├── common/                # Shared utilities
│   └── benchmark/             # Benchmark validation tools
├── My_Ultra_Monitor/          # Real-time monitoring dashboard
│   ├── backend/               # Flask server
│   ├── frontend/              # Web interface
│   └── config/                # Monitoring configuration
└── My_Ultra_Example_Call/     # Example usage scripts
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/My_Ultra.git
   cd My_Ultra
   ```

2. **Install dependencies**:
   ```bash
   # For the main processing engine
   cd My_Ultra_Parallel
   pip install -r requirements.txt
   
   # For the monitoring dashboard
   cd ../My_Ultra_Monitor
   npm install
   ```

3. **Configure API keys**:
   Edit the configuration files in `My_Ultra_Parallel/config/` to add your API keys for the LLM providers you want to use.

## 🚀 Quick Start

### Basic Usage

```python
from My_Ultra_Parallel.main_modular import run_file_processing_simple
from pathlib import Path

# Process a single PDF file
results = run_file_processing_simple(
    input_pdf_dir_path=Path("path/to/pdf/directory"),
    pdf_file_paths=[Path("path/to/document.pdf")]
)

print(results)
```

### Example Script

See `My_Ultra_Example_Call/test_call.py` for a complete example of processing a German invoice.

## 📊 Processing Strategies

### 1. Direct File Strategy
- Sends PDF files directly to LLM providers
- Best for simple documents with clear structure
- Fastest processing time

### 2. Text-First Strategy
- Extracts text from PDFs first, then processes with LLM
- Good for text-heavy documents
- More cost-effective for large documents

### 3. Image-First Strategy
- Converts PDFs to images, then processes with vision-capable LLMs
- Best for complex layouts and visual elements
- Higher accuracy for structured documents

### 4. Hybrid Strategy
- Combines multiple strategies for optimal results
- Automatic fallback and retry mechanisms
- Highest success rate

## 🔧 Configuration

### LLM Providers

The system supports multiple LLM providers:

- **Google Gemini**: Fast and cost-effective
- **OpenAI GPT**: High accuracy and reliability
- **Claude**: Excellent for complex reasoning
- **DeepSeek**: Good balance of speed and accuracy
- **HuggingFace**: Open-source models
- **Ollama**: Local processing capabilities

### Processing Parameters

Configure processing parameters in `config/config_base.py`:

- `MANDATORY_KEYS`: Required fields to extract
- `MAX_WORKERS`: Number of parallel workers
- `MAX_RETRIES`: Retry attempts for failed processing
- `TOKEN_LIMITS`: Maximum tokens per request

## 📈 Monitoring Dashboard

Start the monitoring dashboard:

```bash
cd My_Ultra_Monitor
python backend/server.py
```

Access the dashboard at `http://localhost:8000`

## 🔍 Benchmark Validation

Use benchmark validation to compare extracted data against reference datasets:

```python
from My_Ultra_Parallel.benchmark.benchmark_adapter import BenchmarkAdapter

adapter = BenchmarkAdapter("path/to/benchmark.xlsx")
validation_results = adapter.validate_extracted_data(extracted_data)
```

## 📝 Supported Document Types

- **Invoices**: Extract vendor, customer, amounts, line items
- **Receipts**: Extract merchant, date, total, items
- **Contracts**: Extract parties, terms, dates, amounts
- **Forms**: Extract form fields and values
- **Custom Documents**: Configurable extraction templates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python async/await patterns
- Leverages state-of-the-art LLM APIs
- Inspired by document processing best practices
- Community-driven development approach

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review example scripts in `My_Ultra_Example_Call/` 