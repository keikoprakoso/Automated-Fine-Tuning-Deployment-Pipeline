# 🤖 Automated Fine-Tuning & Deployment Pipeline for LLMs

A complete, production-ready Python backend for automated LLM fine-tuning and deployment. This project demonstrates end-to-end MLOps capabilities for Large Language Models, from dataset preparation to model deployment and monitoring.

**Author:** Keiko Rafi Ananda Prakoso  
**Institution:** Universiti Malaya (Computer Science - AI)  
**Date:** 2024

## 🎯 Project Overview

This pipeline provides a modular, scalable solution for:
- **Data Preprocessing**: Clean and validate CSV datasets for fine-tuning
- **Fine-tuning Orchestration**: Automated OpenAI fine-tuning job management
- **Model Inference**: Production-ready inference with cost tracking
- **API Service**: FastAPI backend with comprehensive endpoints
- **Monitoring Dashboard**: Streamlit-based usage analytics and job tracking

## 🏗️ Architecture

```
fine_tune_pipeline/
│
├── 📁 Core Modules
│   ├── preprocess.py      # Data cleaning and validation
│   ├── fine_tune.py       # Fine-tuning orchestration
│   ├── inference.py       # Model inference and logging
│   └── app.py            # FastAPI server
│
├── 📁 Dashboard
│   └── dashboard.py      # Streamlit monitoring dashboard
│
├── 📁 Data & Logs
│   ├── data/             # Uploaded datasets
│   └── logs/             # Usage logs and analytics
│
├── 📁 Configuration
│   ├── requirements.txt  # Python dependencies
│   ├── env.example       # Environment variables template
│   └── README.md         # This file
│
└── 📁 Database
    └── jobs.db           # SQLite job tracking (auto-created)
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd fine_tune_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your OpenAI API key
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 3. Start the Services

```bash
# Start FastAPI server (Terminal 1)
python app.py

# Start Streamlit dashboard (Terminal 2)
streamlit run dashboard.py
```

- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## 📊 Usage Workflow

### 1. Data Preprocessing

Upload and process your CSV dataset:

```python
from preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Process dataset
summary = preprocessor.process_dataset(
    input_path="data/your_dataset.csv",
    output_path="data/processed_training_data.jsonl"
)

print("Processing Summary:", summary)
```

**Supported CSV Formats:**
- **Q&A Format**: `question,answer` or `prompt,response`
- **Conversation Format**: `role,content` or `message,role`

### 2. Fine-tuning

Create and monitor fine-tuning jobs:

```python
from fine_tune import FineTuningManager

# Initialize manager
manager = FineTuningManager()

# Start fine-tuning
summary = manager.run_fine_tuning_pipeline(
    training_file_path="data/processed_training_data.jsonl",
    model="gpt-3.5-turbo",
    wait_for_completion=False
)

print("Fine-tuning Summary:", summary)

# Check job status
status = manager.get_job_status(summary["job_id"])
print("Job Status:", status)
```

### 3. Model Inference

Generate responses with fine-tuned models:

```python
from inference import InferenceManager

# Initialize inference manager
inference = InferenceManager()

# Generate response
result = inference.generate_response(
    prompt="What is machine learning?",
    model_id="ft:gpt-3.5-turbo:your-org:your-model:1234567890",
    max_tokens=100
)

print("Response:", result["response"])
print("Cost: $", result["estimated_cost_usd"])
```

### 4. API Usage

#### Upload and Preprocess Data
```bash
curl -X POST "http://localhost:8000/preprocess" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/your_dataset.csv"
```

#### Start Fine-tuning
```bash
curl -X POST "http://localhost:8000/fine-tune" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "wait_for_completion": false
  }'
```

#### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "model_id": "ft:gpt-3.5-turbo:your-org:your-model:1234567890",
    "max_tokens": 100
  }'
```

## 📈 Dashboard Features

The Streamlit dashboard provides:

- **Overview**: Key metrics and recent activity
- **Fine-tuning Jobs**: Job status, filtering, and management
- **Models**: Active model listing and testing
- **Inference Logs**: Request history and filtering
- **Usage Analytics**: Cost tracking and performance metrics

## 🔧 API Endpoints

### Data Processing
- `POST /preprocess` - Upload and process CSV datasets

### Fine-tuning
- `POST /fine-tune` - Create fine-tuning job
- `GET /jobs` - List all jobs
- `GET /jobs/{job_id}` - Get job status
- `DELETE /jobs/{job_id}` - Cancel job
- `GET /models` - List active models

### Inference
- `POST /generate` - Generate text response
- `POST /generate/batch` - Batch text generation
- `POST /test-model` - Test model with sample prompts

### Monitoring
- `GET /usage` - Get usage summary
- `GET /logs` - Get recent inference logs
- `GET /health` - Health check

## 💰 Cost Management

The pipeline includes comprehensive cost tracking:

- **Token Usage**: Automatic tracking of input/output tokens
- **Cost Estimation**: Real-time cost calculation based on OpenAI pricing
- **Usage Analytics**: Historical cost analysis and trends
- **Budget Monitoring**: Set alerts and limits (extensible)

## 🛡️ Security & Best Practices

- **Environment Variables**: Secure API key management
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Configurable API rate limits
- **CORS**: Configurable cross-origin resource sharing

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=.

# Code formatting
black .

# Linting
flake8 .
```

## 📝 Example Dataset

Create a sample CSV file for testing:

```csv
question,answer
"What is machine learning?","Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
"How do neural networks work?","Neural networks are computing systems inspired by biological brains, consisting of interconnected nodes that process and transmit information."
"What is fine-tuning?","Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or domain using additional training data."
```

## 🔄 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```bash
OPENAI_API_KEY=your_production_key
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=False
CORS_ORIGINS=["https://yourdomain.com"]
API_RATE_LIMIT=1000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the fine-tuning API
- FastAPI for the excellent web framework
- Streamlit for the interactive dashboard capabilities
- The MLOps community for best practices and inspiration

## 📞 Contact

**Keiko Rafi Ananda Prakoso**  
Computer Science (AI) Student  
Universiti Malaya  
Email: [your-email@example.com]  
LinkedIn: [your-linkedin-profile]

---

**Note**: This project is designed for educational and portfolio purposes. For production use, consider additional security measures, monitoring, and scalability improvements. # Automated-Fine-Tuning-Deployment-Pipeline
