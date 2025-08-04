# Automated Fine-Tuning & Deployment Pipeline for LLMs

A complete, production-ready Python backend for automated LLM fine-tuning and deployment. This project demonstrates end-to-end MLOps capabilities for Large Language Models, from dataset preparation to model deployment and monitoring.

**Author:** Keiko Rafi Ananda Prakoso  
**Institution:** Universiti Malaya (Computer Science - AI)  
**Date:** 2025

## Project Overview

This pipeline provides a modular, scalable solution for:
- **Data Preprocessing**: Clean and validate CSV datasets for fine-tuning
- **Fine-tuning Orchestration**: Automated OpenAI fine-tuning job management
- **Model Inference**: Production-ready inference with cost tracking
- **API Service**: FastAPI backend with comprehensive endpoints
- **Monitoring Dashboard**: Streamlit-based usage analytics and job tracking

## Architecture

```
fine_tune_pipeline/
│
├── Core Modules
│   ├── preprocess.py      # Data cleaning and validation
│   ├── fine_tune.py       # Fine-tuning orchestration
│   ├── inference.py       # Model inference and logging
│   └── app.py            # FastAPI server
│
├── Dashboard
│   └── dashboard.py      # Streamlit monitoring dashboard
│
├── Data & Logs
│   ├── data/             # Uploaded datasets
│   └── logs/             # Usage logs and analytics
│
├── Configuration
│   ├── requirements.txt  # Python dependencies
│   ├── env.example       # Environment variables template
│   └── README.md         # This file
│
└── Database
    └── jobs.db           # SQLite job tracking (auto-created)
```

## Dashboard Features

The Streamlit dashboard provides:

- **Overview**: Key metrics and recent activity
- **Fine-tuning Jobs**: Job status, filtering, and management
- **Models**: Active model listing and testing
- **Inference Logs**: Request history and filtering
- **Usage Analytics**: Cost tracking and performance metrics

## API Endpoints

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

## Cost Management

The pipeline includes comprehensive cost tracking:

- **Token Usage**: Automatic tracking of input/output tokens
- **Cost Estimation**: Real-time cost calculation based on OpenAI pricing
- **Usage Analytics**: Historical cost analysis and trends
- **Budget Monitoring**: Set alerts and limits (extensible)

## Security & Best Practices

- **Environment Variables**: Secure API key management
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: Configurable API rate limits
- **CORS**: Configurable cross-origin resource sharing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
ration

## Contact

**Keiko Rafi Ananda Prakoso**  
Computer Science (AI) Student  
Universiti Malaya  
Email: keikorafi@gmail.com 
LinkedIn: https://www.linkedin.com/in/keiko-prakoso-620811248/
