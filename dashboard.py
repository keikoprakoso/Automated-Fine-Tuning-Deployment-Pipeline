"""
Streamlit Dashboard for LLM Fine-Tuning Pipeline

This module provides a web-based dashboard for monitoring fine-tuning jobs,
model usage, and inference logs with interactive visualizations.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import os
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="LLM Fine-Tuning Pipeline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-pending {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class DashboardManager:
    """Manages dashboard data and visualizations."""
    
    def __init__(self):
        """Initialize the dashboard manager."""
        self.db_path = "jobs.db"
        self.log_file = "logs/inference_logs.csv"
    
    def get_jobs_data(self) -> pd.DataFrame:
        """Get fine-tuning jobs data from database."""
        try:
            if not os.path.exists(self.db_path):
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT 
                    job_id,
                    model_id,
                    status,
                    created_at,
                    updated_at,
                    training_file,
                    validation_file,
                    hyperparameters
                FROM fine_tune_jobs 
                ORDER BY created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert timestamps
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['updated_at'] = pd.to_datetime(df['updated_at'])
            
            return df
        except Exception as e:
            st.error(f"Error loading jobs data: {str(e)}")
            return pd.DataFrame()
    
    def get_models_data(self) -> pd.DataFrame:
        """Get active models data from database."""
        try:
            if not os.path.exists(self.db_path):
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT 
                    model_id,
                    job_id,
                    model_name,
                    status,
                    created_at,
                    deployment_notes
                FROM model_deployments 
                WHERE status = 'active'
                ORDER BY created_at DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert timestamps
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
        except Exception as e:
            st.error(f"Error loading models data: {str(e)}")
            return pd.DataFrame()
    
    def get_inference_logs(self) -> pd.DataFrame:
        """Get inference logs data."""
        try:
            if not os.path.exists(self.log_file):
                return pd.DataFrame()
            
            df = pd.read_csv(self.log_file)
            
            # Convert timestamps
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            st.error(f"Error loading inference logs: {str(e)}")
            return pd.DataFrame()
    
    def get_usage_summary(self, days: int = 30) -> Dict:
        """Get usage summary statistics."""
        try:
            logs_df = self.get_inference_logs()
            
            if logs_df.empty:
                return {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "success_rate": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "average_tokens_per_request": 0,
                    "average_cost_per_request": 0.0
                }
            
            # Filter by date if needed
            if days > 0:
                cutoff_date = datetime.now() - timedelta(days=days)
                logs_df = logs_df[logs_df['timestamp'] >= cutoff_date]
            
            total_requests = len(logs_df)
            successful_requests = len(logs_df[logs_df['status'] == 'success'])
            failed_requests = total_requests - successful_requests
            
            total_tokens = logs_df['total_tokens'].sum()
            total_cost = logs_df['estimated_cost_usd'].sum()
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
                "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0,
                "average_cost_per_request": total_cost / total_requests if total_requests > 0 else 0
            }
        except Exception as e:
            st.error(f"Error calculating usage summary: {str(e)}")
            return {}


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LLM Fine-Tuning Pipeline Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize dashboard manager
    dashboard = DashboardManager()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Preprocessing", "Fine-tuning Jobs", "Models", "Inference", "Inference Logs", "Usage Analytics"]
    )
    
    # Overview page
    if page == "Overview":
        show_overview_page(dashboard)
    
    # Data Preprocessing page
    elif page == "Data Preprocessing":
        show_preprocessing_page(dashboard)
    
    # Fine-tuning Jobs page
    elif page == "Fine-tuning Jobs":
        show_jobs_page(dashboard)
    
    # Models page
    elif page == "Models":
        show_models_page(dashboard)
    
    # Inference page
    elif page == "Inference":
        show_inference_page(dashboard)
    
    # Inference Logs page
    elif page == "Inference Logs":
        show_logs_page(dashboard)
    
    # Usage Analytics page
    elif page == "Usage Analytics":
        show_analytics_page(dashboard)


def show_preprocessing_page(dashboard: DashboardManager):
    """Display the data preprocessing page."""
    st.header("📊 Data Preprocessing")
    
    st.write("""
    This page helps you prepare your data for fine-tuning. Upload a CSV file with your training data
    and convert it to the OpenAI fine-tuning format (JSONL).
    """)
    
    # File upload section
    st.subheader("📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your training data. Supported formats: Q&A (question,answer) or conversation data."
    )
    
    if uploaded_file is not None:
        # Show file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        
        # Preview the data
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Data Preview")
            st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            st.dataframe(df.head())
            
            # Process button
            if st.button("🔄 Process Dataset", type="primary"):
                with st.spinner("Processing dataset..."):
                    try:
                        # Save uploaded file
                        input_path = f"data/{uploaded_file.name}"
                        output_path = f"data/processed_{uploaded_file.name.replace('.csv', '.jsonl')}"
                        
                        os.makedirs("data", exist_ok=True)
                        with open(input_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Import and use preprocessor
                        from preprocess import DataPreprocessor
                        preprocessor = DataPreprocessor()
                        summary = preprocessor.process_dataset(input_path, output_path)
                        
                        st.success("✅ Dataset processed successfully!")
                        st.json(summary)
                        
                        # Download link
                        if os.path.exists(output_path):
                            with open(output_path, "r") as f:
                                st.download_button(
                                    label="📥 Download Processed Data",
                                    data=f.read(),
                                    file_name=output_path.split("/")[-1],
                                    mime="application/json"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Sample dataset section
    st.subheader("📚 Sample Dataset")
    
    if st.button("🔄 Process Sample Dataset"):
        with st.spinner("Processing sample dataset..."):
            try:
                from preprocess import DataPreprocessor
                preprocessor = DataPreprocessor()
                
                input_path = "data/sample_dataset.csv"
                output_path = "data/processed_sample_dataset.jsonl"
                
                if os.path.exists(input_path):
                    summary = preprocessor.process_dataset(input_path, output_path)
                    st.success("✅ Sample dataset processed successfully!")
                    st.json(summary)
                    
                    # Download link
                    if os.path.exists(output_path):
                        with open(output_path, "r") as f:
                            st.download_button(
                                label="📥 Download Processed Sample Data",
                                data=f.read(),
                                file_name=output_path.split("/")[-1],
                                mime="application/json"
                            )
                else:
                    st.error("❌ Sample dataset not found. Please upload your own file.")
            
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
    
    # Instructions
    st.subheader("📖 Instructions")
    st.write("""
    **Supported CSV Formats:**
    
    1. **Q&A Format**: Two columns named `question` and `answer`
    2. **Conversation Format**: One column with conversation data
    
    **Example Q&A Format:**
    ```csv
    question,answer
    "What is machine learning?","Machine learning is a subset of AI..."
    "How do neural networks work?","Neural networks are computing systems..."
    ```
    
    **What happens during processing:**
    - Data is cleaned and validated
    - Converted to OpenAI fine-tuning format (JSONL)
    - Ready for fine-tuning jobs
    """)


def show_inference_page(dashboard: DashboardManager):
    """Display the inference page for testing models."""
    st.header("🎯 Model Inference")
    
    st.write("""
    Test your fine-tuned models by generating responses to prompts. Select a model and enter your prompt to see how it performs.
    """)
    
    # Get available models
    models_df = dashboard.get_models_data()
    
    if models_df.empty:
        st.warning("⚠️ No fine-tuned models available. Please create a fine-tuning job first.")
        return
    
    # Model selection
    st.subheader("🤖 Select Model")
    model_options = models_df['model_id'].tolist()
    selected_model = st.selectbox(
        "Choose a model",
        model_options,
        help="Select a fine-tuned model to test"
    )
    
    if selected_model:
        # Show model info
        model_info = models_df[models_df['model_id'] == selected_model].iloc[0]
        st.info(f"**Model:** {model_info['model_name']} | **Created:** {model_info['created_at']}")
        
        # Prompt input
        st.subheader("💬 Enter Prompt")
        prompt = st.text_area(
            "Your prompt",
            placeholder="Enter your prompt here...",
            height=100,
            help="Enter the text you want the model to respond to"
        )
        
        # Generation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4000, value=1000)
        with col2:
            temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        with col3:
            system_message = st.text_input("System Message (optional)", placeholder="You are a helpful assistant...")
        
        # Generate button
        if st.button("🚀 Generate Response", type="primary", disabled=not prompt.strip()):
            if prompt.strip():
                with st.spinner("Generating response..."):
                    try:
                        # Import and use inference manager
                        from inference import InferenceManager
                        inference_manager = InferenceManager()
                        
                        # Generate response
                        response = inference_manager.generate_text(
                            prompt=prompt,
                            model_id=selected_model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            system_message=system_message if system_message else None
                        )
                        
                        if response.get("status") == "success":
                            st.success("✅ Response generated successfully!")
                            
                            # Display response
                            st.subheader("📝 Generated Response")
                            st.write(response["response"])
                            
                            # Show metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Prompt Tokens", response["prompt_tokens"])
                            with col2:
                                st.metric("Completion Tokens", response["completion_tokens"])
                            with col3:
                                st.metric("Total Tokens", response["total_tokens"])
                            with col4:
                                st.metric("Cost (USD)", f"${response['estimated_cost_usd']:.4f}")
                            
                            # Response time
                            st.write(f"⏱️ Response time: {response['response_time_seconds']:.2f} seconds")
                        
                        else:
                            st.error(f"❌ Generation failed: {response.get('error_message', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Batch generation section
    st.subheader("📦 Batch Generation")
    st.write("Generate responses for multiple prompts at once.")
    
    batch_prompts = st.text_area(
        "Batch Prompts (one per line)",
        placeholder="Enter multiple prompts, one per line...",
        height=150,
        help="Enter multiple prompts, each on a separate line"
    )
    
    if batch_prompts and selected_model:
        if st.button("🚀 Generate Batch Responses"):
            prompts = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
            
            if prompts:
                with st.spinner(f"Generating responses for {len(prompts)} prompts..."):
                    try:
                        from inference import InferenceManager
                        inference_manager = InferenceManager()
                        
                        results = []
                        for i, prompt in enumerate(prompts):
                            response = inference_manager.generate_text(
                                prompt=prompt,
                                model_id=selected_model,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            results.append({
                                "prompt": prompt,
                                "response": response.get("response", "Error"),
                                "status": response.get("status", "error"),
                                "tokens": response.get("total_tokens", 0),
                                "cost": response.get("estimated_cost_usd", 0)
                            })
                        
                        # Display results
                        st.success(f"✅ Generated responses for {len(prompts)} prompts!")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Prompt {i+1}: {result['prompt'][:50]}..."):
                                st.write(f"**Response:** {result['response']}")
                                st.write(f"**Status:** {result['status']} | **Tokens:** {result['tokens']} | **Cost:** ${result['cost']:.4f}")
                    
                    except Exception as e:
                        st.error(f"❌ Batch generation failed: {str(e)}")


def show_overview_page(dashboard: DashboardManager):
    """Display the overview page with key metrics."""
    st.header("📊 Overview")
    
    # Get data
    jobs_df = dashboard.get_jobs_data()
    models_df = dashboard.get_models_data()
    usage_summary = dashboard.get_usage_summary(days=30)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Jobs",
            value=len(jobs_df),
            delta=len(jobs_df[jobs_df['status'] == 'succeeded']) if not jobs_df.empty else 0
        )
    
    with col2:
        st.metric(
            label="Active Models",
            value=len(models_df),
            delta=len(models_df) if not models_df.empty else 0
        )
    
    with col3:
        st.metric(
            label="Total Requests (30d)",
            value=usage_summary.get("total_requests", 0),
            delta=f"{usage_summary.get('success_rate', 0):.1f}% success rate"
        )
    
    with col4:
        st.metric(
            label="Total Cost (30d)",
            value=f"${usage_summary.get('total_cost_usd', 0):.2f}",
            delta=f"${usage_summary.get('average_cost_per_request', 0):.4f} avg"
        )
    
    # Recent activity
    st.subheader("Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Recent Jobs**")
        if not jobs_df.empty:
            recent_jobs = jobs_df.head(5)[['job_id', 'status', 'created_at']]
            for _, job in recent_jobs.iterrows():
                status_color = "status-success" if job['status'] == 'succeeded' else \
                              "status-error" if job['status'] == 'failed' else "status-pending"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{job['job_id'][:8]}...</strong><br>
                    <span class="{status_color}">{job['status'].upper()}</span><br>
                    <small>{job['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No jobs found")
    
    with col2:
        st.write("**Recent Models**")
        if not models_df.empty:
            recent_models = models_df.head(5)[['model_id', 'created_at']]
            for _, model in recent_models.iterrows():
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{model['model_id'][:15]}...</strong><br>
                    <span class="status-success">ACTIVE</span><br>
                    <small>{model['created_at'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active models found")


def show_jobs_page(dashboard: DashboardManager):
    """Display the fine-tuning jobs page."""
    st.header("🔧 Fine-tuning Jobs")
    
    # Create new job section
    st.subheader("🚀 Create New Fine-tuning Job")
    
    with st.expander("Create New Job", expanded=True):
        # Get available processed datasets
        processed_files = []
        if os.path.exists("data"):
            for file in os.listdir("data"):
                if file.endswith(".jsonl") and file.startswith("processed_"):
                    processed_files.append(file)
        
        if not processed_files:
            st.warning("⚠️ No processed datasets found. Please process a dataset first in the Data Preprocessing tab.")
        else:
            # Job creation form
            col1, col2 = st.columns(2)
            
            with col1:
                selected_dataset = st.selectbox(
                    "Select Processed Dataset",
                    processed_files,
                    help="Choose a processed dataset for fine-tuning"
                )
                
                base_model = st.selectbox(
                    "Base Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    index=0,
                    help="Choose the base model to fine-tune"
                )
                
                wait_for_completion = st.checkbox(
                    "Wait for completion",
                    value=False,
                    help="Wait for the job to complete (may take 15-60 minutes)"
                )
            
            with col2:
                n_epochs = st.number_input(
                    "Number of Epochs",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of training epochs"
                )
                
                batch_size = st.selectbox(
                    "Batch Size",
                    [1, 2, 4, 8],
                    index=0,
                    help="Training batch size"
                )
                
                learning_rate = st.selectbox(
                    "Learning Rate Multiplier",
                    [0.5, 1.0, 1.5, 2.0],
                    index=1,
                    help="Learning rate multiplier"
                )
            
            # Hyperparameters
            hyperparameters = {
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate
            }
            
            # Create job button
            if st.button("🚀 Start Fine-tuning Job", type="primary"):
                if selected_dataset:
                    with st.spinner("Creating fine-tuning job..."):
                        try:
                            # Import fine-tuning manager
                            from fine_tune import FineTuningManager
                            fine_tune_manager = FineTuningManager()
                            
                            # Create the job
                            training_file_path = f"data/{selected_dataset}"
                            
                            result = fine_tune_manager.run_fine_tuning_pipeline(
                                training_file_path=training_file_path,
                                model=base_model,
                                hyperparameters=hyperparameters,
                                wait_for_completion=wait_for_completion
                            )
                            
                            st.success(f"✅ Fine-tuning job created successfully!")
                            st.info(f"**Job ID:** {result['job_id']}")
                            st.info(f"**Training File ID:** {result['training_file_id']}")
                            st.info(f"**Model:** {result['model']}")
                            
                            if wait_for_completion:
                                st.info("⏳ Waiting for job completion... This may take 15-60 minutes.")
                            else:
                                st.info("📊 You can monitor the job progress below.")
                            
                            # Refresh the page to show the new job
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ Failed to create fine-tuning job: {str(e)}")
    
    # Existing jobs section
    st.subheader("📊 Existing Jobs")
    jobs_df = dashboard.get_jobs_data()
    
    if jobs_df.empty:
        st.info("No fine-tuning jobs found. Create your first job above!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + list(jobs_df['status'].unique())
        )
    
    with col2:
        date_filter = st.date_input(
            "Filter by Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    with col3:
        search_job = st.text_input("Search Job ID", "")
    
    # Apply filters
    filtered_df = jobs_df.copy()
    
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    
    if date_filter:
        filtered_df = filtered_df[filtered_df['created_at'].dt.date == date_filter]
    
    if search_job:
        filtered_df = filtered_df[filtered_df['job_id'].str.contains(search_job, case=False)]
    
    # Display jobs
    st.subheader(f"Jobs ({len(filtered_df)} found)")
    
    for _, job in filtered_df.iterrows():
        with st.expander(f"Job: {job['job_id']} - {job['status'].upper()}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {job['status']}")
                st.write(f"**Created:** {job['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Updated:** {job['updated_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if job['model_id']:
                    st.write(f"**Model ID:** {job['model_id']}")
                st.write(f"**Training File:** {job['training_file']}")
                if job['validation_file']:
                    st.write(f"**Validation File:** {job['validation_file']}")
    
    # Status distribution chart
    st.subheader("Job Status Distribution")
    if not filtered_df.empty:
        status_counts = filtered_df['status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Job Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_models_page(dashboard: DashboardManager):
    """Display the models page."""
    st.header("🤖 Fine-tuned Models")
    
    models_df = dashboard.get_models_data()
    
    if models_df.empty:
        st.info("No active models found. Complete a fine-tuning job to see models here!")
        return
    
    # Display models
    for _, model in models_df.iterrows():
        with st.expander(f"Model: {model['model_id']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Model ID:** {model['model_id']}")
                st.write(f"**Job ID:** {model['job_id']}")
                st.write(f"**Status:** {model['status']}")
            
            with col2:
                st.write(f"**Created:** {model['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                if model['deployment_notes']:
                    st.write(f"**Notes:** {model['deployment_notes']}")
            
            # Test model button
            if st.button(f"Test Model {model['model_id'][:8]}...", key=f"test_{model['model_id']}"):
                st.info("Model testing functionality would be implemented here")


def show_logs_page(dashboard: DashboardManager):
    """Display the inference logs page."""
    st.header("📝 Inference Logs")
    
    logs_df = dashboard.get_inference_logs()
    
    if logs_df.empty:
        st.info("No inference logs found. Start generating responses to see logs here!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_filter = st.selectbox(
            "Filter by Model",
            ["All"] + list(logs_df['model_id'].unique())
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All"] + list(logs_df['status'].unique())
        )
    
    with col3:
        date_filter = st.date_input(
            "Filter by Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    # Apply filters
    filtered_logs = logs_df.copy()
    
    if model_filter != "All":
        filtered_logs = filtered_logs[filtered_logs['model_id'] == model_filter]
    
    if status_filter != "All":
        filtered_logs = filtered_logs[filtered_logs['status'] == status_filter]
    
    if date_filter:
        filtered_logs = filtered_logs[filtered_logs['timestamp'].dt.date == date_filter]
    
    # Display logs
    st.subheader(f"Logs ({len(filtered_logs)} found)")
    
    # Show recent logs in a table
    display_columns = ['timestamp', 'model_id', 'prompt', 'response', 'total_tokens', 'estimated_cost_usd', 'status']
    st.dataframe(
        filtered_logs[display_columns].head(20),
        use_container_width=True
    )
    
    # Download logs
    if st.button("Download Filtered Logs"):
        csv = filtered_logs.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"inference_logs_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def show_analytics_page(dashboard: DashboardManager):
    """Display the usage analytics page."""
    st.header("📈 Usage Analytics")
    
    # Time period selector
    days = st.slider("Select time period (days)", 1, 90, 30)
    
    # Get usage data
    usage_summary = dashboard.get_usage_summary(days=days)
    logs_df = dashboard.get_inference_logs()
    
    if logs_df.empty:
        st.info("No usage data found. Start generating responses to see analytics!")
        return
    
    # Filter logs by time period
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_logs = logs_df[logs_df['timestamp'] >= cutoff_date]
    
    if filtered_logs.empty:
        st.info(f"No data found for the last {days} days")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Requests",
            value=usage_summary["total_requests"],
            delta=f"{usage_summary['success_rate']:.1f}% success rate"
        )
    
    with col2:
        st.metric(
            label="Total Tokens",
            value=f"{usage_summary['total_tokens']:,}",
            delta=f"{usage_summary['average_tokens_per_request']:.0f} avg per request"
        )
    
    with col3:
        st.metric(
            label="Total Cost",
            value=f"${usage_summary['total_cost_usd']:.2f}",
            delta=f"${usage_summary['average_cost_per_request']:.4f} avg per request"
        )
    
    with col4:
        st.metric(
            label="Failed Requests",
            value=usage_summary["failed_requests"],
            delta=f"{usage_summary['failed_requests'] / usage_summary['total_requests'] * 100:.1f}%" if usage_summary['total_requests'] > 0 else "0%"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily usage trend
        st.subheader("Daily Usage Trend")
        daily_usage = filtered_logs.groupby(filtered_logs['timestamp'].dt.date).agg({
            'total_tokens': 'sum',
            'estimated_cost_usd': 'sum',
            'timestamp': 'count'
        }).reset_index()
        daily_usage.columns = ['date', 'total_tokens', 'total_cost', 'requests']
        
        fig = px.line(
            daily_usage,
            x='date',
            y='total_cost',
            title="Daily Cost Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model usage distribution
        st.subheader("Model Usage Distribution")
        model_usage = filtered_logs['model_id'].value_counts()
        
        fig = px.pie(
            values=model_usage.values,
            names=model_usage.index,
            title="Requests by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Token usage vs cost scatter plot
    st.subheader("Token Usage vs Cost")
    fig = px.scatter(
        filtered_logs,
        x='total_tokens',
        y='estimated_cost_usd',
        color='model_id',
        title="Token Usage vs Cost Relationship"
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main() 