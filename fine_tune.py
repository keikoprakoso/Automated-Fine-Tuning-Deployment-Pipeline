"""
Fine-Tuning Orchestration Module for LLM Pipeline

This module handles OpenAI fine-tuning API calls, job management, and model deployment.
Provides comprehensive fine-tuning workflow with status monitoring and error handling.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import os
import time
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path
import openai
from datetime import datetime
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuningManager:
    """
    Manages OpenAI fine-tuning jobs and model deployment.
    
    Handles file uploads, job creation, status monitoring, and model retrieval.
    Stores job and model information in a local database for tracking.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FineTuningManager.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.db_path = "jobs.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for job tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fine_tune_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    model_id TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    training_file TEXT,
                    validation_file TEXT,
                    hyperparameters TEXT,
                    result_files TEXT,
                    error_message TEXT
                )
            ''')
            
            # Create model deployments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT UNIQUE NOT NULL,
                    job_id TEXT NOT NULL,
                    model_name TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployment_notes TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI.
        
        Args:
            file_path (str): Path to the JSONL training file
            
        Returns:
            str: OpenAI file ID
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If upload fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training file not found: {file_path}")
            
            logger.info(f"Uploading training file: {file_path}")
            
            with open(file_path, 'rb') as file:
                response = openai.File.create(
                    file=file,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            logger.info(f"File uploaded successfully. File ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise
    
    def create_fine_tune_job(self, 
                           training_file_id: str, 
                           model: str = "gpt-3.5-turbo",
                           validation_file_id: Optional[str] = None,
                           hyperparameters: Optional[Dict] = None) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            training_file_id (str): OpenAI file ID for training data
            model (str): Base model to fine-tune
            validation_file_id (str, optional): OpenAI file ID for validation data
            hyperparameters (dict, optional): Fine-tuning hyperparameters
            
        Returns:
            str: Fine-tuning job ID
            
        Raises:
            Exception: If job creation fails
        """
        try:
            logger.info(f"Creating fine-tuning job for model: {model}")
            
            # Prepare job parameters
            job_params = {
                "training_file": training_file_id,
                "model": model
            }
            
            if validation_file_id:
                job_params["validation_file"] = validation_file_id
            
            if hyperparameters:
                job_params["hyperparameters"] = hyperparameters
            
            # Create the fine-tuning job
            response = openai.FineTuningJob.create(**job_params)
            job_id = response.id
            
            # Store job information in database
            self._store_job_info(job_id, training_file_id, validation_file_id, hyperparameters)
            
            logger.info(f"Fine-tuning job created successfully. Job ID: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Fine-tuning job creation failed: {str(e)}")
            raise
    
    def _store_job_info(self, job_id: str, training_file: str, 
                       validation_file: Optional[str] = None, 
                       hyperparameters: Optional[Dict] = None):
        """Store job information in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fine_tune_jobs 
                (job_id, status, training_file, validation_file, hyperparameters)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                job_id,
                'created',
                training_file,
                validation_file,
                json.dumps(hyperparameters) if hyperparameters else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store job info: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get the current status of a fine-tuning job.
        
        Args:
            job_id (str): Fine-tuning job ID
            
        Returns:
            Dict: Job status information
        """
        try:
            logger.info(f"Checking status for job: {job_id}")
            
            response = openai.FineTuningJob.retrieve(job_id)
            
            # Update database with current status
            self._update_job_status(job_id, response.status, response.fine_tuned_model)
            
            status_info = {
                "job_id": job_id,
                "status": response.status,
                "model_id": response.fine_tuned_model,
                "created_at": response.created_at,
                "finished_at": response.finished_at,
                "training_file": response.training_file,
                "validation_file": response.validation_file,
                "result_files": response.result_files,
                "error": response.error
            }
            
            logger.info(f"Job {job_id} status: {response.status}")
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            raise
    
    def _update_job_status(self, job_id: str, status: str, model_id: Optional[str] = None):
        """Update job status in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE fine_tune_jobs 
                SET status = ?, model_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            ''', (status, model_id, job_id))
            
            # If job is completed and we have a model ID, add to deployments
            if status == 'succeeded' and model_id:
                cursor.execute('''
                    INSERT OR REPLACE INTO model_deployments 
                    (model_id, job_id, model_name, status)
                    VALUES (?, ?, ?, ?)
                ''', (model_id, job_id, f"ft-{model_id}", 'active'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
    
    def list_jobs(self, limit: int = 10) -> List[Dict]:
        """
        List recent fine-tuning jobs.
        
        Args:
            limit (int): Maximum number of jobs to return
            
        Returns:
            List[Dict]: List of job information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT job_id, model_id, status, created_at, updated_at, training_file
                FROM fine_tune_jobs 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            jobs = []
            for row in cursor.fetchall():
                jobs.append({
                    "job_id": row[0],
                    "model_id": row[1],
                    "status": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "training_file": row[5]
                })
            
            conn.close()
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {str(e)}")
            return []
    
    def get_active_models(self) -> List[Dict]:
        """
        Get list of active fine-tuned models.
        
        Returns:
            List[Dict]: List of active model information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_id, job_id, model_name, created_at, deployment_notes
                FROM model_deployments 
                WHERE status = 'active'
                ORDER BY created_at DESC
            ''')
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    "model_id": row[0],
                    "job_id": row[1],
                    "model_name": row[2],
                    "created_at": row[3],
                    "deployment_notes": row[4]
                })
            
            conn.close()
            return models
            
        except Exception as e:
            logger.error(f"Failed to get active models: {str(e)}")
            return []
    
    def wait_for_completion(self, job_id: str, check_interval: int = 60) -> Dict:
        """
        Wait for fine-tuning job to complete.
        
        Args:
            job_id (str): Fine-tuning job ID
            check_interval (int): Seconds between status checks
            
        Returns:
            Dict: Final job status information
        """
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            status_info = self.get_job_status(job_id)
            status = status_info["status"]
            
            if status in ['succeeded', 'failed', 'cancelled']:
                logger.info(f"Job {job_id} completed with status: {status}")
                return status_info
            
            logger.info(f"Job {job_id} status: {status}. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running fine-tuning job.
        
        Args:
            job_id (str): Fine-tuning job ID
            
        Returns:
            bool: True if cancellation was successful
        """
        try:
            logger.info(f"Cancelling job: {job_id}")
            
            response = openai.FineTuningJob.cancel(job_id)
            
            # Update database status
            self._update_job_status(job_id, 'cancelled')
            
            logger.info(f"Job {job_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a fine-tuned model.
        
        Args:
            model_id (str): Model ID to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            logger.info(f"Deleting model: {model_id}")
            
            # Note: OpenAI doesn't currently support model deletion via API
            # This would be implemented when the feature becomes available
            logger.warning("Model deletion not yet supported by OpenAI API")
            
            # Mark as inactive in our database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE model_deployments 
                SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                WHERE model_id = ?
            ''', (model_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model {model_id} marked as deleted in database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False
    
    def run_fine_tuning_pipeline(self, 
                                training_file_path: str,
                                model: str = "gpt-3.5-turbo",
                                validation_file_path: Optional[str] = None,
                                hyperparameters: Optional[Dict] = None,
                                wait_for_completion: bool = False) -> Dict:
        """
        Complete fine-tuning pipeline from file upload to job creation.
        
        Args:
            training_file_path (str): Path to training JSONL file
            model (str): Base model to fine-tune
            validation_file_path (str, optional): Path to validation JSONL file
            hyperparameters (dict, optional): Fine-tuning hyperparameters
            wait_for_completion (bool): Whether to wait for job completion
            
        Returns:
            Dict: Pipeline execution summary
        """
        logger.info("Starting fine-tuning pipeline")
        
        try:
            # Upload training file
            training_file_id = self.upload_training_file(training_file_path)
            
            # Upload validation file if provided
            validation_file_id = None
            if validation_file_path:
                validation_file_id = self.upload_training_file(validation_file_path)
            
            # Create fine-tuning job
            job_id = self.create_fine_tune_job(
                training_file_id=training_file_id,
                model=model,
                validation_file_id=validation_file_id,
                hyperparameters=hyperparameters
            )
            
            # Wait for completion if requested
            final_status = None
            if wait_for_completion:
                final_status = self.wait_for_completion(job_id)
            
            # Prepare summary
            summary = {
                "job_id": job_id,
                "training_file_id": training_file_id,
                "validation_file_id": validation_file_id,
                "model": model,
                "status": "created",
                "final_status": final_status,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Fine-tuning pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Fine-tuning pipeline failed: {str(e)}")
            raise


def main():
    """Example usage of the FineTuningManager."""
    try:
        # Initialize manager
        manager = FineTuningManager()
        
        # Example: Run fine-tuning pipeline
        summary = manager.run_fine_tuning_pipeline(
            training_file_path="data/processed_training_data.jsonl",
            model="gpt-3.5-turbo",
            wait_for_completion=False
        )
        
        print("Fine-tuning Summary:", summary)
        
        # List recent jobs
        jobs = manager.list_jobs(limit=5)
        print(f"Recent jobs: {len(jobs)}")
        
        # Get active models
        models = manager.get_active_models()
        print(f"Active models: {len(models)}")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")


if __name__ == "__main__":
    main() 