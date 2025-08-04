"""
Data Preprocessing Module for LLM Fine-Tuning Pipeline

This module handles data cleaning, validation, and preparation for OpenAI fine-tuning.
Supports Q&A and conversation log datasets in CSV format.

Author: Keiko Rafi Ananda Prakoso
Date: 2024
"""

import pandas as pd
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing for LLM fine-tuning datasets.
    
    Supports both Q&A format and conversation log formats.
    Validates data quality and converts to OpenAI fine-tuning format.
    """
    
    def __init__(self, max_tokens_per_example: int = 2048):
        """
        Initialize the DataPreprocessor.
        
        Args:
            max_tokens_per_example (int): Maximum tokens per training example
        """
        self.max_tokens_per_example = max_tokens_per_example
        
    def load_csv_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV dataset from file path.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid CSV
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """
        Detect the format of the dataset (Q&A or conversation).
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            str: 'qa' or 'conversation'
        """
        columns = df.columns.str.lower()
        
        # Check for Q&A format indicators
        qa_indicators = ['question', 'answer', 'prompt', 'response', 'input', 'output']
        if any(indicator in columns for indicator in qa_indicators):
            return 'qa'
        
        # Check for conversation format indicators
        conv_indicators = ['message', 'role', 'content', 'user', 'assistant', 'system']
        if any(indicator in columns for indicator in conv_indicators):
            return 'conversation'
        
        # Default to Q&A if we can't determine
        logger.warning("Could not determine format, defaulting to Q&A")
        return 'qa'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}]', '', text)
        
        return text
    
    def validate_qa_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean Q&A format data.
        
        Args:
            df (pd.DataFrame): Raw Q&A dataset
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Cleaned dataframe and validation errors
        """
        errors = []
        cleaned_df = df.copy()
        
        # Identify question and answer columns
        columns = df.columns.str.lower()
        question_col = None
        answer_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['question', 'prompt', 'input']):
                question_col = col
            elif any(word in col_lower for word in ['answer', 'response', 'output']):
                answer_col = col
        
        if not question_col or not answer_col:
            errors.append("Could not identify question and answer columns")
            return cleaned_df, errors
        
        # Clean and validate data
        cleaned_df[question_col] = cleaned_df[question_col].apply(self.clean_text)
        cleaned_df[answer_col] = cleaned_df[answer_col].apply(self.clean_text)
        
        # Remove rows with empty questions or answers
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[question_col, answer_col])
        cleaned_df = cleaned_df[cleaned_df[question_col] != '']
        cleaned_df = cleaned_df[cleaned_df[answer_col] != '']
        
        removed_count = initial_count - len(cleaned_df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty questions or answers")
        
        # Check for reasonable text lengths
        long_questions = cleaned_df[cleaned_df[question_col].str.len() > 1000]
        long_answers = cleaned_df[cleaned_df[answer_col].str.len() > 2000]
        
        if len(long_questions) > 0:
            errors.append(f"Found {len(long_questions)} questions longer than 1000 characters")
        
        if len(long_answers) > 0:
            errors.append(f"Found {len(long_answers)} answers longer than 2000 characters")
        
        return cleaned_df, errors
    
    def validate_conversation_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and clean conversation format data.
        
        Args:
            df (pd.DataFrame): Raw conversation dataset
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Cleaned dataframe and validation errors
        """
        errors = []
        cleaned_df = df.copy()
        
        # Identify role and content columns
        columns = df.columns.str.lower()
        role_col = None
        content_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'role' in col_lower:
                role_col = col
            elif 'content' in col_lower or 'message' in col_lower:
                content_col = col
        
        if not role_col or not content_col:
            errors.append("Could not identify role and content columns")
            return cleaned_df, errors
        
        # Clean content
        cleaned_df[content_col] = cleaned_df[content_col].apply(self.clean_text)
        
        # Validate roles
        valid_roles = ['user', 'assistant', 'system']
        invalid_roles = cleaned_df[~cleaned_df[role_col].str.lower().isin(valid_roles)]
        
        if len(invalid_roles) > 0:
            errors.append(f"Found {len(invalid_roles)} rows with invalid roles")
            cleaned_df = cleaned_df[cleaned_df[role_col].str.lower().isin(valid_roles)]
        
        # Remove empty content
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=[content_col])
        cleaned_df = cleaned_df[cleaned_df[content_col] != '']
        
        removed_count = initial_count - len(cleaned_df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with empty content")
        
        return cleaned_df, errors
    
    def convert_to_openai_format(self, df: pd.DataFrame, format_type: str) -> List[Dict]:
        """
        Convert cleaned dataset to OpenAI fine-tuning format.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            format_type (str): 'qa' or 'conversation'
            
        Returns:
            List[Dict]: List of training examples in OpenAI format
        """
        training_data = []
        
        if format_type == 'qa':
            # Convert Q&A format to chat completion format
            columns = df.columns.str.lower()
            question_col = None
            answer_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['question', 'prompt', 'input']):
                    question_col = col
                elif any(word in col_lower for word in ['answer', 'response', 'output']):
                    answer_col = col
            
            for _, row in df.iterrows():
                training_example = {
                    "messages": [
                        {"role": "user", "content": row[question_col]},
                        {"role": "assistant", "content": row[answer_col]}
                    ]
                }
                training_data.append(training_example)
        
        elif format_type == 'conversation':
            # Group conversation by session/conversation ID
            # For simplicity, we'll treat each row as a separate conversation
            columns = df.columns.str.lower()
            role_col = None
            content_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'role' in col_lower:
                    role_col = col
                elif 'content' in col_lower or 'message' in col_lower:
                    content_col = col
            
            for _, row in df.iterrows():
                training_example = {
                    "messages": [
                        {"role": row[role_col].lower(), "content": row[content_col]}
                    ]
                }
                training_data.append(training_example)
        
        logger.info(f"Converted {len(training_data)} examples to OpenAI format")
        return training_data
    
    def save_training_data(self, training_data: List[Dict], output_path: str) -> None:
        """
        Save training data to JSONL file for OpenAI fine-tuning.
        
        Args:
            training_data (List[Dict]): Training examples in OpenAI format
            output_path (str): Path to save the JSONL file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in training_data:
                    f.write(json.dumps(example) + '\n')
            
            logger.info(f"Saved {len(training_data)} training examples to {output_path}")
        except Exception as e:
            raise Exception(f"Error saving training data: {str(e)}")
    
    def process_dataset(self, input_path: str, output_path: str) -> Dict:
        """
        Complete dataset processing pipeline.
        
        Args:
            input_path (str): Path to input CSV file
            output_path (str): Path to output JSONL file
            
        Returns:
            Dict: Processing summary with statistics and any errors
        """
        logger.info("Starting dataset processing pipeline")
        
        # Load dataset
        df = self.load_csv_dataset(input_path)
        
        # Detect format
        format_type = self.detect_format(df)
        logger.info(f"Detected format: {format_type}")
        
        # Validate and clean data
        if format_type == 'qa':
            cleaned_df, errors = self.validate_qa_data(df)
        else:
            cleaned_df, errors = self.validate_conversation_data(df)
        
        # Convert to OpenAI format
        training_data = self.convert_to_openai_format(cleaned_df, format_type)
        
        # Save processed data
        self.save_training_data(training_data, output_path)
        
        # Prepare summary
        summary = {
            "original_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "training_examples": len(training_data),
            "format_type": format_type,
            "errors": errors,
            "output_path": output_path
        }
        
        logger.info("Dataset processing completed successfully")
        return summary


def main():
    """Example usage of the DataPreprocessor."""
    preprocessor = DataPreprocessor()
    
    # Example processing
    try:
        summary = preprocessor.process_dataset(
            input_path="data/sample_dataset.csv",
            output_path="data/processed_training_data.jsonl"
        )
        print("Processing Summary:", summary)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    main() 