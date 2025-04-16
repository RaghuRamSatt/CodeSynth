"""
Evaluator for benchmarking LLM model performance
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import chardet


from evaluation.metrics import ComprehensiveEvaluator
from evaluation.test_suite import TestSuite
from components.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates and compares LLM model performance for data analysis tasks.
    """
    
    def __init__(self, agents, config_path=None):
        """
        Initialize the model evaluator.
        
        Args:
            agents: Dictionary of agent instances with agent_id as keys
            config_path: Path to configuration file
        """
        self.agents = agents
        self.execution_engine = ExecutionEngine()
        self.evaluator = ComprehensiveEvaluator(self.execution_engine)
        self.test_suite = TestSuite()
        
        # Create directories for evaluation results
        self.results_dir = "evaluation/results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def evaluate_model(self, agent_id, test_cases, verbose=False):
        """
        Evaluate a single model on the test cases.
        
        Args:
            agent_id: ID of the agent to evaluate
            test_cases: List of test cases
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with evaluation results
        """

        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return None
            
        agent = self.agents[agent_id]
        
        # Initialize results structure
        results = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {},
            "test_cases": []
        }
        
        # Track metrics across all test cases
        all_scores = []
        all_execution_times = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            if verbose:
                logger.info(f"Evaluating test case {i+1}/{len(test_cases)} for model {agent_id}")
                
            # # Load dataset
            # try:
            #     dataset = pd.read_csv(test_case["dataset_path"])
            #     # Add the path as an attribute to the dataset
            #     dataset.test_case_path = test_case["dataset_path"]
            # except Exception as e:
            #     logger.error(f"Error loading dataset {test_case['dataset_path']}: {e}")
            #     continue

            # Load dataset with encoding detection
            try:
                # Detect file encoding
                with open(test_case["dataset_path"], 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding']
                
                # Load with detected encoding
                dataset = pd.read_csv(test_case["dataset_path"], encoding=encoding)
                logger.info(f"Loaded {test_case['dataset']} with detected encoding: {encoding}")
                
                # Add the path as an attribute to the dataset
                dataset.test_case_path = test_case["dataset_path"]
            except Exception as e:
                logger.error(f"Error loading dataset {test_case['dataset_path']}: {e}")
                continue
                
            # Prepare dataset info for the agent
            dataset_info = {
                "name": test_case["dataset"],
                "shape": dataset.shape,
                "columns": [
                    {
                        "name": col,
                        "type": str(dataset[col].dtype),
                        "description": "",
                        "sample": str(dataset[col].iloc[0]) if len(dataset) > 0 else ""
                    }
                    for col in dataset.columns
                ],
                "sample": dataset.head(5).to_string()
            }
            
            # Generate code
            start_time = time.time()
            code = agent.generate_code(test_case["query"], dataset_info)
            generation_time = time.time() - start_time
            
            # Get token counts (this depends on your agent implementation)
            prompt_tokens = agent.last_prompt_tokens if hasattr(agent, "last_prompt_tokens") else 0
            completion_tokens = agent.last_completion_tokens if hasattr(agent, "last_completion_tokens") else 0
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            # Evaluate the generated code
            eval_results = self.evaluator.evaluate(
                code=code,
                prompt=test_case["query"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                dataset=dataset
            )
            
            # Store the results for this test case
            test_result = {
                "test_case": test_case,
                "generated_code": code,
                "generation_time": generation_time,
                "evaluation": eval_results,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
            
            results["test_cases"].append(test_result)
            
            # Collect metrics for overall assessment
            all_scores.append(eval_results["overall_score"])
            all_execution_times.append(generation_time)
            
        # Calculate overall metrics
        results["overall_metrics"] = {
            "avg_score": sum(all_scores) / max(len(all_scores), 1),
            "min_score": min(all_scores) if all_scores else 0,
            "max_score": max(all_scores) if all_scores else 0,
            "avg_generation_time": sum(all_execution_times) / max(len(all_execution_times), 1),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
        
        # Save results to file
        results_file = os.path.join(self.results_dir, f"{agent_id}_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        if verbose:
            logger.info(f"Evaluation complete for {agent_id}. Average score: {results['overall_metrics']['avg_score']:.4f}")
            
        return results
        
    def compare_models(self, test_cases=None, categories=None, limit=10, verbose=False):
        """
        Compare multiple models on the same test cases.
        
        Args:
            test_cases: Specific test cases to use (if None, uses test suite)
            categories: Categories of test cases to include
            limit: Maximum number of test cases to use
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with comparative results
        """
        # Get test cases if not provided
        if test_cases is None:
            test_cases = self.test_suite.get_test_cases(categories=categories, limit=limit)
            
        # Initialize results structure
        comparative_results = {
            "timestamp": datetime.now().isoformat(),
            "test_cases": test_cases,
            "model_results": {}
        }
        
        # Evaluate each model
        for agent_id in self.agents:
            logger.info(f"Evaluating model: {agent_id}")
            model_result = self.evaluate_model(agent_id, test_cases, verbose)
            comparative_results["model_results"][agent_id] = model_result
            
        # Save comparative results
        results_file = os.path.join(self.results_dir, f"comparison_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump(comparative_results, f, indent=4)
            
        return comparative_results
        
    def generate_comparison_plots(self, comparative_results, output_dir="evaluation/plots"):
        """
        Generate comparison plots for multiple models.
        
        Args:
            comparative_results: Results from compare_models
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        # Extract model names and scores
        models = list(comparative_results["model_results"].keys())
        avg_scores = [comparative_results["model_results"][model]["overall_metrics"]["avg_score"] 
                     for model in models]
        
        # 1. Overall score comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=avg_scores)
        plt.title("Average Score Comparison")
        plt.ylabel("Average Score")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.tight_layout()
        
        score_plot = os.path.join(output_dir, "overall_score_comparison.png")
        plt.savefig(score_plot)
        plot_files.append(score_plot)
        plt.close()
        
        # 2. Token efficiency comparison
        token_efficiencies = []
        
        for model in models:
            metrics = comparative_results["model_results"][model]["overall_metrics"]
            score = metrics["avg_score"]
            total_tokens = metrics["total_tokens"]
            efficiency = score / max(total_tokens, 1) * 1000
            token_efficiencies.append(efficiency)
            
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=token_efficiencies)
        plt.title("Token Efficiency Comparison")
        plt.ylabel("Score per 1000 tokens")
        plt.xlabel("Model")
        plt.tight_layout()
        
        efficiency_plot = os.path.join(output_dir, "token_efficiency_comparison.png")
        plt.savefig(efficiency_plot)
        plot_files.append(efficiency_plot)
        plt.close()
        
        # 3. Generation time comparison
        generation_times = [comparative_results["model_results"][model]["overall_metrics"]["avg_generation_time"] 
                          for model in models]
                          
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=generation_times)
        plt.title("Average Generation Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Model")
        plt.tight_layout()
        
        time_plot = os.path.join(output_dir, "generation_time_comparison.png")
        plt.savefig(time_plot)
        plot_files.append(time_plot)
        plt.close()
        
        # 4. Score by category (heatmap)
        # Organize scores by category for each model
        categories = set()
        for test_case in comparative_results["test_cases"]:
            categories.add(test_case["category"])
            
        categories = list(categories)
        category_scores = {model: {category: [] for category in categories} for model in models}
        
        # Collect scores by category
        for model in models:
            for test_result in comparative_results["model_results"][model]["test_cases"]:
                category = test_result["test_case"]["category"]
                score = test_result["evaluation"]["overall_score"]
                category_scores[model][category].append(score)
                
        # Calculate average scores by category
        avg_category_scores = {model: {category: sum(scores) / max(len(scores), 1) 
                                     for category, scores in model_cats.items()}
                             for model, model_cats in category_scores.items()}
                             
        # Create score matrix for heatmap
        score_matrix = []
        for category in categories:
            category_row = [avg_category_scores[model][category] for model in models]
            score_matrix.append(category_row)
            
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                       xticklabels=models, yticklabels=categories, vmin=0, vmax=1)
        plt.title("Score by Category")
        plt.tight_layout()
        
        category_plot = os.path.join(output_dir, "category_score_comparison.png")
        plt.savefig(category_plot)
        plot_files.append(category_plot)
        plt.close()
        
        return plot_files