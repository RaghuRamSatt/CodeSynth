"""
DS-1000 Integration for evaluation framework
"""

import os
import sys
import json
import gzip
import logging
from typing import Dict, List, Any
from datetime import datetime
import time
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Add DS-1000 path to system path - adjust this to your actual path
DS1000_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DS-1000-main")
sys.path.append(DS1000_PATH)

class DS1000Evaluator:
    """Evaluator for DS-1000 benchmark tasks"""
    
    def __init__(self, results_dir="evaluation/results"):
        """Initialize the DS1000 evaluator"""
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load the DS-1000 dataset
        try:
            ds1000_path = os.path.join(DS1000_PATH, "data/ds1000.jsonl.gz")
            self.problems = [json.loads(l) for l in gzip.open(ds1000_path, "rt").readlines()]
            logger.info(f"Loaded {len(self.problems)} problems from DS-1000")
        except Exception as e:
            logger.error(f"Failed to load DS-1000 dataset: {e}")
            self.problems = []
    
    def filter_problems(self, libraries=None, limit=None):
        """
        Filter problems by library and limit
        
        Args:
            libraries: List of libraries to include (numpy, pandas, etc.)
            limit: Maximum number of problems to return
            
        Returns:
            Filtered list of problems
        """
        filtered = self.problems
        
        if libraries:
            filtered = [p for p in filtered if p["metadata"]["library"].lower() in 
                        [lib.lower() for lib in libraries]]
            
        if limit and limit > 0:
            filtered = filtered[:limit]
            
        return filtered
    
    def evaluate_solution(self, problem, solution):
        """
        Evaluate a solution using DS-1000's evaluation logic
        
        Args:
            problem: DS-1000 problem
            solution: Generated solution code
            
        Returns:
            Dictionary with evaluation results
        """
        import tempfile
        import subprocess
        
        # Create a temporary Python file to run the evaluation
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            # Write evaluation code
            f.write(f"""
import sys
import json

# Solution code to evaluate
solution = '''{solution}'''

# Code context from the problem
{problem['code_context']}

# Run evaluation
try:
    test_results = {{'execution': test_execution(solution)}}
    
    # Run string test if it exists
    if 'test_string' in globals():
        test_results['string'] = test_string(solution)
    else:
        test_results['string'] = True
        
    # Solution is correct if both tests pass
    test_results['correct'] = test_results['execution'] and test_results['string']
    
    # Write results to stdout
    print(json.dumps(test_results))
    sys.exit(0)
except Exception as e:
    # Handle evaluation errors
    print(json.dumps({{'error': str(e), 'execution': False, 'string': False, 'correct': False}}))
    sys.exit(1)
""")
            temp_file = f.name
        
        try:
            # Run the evaluation in a separate process to avoid interference
            process = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30  # Set timeout to prevent infinite execution
            )
            
            # Parse results
            if process.returncode == 0 and process.stdout:
                try:
                    results = json.loads(process.stdout.strip())
                    return results
                except json.JSONDecodeError:
                    return {
                        "error": "Failed to parse evaluation results",
                        "execution": False,
                        "string": False,
                        "correct": False
                    }
            else:
                return {
                    "error": process.stderr if process.stderr else "Evaluation failed",
                    "execution": False,
                    "string": False,
                    "correct": False
                }
                
        except subprocess.TimeoutExpired:
            return {
                "error": "Evaluation timed out",
                "execution": False,
                "string": False,
                "correct": False,
                "timeout": True
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def evaluate_model(self, agent, libraries=None, limit=10, verbose=False):
        """
        Evaluate a model on DS-1000 benchmark
        
        Args:
            agent: Agent instance to evaluate
            libraries: Specific libraries to test (numpy, pandas, etc.)
            limit: Maximum number of problems to test
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        # Get filtered problems
        problems = self.filter_problems(libraries, limit)
        
        if not problems:
            logger.error(f"No DS-1000 problems found matching criteria")
            return None
            
        if verbose:
            logger.info(f"Evaluating agent on {len(problems)} DS-1000 problems")
        
        # Initialize results
        results = {
            "agent_id": agent.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {},
            "test_cases": []
        }
        
        # Track metrics
        correct_count = 0
        execution_times = []
        libraries_results = {}
        
        # Evaluate each problem
        for i, problem in enumerate(problems):
            library = problem["metadata"]["library"]
            
            if verbose:
                logger.info(f"Evaluating problem {i+1}/{len(problems)} (Library: {library})")
            
            # Extract problem info
            prompt = problem["prompt"]
            
            # Create dataset info (simplified - DS-1000 handles the data internally)
            dataset_info = {
                "name": f"DS-1000 {library} problem",
                "shape": None,
                "columns": []
            }
            
            # Generate code
            start_time = time.time()
            solution = agent.generate_code(prompt, dataset_info)
            generation_time = time.time() - start_time
            execution_times.append(generation_time)
            
            # Evaluate solution
            eval_result = self.evaluate_solution(problem, solution)
            
            # Update library-specific stats
            if library not in libraries_results:
                libraries_results[library] = {"total": 0, "correct": 0}
            libraries_results[library]["total"] += 1
            if eval_result.get("correct", False):
                libraries_results[library]["correct"] += 1
                correct_count += 1
            
            # Store result
            test_case = {
                "problem_id": i,
                "library": library,
                "prompt": prompt,
                "generated_code": solution,
                "generation_time": generation_time,
                "evaluation": eval_result,
                "correct": eval_result.get("correct", False)
            }
            
            results["test_cases"].append(test_case)
        
        # Calculate overall metrics
        total_problems = len(problems)
        accuracy = correct_count / max(total_problems, 1)
        
        # Calculate library-specific metrics
        library_metrics = {}
        for lib, stats in libraries_results.items():
            lib_accuracy = stats["correct"] / max(stats["total"], 1)
            library_metrics[lib] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": lib_accuracy
            }
        
        results["overall_metrics"] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_problems": total_problems,
            "avg_generation_time": sum(execution_times) / max(len(execution_times), 1),
            "library_metrics": library_metrics
        }
        
        # Save results to file
        agent_id = agent.__class__.__name__.lower()
        results_file = os.path.join(self.results_dir, f"ds1000_{agent_id}_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        if verbose:
            logger.info(f"DS-1000 evaluation complete. Accuracy: {accuracy:.4f} ({correct_count}/{total_problems})")
            for lib, metrics in library_metrics.items():
                logger.info(f"  {lib}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            
        return results