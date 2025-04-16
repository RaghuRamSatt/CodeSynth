"""
Command line interface for model evaluation
"""

import argparse
import os
import logging
import json
import sys
import pandas as pd

from agents.claude_agent import ClaudeAgent
from agents.base_agent import BaseAgent
from evaluation.evaluator import ModelEvaluator
from evaluation.test_suite import TestSuite
from evaluation.ds1000_evaluator import DS1000Evaluator


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_agents():
    """Get dictionary of available agent instances."""
    agents = {}
    
    # Try to initialize Claude agent
    claude_agent = ClaudeAgent()
    if claude_agent.initialize():
        agents["claude"] = claude_agent
        logger.info("Claude agent initialized successfully")
    else:
        logger.warning("Failed to initialize Claude agent")
    
    # Add other agents as they are implemented
    # For example:
    # openai_agent = OpenAIAgent()
    # if openai_agent.initialize():
    #     agents["openai"] = openai_agent
    
    return agents

def run_evaluation(args):
    """Run the evaluation based on command line arguments."""
    # Get available agents
    agents = get_available_agents()
    
    if not agents:
        logger.error("No agents could be initialized. Check API keys and connections.")
        return False
    
    # Initialize the evaluator
    evaluator = ModelEvaluator(agents)
    test_suite = TestSuite()
    
    # Get test cases
    if args.categories:
        categories = args.categories.split(',')
    else:
        categories = None
        
    test_cases = test_suite.get_test_cases(categories=categories, limit=args.limit)
    
    if not test_cases:
        logger.error("No test cases found for the specified categories.")
        return False
        
    logger.info(f"Running evaluation with {len(test_cases)} test cases")
    
    # Run comparison if multiple agents or single agent evaluation
    if args.compare and len(agents) > 1:
        logger.info(f"Comparing {len(agents)} models: {', '.join(agents.keys())}")
        results = evaluator.compare_models(test_cases=test_cases, verbose=args.verbose)
        
        # Generate comparison plots
        plot_files = evaluator.generate_comparison_plots(results)
        logger.info(f"Comparative evaluation complete. Results saved to {evaluator.results_dir}")
        logger.info(f"Generated {len(plot_files)} comparison plots")
        
    elif args.agent in agents:
        logger.info(f"Evaluating single agent: {args.agent}")
        results = evaluator.evaluate_model(args.agent, test_cases, verbose=args.verbose)
        logger.info(f"Evaluation complete for {args.agent}. Results saved to {evaluator.results_dir}")
        
    else:
        logger.error(f"Agent '{args.agent}' not found or not initialized.")
        logger.info(f"Available agents: {', '.join(agents.keys())}")
        return False
    
    return True

def run_ds1000_evaluation(args):
    """Run evaluation on DS-1000 benchmark."""
    # Get available agents
    agents = get_available_agents()
    
    if not agents:
        logger.error("No agents could be initialized. Check API keys and connections.")
        return False
    
    # Initialize the DS-1000 evaluator
    ds1000_evaluator = DS1000Evaluator()
    
    # Parse libraries
    libraries = None
    if args.libraries:
        libraries = args.libraries.split(',')
    
    # Run evaluation
    if args.compare and len(agents) > 1:
        logger.info(f"Comparing {len(agents)} models on DS-1000: {', '.join(agents.keys())}")
        results = {}
        
        for agent_id, agent in agents.items():
            logger.info(f"Evaluating {agent_id} on DS-1000")
            result = ds1000_evaluator.evaluate_model(
                agent,
                libraries=libraries,
                limit=args.limit,
                verbose=args.verbose
            )
            results[agent_id] = result
        
        # Show comparison table
        comparison = []
        for agent_id, result in results.items():
            comparison.append({
                "Model": agent_id,
                "Accuracy": f"{result['overall_metrics']['accuracy']:.4f}",
                "Correct": result['overall_metrics']['correct_count'],
                "Total": result['overall_metrics']['total_problems']
            })
        
        # Print as table
        comparison_df = pd.DataFrame(comparison)
        logger.info("\nDS-1000 Comparison Results:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
            
    elif args.agent in agents:
        logger.info(f"Evaluating {args.agent} on DS-1000")
        result = ds1000_evaluator.evaluate_model(
            agents[args.agent],
            libraries=libraries,
            limit=args.limit,
            verbose=args.verbose
        )
        
        # Show library-specific results
        if result:
            logger.info("\nLibrary-specific results:")
            lib_results = []
            for lib, metrics in result["overall_metrics"]["library_metrics"].items():
                lib_results.append({
                    "Library": lib,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "Correct": metrics['correct'],
                    "Total": metrics['total']
                })
            
            lib_df = pd.DataFrame(lib_results)
            logger.info(f"\n{lib_df.to_string(index=False)}")
    
    else:
        logger.error(f"Agent '{args.agent}' not found or not initialized.")
        logger.info(f"Available agents: {', '.join(agents.keys())}")
        return False
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate LLM models for data analysis code generation")
    
    # Existing arguments
    parser.add_argument("--agent", type=str, default="claude", 
                       help="Agent to evaluate (default: claude)")
    
    parser.add_argument("--compare", action="store_true",
                       help="Compare all available agents")
    
    parser.add_argument("--categories", type=str,
                       help="Comma-separated list of test categories to include")
    
    parser.add_argument("--limit", type=int, default=5,
                       help="Maximum number of test cases to evaluate (default: 5)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress information")
    
    # DS-1000 arguments
    parser.add_argument("--ds1000", action="store_true",
                       help="Use DS-1000 benchmark for evaluation")
    
    parser.add_argument("--libraries", type=str,
                       help="Comma-separated list of libraries for DS-1000 (numpy,pandas,matplotlib,sklearn,pytorch,tensorflow,scipy)")
    
    args = parser.parse_args()
    
    # Run appropriate evaluation
    if args.ds1000:
        success = run_ds1000_evaluation(args)
    else:
        success = run_evaluation(args)
    
    if success:
        logger.info("Evaluation completed successfully")
        return 0
    else:
        logger.error("Evaluation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())