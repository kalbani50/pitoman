import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timezone
import random
from deap import base, creator, tools, algorithms
import pandas as pd
import multiprocessing
from scipy.stats import norm

class GeneticOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.population = None
        self.best_solution = None
        self.fitness_history = []
        self.generation_stats = []
        
    async def optimize_strategy(self, data: pd.DataFrame, strategy_params: Dict) -> Dict:
        """Optimizes trading strategy using genetic algorithm"""
        try:
            # Create types
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Initialize toolbox
            toolbox = self._setup_toolbox(strategy_params)
            
            # Run optimization
            population, logbook = await self._run_optimization(
                toolbox,
                data,
                strategy_params
            )
            
            # Get best solution
            best_individual = tools.selBest(population, k=1)[0]
            
            # Analyze results
            analysis = self._analyze_optimization_results(
                population,
                logbook,
                best_individual
            )
            
            return {
                'best_params': self._decode_individual(
                    best_individual,
                    strategy_params
                ),
                'fitness': best_individual.fitness.values[0],
                'analysis': analysis
            }
            
        except Exception as e:
            logging.error(f"Error optimizing strategy: {e}")
            raise
            
    def _setup_toolbox(self, strategy_params: Dict) -> base.Toolbox:
        """Sets up genetic algorithm toolbox"""
        try:
            toolbox = base.Toolbox()
            
            # Register gene generation functions
            for param, bounds in strategy_params.items():
                if isinstance(bounds, tuple):
                    toolbox.register(
                        f"attr_{param}",
                        random.uniform,
                        bounds[0],
                        bounds[1]
                    )
                    
            # Register individual and population creation
            n_params = len(strategy_params)
            toolbox.register(
                "individual",
                tools.initCycle,
                creator.Individual,
                (getattr(toolbox, f"attr_{param}") for param in strategy_params),
                n=1
            )
            toolbox.register(
                "population",
                tools.initRepeat,
                list,
                toolbox.individual
            )
            
            # Register genetic operators
            toolbox.register(
                "evaluate",
                self._evaluate_individual
            )
            toolbox.register(
                "mate",
                tools.cxTwoPoint
            )
            toolbox.register(
                "mutate",
                tools.mutGaussian,
                mu=0,
                sigma=1,
                indpb=0.2
            )
            toolbox.register(
                "select",
                tools.selTournament,
                tournsize=3
            )
            
            return toolbox
            
        except Exception as e:
            logging.error(f"Error setting up toolbox: {e}")
            raise
            
    async def _run_optimization(self,
                              toolbox: base.Toolbox,
                              data: pd.DataFrame,
                              strategy_params: Dict) -> Tuple[List, tools.Logbook]:
        """Runs genetic optimization"""
        try:
            # Create initial population
            population = toolbox.population(n=self.config['population_size'])
            
            # Create hall of fame
            hof = tools.HallOfFame(1)
            
            # Create statistics tracker
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run evolution
            population, logbook = algorithms.eaSimple(
                population,
                toolbox,
                cxpb=self.config['crossover_prob'],
                mutpb=self.config['mutation_prob'],
                ngen=self.config['n_generations'],
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            
            return population, logbook
            
        except Exception as e:
            logging.error(f"Error running optimization: {e}")
            raise
            
    def _evaluate_individual(self,
                           individual: List,
                           data: pd.DataFrame,
                           strategy_params: Dict) -> Tuple[float]:
        """Evaluates fitness of individual"""
        try:
            # Decode parameters
            params = self._decode_individual(individual, strategy_params)
            
            # Run strategy with parameters
            returns = self._run_strategy_backtest(data, params)
            
            # Calculate fitness metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            profit_factor = self._calculate_profit_factor(returns)
            
            # Combine metrics into single fitness
            fitness = (
                sharpe_ratio * self.config['sharpe_weight'] +
                (1 - max_drawdown) * self.config['drawdown_weight'] +
                profit_factor * self.config['profit_weight']
            )
            
            return (fitness,)
            
        except Exception as e:
            logging.error(f"Error evaluating individual: {e}")
            raise
            
    def _decode_individual(self,
                          individual: List,
                          strategy_params: Dict) -> Dict:
        """Decodes individual into strategy parameters"""
        return {
            param: value
            for param, value in zip(strategy_params.keys(), individual)
        }
        
    def _analyze_optimization_results(self,
                                   population: List,
                                   logbook: tools.Logbook,
                                   best_individual: List) -> Dict:
        """Analyzes optimization results"""
        try:
            # Get statistics
            gen_stats = logbook.select("gen")
            fit_stats = logbook.select("max")
            
            # Calculate convergence
            convergence = self._calculate_convergence(fit_stats)
            
            # Analyze population diversity
            diversity = self._analyze_population_diversity(population)
            
            # Generate parameter distribution
            param_dist = self._analyze_parameter_distribution(population)
            
            return {
                'convergence': convergence,
                'diversity': diversity,
                'parameter_distribution': param_dist,
                'generations': len(gen_stats),
                'final_fitness_stats': logbook[-1]
            }
            
        except Exception as e:
            logging.error(f"Error analyzing optimization results: {e}")
            raise
            
    def _calculate_convergence(self, fitness_history: List) -> Dict:
        """Calculates optimization convergence metrics"""
        try:
            # Calculate improvement rate
            improvements = np.diff(fitness_history)
            improvement_rate = np.mean(improvements > 0)
            
            # Calculate convergence speed
            convergence_speed = len(fitness_history) / np.argmax(fitness_history)
            
            # Check for premature convergence
            premature_convergence = self._detect_premature_convergence(
                fitness_history
            )
            
            return {
                'improvement_rate': improvement_rate,
                'convergence_speed': convergence_speed,
                'premature_convergence': premature_convergence
            }
            
        except Exception as e:
            logging.error(f"Error calculating convergence: {e}")
            raise
            
    async def generate_optimization_report(self) -> Dict:
        """Generates comprehensive optimization report"""
        try:
            report = {
                'optimization_results': {
                    'best_solution': self.best_solution,
                    'fitness_history': self.fitness_history,
                    'generation_stats': self.generation_stats
                },
                'analysis': {
                    'convergence': self._calculate_convergence(
                        self.fitness_history
                    ),
                    'population_diversity': self._analyze_population_diversity(
                        self.population
                    ),
                    'parameter_sensitivity': self._analyze_parameter_sensitivity()
                },
                'recommendations': self._generate_optimization_recommendations(),
                'timestamp': datetime.now(timezone.utc)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating optimization report: {e}")
            raise
