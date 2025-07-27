"""
Evolutionary Algorithm Engine for NEXUS

This module implements advanced evolutionary algorithms for:
- Neural architecture search (NAS)
- Hyperparameter optimization
- Strategy evolution
- Multi-objective optimization
- Self-adaptive mutation rates
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import random
from copy import deepcopy
import pickle
from pathlib import Path
import json
import time

from ..config import AIConfig
from ..intelligence.transformer import MarketTransformer  # Updated to use the actual existing class


@dataclass
class Individual:
    """Individual in the evolutionary population."""
    genome: Dict[str, Any]
    fitness: float = 0.0
    age: int = 0
    wins: int = 0
    trades: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    birth_generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not hasattr(self, 'id'):
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for individual."""
        return f"ind_{int(time.time() * 1000000) % 1000000}_{random.randint(1000, 9999)}"

    def get_win_rate(self) -> float:
        """Calculate win rate."""
        return self.wins / max(self.trades, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'genome': self.genome,
            'fitness': self.fitness,
            'age': self.age,
            'wins': self.wins,
            'trades': self.trades,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'birth_generation': self.birth_generation,
            'parent_ids': self.parent_ids,
            'mutations': self.mutations
        }


class EvolutionEngine:
    """Advanced evolutionary optimizer for NEXUS."""

    def __init__(self, config: AIConfig = None, fitness_function: Callable = None, meta_strategy=None, population_size: int = None, mutation_rate: float = None, crossover_rate: float = None):
        self.config = config
        self.fitness_function = fitness_function
        self.meta_strategy = meta_strategy
        # Evolution parameters
        if population_size is not None:
            self.population_size = population_size
        elif isinstance(config, dict) or (hasattr(config, 'get') and callable(config.get)):
            self.population_size = config.get("population_size", 100)
        else:
            self.population_size = getattr(config, "population_size", 100)
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        elif isinstance(config, dict) or (hasattr(config, 'get') and callable(config.get)):
            self.mutation_rate = config.get("mutation_rate", 0.05)
        else:
            self.mutation_rate = getattr(config, "mutation_rate", 0.05)
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        elif isinstance(config, dict) or (hasattr(config, 'get') and callable(config.get)):
            self.crossover_rate = config.get("crossover_rate", 0.7)
        else:
            self.crossover_rate = getattr(config, "crossover_rate", 0.7)
        self.selection_pressure = getattr(config, "selection_pressure", 2.0) if config is not None else 2.0

        # Population management
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

        # Advanced features
        self.adaptive_rates = True
        self.species_count = 5  # Number of species for speciation
        self.elitism_rate = 0.1  # Top % to preserve

        # Genome structure definition
        self.genome_structure = self._define_genome_structure()

    def _define_genome_structure(self) -> Dict[str, Dict[str, Any]]:
        """Define the structure of the genome for neural architecture search."""
        return {
            # Transformer architecture
            'hidden_dim': {'type': 'int', 'range': (256, 1024), 'step': 64},
            'num_layers': {'type': 'int', 'range': (4, 12), 'step': 1},
            'num_heads': {'type': 'int', 'range': (8, 32), 'step': 4},
            'dropout': {'type': 'float', 'range': (0.05, 0.3), 'step': 0.05},

            # Learning parameters
            'learning_rate': {'type': 'float', 'range': (1e-5, 1e-2), 'log': True},
            'batch_size': {'type': 'int', 'range': (32, 128), 'step': 16},
            'sequence_length': {'type': 'int', 'range': (50, 200), 'step': 10},

            # RL parameters
            'gamma': {'type': 'float', 'range': (0.9, 0.999), 'step': 0.01},
            'gae_lambda': {'type': 'float', 'range': (0.9, 0.99), 'step': 0.01},
            'clip_ratio': {'type': 'float', 'range': (0.1, 0.3), 'step': 0.05},

            # Trading strategy parameters
            'risk_threshold': {'type': 'float', 'range': (0.01, 0.05), 'step': 0.005},
            'confidence_threshold': {'type': 'float', 'range': (0.6, 0.9), 'step': 0.05},
            'position_sizing_aggression': {'type': 'float', 'range': (0.5, 2.0), 'step': 0.1},

            # Technical indicator weights
            'rsi_weight': {'type': 'float', 'range': (0.0, 2.0), 'step': 0.1},
            'macd_weight': {'type': 'float', 'range': (0.0, 2.0), 'step': 0.1},
            'bollinger_weight': {'type': 'float', 'range': (0.0, 2.0), 'step': 0.1},
            'volume_weight': {'type': 'float', 'range': (0.0, 1.5), 'step': 0.1},

            # Market regime parameters
            'trend_sensitivity': {'type': 'float', 'range': (0.1, 1.0), 'step': 0.1},
            'volatility_threshold': {'type': 'float', 'range': (0.01, 0.1), 'step': 0.01},
            'regime_change_sensitivity': {'type': 'float', 'range': (0.1, 0.9), 'step': 0.1},
        }

    def create_random_individual(self) -> Individual:
        """Create a random individual with valid genome."""
        genome = {}

        for param_name, param_info in self.genome_structure.items():
            if param_info['type'] == 'int':
                min_val, max_val = param_info['range']
                step = param_info.get('step', 1)
                value = random.randrange(min_val, max_val + 1, step)

            elif param_info['type'] == 'float':
                min_val, max_val = param_info['range']

                if param_info.get('log', False):
                    # Log-scale sampling for learning rates
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    value = 10 ** np.random.uniform(log_min, log_max)
                else:
                    step = param_info.get('step', 0.01)
                    n_steps = int((max_val - min_val) / step)
                    value = min_val + random.randint(0, n_steps) * step
                    value = round(value, 3)

            genome[param_name] = value

        return Individual(
            genome=genome,
            birth_generation=self.generation
        )

    def initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            self.population.append(individual)

    def mutate(self, individual: Individual, mutation_strength: float = 1.0) -> Individual:
        """Mutate an individual's genome."""
        mutated = deepcopy(individual)
        mutated.id = mutated._generate_id()
        mutated.parent_ids = [individual.id]
        mutated.birth_generation = self.generation
        mutated.mutations = []

        # Adaptive mutation rate based on fitness diversity
        diversity = self._calculate_population_diversity()
        adaptive_rate = self.mutation_rate * (1.0 + (1.0 - diversity)) * mutation_strength

        for param_name, param_info in self.genome_structure.items():
            if random.random() < adaptive_rate:
                mutated.mutations.append(param_name)

                if param_info['type'] == 'int':
                    min_val, max_val = param_info['range']
                    step = param_info.get('step', 1)
                    current_val = mutated.genome[param_name]

                    # Gaussian mutation with bounds
                    mutation_strength_val = max(1, int(mutation_strength * step * 3))
                    new_val = current_val + random.randint(-mutation_strength_val, mutation_strength_val)
                    new_val = max(min_val, min(max_val, new_val))

                    # Ensure step compliance
                    new_val = min_val + ((new_val - min_val) // step) * step
                    mutated.genome[param_name] = new_val

                elif param_info['type'] == 'float':
                    min_val, max_val = param_info['range']
                    current_val = mutated.genome[param_name]

                    if param_info.get('log', False):
                        # Log-space mutation for learning rates
                        log_val = np.log10(current_val)
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        log_range = log_max - log_min
                        mutation = np.random.normal(0, log_range * 0.1 * mutation_strength)
                        new_log_val = np.clip(log_val + mutation, log_min, log_max)
                        new_val = 10 ** new_log_val
                    else:
                        step = param_info.get('step', 0.01)
                        range_size = max_val - min_val
                        mutation = np.random.normal(0, range_size * 0.1 * mutation_strength)
                        new_val = np.clip(current_val + mutation, min_val, max_val)

                        # Ensure step compliance
                        new_val = min_val + round((new_val - min_val) / step) * step

                    mutated.genome[param_name] = round(new_val, 6)

        return mutated

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        child1.id = child1._generate_id()
        child2.id = child2._generate_id()
        child1.parent_ids = [parent1.id, parent2.id]
        child2.parent_ids = [parent1.id, parent2.id]
        child1.birth_generation = self.generation
        child2.birth_generation = self.generation

        # Multi-point crossover
        crossover_points = sorted(random.sample(
            range(len(self.genome_structure)),
            k=max(1, len(self.genome_structure) // 3)
        ))

        param_names = list(self.genome_structure.keys())
        swap = False

        for i, param_name in enumerate(param_names):
            if i in crossover_points:
                swap = not swap

            if swap:
                child1.genome[param_name], child2.genome[param_name] = \
                    child2.genome[param_name], child1.genome[param_name]

        return child1, child2

    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def species_distance(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> float:
        """Calculate distance between two genomes for speciation."""
        distance = 0.0

        for param_name in self.genome_structure:
            val1, val2 = genome1[param_name], genome2[param_name]
            param_info = self.genome_structure[param_name]

            if param_info['type'] == 'int':
                range_size = param_info['range'][1] - param_info['range'][0]
                normalized_diff = abs(val1 - val2) / range_size
            else:  # float
                if param_info.get('log', False):
                    log1, log2 = np.log10(val1), np.log10(val2)
                    log_range = np.log10(param_info['range'][1]) - np.log10(param_info['range'][0])
                    normalized_diff = abs(log1 - log2) / log_range
                else:
                    range_size = param_info['range'][1] - param_info['range'][0]
                    normalized_diff = abs(val1 - val2) / range_size

            distance += normalized_diff ** 2

        return np.sqrt(distance / len(self.genome_structure))

    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity for adaptive mutation."""
        if len(self.population) < 2:
            return 1.0

        total_distance = 0.0
        count = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self.species_distance(
                    self.population[i].genome,
                    self.population[j].genome
                )
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 1.0

    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation."""
        # Evaluate fitness for all individuals
        for individual in self.population:
            if individual.fitness == 0.0:  # Not evaluated yet
                individual.fitness = self.fitness_function(individual.genome)

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = deepcopy(self.population[0])

        # Record statistics
        fitnesses = [ind.fitness for ind in self.population]
        diversity = self._calculate_population_diversity()

        self.fitness_history.append(np.mean(fitnesses))
        self.diversity_history.append(diversity)

        # Create next generation
        new_population = []

        # Elitism - preserve top individuals
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        new_population.extend(deepcopy(ind) for ind in self.population[:elite_count])

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(new_population) < self.population_size - 1:
                # Crossover
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)

                # Possible mutation
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)

                new_population.extend([child1, child2])
            else:
                # Mutation only
                parent = self.tournament_selection()
                child = self.mutate(parent)
                new_population.append(child)

        # Trim to exact population size
        new_population = new_population[:self.population_size]

        # Age existing individuals
        for ind in new_population:
            if ind.birth_generation < self.generation:
                ind.age += 1

        self.population = new_population
        self.generation += 1

        return {
            'generation': self.generation,
            'best_fitness': self.best_individual.fitness,
            'mean_fitness': np.mean(fitnesses),
            'diversity': diversity,
            'population_size': len(self.population)
        }

    def save_population(self, filepath: Path):
        """Save population to file."""
        data = {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_population(self, filepath: Path):
        """Load population from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.generation = data['generation']
        self.fitness_history = data['fitness_history']
        self.diversity_history = data['diversity_history']

        # Reconstruct population
        self.population = []
        for ind_data in data['population']:
            ind = Individual(genome=ind_data['genome'])
            for key, value in ind_data.items():
                if key != 'genome':
                    setattr(ind, key, value)
            self.population.append(ind)

        # Reconstruct best individual
        if data['best_individual']:
            best_data = data['best_individual']
            self.best_individual = Individual(genome=best_data['genome'])
            for key, value in best_data.items():
                if key != 'genome':
                    setattr(self.best_individual, key, value)


class NeuralArchitectureSearch:
    """Neural Architecture Search for NEXUS models."""

    def __init__(self, config: AIConfig):
        self.config = config
        self.search_space = self._define_search_space()

    def _define_search_space(self) -> Dict[str, Any]:
        """Define the neural architecture search space."""
        return {
            'transformer_layers': list(range(4, 13)),
            'attention_heads': [8, 12, 16, 20, 24, 32],
            'hidden_dims': [256, 384, 512, 640, 768, 1024],
            'feed_forward_dims': [512, 1024, 1536, 2048, 3072, 4096],
            'activation_functions': ['relu', 'gelu', 'swish', 'mish'],
            'normalization': ['layer_norm', 'batch_norm', 'rms_norm'],
            'dropout_rates': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'attention_dropout': [0.0, 0.05, 0.1, 0.15, 0.2],
        }

    def create_model_from_genome(self, genome: Dict[str, Any]) -> MarketTransformer:
        """Create a model instance from a genome."""
        # Extract architecture parameters
        model_config = AIConfig(
            hidden_dim=genome.get('hidden_dim', 512),
            num_layers=genome.get('num_layers', 8),
            num_heads=genome.get('num_heads', 16),
            dropout=genome.get('dropout', 0.1),
            learning_rate=genome.get('learning_rate', 1e-4),
            batch_size=genome.get('batch_size', 64),
            sequence_length=genome.get('sequence_length', 100),
            gamma=genome.get('gamma', 0.99),
            gae_lambda=genome.get('gae_lambda', 0.95),
            clip_ratio=genome.get('clip_ratio', 0.2)
        )

        # Assume input_dim based on number of technical indicators
        input_dim = 20  # Basic OHLCV + technical indicators

        return MarketTransformer(
            config=model_config,
            input_dim=input_dim,
            num_actions=3,  # Hold, Call, Put
            num_assets=5
        )


import random
from typing import List, Dict

class NeuroEvolution:
    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self._initialize_population()

    def _initialize_population(self) -> List[Dict]:
        """Initialize a population with random strategies."""
        return [self._random_strategy() for _ in range(self.population_size)]

    def _random_strategy(self) -> Dict:
        """Generate a random strategy."""
        return {
            "risk": random.uniform(0.01, 0.1),
            "timeframe": random.choice([60, 300, 900]),
            "take_profit": random.uniform(0.02, 0.1),
            "stop_loss": random.uniform(0.01, 0.05),
        }

    def evolve(self):
        """Perform evolution: selection, crossover, and mutation."""
        selected = self._select()
        offspring = self._crossover(selected)
        mutated = self._mutate(offspring)
        self.population = mutated

    def _select(self) -> List[Dict]:
        """Select the top strategies based on fitness."""
        return sorted(self.population, key=lambda x: self._fitness(x), reverse=True)[:self.population_size // 2]

    def _fitness(self, strategy: Dict) -> float:
        """Calculate fitness based on strategy performance."""
        return random.uniform(0, 1)  # Placeholder for actual performance metrics

    def _crossover(self, selected: List[Dict]) -> List[Dict]:
        """Perform crossover to generate offspring."""
        offspring = []
        for _ in range(len(selected) // 2):
            parent1, parent2 = random.sample(selected, 2)
            child = {
                "risk": (parent1["risk"] + parent2["risk"]) / 2,
                "timeframe": random.choice([parent1["timeframe"], parent2["timeframe"]]),
                "take_profit": (parent1["take_profit"] + parent2["take_profit"]) / 2,
                "stop_loss": (parent1["stop_loss"] + parent2["stop_loss"]) / 2,
            }
            offspring.append(child)
        return offspring

    def _mutate(self, offspring: List[Dict]) -> List[Dict]:
        """Perform mutation on offspring."""
        for child in offspring:
            if random.random() < self.mutation_rate:
                child["risk"] += random.uniform(-0.01, 0.01)
                child["take_profit"] += random.uniform(-0.01, 0.01)
                child["stop_loss"] += random.uniform(-0.01, 0.01)
        return offspring
