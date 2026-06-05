"""
Evolutionary Graph Mutation — Phase 6.

Maintains a population of PC graph topologies and evolves them
via mutation, crossover, and tournament selection.

Genome encodes:
- hidden_dim: size of hidden layers
- n_hidden_layers: 1 or 2
- eta_infer, eta_learn: learning hyperparameters
- activation: 'tanh' or 'identity'
- use_skip: direct obs_in → obs_out edge

Usage:
    from fabricpc_extensions.evolution import Population, PCGenome

    pop = Population(size=10, seed=42)
    pop.evaluate_all(...)
    pop.evolve(generations=20)
"""

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from fabricpc_extensions.ansi import C, dim, metric


@dataclass
class PCGenome:
    """A genome encoding a predictive-coding network topology."""

    hidden_dim: int = 16
    n_hidden_layers: int = 1
    eta_infer: float = 0.05
    eta_learn: float = 0.001
    activation: str = "tanh"
    use_skip: bool = False

    def mutate(self, rng: random.Random, rate: float = 0.3):
        """Apply random mutations to this genome in-place."""
        if rng.random() < rate:
            self.hidden_dim += rng.choice([-2, 2])
            self.hidden_dim = max(4, min(64, self.hidden_dim))
        if rng.random() < rate:
            self.n_hidden_layers = 2 if self.n_hidden_layers == 1 else 1
        if rng.random() < rate:
            self.eta_infer *= rng.uniform(0.8, 1.25)
            self.eta_infer = max(0.005, min(0.5, self.eta_infer))
        if rng.random() < rate:
            self.eta_learn *= rng.uniform(0.8, 1.25)
            self.eta_learn = max(0.0001, min(0.01, self.eta_learn))
        if rng.random() < rate:
            self.activation = "identity" if self.activation == "tanh" else "tanh"
        if rng.random() < rate:
            self.use_skip = not self.use_skip

    def crossover(self, other: "PCGenome", rng: random.Random) -> "PCGenome":
        """Return a new genome by crossing over with another."""
        child = PCGenome()
        child.hidden_dim = rng.choice([self.hidden_dim, other.hidden_dim])
        child.n_hidden_layers = rng.choice([self.n_hidden_layers, other.n_hidden_layers])
        child.eta_infer = (self.eta_infer + other.eta_infer) / 2
        child.eta_learn = (self.eta_learn + other.eta_learn) / 2
        child.activation = rng.choice([self.activation, other.activation])
        child.use_skip = rng.choice([self.use_skip, other.use_skip])
        return child

    def to_dict(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim,
            "n_hidden_layers": self.n_hidden_layers,
            "eta_infer": round(self.eta_infer, 4),
            "eta_learn": round(self.eta_learn, 6),
            "activation": self.activation,
            "use_skip": self.use_skip,
        }


class Population:
    """
    Manages a population of genomes with evaluation and evolution.

    Each generation:
    1. Evaluate fitness of all individuals
    2. Tournament selection for parents
    3. Crossover and mutation to create offspring
    4. Elitism: keep top-k individuals
    """

    def __init__(
        self,
        size: int = 20,
        seed: int = 42,
        elite_size: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
    ):
        self.size = size
        self.elite_size = min(elite_size, size)
        self.tournament_size = min(tournament_size, size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = random.Random(seed)

        self.genomes: List[PCGenome] = [self._random_genome() for _ in range(size)]
        self.fitness: List[float] = [0.0] * size
        self.generation: int = 0
        self.history: List[Dict] = []

    def _random_genome(self) -> PCGenome:
        return PCGenome(
            hidden_dim=self.rng.choice([8, 12, 16, 20, 24]),
            n_hidden_layers=self.rng.choice([1, 2]),
            eta_infer=self.rng.uniform(0.01, 0.1),
            eta_learn=self.rng.uniform(0.0005, 0.005),
            activation=self.rng.choice(["tanh", "identity"]),
            use_skip=self.rng.random() < 0.3,
        )

    def evaluate_all(
        self,
        fitness_fn: Callable[[PCGenome, int], float],
        verbose: bool = True,
        eval_episodes: int = 1,
    ):
        """Evaluate every genome in the population."""
        new_fitness = []
        for i, genome in enumerate(self.genomes):
            fit = fitness_fn(genome, eval_episodes)
            new_fitness.append(fit)
            if verbose and (i + 1) % 5 == 0:
                print(metric(f"Evaluated {i+1}/{self.size}", f"fitness={fit:.4f}", C.GREEN))
        self.fitness = new_fitness

    def _tournament_select(self) -> PCGenome:
        """Select a genome via tournament selection."""
        contestants = self.rng.sample(range(self.size), self.tournament_size)
        winner = max(contestants, key=lambda i: self.fitness[i])
        return deepcopy(self.genomes[winner])

    def evolve(self):
        """Create the next generation."""
        best_idx = max(range(self.size), key=lambda i: self.fitness[i])
        best_genome = deepcopy(self.genomes[best_idx])
        best_fitness = self.fitness[best_idx]

        ranked = sorted(
            range(self.size), key=lambda i: self.fitness[i], reverse=True
        )
        elites = [deepcopy(self.genomes[i]) for i in ranked[: self.elite_size]]

        next_genomes = elites[:]

        while len(next_genomes) < self.size:
            parent1 = self._tournament_select()
            if self.rng.random() < self.crossover_rate:
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2, self.rng)
            else:
                child = deepcopy(parent1)
            child.mutate(self.rng, self.mutation_rate)
            next_genomes.append(child)

        self.genomes = next_genomes[: self.size]
        self.generation += 1

        self.history.append({
            "generation": self.generation,
            "best_fitness": round(best_fitness, 4),
            "avg_fitness": round(float(np.mean(self.fitness)), 4),
            "best_genome": best_genome.to_dict(),
        })

        print(f"  Generation {self.generation}: "
              f"best={best_fitness:.4f} "
              f"avg={float(np.mean(self.fitness)):.4f}")

    def stats(self) -> Dict:
        return {
            "generation": self.generation,
            "population_size": self.size,
            "best_fitness": max(self.fitness) if self.fitness else 0.0,
            "avg_fitness": float(np.mean(self.fitness)) if self.fitness else 0.0,
            "std_fitness": float(np.std(self.fitness)) if self.fitness else 0.0,
        }
