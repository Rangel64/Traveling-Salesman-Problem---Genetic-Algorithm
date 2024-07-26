import random
import numpy as np

def fitness(x):
    # return 3*np.e**(-0.5*(x**2))*np.sin(10*x) -(x**2) + 3
    return -np.abs(x * np.sin(np.sqrt(np.abs(x))))

def crossover(parent1, parent2):
    # Ponto de cruzamento
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(chromosome, mutation_rate=0.01):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = '1' if mutated_chromosome[i] == '0' else '0'
    return ''.join(mutated_chromosome)

# Parâmetros do algoritmo
population_size = 1000
chromosome_length = 20  # 10 bits para a parte inteira e 10 bits para a parte decimal
mutation_rate = 0.01
num_generations = 1000

# Inicialização da população
population = [''.join(random.choices('01', k=chromosome_length)) for _ in range(population_size)]

# Algoritmo genético
for generation in range(num_generations):
    # Avaliação da população
    evaluated_population = [(chromosome, fitness(int(chromosome[:10], 2) + int(chromosome[10:], 2)/1023)) for chromosome in population]
    # Seleção dos pais
    sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
    parents = [chromosome for chromosome, _ in sorted_population[:2]]

    # Reprodução
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = random.choices(parents, k=2)
        child1, child2 = crossover(parent1, parent2)
        offspring.extend([mutation(child1, mutation_rate), mutation(child2, mutation_rate)])

    # Substituição da população
    population = offspring

# Melhor solução encontrada
best_solution, best_fitness = sorted(evaluated_population, key=lambda x: x[1], reverse=True)[0]
print(f'Melhor solucao: f({int(best_solution[:10], 2) + int(best_solution[10:], 2)/1023}) = {best_fitness}')
