import random as rd
import networkx as nx
import matplotlib.pyplot as plt
import folium
import pandas as pd
import numpy as np

file_path = 'Distance_Matrix.csv'
distances_df = pd.read_csv(file_path, index_col=0)

p = distances_df.values
c = distances_df.index.tolist()

def fitness(cromossomo, p, max_distance):
    penalty_start_end = 1000000 
    penalty_distance = 1000 

    if cromossomo[0] != 0 or cromossomo[-1] != 0:
        return penalty_start_end

    z = 0
    for i in range(len(cromossomo) - 1):
        m = cromossomo[i]
        n = cromossomo[i + 1]
        distance = p[m][n]
        z += distance
        if distance > max_distance:
            z += penalty_distance

    return z

def crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(rd.sample(range(1, size - 1), 2))
    
    child1 = [None] * size
    child2 = [None] * size
    
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]
    
    def fill_child(child, parent, point1, point2):
        for i in range(point1, point2):
            if parent[i] not in child:
                pos = i
                while child[pos] is not None:
                    pos = parent.index(child[pos])
                child[pos] = parent[i]
        
        for i in range(size):
            if child[i] is None:
                child[i] = parent[i]

    fill_child(child1, parent2, point1, point2)
    fill_child(child2, parent1, point1, point2)

    return child1, child2

def mutation(chromosome, mutation_rate=0.00001):
    if rd.uniform(0, 1) < mutation_rate:
        i, j = rd.sample(range(1, len(chromosome) - 1), 2) 
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def tournament_selection(population, tournament_size=5):
    
    tournament = rd.sample(population, tournament_size)
    
    sorted_population = sorted(tournament, key=lambda x: x[1], reverse=False)

    parents = [chromosome for chromosome, _ in sorted_population[:2]]
    
    return parents

max_distance = 500


population_size = 10000
mutation_rate = 0.001
patience = 25
last_fitness = None
num_generations = 1000
counter = 0


population = [[0] + rd.sample(range(1, len(c)), len(c)-1) + [0] for _ in range(population_size)]


for generation in range(num_generations):
    
    evaluated_population = [(chromosome, fitness(chromosome, p, max_distance)) for chromosome in population]
    
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = tournament_selection(evaluated_population)
        child1, child2 = crossover(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)

  
    population = [mutation(chromosome, mutation_rate) for chromosome in offspring]

    
    evaluated_population = [(chromosome, fitness(chromosome, p, max_distance)) for chromosome in population]

    
    best_solution = sorted(evaluated_population, key=lambda x: x[1], reverse=False)[0]
    
    if(best_solution==last_fitness):
        counter = counter + 1
        if(counter>=patience):
            mutation_rate = 1.1
    else:
        mutation_rate=0.001
        
    last_fitness = best_solution
    
    print(f'Generation {generation}: Best solution = {best_solution[0]}, Fitness = {best_solution[1]}, Mutation Rate = {mutation_rate}')

# Melhor solução final
best_solution = sorted(evaluated_population, key=lambda x: x[1], reverse=False)[0]
best_chromosome = best_solution[0]
best_chromosome_cities = [c[i] for i in best_chromosome]

G = nx.Graph()

for city in c:
    G.add_node(city)

for i in range(len(p)):
    for j in range(i + 1, len(p[i])):
        if p[i][j] > 0: 
            G.add_edge(c[i], c[j], weight=p[i][j])

edges_in_route = [(best_chromosome_cities[i], best_chromosome_cities[i + 1]) for i in range(len(best_chromosome_cities) - 1)]

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(30, 20))

nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if (u, v) not in edges_in_route and (v, u) not in edges_in_route], edge_color='grey', width=2)

nx.draw_networkx_edges(G, pos, edgelist=edges_in_route, edge_color='red', width=2)

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})

plt.title('Grafo de Cidades com Rota Específica Destacada')
plt.show()

coords = {
    "Uberaba": (-19.7472, -47.9381),
    "Uberlândia": (-18.9128, -48.2755),
    "Araguari": (-18.6467, -48.1936),
    "Araxá": (-19.5902, -46.9433),
    "Patos de Minas": (-18.5789, -46.5183),
    "Ituiutaba": (-18.9746, -49.4653),
    "Monte Carmelo": (-18.7302, -47.4913),
    "Frutal": (-20.0244, -48.9351),
    "Prata": (-19.3084, -48.9276),
    "Iturama": (-19.7279, -50.1959),
    "Campina Verde": (-19.5386, -49.4863),
    "Sacramento": (-19.8624, -47.4503),
    "Conceição das Alagoas": (-19.9171, -48.3837),
    "Perdizes": (-19.3511, -47.2962),
    "Ibiá": (-19.4788, -46.5383),
    "Coromandel": (-18.4731, -47.1972),
    "Paracatu": (-17.2252, -46.8711),
    "Vazante": (-17.9829, -46.9053),
    "Serra do Salitre": (-19.1081, -46.6955),
    "Rio Paranaíba": (-19.1866, -46.2454),
    "Santa Vitória": (-18.8412, -50.1208)
}


m = folium.Map(location=coords['Uberaba'], zoom_start=7)

for city, coord in coords.items():
    folium.Marker(location=coord, popup=city).add_to(m)

route_coords = [coords[city] for city in best_chromosome_cities]
folium.PolyLine(route_coords, color='red', weight=5, opacity=0.7).add_to(m)

m.save('rota_especifica1.html')

print(f'Best solution = {best_chromosome_cities}, Fitness = {best_solution[1]}')
