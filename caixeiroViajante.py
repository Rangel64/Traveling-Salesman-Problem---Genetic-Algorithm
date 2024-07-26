import random as rd
import networkx as nx
import matplotlib.pyplot as plt
import folium
import pandas as pd
# import numpy as np

file_path = 'Distance_Matrix.csv'
distances_df = pd.read_csv(file_path, index_col=0)

p = distances_df.values
c = distances_df.index.tolist()

def fitness(cromossomo, p, max_distance):
    penalty_start_end = 1000000  # Penalidade para indivíduos que não começam/terminam com Uberaba
    penalty_distance = 1000  # Penalidade para distâncias maiores que 200 km

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

def crossover(parent1, parent2): #Operador PMX – PartiallyMatched Crossover
    size = len(parent1)
    point1, point2 = sorted(rd.sample(range(1, size - 1), 2))  # Evitar o primeiro e o último elemento (0)
    
    # Inicializar os filhos com None
    child1 = [None] * size
    child2 = [None] * size
    
    # Copiar segmento dos pais para os filhos
    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]
    
    # Função para preencher os filhos fora do segmento copiado
    def fill_child(child, parent, point1, point2):
        for i in range(point1, point2):
            if parent[i] not in child:
                pos = i
                while child[pos] is not None:
                    pos = parent.index(child[pos])
                child[pos] = parent[i]
        
        # Preencher os None restantes
        for i in range(size):
            if child[i] is None:
                child[i] = parent[i]

    # Preencher os filhos
    fill_child(child1, parent2, point1, point2)
    fill_child(child2, parent1, point1, point2)

    return child1, child2

def mutation(chromosome, mutation_rate=0.00001):
    if rd.uniform(0, 1) < mutation_rate:
        i, j = rd.sample(range(1, len(chromosome) - 1), 2)  # Evitar o primeiro e o último elemento (0)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def tournament_selection(population, tournament_size=5):
    
    tournament = rd.sample(population, tournament_size)
    
    sorted_population = sorted(tournament, key=lambda x: x[1], reverse=False)

    parents = [chromosome for chromosome, _ in sorted_population[:2]]
    
    return parents

# p = [
#     [0, 108, 147, 109, 221, 192, 188, 98, 123, 246, 158, 74, 62, 146, 156, 227, 326, 327, 186, 300, 215],
#     [108, 0, 29, 172, 233, 142, 98, 190, 78, 295, 203, 182, 161, 218, 239, 174, 379, 381, 252, 314, 167],
#     [147, 29, 0, 211, 262, 170, 99, 229, 106, 324, 231, 222, 201, 247, 268, 176, 408, 409, 281, 343, 195],
#     [109, 172, 211, 0, 170, 282, 250, 207, 207, 328, 276, 109, 138, 83, 83, 112, 233, 234, 112, 231, 275],
#     [221, 233, 262, 170, 0, 278, 139, 320, 177, 418, 325, 248, 227, 160, 120, 88, 183, 139, 56, 119, 353],
#     [192, 142, 170, 282, 278, 0, 121, 262, 90, 197, 105, 250, 229, 286, 307, 189, 419, 391, 283, 349, 26],
#     [188, 98, 99, 250, 139, 121, 0, 258, 104, 318, 226, 216, 195, 241, 162, 87, 226, 228, 99, 161, 147],
#     [98, 190, 229, 207, 320, 262, 258, 0, 203, 235, 170, 110, 125, 168, 249, 320, 416, 418, 278, 378, 236],
#     [123, 78, 106, 207, 177, 90, 104, 203, 0, 285, 193, 197, 176, 218, 239, 182, 387, 380, 241, 303, 108],
#     [246, 295, 324, 328, 418, 197, 318, 235, 285, 0, 196, 320, 299, 356, 377, 448, 544, 546, 406, 506, 223],
#     [158, 203, 231, 276, 325, 105, 226, 170, 193, 196, 0, 232, 211, 268, 289, 360, 456, 458, 318, 418, 132],
#     [74, 182, 222, 109, 248, 250, 216, 110, 197, 320, 232, 0, 39, 148, 128, 199, 295, 297, 157, 257, 291],
#     [62, 161, 201, 138, 227, 229, 195, 125, 176, 299, 211, 39, 0, 127, 147, 218, 314, 316, 176, 276, 270],
#     [146, 218, 247, 83, 160, 286, 241, 168, 218, 356, 268, 148, 127, 0, 100, 111, 206, 207, 87, 187, 254],
#     [156, 239, 268, 83, 120, 307, 162, 249, 239, 377, 289, 128, 147, 100, 0, 59, 145, 146, 79, 119, 275],
#     [227, 174, 176, 112, 88, 189, 87, 320, 182, 448, 360, 199, 218, 111, 59, 0, 86, 87, 41, 80, 270],
#     [326, 379, 408, 233, 183, 419, 226, 416, 387, 544, 456, 295, 314, 206, 145, 86, 0, 44, 45, 111, 509],
#     [327, 381, 409, 234, 139, 391, 228, 418, 380, 546, 458, 297, 316, 207, 146, 87, 44, 0, 48, 110, 511],
#     [186, 252, 281, 112, 56, 283, 99, 278, 241, 406, 318, 157, 176, 87, 79, 41, 45, 48, 0, 86, 314],
#     [300, 314, 343, 231, 119, 349, 161, 378, 303, 506, 418, 257, 276, 187, 119, 80, 111, 110, 86, 0, 422],
#     [215, 167, 195, 275, 353, 26, 147, 236, 108, 223, 132, 291, 270, 254, 275, 270, 509, 511, 314, 422, 0]
# ]

max_distance = 500

# Parâmetros do algoritmo
population_size = 10000
mutation_rate = 0.001
patience = 25
last_fitness = None
num_generations = 1000
counter = 0

# Lista de cidades
# c = [
#     "Uberaba", "Uberlândia", "Araguari", "Araxá", "Patos de Minas", "Ituiutaba", 
#     "Monte Carmelo", "Frutal", "Prata", "Iturama", "Campina Verde", "Sacramento", 
#     "Conceição das Alagoas", "Perdizes", "Ibiá", "Coromandel", "Paracatu", 
#     "Vazante", "Serra do Salitre", "Rio Paranaíba", "Santa Vitória"
# ]

# Inicialização da população com Uberaba no início e no fim
population = [[0] + rd.sample(range(1, len(c)), len(c)-1) + [0] for _ in range(population_size)]

# Algoritmo genético
for generation in range(num_generations):
    # Avaliação da população
    evaluated_population = [(chromosome, fitness(chromosome, p, max_distance)) for chromosome in population]
    
    # Seleção dos pais
    # sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=False)
    # parents = [chromosome for chromosome, _ in sorted_population[:2]]
    
    # Reprodução
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = tournament_selection(evaluated_population)
        child1, child2 = crossover(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)

    # Mutação
    population = [mutation(chromosome, mutation_rate) for chromosome in offspring]

    # Avaliação final
    evaluated_population = [(chromosome, fitness(chromosome, p, max_distance)) for chromosome in population]

    # Melhor solução
    best_solution = sorted(evaluated_population, key=lambda x: x[1], reverse=False)[0]
    
    if(best_solution==last_fitness):
        counter = counter + 1
        if(counter>=patience):
            mutation_rate = 1.1
    else:
        mutation_rate=0.001
        
    last_fitness = best_solution
    
        
    # Exibir progresso
    # if generation % 100 == 0:
    print(f'Generation {generation}: Best solution = {best_solution[0]}, Fitness = {best_solution[1]}, Mutation Rate = {mutation_rate}')

# Melhor solução final
best_solution = sorted(evaluated_population, key=lambda x: x[1], reverse=False)[0]
best_chromosome = best_solution[0]
best_chromosome_cities = [c[i] for i in best_chromosome]

# # Criar o grafo
# G = nx.Graph()

# # Adicionar nós
# for city in c:
#     G.add_node(city)

# # Adicionar arestas
# for i in range(len(p)):
#     for j in range(i + 1, len(p[i])):
#         if p[i][j] > 0:  # Só adiciona aresta se a distância for maior que 0
#             G.add_edge(c[i], c[j], weight=p[i][j])

# # Arestas da rota específica
# edges_in_route = [(best_chromosome_cities[i], best_chromosome_cities[i + 1]) for i in range(len(best_chromosome_cities) - 1)]

# # Desenhar o grafo
# pos = nx.spring_layout(G, seed=42)  # Layout para melhor visualização

# plt.figure(figsize=(30, 20))

# # Desenhar todas as arestas em cinza
# nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if (u, v) not in edges_in_route and (v, u) not in edges_in_route], edge_color='grey', width=2)

# # Desenhar as arestas da rota específica em vermelho
# nx.draw_networkx_edges(G, pos, edgelist=edges_in_route, edge_color='red', width=2)

# # Desenhar os nós e os rótulos
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold')
# nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})

# plt.title('Grafo de Cidades com Rota Específica Destacada')
# plt.show()

# # Coordenadas das cidades (latitude, longitude)
# coords = {
#     "Uberaba": (-19.7472, -47.9381),
#     "Uberlândia": (-18.9128, -48.2755),
#     "Araguari": (-18.6467, -48.1936),
#     "Araxá": (-19.5902, -46.9433),
#     "Patos de Minas": (-18.5789, -46.5183),
#     "Ituiutaba": (-18.9746, -49.4653),
#     "Monte Carmelo": (-18.7302, -47.4913),
#     "Frutal": (-20.0244, -48.9351),
#     "Prata": (-19.3084, -48.9276),
#     "Iturama": (-19.7279, -50.1959),
#     "Campina Verde": (-19.5386, -49.4863),
#     "Sacramento": (-19.8624, -47.4503),
#     "Conceição das Alagoas": (-19.9171, -48.3837),
#     "Perdizes": (-19.3511, -47.2962),
#     "Ibiá": (-19.4788, -46.5383),
#     "Coromandel": (-18.4731, -47.1972),
#     "Paracatu": (-17.2252, -46.8711),
#     "Vazante": (-17.9829, -46.9053),
#     "Serra do Salitre": (-19.1081, -46.6955),
#     "Rio Paranaíba": (-19.1866, -46.2454),
#     "Santa Vitória": (-18.8412, -50.1208)
# }

# # Criar um mapa centrado na primeira cidade da rota
# m = folium.Map(location=coords['Uberaba'], zoom_start=7)

# # Adicionar os pontos no mapa
# for city, coord in coords.items():
#     folium.Marker(location=coord, popup=city).add_to(m)

# # Adicionar a rota no mapa
# route_coords = [coords[city] for city in best_chromosome_cities]
# folium.PolyLine(route_coords, color='red', weight=5, opacity=0.7).add_to(m)

# # Exibir o mapa
# m.save('rota_especifica1.html')

# print(f'Best solution = {best_chromosome_cities}, Fitness = {best_solution[1]}')
