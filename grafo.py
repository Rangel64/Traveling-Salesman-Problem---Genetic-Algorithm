import folium

# Coordenadas das cidades (latitude, longitude)
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

# Criar um mapa centrado na primeira cidade da rota
m = folium.Map(location=coords['Uberaba'], zoom_start=7)

# Adicionar os pontos no mapa
for city, coord in coords.items():
    folium.Marker(location=coord, popup=city).add_to(m)

# Adicionar a rota no mapa
route_coords = [coords[city] for city in rota_especifica]
folium.PolyLine(route_coords, color='red', weight=5, opacity=0.7).add_to(m)

# Exibir o mapa
m.save('rota_especifica.html')