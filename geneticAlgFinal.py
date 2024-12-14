import numpy as np
import matplotlib.pyplot as plt

pop_size = 1000  # Populationsgröße
gen_len = 4  # Länge eines Gens (4 Parameter: a, b, c, d)
gene = np.random.uniform(-1, 1, (pop_size, gen_len))  # Initialisiere Population mit zufälligen Floats
new_gene = np.zeros((pop_size, gen_len)) # genes der neuen generation
fitns = np.ones(pop_size)

rek_rate = 0.8  # Rekombinationsrate
mut_rate = 0.2  # Mutationsrate

# Funktion f(x) = ax^3 + bx^2 + cx + d
def f(x, params):
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d

# Funktion g(x) = e^x
def g(x):
    return np.exp(x)

# Fitness-Funktion (quadratischer Fehler)
def fitness(params, x_values):
    return np.sum((f(x_values, params) - g(x_values))**2)

# Initialisiere Population
def initialize_population(pop_size, gen_len):
    return np.random.uniform(-1, 1, (pop_size, gen_len)) # Initialisiere Population mit zufälligen Floats

# Selektion
def rank_select():
    return np.random.randint(0, pop_size // 2) #auswahl aus der oberen hälfte der population

# Crossover
def crossover():
    global gene #globale variable
    split = np.random.randint(1, gen_len)#zufälliger splitpunkt
    sel1 = rank_select()#selektiere Individuum 1
    sel2 = rank_select()# selektiere Individuum 2
    return np.append(gene[sel1][:split], gene[sel2][split:]), np.append(gene[sel2][:split], gene[sel1][split:])#erzeuge zwei neue Individuen

# Mutation
def mutate():
    global gene, pop_size
    for i in range(int(pop_size * mut_rate)): # pop_size * mut_rate mal
        indi = np.random.randint(0, pop_size)#zufällige auswahl eines individuums
        bit = np.random.randint(0, gen_len)#zufällige auswahl eines bits
        gene[indi][bit] += np.random.uniform(-0.1, 0.1)  # Zufällige Änderung des Parameters 
        #genlänge in bits umwandeln formel binäcode zu parameter 
        #greycode verwebdeb
        #greycode zu dezimal code dann zu zahl 

# Auswertung der Fitness
def eval_fitness():
    global fitns, pop_size, gene
    x_values = np.linspace(-1, 1, 100) #Werte zwischen -1 und 1
    for i in range(pop_size): #für jedes individuum
        fitns[i] = -fitness(gene[i], x_values) #fitness berechnen
    sorted_indices = np.flip(np.argsort(fitns)) #sortiere gene[] nach fitns[] absteigend
    gene = gene[sorted_indices] #sortierte gene
    fitns = fitns[sorted_indices] #sortiere fit

if __name__ == "__main__":
    gene = initialize_population(pop_size, gen_len)
    graph = np.array([])#  Array für die besten Fitnesswerte
    for i in range(10): #initialisiere das Array zufällig
        mutate()  # initialisiere das Array zufällig
    for gen in range(100):  # Generation
        eval_fitness()  # Fitness auswerten
        graph = np.append(graph, -fitns[0]) #bester fitness der aktuellen generation speichern
        for i in range(pop_size):
            if i < pop_size * rek_rate:
                # erzeuge zwei neue Individuen
                new_gene[i], new_gene[i + 1] = crossover()
                i = i + 1  # weil zwei Individuen erzeugt wurden
            else:
                # selektiere ein Individuum
                new_gene[i] = np.copy(gene[rank_select()])

        gene = np.copy(new_gene)#kopiere neue gene
        mutate() 

    best_params = gene[0]# wählt den besten fit aus 

    # Zeichne die beste gefundene Lösung
    x_values = np.linspace(-1, 1, 100) #Werte zwischen -1 und 1
    plt.plot(x_values, f(x_values, best_params), label='Optimierte Funktion f(x)')
    plt.plot(x_values, g(x_values), label='Ziel-Funktion g(x)')
    plt.legend()
    plt.show()

    # Vergleiche die Koeffizienten mit denen der Taylor-Reihe von e^x um 0
    taylor_coeffs = [1 / np.math.factorial(i) for i in range(4)]  # Taylor-Reihe von e^x um 0

    print("Optimierte Koeffizienten:")
    for i, coeff in enumerate(best_params):
        print(f"  x^{3 - i}: {coeff:.6f}")

    print("\nTaylor-Koeffizienten:")
    for i, coeff in enumerate(reversed(taylor_coeffs)):
        print(f"  x^{3 - i}: {coeff:.6f}")