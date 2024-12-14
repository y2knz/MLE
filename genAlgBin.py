import numpy as np
import matplotlib.pyplot as plt

pop_size = 1000  # Reduzierte Populationsgröße
gen_len = 64  # Länge eines Gens (16 Bits pro Parameter, insgesamt 4 Parameter)
gene = np.random.uniform(-1, 1, (pop_size, 4))  # Initialisiere Population mit zufälligen Dezimalwerten
new_gene = np.zeros((pop_size, 4))
fitns = np.ones(pop_size)

rek_rate = 0.8  # Rekombinationsrate
mut_rate = 0.05  # Mutationsrate

qmin = -1  # Minimaler Wert der Parameter
qmax = 1   # Maximaler Wert der Parameter

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
def initialize_population(pop_size, num_params):
    return np.random.uniform(-1, 1, (pop_size, num_params))

# Hilfsfunktionen für Gray-Code
def bin_to_gray(bin_value):
    return bin_value ^ (bin_value >> 1)

def gray_to_bin(gray_value):
    mask = gray_value
    while mask != 0:
        mask >>= 1
        gray_value ^= mask
    return gray_value

def gray_to_real(gray_value, qmin, qmax, n_bits):
    bin_value = gray_to_bin(gray_value)
    return qmin + (qmax - qmin) * bin_value / (2**n_bits - 1)

def real_to_gray(real_value, qmin, qmax, n_bits):
    bin_value = int((real_value - qmin) * (2**n_bits - 1) / (qmax - qmin))
    return bin_to_gray(bin_value)

# Konvertierung zwischen Binär und Dezimal
def decimal_to_binary(decimal, gene_length):
    return np.array(list(np.binary_repr(decimal, width=gene_length)), dtype=int)

def binary_to_decimal(binary_gene):
    return int("".join(str(int(bit)) for bit in binary_gene), 2)

# Selektion
def rank_select():
    return np.random.randint(0, pop_size // 2)

# Crossover
def crossover():
    global gene
    split = np.random.randint(1, gen_len // 4)
    sel1 = rank_select()
    sel2 = rank_select()
    child1 = np.zeros(4)
    child2 = np.zeros(4)
    for i in range(4):
        parent1_gray = real_to_gray(gene[sel1][i], qmin, qmax, gen_len // 4)
        parent2_gray = real_to_gray(gene[sel2][i], qmin, qmax, gen_len // 4)
        parent1_bin = decimal_to_binary(parent1_gray, gen_len // 4)
        parent2_bin = decimal_to_binary(parent2_gray, gen_len // 4)
        child1_bin = np.append(parent1_bin[:split], parent2_bin[split:])
        child2_bin = np.append(parent2_bin[:split], parent1_bin[split:])
        child1_gray = binary_to_decimal(child1_bin)
        child2_gray = binary_to_decimal(child2_bin)
        child1[i] = gray_to_real(child1_gray, qmin, qmax, gen_len // 4)
        child2[i] = gray_to_real(child2_gray, qmin, qmax, gen_len // 4)
    return child1, child2

# Mutation
def mutate():
    global gene, pop_size
    for i in range(int(pop_size * mut_rate)):
        indi = np.random.randint(0, pop_size)
        param_index = np.random.randint(0, 4)
        param_value = real_to_gray(gene[indi][param_index], qmin, qmax, gen_len // 4)
        bin_value = decimal_to_binary(param_value, gen_len // 4)
        bit = np.random.randint(0, gen_len // 4)
        bin_value[bit] = 1 - bin_value[bit]  # Bit kippen
        gray_value = binary_to_decimal(bin_value)
        gene[indi][param_index] = gray_to_real(gray_value, qmin, qmax, gen_len // 4)

# Auswertung der Fitness
def eval_fitness():
    global fitns, pop_size, gene
    x_values = np.linspace(-1, 1, 100)
    for i in range(pop_size):
        fitns[i] = -fitness(gene[i], x_values)
    sorted_indices = np.flip(np.argsort(fitns))
    gene = gene[sorted_indices]
    fitns = fitns[sorted_indices]

if __name__ == "__main__":
    gene = initialize_population(pop_size, 4)
    graph = np.array([])
    for i in range(10):
        mutate()  # initialisiere das Array zufällig
    for gen in range(200):  # Erhöhte Anzahl der Generationen
        eval_fitness()  # Fitness auswerten
        graph = np.append(graph, -fitns[0])  # bester fitness der aktuellen generation speichern
        for i in range(pop_size):
            if i < pop_size * rek_rate:
                # erzeuge zwei neue Individuen
                new_gene[i], new_gene[i + 1] = crossover()
                i = i + 1  # weil zwei Individuen erzeugt wurden
            else:
                # selektiere ein Individuum
                new_gene[i] = np.copy(gene[rank_select()])

        gene = np.copy(new_gene)  # kopiere neue gene
        mutate()

    best_params = gene[0]  # wählt den besten fit aus

    # Zeichne die beste gefundene Lösung
    x_values = np.linspace(-1, 1, 100)
    plt.plot(x_values, f(x_values, best_params), label="Optimierte Funktion f(x)")
    plt.plot(x_values, g(x_values), label="Ziel-Funktion g(x)")
    plt.legend()
    plt.show()

    print("Optimierte Koeffizienten:")
    for i, coeff in enumerate(best_params):
        print(f"  x^{3 - i}: {coeff:.6f}")

    # Vergleiche die Koeffizienten mit denen der Taylor-Reihe von e^x um 0
    taylor_coeffs = [1 / np.math.factorial(i) for i in range(4)]  # Taylor-Reihe von e^x um 0

    print("\nTaylor-Koeffizienten:")
    for i, coeff in enumerate(reversed(taylor_coeffs)):
        print(f"  x^{3 - i}: {coeff:.6f}")