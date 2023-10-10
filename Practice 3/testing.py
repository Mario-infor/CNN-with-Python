import numpy as np


# Función objetivo como la suma de tres gaussianas
def objective_function(x):
    # Define las tres gaussianas
    gaussian1 = lambda x: np.exp(-((x - 1) ** 2) / 2)
    gaussian2 = lambda x: np.exp(-((x - 3) ** 2) / 2)
    gaussian3 = lambda x: np.exp(-((x - 5) ** 2) / 2)

    # Suma de las tres gaussianas
    return gaussian1(x) + gaussian2(x) + gaussian3(x)


# Inicialización de parámetros
x = 0.0  # Valor inicial de x
learning_rate = 0.01
iterations = 1000  # Número máximo de iteraciones
convergence_threshold = 1e-5  # Criterio de convergencia

# Historial para el seguimiento del progreso
parameter_history = []
objective_history = []

# Descenso del gradiente
for i in range(iterations):
    gradient = np.gradient(objective_function(x))  # Gradiente de la función objetivo
    x -= learning_rate * gradient

    parameter_history.append(x)
    objective_history.append(objective_function(x))

    # Criterio de convergencia
    if np.linalg.norm(gradient) < convergence_threshold:
        break

print(f"Resultado: x = {x}, f(x) = {objective_function(x)}")
