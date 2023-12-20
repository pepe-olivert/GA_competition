import numpy as np
import random
import itertools

class GA:
    def __init__(self,time_deadline,problem_path,**kwargs):
        """
        Inicializador de los objetos de la clase. Usar
        este método para hacer todo el trabajo previo y necesario
        para configurar el algoritmo genético
        Args:
            problem_path: Cadena de texto que determina la ruta en la que se encuentra la definición del problema
            time_deadline: Límite de tiempo que el algoritmo genético puede computar
        """
        self.problem_path = problem_path
        self.best_solution = None 
        self.time_deadline = time_deadline 
        self.best_fitness = None
        
        #TODO : Completar método para configurar el algoritmo genético (e.g., seleccionar cruce, mutación, etc.)
        

    def read_problem_instance(self,problem_path):
        
        with open(problem_path, "r") as f:
            text = f.read()
        lines = text.strip().split('\n')
        num_locations = int(lines[0].split()[1])
        num_vehicles = int(lines[1].split()[1])

        matrix_lines = [line.split() for line in lines[3:]]
        distance_matrix = np.array(matrix_lines, dtype=int)

        return num_locations,num_vehicles,distance_matrix

    def get_best_solution(self):
        """
        Método para devolver la mejor solución encontrada hasta
        el momento
        """
        
        return self.best_solution


    def run(self,individuals=300):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        n_location,n_vehicles,instance = self.read_problem_instance(self.problem_path)
        population = self.create_population(n_location,n_vehicles,individuals)
        for s in population:
            pass
            """
            Aqui hay que calcular el fitness de la solución
            Comparas con self.best_fitness si > lo cambias y actualizas self.best_solution
            """
        
        pass

    def fitness(self,solution,dist_matrix):

        """
        If the solution is in format : [1,3,6,0,10,5,9,0,7,4,8,2] We should calculate the fitness of that solution
        as 1/total distance traveled by that salesman
        """
    
        total_distance = 0
        origin = 0
        for n in solution:
            destination=n
            total_distance += dist_matrix[origin][destination]
            origin = n

        destination=0
        total_distance+=dist_matrix[origin][destination]
        return 1/total_distance
    
    def create_individual(self,n_locations,n_vehicles): ## THIS IS RANDOM SEARCH FUNCTION
        aux = [0]*(n_vehicles-1)
        rnge = list(range(1,n_locations))
        individual = aux+rnge
        random.shuffle(individual)

        return individual
    
    def random_search(self,n_locations,n_vehicles):
        return self.create_individual(n_locations,n_vehicles)
    
    def create_population(self,n_locations,n_vehicles,n_individuals):
        population=[]
        for i in range(n_individuals):
            population.append(self.create_individual(n_locations,n_vehicles))
        return population
    


    def greedy_heuristic(self,dist_matrix, n_vehicles,n_locations):
      
        routes = [[] for _ in range(n_vehicles)]
        visited = set()
        visited.add(0)  # Assuming the depot is at index 0

        for route in routes:   # Initialize routes with the depot
            route.append(0)

        # Assign locations to vehicles
        while len(visited) < n_locations:
            for route in routes:
                if len(visited) == n_locations:
                    break
                last_location = route[-1]
                closest_distance = float('inf')
                closest_location = None
                for i in range(n_locations):
                    if i not in visited and dist_matrix[last_location][i] < closest_distance:
                        closest_distance = dist_matrix[last_location][i]
                        closest_location = i
                route.append(closest_location)
                visited.add(closest_location)

        for route in routes:
            route.append(0)
        routes = [r[:-1] for r in routes]
        extended = list(itertools.chain.from_iterable(routes))
        extended = extended[1:]
        

        return extended

    def translate_solution(self,solution):
        """
        [1,3,6,0,10,5,9,0,7,4,8,2] to [[0,1,3,6,0],[0,10,5,9,0],[0,7,4,8,2,0]]
        """
        final = []
        aux = [0]
        for s in solution:
            if s > 0:
                aux.append(s)
            else:
                aux.append(0)
                final.append(aux)
                aux=[0]
            
        aux.append(0)
        final.append(aux)
        return final

if __name__ == '__main__':
    pass
    ##print(GA(100,'prueba').translate_solution([1,3,6,0,10,5,9,0,7,4,8,2]))

   


