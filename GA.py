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


    def run(self,individuals=300, crossovers= 10, max_iter=100, objective_error=1000):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        n_location,n_vehicles,instance = self.read_problem_instance(self.problem_path)
        population = self.create_population(n_location,n_vehicles,individuals)
        fitness = []
        n_iter = 0
        while self.best_fitness != None and self.best_fitness > objective_error and n_iter < max_iter:
            for s in population:
                f = self.fitness(s,instance)
                fitness.append((f, s))
                if self.best_fitness == None or f > self.best_fitness:
                    self.best_fitness = f
                    self.best_solution = s
            for i in range(crossovers):
                parent1, parent2 = self.select_parents(fitness)
                child1, child2 = self.crossover(parent1, parent2)
                _child1, n1 = self.extract_chromosome(child1)
                _child2, n2 = self.extract_chromosome(child2)
                _child1[n1] = self.in_route_mutation(_child1[n1])
                _child2[n2] = self.in_route_mutation(_child2[n2])
                child1 = self.inverted_transformation(_child1)
                child2 = self.inverted_transformation(_child2)
                child1 = self.cross_route_mutation(child1)
                child2 = self.cross_route_mutation(child2)
                f1 = self.fitness(child1, instance)
                f2 = self.fitness(child2, instance)
                fitness.append((f1, child1))
                fitness.append((f2, child2))
                if self.best_fitness == None or f1 > self.best_fitness:
                    self.best_fitness = f1
                    self.best_solution = child1
                if self.best_fitness == None or f2 > self.best_fitness:
                    self.best_fitness = f2
                    self.best_solution = child2
            population = self.select_population(fitness, individuals)
            n_iter += 1

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
    
    def create_individual(self,n_locations,n_vehicles): 
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
        This function will translate our representation to the solution format imposed by the tournament.
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
    

    def transform_solution(self,solution):

        final = []
        aux = []
        for s in solution:
            if s > 0:
                aux.append(s)
            else:
                
                final.append(aux) 
                aux=[] 
        
        final.append(aux)

        return final
    
    def inverted_transformation(self,solution):
        final = []
        for i,elem in enumerate(solution):
            for n in elem:
                final.append(n)
            if i !=len(solution)-1:
                final.append(0)

        return final

    def extract_chromosome(self,solution):
        """
        [1,3,6,0,10,5,9,0,7,4,8,2] to [1,3,6],[10,5,9],[7,4,8,2]
        This function will provide one of the routes of one salesman of the solution randomly.
        Later, we will apply In-route mutation in that salesman/chromosome.
        """
        final = self.transform_solution(solution)
        n = random.randint(0,len(final)-1)
        return final, n
    
    def in_route_mutation(self,chromosome):
        """
        This function applies In-route mutation where we act in one salesman in order to generate
        the new population. An example of this function will be:
        [10,13,14,8,9,16,12] to [10,9,8,14,13,16,12]
        """
        chromosome_length = len(chromosome)
    
        # Choose random indices for the subsection
        start_idx = random.randint(0, chromosome_length - 1)
        end_idx = random.randint(start_idx + 1, chromosome_length)
        
        # Select the subsection to be reversed
        subsection = chromosome[start_idx:end_idx]
        
        # Perform in-route mutation by reversing the subsection
        mutated_chromosome = chromosome[:start_idx] + subsection[::-1] + chromosome[end_idx:]
        
        return mutated_chromosome
    
    def cross_route_mutation(self,solution):
        """
        This function applies Cross-route mutation where we mutate the routes of different
        salesmen in order to generate the new population
        """

        # Ensure there are at least two salesmen for mutation
        final = self.transform_solution(solution)

        if len(final) < 2:
            return final
        
        # Choose two distinct random indices representing salesmen
        salesman_1_idx, salesman_2_idx = random.sample(range(len(final)), 2)
        
        # Select random subsections from the chosen salesmen
        salesman_1 = final[salesman_1_idx]
        salesman_2 = final[salesman_2_idx]
        
        # Ensure the subsections are not empty
        if len(salesman_1) == 0 or len(salesman_2) == 0:
            return final
        
        # Choose random subsections within the salesmen
        start_idx_1 = random.randint(0, len(salesman_1) - 1)
        end_idx_1 = random.randint(start_idx_1 + 1, len(salesman_1))
        
        start_idx_2 = random.randint(0, len(salesman_2) - 1)
        end_idx_2 = random.randint(start_idx_2 + 1, len(salesman_2))
        
        # Swap the subsections between the two salesmen
        mutated_final = final.copy()
        mutated_final[salesman_1_idx] = salesman_1[:start_idx_1] + salesman_2[start_idx_2:end_idx_2] + salesman_1[end_idx_1:]
        mutated_final[salesman_2_idx] = salesman_2[:start_idx_2] + salesman_1[start_idx_1:end_idx_1] + salesman_2[end_idx_2:]
        
        return self.inverted_transformation(mutated_final)
    
    def crossover(self,solution1,solution2):
        """
        This function applies our own crossover to two different solutions and returns two childs.
        The mechanism that has been followed will be explained in the paper.
        """

        
        n = len(solution1)
        c1 = [0] * n
        c2 = [0] * n

        c1[0] = solution2[n - 1]
        c2[0] = solution1[n - 1]
        c1[n - 1] = solution2[0]
        c2[n - 1] = solution1[0]

        for i in range(n):
            for j in range(1, n - 1):
                if solution2[i] == solution1[j]:
                    c1[j] = solution2[j]
                if solution1[i] == solution2[j]:
                    c2[j] = solution1[j]

        return c1, c2

if __name__ == '__main__':
    pass
    #print(GA(100,'prueba').extract_chromosome([1,3,6,0,10,5,9,0,7,4,8,2]))
    #print(GA(100,'prueba').in_route_mutation([1,3,6]))
    #print(GA(100,'prueba').crossover([1,3,6,0,10,5,9,0,7,4,8,2],[8,7,4,0,1,3,6,0,10,5,9,2]))
    #print(GA(100,'prueba').cross_route_mutation([1,3,6,0,10,5,9,0,7,4,8,2]))
    

