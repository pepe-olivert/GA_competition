import numpy as np
import random
import itertools

class GA:
    def __init__(self,time_deadline,problem_path,**kwargs): #the default values for these parameters should be the best values found in the experimental optimization of the algorithm.
        self.problem_path = problem_path
        self.best_solution = None 
        self.time_deadline = time_deadline 
        self.best_fitness = None
    
    #TODO : Completar método para configurar el algoritmo genético (e.g., seleccionar cruce, mutación, etc.)

    def read_problem_instance(self,problem_path): #Process the .txt file with the instance of the problem
        with open(problem_path, "r") as f:
            text = f.read()
        lines = text.strip().split('\n')
        num_locations = int(lines[0].split()[1])
        num_vehicles = int(lines[1].split()[1])

        matrix_lines = [line.split() for line in lines[3:]]
        distance_matrix = np.array(matrix_lines, dtype=int)

        return num_locations,num_vehicles,distance_matrix


# ------------------- Auxiliar methods ------------------- 

    def translate_solution(self,solution):
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
    
#-------------------------------------- 
    
    def create_individual(self,n_locations,n_vehicles): 
        aux = [0]*(n_vehicles-1)
        rnge = list(range(1,n_locations))
        individual = aux+rnge
        random.shuffle(individual)
        return individual
    
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
        population = list(itertools.chain.from_iterable(routes))
        population = population[1:]

        return population
    
    def fitness(self,solution,dist_matrix):

        total_distance = 0
        origin = 0
        for n in solution:
            destination=n
            total_distance += dist_matrix[origin][destination]
            origin = n

        destination=0
        total_distance+=dist_matrix[origin][destination]
        return 1/total_distance
    
    def select_parent(self,population_fitness, n, m):
        if n<m and n%2 == 0:
            #Randomly sample m individuals from population
            sampled_individuals = random.sample(population_fitness, m)

            #Sort according to fitness (best individuals first)
            sorted_individuals = sorted(sampled_individuals, key=lambda x: x[0], reverse=True)

            #Select the top n individuals from the m individuals sampled
            selected_parents = sorted_individuals[:n]

            return selected_parents
        
        else:
            print("n is not less than m or n is not even")


    def inspired_crossover_DPX(self,parent1,parent2):
        n = len(parent1)
        c1 = [0] * n
        c2 = [0] * n

        #Copy ends of the parents to opposite positions in the children
        c1[0] = parent2[n - 1]
        c2[0] = parent1[n - 1]
        c1[n - 1] = parent2[0]
        c2[n - 1] = parent1[0]
        
        #Swap remaining cities as given algorithm from paper
        for i in range(n):
            for j in range(1, n - 1):
                if parent2[i] == parent1[j]:
                    c1[j] = parent2[j]
                if parent1[i] == parent2[j]:
                    c2[j] = parent1[j]

        return c1, c2
    
    def extract_chromosome(self,solution):
        final = self.transform_solution(solution)
        n = random.randint(0,len(final)-1)
        return final, n
    
    
    def in_route_mutation(self,chromosome):

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
        #Ensure there are at least two salesmen for mutation
        final = self.transform_solution(solution)

        if len(final) < 2:
            return final
        
        #Choose two distinct random indices representing salesmen
        salesman_1_idx, salesman_2_idx = random.sample(range(len(final)), 2)
        
        #Select random subsections from the chosen salesmen
        salesman_1 = final[salesman_1_idx]
        salesman_2 = final[salesman_2_idx]
        
        #Ensure the subsections are not empty
        if len(salesman_1) == 0 or len(salesman_2) == 0:
            return final
        
        #Choose random subsections within the salesmen
        start_idx_1 = random.randint(0, len(salesman_1) - 1)
        end_idx_1 = random.randint(start_idx_1 + 1, len(salesman_1))
        
        start_idx_2 = random.randint(0, len(salesman_2) - 1)
        end_idx_2 = random.randint(start_idx_2 + 1, len(salesman_2))
        
        #Swap the subsections between the two salesmen
        mutated_final = final.copy()
        mutated_final[salesman_1_idx] = salesman_1[:start_idx_1] + salesman_2[start_idx_2:end_idx_2] + salesman_1[end_idx_1:]
        mutated_final[salesman_2_idx] = salesman_2[:start_idx_2] + salesman_1[start_idx_1:end_idx_1] + salesman_2[end_idx_2:]
        
        return self.inverted_transformation(mutated_final)
    
    def replace_cmin(self, fitness, offspring):
        # CD/RW strategy for replacing cmin
        cmin = min(fitness, key=lambda x: self.contribution_of_diversity(x[1], [ind[1] for ind in fitness]))
        fitness_without_cmin = [ind for ind in fitness if ind != cmin]

        # Check if offspring provides more diversity than cmin
        if self.contribution_of_diversity(offspring, [ind[1] for ind in fitness_without_cmin]) > self.contribution_of_diversity(cmin[1], [ind[1] for ind in fitness]):
            fitness.remove(cmin)
        else:
            # Using replace by worst otherwise
            self.replace_by_worst(fitness, offspring)
        return [ind[1] for ind in fitness]

    def replace_by_worst(self, fitness, offspring):
        # RW strategy for replacing the worst individual
        worst_individual = max(fitness, key=lambda x: x[0])
        offspring_fitness = next(fit[0] for fit in fitness if fit[1] == offspring)

        if offspring_fitness < worst_individual[0]:
            fitness.remove(worst_individual)

    
    def run(self,individuals=300, crossovers= 10, max_iter=100, objective_error=1000):

        '''Initialize population'''
        n_location,n_vehicles,instance = self.read_problem_instance(self.problem_path)          
        population = self.create_population(n_location,n_vehicles,individuals) 

#each of the starting populations was seeded with a solution produced by a simple greedy heuristic 
#in order to give the GA a good starting point (paper)
        #population = self.greedy_heuristic(instance,n_vehicles,n_locations)

        '''Evaluate each solution and creating list with the fitness'''
        fitness = []
        for s in population:                                                                                                                                                            
            f = self.fitness(s,instance)
            fitness.append((f, s))
            if self.best_fitness == None or f > self.best_fitness:          
                self.best_fitness = f
                self.best_solution = s

        n_iter = 0
        while (self.best_fitness != None and self.best_fitness > objective_error and n_iter < max_iter) or  self.best_fitness == None: #Termination condition
            '''SELECT parent'''
            parent1, parent2 = self.select_parents(fitness, n_location, n_vehicles)                                                    

            for i in range(crossovers):  #Iterations specified in the configuration (10 by default)

                '''CROSSOVER'''                                                                                              
                child1, child2 = self.inspired_crossover_DPX(parent1, parent2)                                                         

                '''MUTATION'''   
                _child1, n1 = self.extract_chromosome(child1)
                _child2, n2 = self.extract_chromosome(child2)

                _child1[n1] = self.in_route_mutation(_child1[n1])
                _child2[n2] = self.in_route_mutation(_child2[n2])

                child1 = self.inverted_transformation(_child1)
                child2 = self.inverted_transformation(_child2)

                child1 = self.cross_route_mutation(child1)
                child2 = self.cross_route_mutation(child2)

                '''EVALUATION'''
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
            
            selected_offspring = random.choice([child1, child2])      #This will continue with the diversity of the population but other methods can be implemented such as: selecting child with best fitness or by alternancy in each iteration.
            population = self.replace_cmin(fitness, selected_offspring)
            
            n_iter += 1

        return self.translate_solution(self.best_solution)
        
    
    
    if __name__ == '__main__':
        pass