import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import threading
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import argparse


class GA:

    def __init__(self,time_deadline,problem_path,init="cluster",mode="agglomerative",**kwargs): 
        """
        Initialize an instance of class GA given a time deadline and an instance path.

        :param time_deadline: Max time of the execution.
        :type time_deadline: int
        :param problem_path: Path to a certain instance.
        :type problem_path: str
        :param init: Parameter to use cluster initialization or not.
        :type init: str
        :param mode: Specific parameter that indicates the mode of the clusters.
        :type mode: str
        """
        self.problem_path = problem_path
        self.best_solution = None 
        self.time_deadline = time_deadline 
        self.best_fitness = None
        self.init = init
        self.mode = mode

        self.evolution = []
        
        

        
    def get_best_solution(self):  
        """
        Function to get the best solution of the execution of the GA and translated into the desired form.
        """
        return self.translate_solution(self.best_solution)

    def read_problem_instance(self): 
        """
        Function to read the problem instance given the instance path. You get the number of vehicles,
        the number of locations and the distance matrix for future purposes.
        """
        with open(self.problem_path, "r") as f:
            text = f.read()
        lines = text.strip().split('\n')
        num_locations = int(lines[0].split()[1])
        num_vehicles = int(lines[1].split()[1])

        matrix_lines = [line.split() for line in lines[3:]]
        distance_matrix = np.array(matrix_lines, dtype=float)

        return num_locations,num_vehicles,distance_matrix


    ''' ------------------- Auxiliar methods ------------------- '''

    def translate_solution(self,solution):
        """
        Function to translate the solution to the desired form.

        :param solution: A given solution to translate.
        :type solution: str

        :return: The new solution.
        :rtype: list
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
        """
        Function to encode a given solution in order to compute different crossovers and mutations.

        :param solution: A given solution to translate.
        :type solution: str

        :return: The new solution.
        :rtype: list
        """
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
        """
        Function to invert the solution.

        :param solution: A given solution to translate.
        :type solution: str

        :return: The new solution.
        :rtype: list
        """

        final = []
        for i,elem in enumerate(solution):
            for n in elem:
                final.append(n)
            if i !=len(solution)-1:
                final.append(0)
        return final
    
    def create_individual(self,n_locations,n_vehicles): 
        """
        Function to randomly create an individual, given the locations and vehicles.

        :param n_locations: Number of location.
        :type n_locations: int
        :param n_vehicles: Number of vehicles.
        :type n_vehicles: int

        :return: New individual.
        :rtype: list
        """

        aux = [0]*(n_vehicles-1)
        rnge = list(range(1,n_locations))
        individual = aux+rnge
        random.shuffle(individual)
        return individual
    
    def create_population(self,n_locations,n_vehicles,n_individuals):
        """
        Function to create a population.

        :param n_locations: Number of location.
        :type n_locations: int
        :param n_vehicles: Number of vehicles.
        :type n_vehicles: int
        :param n_individuals: Number of individuals.
        :type n_individuals: int 

        :return: The population.
        :rtype: list
        """

        population=[]
        for i in range(n_individuals):
            population.append(self.create_individual(n_locations,n_vehicles))
        return population
    
    def cluster_population(self,mode,individuals,n_location,n_vehicles,distance_matrix):
        """
        Function to create a population built upon clustering.

        :param n_locations: Number of location.
        :type n_locations: int
        :param n_vehicles: Number of vehicles.
        :type n_vehicles: int
        :param n_individuals: Number of individuals.
        :type n_individuals: int 
        :param distance_matrix: Distance matrix.
        :type distance_matrix: list

        :return: The population.
        :rtype: list
        """

        labels,locations,vehicles = self.cluster_initialization(mode=mode,n_location=n_location,n_vehicles=n_vehicles,distance_matrix=distance_matrix)
        locs = np.arange(1,locations)
        population_salesman = {str(i):[locs[j] for j in range(locations-1) if labels[j]==i] for i in range(vehicles)}
        
        population_initial = [[]]
        individuals = individuals - 1
        for i in range(vehicles-1):
            for j in range(locations-1):
                if labels[j]==i:
                    population_initial[0].append(locs[j])
            population_initial[0].append(0)
        population_initial[0].extend(population_salesman[str(vehicles-1)])
        
        i = 1
        while individuals > 0:
            tours_altered = random.randint(1,vehicles)
            ind = population_salesman.copy()

            population_initial.append([])
            for j in range(tours_altered):
                vehicle = random.randint(0,vehicles-1)
                ind[str(vehicle)] = random.sample(ind[str(vehicle)],len(ind[str(vehicle)]))  
            
            for j in range(vehicles-1):
                population_initial[i].extend(ind[str(j)]+[0])
            population_initial[i].extend(ind[str(vehicles-1)])
            i += 1
            individuals -= 1
        
        return population_initial
    
    def cluster_initialization(self,mode,n_location,n_vehicles,distance_matrix): 
        """
        Function that initiatializates clustering

        :param mode: Different modes of clusters.
        :type mode: str
        :type n_locations: int
        :param n_vehicles: Number of vehicles.
        :type n_vehicles: int
        :param n_individuals: Number of individuals.
        :type n_individuals: int 
        :param distance_matrix: Distance matrix.
        :type distance_matrix: list
        
        :return: The labels of each city. Each cluster will be assigned initially to a salesman.
        :rtype: list
        """

        distance_matrix=np.array(distance_matrix)
        distance_matrix=distance_matrix[1:,1:] #Skip the initial node distances
        
        if mode == "agglomerative":
            c =AgglomerativeClustering(n_clusters=n_vehicles,metric="precomputed",linkage="complete")
        
        if mode == "spectral":
            transf_matrix = distance_matrix
            non_zero_indices = transf_matrix!=0
            transf_matrix[non_zero_indices] = 9999 / transf_matrix[non_zero_indices]
            c = SpectralClustering(n_clusters=n_vehicles,affinity="precomputed").fit(transf_matrix)
        return c.fit(distance_matrix).labels_,n_location,n_vehicles


    
    
    def greedy_heuristic(self,dist_matrix, n_vehicles,n_locations):
        """
        Function that applies the greedy heuristic search.

        :param n_vehicles: Number of vehicles.
        :type n_vehicles: int
        :param n_individuals: Number of individuals.
        :type n_individuals: int 
        :param dist_matrix: Distance matrix.
        :type dist_matrix: list
        """
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
        """
        Function that, if the solution is in format : [1,3,6,0,10,5,9,0,7,4,8,2] (permutation representation) We should calculate the fitness of that solution
        as 1/total distance traveled by that vehicle.

        :param solution: A given solution.
        :type solutions: list
        :param dist_matrix: Distance matrix.
        :type dist_matrix: list

        :return: The compute of our fitness.
        :rtype: float
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
    
    def select_parent(self,population_fitness, n=4, m=100):
        """
        Select the optimal parent.

        :param population_fitness: The population of the fitness.
        :type population_fitness: list
        :param n: Random number of individuals of the m samples.
        :type n: int
        :param m: Random number of individuals.
        :type m: int

        :return: The selected parents.
        :rtype: list
        """
        if n<m and n%2 == 0:
            #Randomly sample m individuals from population
            sampled_individuals = random.sample(population_fitness, m)

            #Sort according to fitness (best individuals first)
            sorted_individuals = sorted(sampled_individuals, key=lambda x: x[0], reverse=True)

            #Select the top n individuals from the m individuals sampled
            selected_parents = sorted_individuals[:n]
            selected_parents = [parent[1] for parent in selected_parents]

            return selected_parents
        
        else:
            print("n is not less than m or n is not even")

    def inspired_crossover_DPX(self,parent1,parent2):
        """
        Function that computes the a crossover inspired in DPX.

        :param parent1: Parent 1.
        :type parent1: list
        :param parent2: Parent 2.
        :type parent2: list

        :return: childs
        :rtype: list
        """
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
        """
        Function to extract a chromosome of a solution. A chromose means a salesman.

        :param solution: A given solution.
        :type solution: list

        :return: The chromosome and the index.
        :rtype: list,int
        """

        final = self.transform_solution(solution)
        n = random.randint(0,len(final)-1)
        return final, n
    
    def inversion_mutaion(self,chromosome):
        """
        Function that computes the inversion mutation given a chromosome.

        :param chromosome: A given solution.
        :type chromosome: list

        :return: The returned from computing this mutation.
        :rtype: list
        """
        index1, index2 = random.sample(range(len(chromosome)), 2)
        start_index = min(index1, index2)
        end_index = max(index1, index2)
        inverted = chromosome[:start_index] + list(reversed(chromosome[start_index:end_index + 1])) + chromosome[end_index + 1:]
        return inverted
    
    def in_route_mutation(self,chromosome):
        """
        Function that implements the in route mutation, mutation operator shown in a paper, 
        optimal for MTSP.

        :param chromosome: A given chromosome.
        :type chromosome: list

        :return mutated_crossover: The returned chromosome when applied this mutation operator.
        :rtype mutated_crossover: list
        """
        
        chromosome_length = len(chromosome)
        if chromosome_length != 0:

            # Choose random indices for the subsection
            start_idx = random.randint(0, chromosome_length - 1)
            end_idx = random.randint(start_idx + 1, chromosome_length)
            
            # Select the subsection to be reversed
            subsection = chromosome[start_idx:end_idx]
            
            # Perform in-route mutation by reversing the subsection
            mutated_chromosome = chromosome[:start_idx] + subsection[::-1] + chromosome[end_idx:]
        else: mutated_chromosome = chromosome
        
        return mutated_chromosome
    
    def cross_route_mutation(self,solution):
        """
        This function applies Cross-route mutation where we mutate the routes of different
        salesmen in order to generate the new population

        :param solution: A given solution.
        :type solution: list

        :return final: The returned chromosome when applied this mutation operator.
        :rtype final: list
        """

        #Ensure there are at least two salesmen for mutation
        final = self.transform_solution(solution)

        if len(final) < 2:
            return self.inverted_transformation(final)
        
        #Choose two distinct random indices representing salesmen
        salesman_1_idx, salesman_2_idx = random.sample(range(len(final)), 2)
        
        #Select random subsections from the chosen salesmen
        salesman_1 = final[salesman_1_idx]
        salesman_2 = final[salesman_2_idx]
        
        #Ensure the subsections are not empty
        if len(salesman_1) == 0 or len(salesman_2) == 0:
            return self.inverted_transformation(final)
        
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
    
    def inroute_opt2_mutation(self,chromosome):
        """
        Function to apply new type of mutation.

        :param chromosome: A given chromosome.
        :type chromosome: list

        :return mutated_salesmen: The returned chromosome when applied this mutation operator.
        :rtype mutated_salesmen: list
        """
        new = np.array([0]+chromosome)
        idx = np.where(new == 0)[0]
    
        separations = np.split(new, idx)
        subroutes = [subarray.tolist() for subarray in separations if len(subarray) > 0]
        mutated_salesmen = []
        for i,salesman in enumerate(subroutes):
            if len(salesman) > 3:
                salesman_modified = salesman[1:]
                position1, position2 = random.sample(range(len(salesman_modified)), 2)
                salesman_modified[position1], salesman_modified[position2] = salesman_modified[position2], salesman_modified[position1]
                if i==0:
                    mutated_salesmen+=salesman_modified
                else:
                    mutated_salesmen+=([0]+salesman_modified)
            else:
                if i==0:
                    mutated_salesmen+=salesman[1:]
                else:
                    mutated_salesmen+=(salesman)
        return mutated_salesmen
    
    def replace_cmin(self, fitness, offspring):
        """
        Replace the least contributing individual in terms of diversity with offspring 
        if it provides more diversity.

        :param fitness: The fitness.
        :type fitness: list
        :param offspring: Children.
        :type offspring: list
        """
        # CD/RW strategy for replacing cmin
        cmin = min(fitness, key=lambda x: self.contribution_of_diversity(x[1], [ind[1] for ind in fitness]))
        fitness_without_cmin = [ind for ind in fitness if ind != cmin]

        # Check if offspring provides more diversity than cmin
        if self.contribution_of_diversity(offspring, [ind[1] for ind in fitness_without_cmin]) > self.contribution_of_diversity(cmin[1], [ind[1] for ind in fitness]):
            fitness.remove(cmin)
        else:
            # Using replace by worst otherwise
            self.replace_by_worst(fitness, offspring)
        return [ind[1] for ind in fitness], fitness

    def replace_by_worst(self, fitness, offspring):
        """
        Replace the worst individual in the population with offspring.

        :param fitness: The fitness.
        :type fitness: list
        :param offspring: Children.
        :type offspring: list
        """
        # RW strategy for replacing the worst individual
        worst_individual = max(fitness, key=lambda x: x[0])
        offspring_fitness = next(fit[0] for fit in fitness if fit[1] == offspring)

        if offspring_fitness < worst_individual[0]:
            fitness.remove(worst_individual)

    def contribution_of_diversity(self, individual, population):
        """
        Calculate the contribution of diversity of an individual to a population based 
        on the euclidean distance of the vectors.

        :param individual: An individual.
        :type individual: list
        :param population: The population.
        :type population: list

        :return: List of the mean of distances.
        :rtype: list
        """
        # Calculate the contribution of diversity of an individual to a population based on the euclidean distance of the vectors
        distances = [np.linalg.norm(np.array(individual) - np.array(ind)) for ind in population]
        return float(np.sum(distances) / len(distances))
    
    def fitness_proportion_ranking_selection(self, fitness, k = 2):
        """
        Function to apply fitness proportion ranking selection to select the next population.

        :param fitness: Fitness of the population
        :type fitness: list

        :return: list of selected individuals to pass to the next generation.
        :rtype: list
        """

        # Calculate selection probability for each individual
        cummulative = sum(x[0] for x in fitness)
        selection = []
        selection_fitness = []
        while len(selection) < k:
            i = random.randint(0, len(fitness)-1)
            f, ind = fitness[i]
            r = random.random()
            p = f/cummulative
            if r < p:
                selection.append(ind)
                selection_fitness.append((f, ind))
        # Select two individuals
        return selection, selection_fitness
    
    def inverse_fitness_proportion_ranking_selection(self, fitness, k = 2):
        """
        Function to apply inversed fitness proportion ranking selection to select the next population.

        :param fitness: Fitness of the population
        :type fitness: list

        :return: list of selected individuals to pass to the next generation.
        :rtype: list
        """
        
        # Calculate selection probability for each individual
        cummulative = sum(x[0] for x in fitness)
        selection = []
        while len(selection) < k:
            i = random.randint(0, len(fitness)-1)
            f, ind = fitness[i]
            r = random.random()
            p = 1 - f/cummulative
            if r < p:
                selection.append(i)
       
        # Select two individuals
        return selection
    
    def linear_ranking_selection(self,fitness, s = 1.5, k = 2):
        """
        Function to apply linear ranking selection to select the next population.

        :param fitness: Fitness of the population
        :type fitness: list

        :return: list of selected individuals to pass to the next generation.
        :rtype: list
        """


        # Sort individuals by fitness
        sorted_fitness = sorted(fitness, key=lambda x: x[0])
        # Initialize variables
        selected = []
        selection_fitness = []
        for i, (f, ind) in enumerate(sorted_fitness):
            # Calculate selection probability
            p = ((2 - s) / len(sorted_fitness)) + ((2 * i * (s - 1)) / (len(sorted_fitness) * (len(sorted_fitness) - 1)))
            # Random number for selection
            r = random.random()
            if r < p:
                selected.append(ind)
                selection_fitness.append((f, ind))
            # Break the loop if we have selected enough individuals
            if len(selected) == k:
                break
        # If not enough individuals are selected, fill the rest randomly
        if len(selected) < k:
            remaining = k - len(selected)
            random_selection = random.sample(sorted_fitness[-remaining:], remaining)
            selected.extend(ind for f, ind in random_selection)
            selection_fitness.extend(random_selection)

        return selected, selection_fitness
    
    def exponential_ranking_selection(self,fitness, c = 0.5, k = 2):
        """
        Function to apply exponential ranking selection to select the next population.

        :param fitness: Fitness of the population
        :type fitness: list

        :return: list of selected individuals to pass to the next generation.
        :rtype: list
        """

        f = sorted(fitness)
        selected = []
        for i, ft in enumerate(f):
            p = ((c-1)/(c*len(f) -1))(c**(len(f)-i-1))
            r = random.random()
            if r < p: 
                selected.append(ft)
            if len(selected) == k: break
        if len(selected)<k:
            selected = selected + list(f[-(k-len(selected)):])
        return selected
    
    def tournament_selection(self,fitness, k = 20, n= 2, p = 1):
        """
        Function to apply tournament selection to select the next population.

        :param fitness: Fitness of the population
        :type fitness: list

        :return: list of selected individuals to pass to the next generation.
        :rtype: list
        """
        
        selected = []
        while len(selected) < n:
            sample = random.sample(fitness, k = k)
            r = random.random()
            if r < p:
                for j in range(1, len(sample)):
                    best = sorted(sample)[-j]
                    if best not in selected:
                        selected.append(best)
                        break
        return selected
    
    def PMX_crossover(self, parent1, parent2, seed=42):
        '''
        Function to apply the PMX crossover

        :param parent1: First parent.
        :type parent1: list
        :param parent2: First parent.
        :type parent2: list

        :return: The children.
        :rtype: list,list
        '''
        rng = np.random.default_rng(seed=seed)
        next = -1
        for i in range(len(parent1)):
            if parent1[i] == 0:
                parent1[i]= next
                next -= 1
        
        next = -1
        for i in range(len(parent2)):
            if parent2[i] == 0:
                parent2[i]= next
                next -= 1
        parent1, parent2 = np.array(parent1, dtype=int), np.array(parent2, dtype = int)

        cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(parent1)+1), size=2, replace=False))

        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

            # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
                candidate = p2[i]
                while candidate in p1[cutoff_1:cutoff_2] and candidate: # allows for several successive mappings
                    #print(f"Candidate {candidate} not valid in position {i}") # DEBUGONLY
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[i] = candidate
            offspring = list(offspring)
            offspring = [x*int(x > 0) for x in offspring]
            return offspring

        offspring1 = PMX_one_offspring(parent1, parent2)
        offspring2 = PMX_one_offspring(parent2, parent1)

        return offspring1, offspring2
    
    def flip_insert_mutation(self,chromosome):
        """
        Function to apply flip insert mutation.

        :param chromosome: A given chromosome.
        :type chromosome: list
        """
        new = np.array([0]+chromosome)
        idx = np.where(new == 0)[0]
    
        separations = np.split(new, idx)
        subroutes = [subarray.tolist()[1:] for subarray in separations if len(subarray) > 0]
        mutated_salesman = []
        new_salesman = [0]*len(chromosome)
        for sub in subroutes: 
            mutated_salesman.extend(sub)
      
        id1, id2 = random.sample(range(1, len(mutated_salesman)-1), 2)
        mutated_salesman[id1],mutated_salesman[id2] = mutated_salesman[id2],mutated_salesman[id1]
        
        mutated_salesman[id1:id2+1] = reversed(mutated_salesman[id1:id2+1])

        new_ids = []
        for id in idx[1:]:
            if id-1 < len(mutated_salesman)-1 and id-1 > 1:
                id_n = id-1 + random.choice([-1,0,1])
            elif id-1 >= len(mutated_salesman)-1:
                id_n = id-1 - random.choice([0,1])
            elif id-1 <= 1:
                id_n = id-1 + random.choice([0,1])
            new_ids.append(id_n)
        
      
        j = 0
        for i,value in enumerate(mutated_salesman):
            if i in new_ids:
                new_salesman[j] = 0
                j += 1
            new_salesman[j] = value
            j += 1
            
        return new_salesman
    
    
    def run(self,individuals=300, crossovers= 1, max_iter=10000000, objective_value=0.2,proba_selection = [0.5,0.5]):
        """
        Function to run the algorithm.

        """
        

        '''Initialize population'''
        n_location,n_vehicles,instance = self.read_problem_instance()     
        if self.init == "normal":     
            population = self.create_population(n_location,n_vehicles,individuals) 
        else:
            population = self.cluster_population(mode=self.mode,individuals=individuals,n_location=n_location,n_vehicles=n_vehicles,distance_matrix=instance)
        
        '''Evaluate each solution and creating list with the fitness'''
        fitness = []
        self.evolution = []
        
        for s in population:                                                                                                                                                          
            f = self.fitness(s,instance)
            fitness.append((f, s))
            if self.best_fitness == None or f > self.best_fitness:          
                self.best_fitness = f
                self.best_solution = s
                
        n_iter = 0
        while (self.best_fitness is not None and self.best_fitness < objective_value and n_iter < max_iter) or  self.best_fitness is None: #Termination condition
            print("Iteration: ", n_iter, end = "\r")
            self.evolution.append(self.best_fitness)
           
                    
            for i in range(crossovers):  #Iterations specified in the configuration (10 by default)
                
                
                '''SELECT parent'''
                if random.random() < proba_selection[0]:
                    parents, _ = self.fitness_proportion_ranking_selection(fitness, k = 2)
                    parent1 =  parents[0]        
                    parent2 =  parents[1] 
                else:
                    parents = self.select_parent(fitness, n = 2)
                    parent1 =  parents[0]        
                    parent2 =  parents[1]   

                                                   

                '''CROSSOVER'''                                                                                            
                child1, child2 = self.inspired_crossover_DPX(parent1, parent2)
                child3, child4 = self.PMX_crossover(parent1, parent2)

                
                '''MUTATION''' 
              
                _child1, n1 = self.extract_chromosome(child1)
                _child2, n2 = self.extract_chromosome(child2)
                _child3, n3 = self.extract_chromosome(child3)
               
                _child1[n1] = self.in_route_mutation(_child1[n1])
                _child2[n2] = self.in_route_mutation(_child2[n2])
                _child3[n3] = self.in_route_mutation(_child3[n3])

                child1 = self.inverted_transformation(_child1)
                child2 = self.inverted_transformation(_child2)
                child3 = self.inverted_transformation(_child3)

                if random.random() < 0.5:
                    child1 = self.cross_route_mutation(child1)
                    child2 = self.cross_route_mutation(child2)
                    child3 = self.cross_route_mutation(child3)
                
                if random.random() < 0.5:
                    child1 = self.inroute_opt2_mutation(child1)
                    child2 = self.inroute_opt2_mutation(child2)
                    child3 = self.inroute_opt2_mutation(child3)

                if random.random() < 0.5:
                    child1 = self.flip_insert_mutation(child1)
                    child2 = self.flip_insert_mutation(child2)
                    child3 = self.flip_insert_mutation(child3)

                '''EVALUATION'''
                
                f1 = self.fitness(child1, instance)
                f2 = self.fitness(child2, instance)
                f3 = self.fitness(child3, instance)

                idx1, idx2, idx3 = self.inverse_fitness_proportion_ranking_selection(fitness+ [(f1, child1), (f2, child2), (f3, child3)], k = 3)
                if idx1 < len(fitness): fitness[idx1] = (f1, child1)
                if idx2 < len(fitness): fitness[idx2] = (f2, child2)
                if idx3 < len(fitness): fitness[idx3] = (f3, child3)

                if self.best_fitness == None or f1 > self.best_fitness:
                    self.best_fitness = f1
                    self.best_solution = child1

                if self.best_fitness == None or f2 > self.best_fitness:
                    self.best_fitness = f2
                    self.best_solution = child2

                if self.best_fitness == None or f3 > self.best_fitness:
                    self.best_fitness = f3
                    self.best_solution = child3
            
                

                    
            n_iter += 1

        return self.translate_solution(self.best_solution)

    


if __name__ == '__main__':
    a = GA(time_deadline=180,problem_path='instances/instance1.txt')
    try:print(a.run(),1/a.best_fitness)
    except:print(a.get_best_solution(),1/a.best_fitness)

    