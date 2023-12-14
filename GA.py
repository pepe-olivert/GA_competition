import numpy as np


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
        #TODO: Completar método para configurar el algoritmo genético (e.g., seleccionar cruce, mutación, etc.)
        

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
        #TODO
        pass


    def run(self):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        self.read_problem_instance(self.problem_path)
        pass

    def fitness(self,solution,dist_matrix):
        """
        If the solution is in format : [[0,2,3,4,0][0,5,6,7,0]] We should calculate the fitness of that solution
        as 1/total distance traveled by that salesman
        """
        total_distance = 0
        for salesman in solution:
            for i in range(len(salesman) - 1):
                origin = salesman[i]
                destination = salesman[i + 1]
                total_distance += dist_matrix[origin][destination]
        return 1 / total_distance
