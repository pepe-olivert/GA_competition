
class GA:

    def read_problem_instance(self,problem_path):
        """
        TODO: Implementar método para leer una instancia del problema
        y ajustar los atributos internos del objeto necesarios
        """
        pass

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
        self.best_solution = None #Atributo para guardar la mejor solución encontrada
        self.time_deadline = time_deadline # Límite de tiempo (en segundos) para el cómputo del algoritmo genético
        #TODO: Completar método para configurar el algoritmo genético (e.g., seleccionar cruce, mutación, etc.)



    

