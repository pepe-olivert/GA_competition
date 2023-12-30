import click
from GA import GA
from func_timeout import func_timeout, FunctionTimedOut #Requires pip install
import time

@click.command()
@click.option(
    "-d",
    "--deadline",
    type=int,
    default=180,
    help="Execution deadline"
)
@click.option(
    "-i",
    "--instance",
    type=click.Path(exists=True),
    required=True,
    help="Path to the problem instance to be solved"
)
def run_genetic_algorithm(deadline, instance):
    ga = GA(deadline, instance)
    total_time = None
    try:
        t1 = time.time()
        func_timeout(deadline,ga.run)
        total_time = time.time() - t1
    except FunctionTimedOut:
        total_time = deadline
    #TODO: Lo que quieras hacer después de ejecutar el algoritmo genético
    

    
        
        
    
if __name__ == "__main__":
    run_genetic_algorithm()
    
    