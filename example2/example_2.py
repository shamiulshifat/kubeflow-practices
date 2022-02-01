import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components


#setup
EXPERIMENT_NAME = 'A Simple kubeflow pipeline'        # Name of the experiment in the UI
BASE_IMAGE = 'python:3.7'    # Base image used for components in the pipeline

# a simple python function
@dsl.python_component(
    name='add_op',
    description='adds three numbers',
    base_image=BASE_IMAGE  # we can define the base image here, or when you build in the next step. 
)
def add(a: float, b: float, c:float) -> float:
    '''Calculates sum of two arguments'''
    print(a, '+', b, '+', c,'=', a + b +c)
    return a + b + c

#build a component using the function
# Convert the function to a pipeline operation.
add_op = components.func_to_container_op(
    add,
    base_image=BASE_IMAGE, 
)

#build a pipeline using the component

@dsl.pipeline(
   name='Calculation pipeline',
   description='A demo pipeline that performs arithmetic calculations.'
)
def calc_pipeline(
   a: float =0,
   b: float =7,
   c: float=9
):
    #Passing pipeline parameter and a constant value as operation arguments
    add_task = add_op(a, b, 4) #Returns a dsl.ContainerOp class instance. 
    
    #You can create explicit dependency between the tasks using xyz_task.after(abc_task)
    add_2_task = add_op(a, b, 7)
    
    add_3_task = add_op(7, add_task.output, add_2_task.output)

    add_4_task = add_op(add_task.output, add_3_task.output, 6)

    add_5_task = add_op(add_3_task.output, add_4_task.output, 86)

#compile to yaml file
# Compile the pipeline
pipeline_func = calc_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline2.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)