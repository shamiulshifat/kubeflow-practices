from typing import NamedTuple
import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op


# a simple python function
#consuming data
@func_to_container_op
def print_small_text(text: str):
    '''Print small text'''
    print(text)

def constant_to_consumer_pipeline():
    '''Pipeline that passes small constant string to to consumer'''
    consume_task = print_small_text('Hello world') # Passing constant as argument to consumer


#producing data
@func_to_container_op
def produce_one_small_output(text:str) -> str:
    return text

#save data in txt
@func_to_container_op
def save_text(text:str):
    data=open("textdata.txt", "w")
    temp=data.write(text)
    data.close()
    print(temp)


#build pipeline
@dsl.pipeline(
   name='simple text flow pipeline',
   description='A demo pipeline for text print.'
)
def make_pipeline(
    text:str
):
    '''Pipeline that passes small data from producer to consumer'''
    produce_task = produce_one_small_output(text)
    # Passing producer task output as argument to consumer
    consume_task1 = print_small_text(produce_task.output) # task.output only works for single-output components
    consume_task2 = print_small_text(produce_task.outputs['output']) # task.outputs[...] always works
    save_task=save_text(produce_task.output)




#compile to yaml file
# Compile the pipeline
pipeline_func = make_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline3.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)