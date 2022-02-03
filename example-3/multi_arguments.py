from typing import NamedTuple
import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op


#function 1
@func_to_container_op
def produce_two_small_outputs() -> NamedTuple('Outputs', [('text', str), ('number', int)]):
    return ("data 1", 42)

#function 2
@func_to_container_op
def consume_two_arguments(text:str, number:int):
    print('Text={}'.format(text))
    print('Number={}'.format(str(number)))
    

#function 3
@func_to_container_op
def produce_one_small_output() -> str:
    return 'no inputs taken'


#build pipeline
@dsl.pipeline(
   name='multi-arguments I/O pipeline',
   description='A demo pipeline for I/O.'
)
def make_pipeline(text:str):
    '''pipeline that passes data from producer to consumer'''
    produce1_task = produce_one_small_output()
    produce2_task = produce_two_small_outputs()

    consume_task1 = consume_two_arguments(produce1_task.output, 42)
    consume_task2 = consume_two_arguments(text, produce2_task.outputs['number'])
    consume_task3 = consume_two_arguments(produce2_task.outputs['text'], produce2_task.outputs['number'])
    consume_task4 = consume_two_arguments(produce1_task.output, produce2_task.outputs['number'])


##compile to yaml file
# Compile the pipeline
pipeline_func = make_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline4.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)
