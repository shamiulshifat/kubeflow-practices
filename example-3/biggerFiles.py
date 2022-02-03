from typing import NamedTuple
import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op


# Writing bigger data
@func_to_container_op
def repeat_line(line: str, output_text_path: OutputPath(str), count: int = 10):
    '''Repeat the line specified number of times'''
    with open(output_text_path, 'w') as writer:
        for i in range(count):
            writer.write(line + '\n')


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath()): # The "text" input is untyped so that any data can be printed
    '''Print text'''
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')

#processing bigger files
@func_to_container_op
def split_text_lines(source_path: InputPath(str), odd_lines_path: OutputPath(str), even_lines_path: OutputPath(str)):
    with open(source_path, 'r') as reader:
        with open(odd_lines_path, 'w') as odd_writer:
            with open(even_lines_path, 'w') as even_writer:
                while True:
                    line = reader.readline()
                    if line == "":
                        break
                    odd_writer.write(line)
                    line = reader.readline()
                    if line == "":
                        break
                    even_writer.write(line)

#processing pre-opened bigger files
@func_to_container_op
def split_text_lines2(source_file: InputTextFile(str), odd_lines_file: OutputTextFile(str), even_lines_file: OutputTextFile(str)):
    while True:
        line = source_file.readline()
        if line == "":
            break
        odd_lines_file.write(line)
        line = source_file.readline()
        if line == "":
            break
        even_lines_file.write(line)


#build pipeline
@dsl.pipeline(
   name='Bigger files I/O pipeline',
   description='A demo pipeline for handling bigger I/O.'
)
def make_pipeline():
    repeat_lines_task = repeat_line(line='Hello', count=5000)
    print_text(repeat_lines_task.output) 
    ##########################################
    text = '\n'.join(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])
    split_text_task = split_text_lines(text)
    print_text(split_text_task.outputs['odd_lines'])
    print_text(split_text_task.outputs['even_lines'])
    ############################################
    split_text_task2 = split_text_lines2(text)
    print_text(split_text_task2.outputs['odd_lines']).set_display_name('Odd lines')
    print_text(split_text_task2.outputs['even_lines']).set_display_name('Even lines')

##compile to yaml file
# Compile the pipeline
pipeline_func = make_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline5.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)




