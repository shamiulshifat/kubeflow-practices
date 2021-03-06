from kfp import dsl
from kfp import compiler
#1st component preprocess

def data_process(dataset_url):

    return dsl.ContainerOp(
        name='supa-mega IO test',
        image='shamiulshifat/supamega:v1',
        arguments=[
            '--dataset_url', dataset_url
        ],
        file_outputs={
            'dataset': '/app/salary_modified.csv'
        }
    )



# create complete pipeline

@dsl.pipeline(
   name='supabase-mega Pipeline',
   description='An example pipeline that receives and transmits dataset.'
)
def supamega_pipeline(dataset_url:str):
    _data_process = data_process(dataset_url)


#compile yaml file
# Compile the pipeline
pipeline_func = supamega_pipeline
pipeline_filename =  'supamegapipeline3.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)
