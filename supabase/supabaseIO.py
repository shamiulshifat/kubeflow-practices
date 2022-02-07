import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components
from kfp.components import func_to_container_op

@func_to_container_op
def data_process(link: str):
    '''Print the original dataset'''
    import pandas as pd
    import numpy as np
    from mega import Mega
    df=pd.read_csv(link)
    print(df.to_string()) 

    #add a new column with random integers
    df['NewNumCol'] = np.random.choice([1, 9, 20], df.shape[0])
    print(df.to_string()) 
    '''Print the modified dataset'''
    #send to mega
    mega = Mega()
    email='shifat@betterdata.ai'
    password='shadia1afshan2raisha3'
    m = mega.login(email, password)
    df.to_csv('salary_modified.csv')
    file_name='salary_modified.csv'
    # upload dataset
    m.upload(file_name)


#build pipeline
@dsl.pipeline(
   name='data in and out test on Kubeflow',
   description='A demo pipeline for receiving and sending dataset on Kubeflow.'
)
def make_pipeline( link:str):
    send_dataset=data_process(link)



#compile to yaml file
# Compile the pipeline
pipeline_func = make_pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline_supa.yaml'
compiler.Compiler().compile(pipeline_func, pipeline_filename)
