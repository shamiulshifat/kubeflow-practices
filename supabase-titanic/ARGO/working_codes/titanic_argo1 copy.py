## this pipeline works on argo perfectly and takes very small time.. mega removed.

from ast import arguments
import kfp
from kfp import dsl
from kfp import compiler
#1st component preprocess
def preprocess_op(traindata_url, testdata_url):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='shamiulshifat/titanicprocess:v1',
        command=[ "python", "dataprocess.py", "--traindata_url", traindata_url, "--testdata_url", testdata_url ],
        file_outputs={
            'X_train': '/app/X_train.npy',
            'X_valid': '/app/X_valid.npy',
            'Y_train': '/app/Y_train.npy',
            'Y_valid': '/app/Y_valid.npy',
            'X_test' : '/app/X_test.npy',
        }
    )

#2nd component train

def train_op(param,X_train, Y_train,X_valid,Y_valid):

    return dsl.ContainerOp(
        name='Train Model',
        image='shamiulshifat/titanictrain:nomega',
        command=[ "python", "train.py", "--param", param, "--X_train", X_train, "--Y_train", Y_train, "--X_valid", X_valid, "--Y_valid", Y_valid],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )


#3rd component test

def deploy_op(X_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='shamiulshifat/deploy:nomega',
        command=[ "python", "deploy.py", "--X_test", X_test, "--model", model],
        file_outputs={
            'output': '/app/titanic_predictions.csv'
        }
    )

# create complete pipeline

@dsl.pipeline(
   name='TITANIC Survivor Prediction Pipeline',
   description='An example pipeline that trains and logs a ML model.'
)
def titanic_pipeline(traindata_url, testdata_url, param_url):
    '''Enter your training dataset and test dataset URL here'''
    _preprocess_op = preprocess_op(traindata_url, testdata_url)
    
    _train_op = train_op(param_url,
        dsl.InputArgumentPath(_preprocess_op.outputs['X_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['Y_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['X_valid']),
        dsl.InputArgumentPath(_preprocess_op.outputs['Y_valid'])

    ).after(_preprocess_op)

    _test_op = deploy_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['X_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)


#compile yaml file



if __name__ == '__main__':
  workflow_dict = kfp.compiler.Compiler()._create_workflow(titanic_pipeline)
  workflow_dict['metadata']['namespace'] = "argo"
  del workflow_dict['spec']['serviceAccountName']
  kfp.compiler.Compiler._write_workflow(workflow_dict, "titanic_pipe_nomega.yaml")
 