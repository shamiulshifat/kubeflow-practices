import kfp
@kfp.components.func_to_container_op
def print_func(param: int):
  print(str(param))
@kfp.components.func_to_container_op
def list_func(param: int) -> list:
  return list(range(param))
@kfp.dsl.pipeline(name='pipeline')
def pipeline(param: int):
  list_func_op = list_func(param)
  with kfp.dsl.ParallelFor(list_func_op.output) as param:
    print_func(param)
if __name__ == '__main__':
  workflow_dict = kfp.compiler.Compiler()._create_workflow(pipeline)
  workflow_dict['metadata']['namespace'] = "argo"
  del workflow_dict['spec']['serviceAccountName']
  kfp.compiler.Compiler._write_workflow(workflow_dict, 'pipe.yaml')