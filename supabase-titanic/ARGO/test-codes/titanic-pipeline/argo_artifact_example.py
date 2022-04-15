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
    artifact_location = ArtifactLocation.s3(
                            bucket="__argo_bucket_name__",
                            endpoint="s3.amazonaws.com",
                            region="us-west-2",
                            insecure = False,
                            access_key_secret=V1SecretKeySelector(name="__secret_name__", key="aws-access-key-id"),
                            secret_key_secret=V1SecretKeySelector(name="__secret_name__", key="aws-secret-access-key"))
    # config pipeline level artifact location
    conf = dsl.PipelineConf()
    conf = conf.set_artifact_location(artifact_location)

    workflow_dict = kfp.compiler.Compiler()._create_workflow(pipeline,pipeline_conf=conf)
    workflow_dict['metadata']['namespace'] = "default"
    del workflow_dict['spec']['serviceAccountName']
    kfp.compiler.Compiler._write_workflow(workflow_dict, "pipe.yaml")