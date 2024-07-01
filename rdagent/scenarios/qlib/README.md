

# Introduction

The entry should be in `rdagent/app/model_proposal/run.py`
- It provides a genenral workflow
- `rdagent/scenarios/qlib/`  provides specific implementation
- We'll use settings (`rdagent/app/model_proposal/conf.py`) to control which conceret implementation will be used.


## About Qlib model implementation

We created a template to run easily run models.
- This is a Pytorch model `https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py` provided by Qlib.
  - Current configuration is from `https://github.com/microsoft/qlib/blob/main/examples/benchmarks/MLP/workflow_config_mlp_Alpha158.yaml`
  - You model structure can be implemented in `model.py`
    - You can find a simple version of `model.py` at the end of end of `pytorch_nn.py`
- We can run it via run `cd tpl ; qrun conf.yaml`

