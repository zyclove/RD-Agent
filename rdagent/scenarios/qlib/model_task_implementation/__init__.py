from rdagent.components.task_implementation.model_implementation.one_shot import ModelTaskGen
from rdagent.components.task_implementation.model_implementation.task import ModelImplTask, ModelTaskImpl
from rdagent.core.proposal import Imp2Feedback


class QlibMOdelTask(ModelImplTask):
    """
    Describe a task to implement a Qlib model
    """


class QlibModelTaskGen(ModelTaskGen):
    """
    Based on you task to generate the model code
    """

    def generate(self, task_l):
        return super().generate(task_l)


class QlibModelTaskImpl(ModelTaskImpl):
    """
    Docker-based Qlib image

    code should be implementd in a folder, 
    Then the folder should be mounted into a Qlib docker image
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def prepare(self):
        pass

    def execute(self):
        """
        docker run  -v <path to model>:<model in the image> qlib_image
        """


class QlibMImp2Feedback(Imp2Feedback):
     ...
