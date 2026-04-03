from fluxvla.engines import PROJECTORS
from .linear_projector import LinearProjector


@PROJECTORS.register_module()
class LinearProjectorInference(LinearProjector):

    def prepare_triton(self, prefix='') -> dict:
        return {
            f'{prefix}_w':
            self.projector.weight.data.T.contiguous().bfloat16().cuda(),
            f'{prefix}_b':
            self.projector.bias.data.bfloat16().cuda(),
        }
