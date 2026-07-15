import numpy as np

class HeterogeneityProgramInterface:
    def __init__(self, _path_template: str, _program_loading_params: dict):
        pass

    def prepare_heterogeneity_program(self, **kwargs) -> object:
        return None

    def decode_state_from_latent(self, latent: np.array) -> None:
        return np.random.uniform([64, 64, 64])
    
