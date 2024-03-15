from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gymnasium as gym
from ray.rllib.utils.typing import ModelConfigDict
import pickle

class ModelRRLIBConfig:
    def __init__(self) -> None:
        pass

    def asDict(self):
        return self.__dict__

class ModelRLLIB(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space = None,
        action_space: gym.spaces.Space = None,
        num_outputs: int = None,
        model_config: ModelConfigDict = None,
        name: str = None):
        super().__init__(
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name
            )
        #super is not calling nn.Module init for unknown reasons
        nn.Module.__init__(self)

    def getModelSpecificParams(self):
        return self.__dict__
    
class ModelConfig:
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        return ''

    def asDict(self):
        return self.__dict__

class Model(nn.Module):
    def __init__(
        self):
        super().__init__()

    @classmethod
    def load_model(cls, cp):
        x = None
        with open(cp/"params.pkl",'rb') as f:
            x = pickle.load(f)
        
        if i == 0:
            dataset = x['dataset'](**x['dataset_config'])
            gen = torch.Generator().manual_seed(x['seed'])
            trainset, valset, testset = random_split(dataset, x['data_split'], generator=gen)
            if not x['shuffle']:
                # Improves data gathering speeds. Selected indices for each set are still random.
                trainset.indices = sorted(trainset.indices)
                valset.indices = sorted(valset.indices)
                testset.indices = sorted(testset.indices)
            dl_kwargs = dict(
                batch_size=1,
                shuffle=x['shuffle'],
                collate_fn=collate_wrapper,
                pin_memory=x['pin_memory'],
                num_workers=0,
                # prefetch_factor=1
                )
            trainloader = DataLoader(trainset, **dl_kwargs)
            valloader = DataLoader(valset, **dl_kwargs)
            testloader = DataLoader(testset, **dl_kwargs)
            inp_sample = next(iter(trainloader)).inp

        enc = x['model'](input_sample=inp_sample, config=x['model_config'])
        dec = AutoEncoder.create_decoder(enc, inp_sample)

        latest_checkpoint = sorted(cp.glob('checkpoint*'), reverse=True)[0]

        enc.load_state_dict(torch.load(latest_checkpoint/'model.pt'))

    def getModelSpecificParams(self):
        return self.__dict__