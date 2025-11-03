from torch import nn
from utils.log import logger

class ExampleArchitecture(nn.Module):

    #recall that the model must require two parameters: model_params and metadata
    def __init__(self, model_params, metadata):
        super(ExampleArchitecture, self).__init__()
        input_size = metadata.get("input_size", 13)  # default input size if not provided
        hidden_size = model_params.get("hidden_size", 64)
        output_size = metadata.get("num_classes", 3)  # default number of classes if not provided
        num_hidden_layers = model_params.get("num_hidden_layers", 2)

        # add all model parameters to logger
        logger.log_dict(model_params)
        layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)