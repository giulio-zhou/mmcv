import torch
import torch.nn as nn

class EnsembleAveraging(nn.Module):
    def __init__(self, models):
        super(EnsembleAveraging, self).__init__()
        self.models = models
        for i, model in enumerate(self.models):
            self.add_module('model_%d' % i, model)

    def forward(self, x):
        model_logits = []
        for i, model in enumerate(self.models):
            device = next(model.parameters()).device
            model_logits.append(model.forward(x.to(device)))
        mean_logits = torch.mean(torch.stack(model_logits), 0)
        return mean_logits
