import torch.nn as nn
import torch


class IncrementalClassifier(nn.Module):
    def __init__(
        self, in_features, initial_out_features=2, masking=True, mask_value=-1000
    ):
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value

        self.classifier = nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)
    
    @torch.no_grad()
    def adapt(self, targets):
        in_features = self.classifier.in_features
        old_classes = self.classifier.out_features
        new_classes = max(self.classifier.out_features, max(targets) + 1)
        device = self.classifier.weight.device

        if self.masking:
            if old_classes != new_classes:
                old_active_units = self.active_units
                self.active_units = torch.zeros(
                    new_classes, dtype=torch.int8, device=device
                )
                self.active_units[: old_active_units.shape[0]] = old_active_units

            if self.training:
                self.active_units[list(targets)] = 1

        if old_classes == new_classes:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = nn.Linear(in_features, new_classes).to(device)
        self.classifier.weight[:old_classes] = old_w
        self.classifier.bias[:old_classes] = old_b

    def forward(self, x):
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask, self.mask_value)
        return out
