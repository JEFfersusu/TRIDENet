import torch
import torch.nn as nn
import torch.nn.functional as F


class SCA_base(nn.Module):

    def __init__(
        self,
        channels,
        groups=4,
        reduction=8,
        topk_ratio=0.75,
        tau=0.5,
        residual_alpha=0.1
    ):
        super(SCA_base, self).__init__()

        self.channels = channels
        self.groups = min(groups, channels)
        self.topk = max(1, int(channels * topk_ratio))
        self.tau = tau
        self.residual_alpha = residual_alpha


        if channels % self.groups != 0:
            self.groups = 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        if self.groups > 1:
            channels_per_group = channels // self.groups
            reduced_per_group = max(1, channels_per_group // reduction)

            self.group_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(channels_per_group * 2, reduced_per_group),
                    nn.ReLU(inplace=True),
                    nn.Linear(reduced_per_group, channels_per_group)
                )
                for _ in range(self.groups)
            ])
        else:
            reduced_dim = max(1, channels // reduction)
            self.single_mlp = nn.Sequential(
                nn.Linear(channels * 2, reduced_dim),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_dim, channels)
            )

    def forward(self, x):
        B, C, H, W = x.shape

        # -----------------------
        # Global descriptor
        # -----------------------
        avg_pooled = self.avg_pool(x).view(B, C)
        max_pooled = self.max_pool(x).view(B, C)
        combined = torch.cat([avg_pooled, max_pooled], dim=1)

        # -----------------------
        # Channel score
        # -----------------------
        if self.groups > 1:
            group_feats = torch.chunk(combined, self.groups, dim=1)
            scores = []
            for mlp, feat in zip(self.group_mlp, group_feats):
                scores.append(mlp(feat))
            score = torch.cat(scores, dim=1)
        else:
            score = self.single_mlp(combined)

        # -----------------------
        # Top-K selection
        # -----------------------
        if self.topk < C:
            _, topk_indices = torch.topk(score, self.topk, dim=1)

            if self.training:
                # ===== Soft Top-K (training) =====
                hard_mask = torch.zeros_like(score)
                hard_mask.scatter_(1, topk_indices, 1.0)

                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(score) + 1e-8) + 1e-8
                )
                soft_score = torch.sigmoid((score + gumbel_noise) / self.tau)

                mask = hard_mask * soft_score

            else:
                # ===== Hard Top-K (eval / test) =====
                mask = torch.zeros_like(score)
                mask.scatter_(1, topk_indices, 1.0)

        else:
            # No Top-K constraint
            if self.training:
                mask = torch.sigmoid(score / self.tau)
            else:
                mask = (score > 0).float()

        # -----------------------
        # Apply mask + residual
        # -----------------------
        mask = mask.view(B, C, 1, 1)
        x_out = x * mask + x * (1.0 - mask) * self.residual_alpha

        return x_out


class SCA(nn.Module):

    def __init__(self, dimension=1, enable_cspm=True, cspm_ratio=0.75):
        super().__init__()
        self.d = dimension
        self.enable_cspm = enable_cspm
        self.cspm_ratio = cspm_ratio
        self.cspm_module = None
        
    def forward(self, x: List[torch.Tensor]):

        concatenated = torch.cat(x, self.d)

        if self.enable_cspm:
            if self.cspm_module is None:
                channels = concatenated.size(1)
                self.cspm_module = SCA_base(
                    channels=channels,
                    groups=min(4, channels),
                    topk_ratio=self.cspm_ratio,
                    tau=0.5,
                    residual_alpha=0.1
                ).to(concatenated.device)
            
            concatenated = self.cspm_module(concatenated)
        
        return concatenated
