import einops
import numpy as np
import tqdm
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

# 继承RealImageTactileDataset，解析 shape_meta，从 zarr 加载数据，滑动窗口采样，构建 obs/extended_obs/action，__getitem__ 返回每个 batch
class RealImageTactileLatentDiffusionDataset(RealImageTactileDataset):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """ 重写 get_normalizer()，在计算归一化参数时额外用 AT encoder 把所有 action 编码成 latent_action，并计算 latent_action 的归一化统计量
            这是因为 LDP 的训练目标是在潜空间 Z 做扩散，需要提前知道 Z 的分布范围来做归一化，而普通 DP 不需要这一步。
        """
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []

        for data in tqdm.tqdm(self, leave=False, desc='Calculating latent action for normalizer'):
            action = data['action'].to(self.at.device).unsqueeze(0)
            action = normalizer['action'].normalize(action)
            latent_action = self.at.encoder(
                self.at.preprocess(action / self.at.act_scale)
            )
            if self.at.use_vq:
                if not self.use_latent_action_before_vq:
                    latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
            else:
                latent_action, _ = self.at.quant_state_without_vq(latent_action)
            if self.at.use_conv_encoder:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
            else:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
            latent_action_all.append(latent_action[0].cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer
