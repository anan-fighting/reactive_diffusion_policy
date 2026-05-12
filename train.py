"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

--config-name 是 Hydra 自己的内置参数，不是 train.py 定义的
而 task=xxx、dataloader.batch_size=4 这种不带 -- 的参数是 Hydra 的"覆盖语法"，会动态覆盖配置文件中的对应字段，也不需要在 train.py 里声明
"""

import sys
# use line-buffering for both stdout and stderr
# 设置行缓冲（buffering=1）输出，print 会立刻输出，不缓存，训练时能实时看到日志，不会卡住不打印
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# 固定在reactive_diffusion_policy/config目录下找配置文件
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'reactive_diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_) # 从配置文件里读类名，根据_target_的值
    print(f"--- Load cls: ", cls)
    workspace: BaseWorkspace = cls(cfg) # 实际实例化的是子类
    workspace.run()

if __name__ == "__main__":
    main()
