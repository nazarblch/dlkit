import datetime
import os
import shutil
import yaml

import torch
from tensorboardX import SummaryWriter


class Writer:

    def __init__(self, dirname, test=False):
        import __main__
        timestamp = datetime.datetime.now()
        dirname = os.path.join(dirname, ('test_' if test else '') + timestamp.strftime('%y%m%d%H%M%S'))
        os.makedirs(dirname)
        script_src = __main__.__file__
        script_dst = os.path.join(dirname, os.path.basename(script_src))
        shutil.copy2(script_src, script_dst)
        self.script = script_dst
        self.main = __main__
        self.writer = SummaryWriter(log_dir=dirname)
        self.dirname = dirname
        self.save_params()

    def save_params(self):
        params = {k: v for k, v in vars(self.main).items()
                  if k.isupper() and isinstance(v, (int, float, str, bool, tuple, list, dict))}
        params['comment'] = getattr(self.main, '__doc__', '')
        params['path'] = self.dirname
        params['script'] = self.script
        with open(os.path.join(self.dirname, 'params.yaml'), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

    def add_scalar(self, tag, value, iteration):
        self.writer.add_scalar(tag, value, iteration)

    def add_image(self, *args, **kwargs):
        self.writer.add_image(*args, **kwargs)

    def add_model(self, tag, model, iteration=None):
        fname = f'{tag}.statedict' if iteration is None else f'{tag}-{iteration}.statedict'
        torch.save(model.state_dict(), os.path.join(self.dirname, fname))
