from datetime import time

import torch as t
# package nn.module and povide function save and load
class basicModule(t.nn.Moudle):
    def __init__(self,opt=None):
        super(basicModule, self).__init__()
        self.model_name=str(type(self)) # module's name

    def load(self,path):
        # load module path
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        # USE module and time as file's name
        if name is None:
            prefix='ckpt/'+self.model_name+'-'
            name=time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name
