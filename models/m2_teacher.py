import torch
import math
from torch.nn import Module
import numpy as np

class M2Teacher(Module):
    def __init__(self,student_model,teacher_model,t_total,args,current_train_iter):

        super(M2Teacher, self).__init__()
        self.total_iters = t_total
        self.param_momentum = 0.99
        self.current_train_iter = current_train_iter
        self.perserving_rate = 0.8

        self.student_encoder = student_model
        self.teacher_encoder = teacher_model
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        #self.momentum_update(m=0)

    @torch.no_grad()
    def momentum_update(self, m):
        for p1, p2 in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            # p2.data = m * p2.data + (1.0 - m) * p1.detach().data
            tmp_prob = np.random.rand()
            if tmp_prob < self.perserving_rate:
                pass
            else:
                p2.data = m * p2.data + (1.0 - m) * p1.detach().data




    def get_param_momentum(self):
        return 1.0 - (1.0 - self.param_momentum) * (
            (math.cos(math.pi * self.current_train_iter / self.total_iters) + 1) * 0.5
        )

    def forward(self,inputs, update_param=True):
        if update_param:
            current_param_momentum = self.get_param_momentum()
            self.momentum_update(current_param_momentum)


        outputs = self.teacher_encoder(**inputs)

        return outputs


    
    