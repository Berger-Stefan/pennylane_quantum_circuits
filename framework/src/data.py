import torch
import itertools

class Data:
    def __init__(self,  domain_dict:dict) -> None:
        self.domain_dict = domain_dict
        self.build_domain()
    
    def build_domain(self):
        tmp = []
        for i in self.domain_dict.values():
            tmp.append(torch.linspace(i[0],i[1],i[2]))
        self.domain = torch.tensor([x for x in itertools.product(*tmp)], requires_grad=True)