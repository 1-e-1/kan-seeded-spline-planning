"""Fixed-shape CUDA Graph inference; identical frozen network and observations."""
import torch


class GraphPolicy:
    def __init__(self,actor):
        assert not actor.training and not any(p.requires_grad for p in actor.parameters())
        self.actor=actor;self.config=actor.config;self.graph=None

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self,batch):
        assert all(v.shape[0]==1 for v in batch.values())
        # Existing encoder's batch=1 optimization dynamically removes masked
        # tokens (CUDA nonzero), which cannot be graph-captured. Its ordinary
        # batched path already supports the equivalent masked computation.
        # Duplicate the sample to use that UNMODIFIED path; return only row 0.
        self.inputs={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in batch.items()}
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.actor(self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.outputs={k:v[:1] for k,v in self.actor(self.inputs).items()}
        torch.cuda.current_stream().wait_stream(stream)

    @torch.inference_mode()
    def __call__(self,batch):
        if self.graph is None:self.prepare(batch)
        assert self.inputs.keys()==batch.keys()
        for key,value in batch.items():
            assert value.shape[0]==1 and value.shape[1:]==self.inputs[key].shape[1:] and value.dtype==self.inputs[key].dtype
            self.inputs[key].copy_(value)
        self.graph.replay()
        return self.outputs
