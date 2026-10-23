import torch
from src.planning.normalized_spline_objective import normalized_smoothness_cost


def test_uniform_time_scaling_does_not_change_shape_cost():
    torch.manual_seed(10)
    q=torch.randn((2,11,8),dtype=torch.float64)
    a=torch.randn_like(q);j=torch.randn_like(q);tcp=torch.randn((2,11,1,3),dtype=torch.float64)
    dt=torch.tensor([.02,.04],dtype=torch.float64)
    x=normalized_smoothness_cost(q,a,j,dt,tcp,tcp_ratio=.02,tcp_jerk_ratio=.0002)
    y=normalized_smoothness_cost(q,a/9,j/27,dt*3,tcp,tcp_ratio=.02,tcp_jerk_ratio=.0002)
    torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)


def test_derivative_gradients_and_fixed_endpoint_scaling():
    torch.manual_seed(12);q=torch.randn((1,9,8),dtype=torch.float64)
    a=torch.randn_like(q,requires_grad=True);j=torch.randn_like(q,requires_grad=True)
    dt=torch.tensor([.1],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a,j:normalized_smoothness_cost(q,a,j,dt),(a,j))
