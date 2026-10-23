import torch
from src.planning.tcp_spline_objective import tcp_acceleration_cost,tcp_jerk_cost


def test_linear_zero_and_quadratic_interior():
    t=torch.linspace(0,1,101,dtype=torch.float64)
    p=torch.zeros((1,101,1,3),dtype=torch.float64);p[0,:,0,0]=t
    assert float(tcp_acceleration_cost(p,torch.tensor(.01)).max())<1e-15
    p[0,:,0,0]=t*t
    torch.testing.assert_close(tcp_acceleration_cost(p,torch.tensor(.01,dtype=torch.float64))[:,1:-1],torch.full((1,99,1),4.,dtype=torch.float64),atol=1e-9,rtol=1e-9)


def test_gradient():
    torch.manual_seed(8)
    p=torch.randn((1,8,1,3),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:tcp_acceleration_cost(x,torch.tensor(.2,dtype=torch.float64)),(p,))


def test_tcp_jerk_polynomial_and_gradient():
    t=torch.linspace(0,1,101,dtype=torch.float64)
    p=torch.zeros((1,101,1,3),dtype=torch.float64);p[0,:,0,0]=t**3
    torch.testing.assert_close(tcp_jerk_cost(p,torch.tensor(.01,dtype=torch.float64))[:,3:-3],
                              torch.full((1,95,1),36.,dtype=torch.float64),atol=1e-7,rtol=1e-7)
    p=torch.randn((1,9,1,3),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:tcp_jerk_cost(x,torch.tensor(.2,dtype=torch.float64)),(p,))
