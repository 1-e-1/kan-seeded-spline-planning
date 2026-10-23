import numpy as np
import pytest
from src.planning.cubic_execution_curve import CubicExecutionCurve
from src.planning.c2_trajectory_buffer import C2TrajectoryBuffer


def curves():
    original=CubicExecutionCurve(np.linspace(0,1,32)[:,None]*np.ones((1,8)),np.zeros(8),np.ones(8),6.)
    state=original.evaluate(2.)
    replacement=CubicExecutionCurve(np.linspace(state['position'],np.ones(8)*.7,32),
        state['position'],np.ones(8)*.7,4.,state['velocity'],state['acceleration'])
    return original,replacement


def test_continuous_handoff_no_change_to_prefix_or_output():
    original,replacement=curves();buffer=C2TrajectoryBuffer(original)
    assert buffer.schedule(replacement,2.,1.,True)['accepted']
    for key,value in buffer.advance(1.5).items():np.testing.assert_allclose(value,original.evaluate(1.5)[key])
    before=original.evaluate(2.);after=buffer.advance(2.)
    for key in ('position','velocity','acceleration'):np.testing.assert_allclose(before[key],after[key],atol=1e-11)
    for key,value in buffer.advance(2.5).items():np.testing.assert_allclose(value,replacement.evaluate(.5)[key])
    final=buffer.advance(10.)
    np.testing.assert_allclose(final['position'],.7)
    assert all(np.max(abs(final[k]))==0 for k in ('velocity','acceleration','jerk'))


@pytest.mark.parametrize('verified,ready,reason',[(False,1.,'not_verified'),(True,2.1,'late_or_stale')])
def test_rejection_keeps_original(verified,ready,reason):
    original,replacement=curves();buffer=C2TrajectoryBuffer(original)
    outcome=buffer.schedule(replacement,2.,ready,verified)
    assert not outcome['accepted'] and outcome['reason']==reason
    np.testing.assert_allclose(buffer.advance(3.)['position'],original.evaluate(3.)['position'])


def test_rejects_boundary_mismatch_and_clock_reversal():
    original,_=curves();buffer=C2TrajectoryBuffer(original)
    wrong=CubicExecutionCurve(np.zeros((32,8)),np.zeros(8),np.ones(8),4.)
    assert buffer.schedule(wrong,2.,1.,True)['reason']=='boundary_mismatch'
    buffer.advance(1.)
    with pytest.raises(ValueError):buffer.advance(.5)
