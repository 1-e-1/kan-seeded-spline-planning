import numpy as np
from src.planning.paired_statistics import paired_success_interval


def test_zero_discordance_does_not_imply_zero_uncertainty():
    small=paired_success_interval(np.ones(35),np.ones(35))
    large=paired_success_interval(np.ones(350),np.ones(350))
    assert small['ci'][0]<-.02 and small['ci'][1]>.02
    assert -.02<large['ci'][0]<0<large['ci'][1]<.02


def test_paired_failure_direction():
    result=paired_success_interval([1,1,0,0],[0,1,1,1])
    assert result['baseline_only_success']==1 and result['candidate_only_success']==2
    assert result['difference']==.25 and result['ci'][0]<=.25<=result['ci'][1]
