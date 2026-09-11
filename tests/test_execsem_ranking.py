import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('ranking',Path('scripts/222_execsem_ranking.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_pair_weights_and_ranking_ties():
    x=np.array([[3.],[2.],[0.],[0.],[1.]])
    grouped=[('a',[0,1],[2]),('b',[3],[4])]
    d,w=m.differences(x,grouped)
    np.testing.assert_array_equal(d[:,0],[3,2,-1])
    np.testing.assert_array_equal(w,[.5,.5,1])
    assert m.evaluate(x[:,0],grouped)['accuracy']==.5
    assert m.evaluate(np.zeros(5),grouped)['accuracy']==.5


def test_fit_recovers_signal_and_problem_offsets_cancel():
    d=np.array([[2.,0.],[3.,1.],[4.,-1.]])
    w=m.fit_direction(d,np.ones(3),.1)
    assert np.all(d@w>0)
    x=np.array([[3.,2.],[1.,2.]])
    grouped=[('a',[0],[1])]
    np.testing.assert_allclose(m.differences(x,grouped)[0],m.differences(x+100,grouped)[0])


def test_groups_never_cross_split_and_skip_missing_class():
    rows=[dict(task_id='a',split='train',label=1),dict(task_id='a',split='train',label=0),dict(task_id='b',split='val',label=1)]
    assert m.groups(rows,'train')==[('a',[0],[1])]
    assert m.groups(rows,'val')==[]
    with pytest.raises(ValueError):m.differences(np.zeros((3,2)),[])
