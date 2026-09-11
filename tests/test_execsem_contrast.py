import importlib.util
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('contrast', 'scripts/221_execsem_contrast.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    rows = []
    for task, ac_ranks, wa_ranks in [('a', [1,1], [4]), ('b', [4], [1]), ('excluded',[1],[])]:
        for label, ranks in [(1, ac_ranks),(0,wa_ranks)]:
            for i, rank in enumerate(ranks):
                for kind in ['j-lens','logit-lens']:
                    rows.append(dict(id=f'{task}-{label}-{i}',task_id=task,label=label,
                                     layer=13,kind=kind,candidate_ranks={'positive':{'correct':rank if kind=='j-lens' else 2,'unavailable':None}}))
    return dict(rows=rows,tokenization={'positive':{'correct':[1],'unavailable':[]}})


def test_equal_problem_weights_paired_baseline_and_exclusions():
    result=m.analyze(fixture(),repetitions=100)
    rows=[r for r in result['summary'] if r['feature']=='correct' and r['metric']=='log_rank']
    assert all(r['n_problems']==2 and abs(r['mean'])<1e-10 for r in rows)
    pairs=[r for r in result['pairs'] if r['task_id']=='a' and r['feature']=='correct' and r['metric']=='log_rank']
    values={r['kind']:r['ac_minus_wa'] for r in pairs}
    assert values['j-lens']==pytest.approx(np.log(4))
    assert values['j-minus-logit']==values['j-lens']
    assert len(result['excluded'])==1
    assert len(result['unsupported_words'])==1
    assert result==m.analyze(fixture(),repetitions=100)


def test_unmatched_readouts_rejected():
    data=fixture(); data['rows'].pop()
    with pytest.raises(ValueError,match='submissions differ'):
        m.analyze(data,repetitions=100)


def test_no_pairs_and_invalid_rank_rejected():
    data=fixture(); data['rows']=[r for r in data['rows'] if r['label']==1]
    with pytest.raises(ValueError,match='No problems'):
        m.analyze(data,repetitions=100)
    data=fixture(); data['rows'][0]['candidate_ranks']['positive']['correct']=None
    with pytest.raises(ValueError,match='rank'):
        m.analyze(data,repetitions=100)
