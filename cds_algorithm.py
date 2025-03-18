'''
    This script implements Algorithm 1 of  Andreas (2020).

    References:
    Andreas (2020) Good-Enough Compositional Data Augmentation.
    Link: https://aclanthology.org/2020.acl-main.676/
'''
import typing
from typing import Tuple
from typing import Dict
from typing import Callable
from typing import Generator
from itertools import batched
from itertools import combinations
from functools import reduce
from collections import defaultdict
from random import shuffle
#io
import pickle
from pathlib import Path
import numpy as np



def batch_fragments(seq: Tuple, min_length: int, max_length: int, max_gaps: int) -> Tuple:
    '''
    Takes a sequence of symbols seq and breaks it into batches of length between 
    min_length and max_length.Gaps are implemented by the combination of 
    non-adjacent batches, where the in-between batch is skipped.

    Output: A tuple of tuples containg the indices defining a fragment.
    '''
    assert len(seq) > 1, 'ERROR: Length of seq must be larger than 1.'
    assert min_length > 1 and isinstance(min_length, int), 'ERROR: min_length must be a positive integer > 1.'
    assert max_length > 1 and isinstance(max_length, int), 'ERROR: max_length must be a positive integer > 1.'
    assert max_length >= min_length, 'ERROR: max_length < min_length.'
    assert max_gaps >= 0 and isinstance(max_gaps, int), 'ERROR: max_gaps must be a positive integer.'

    seq_ind = tuple(range(len(seq))) # create set of indices
    frgs = []
    for frg_len in range(min_length, max_length + 1):
        #shift start of seq batching by off
        for off in range(frg_len):
            #break seq in batches of length frg_len 
            new_frgs = [x for x in batched(seq_ind[off:], frg_len) if len(x) == frg_len]
            #create fragments with gaps 
            for gap in range(1, max_gaps + 1):
                n_frgs = 2 * gap + 1
                for off_g in range(0, gap + 1):
                    #create tuples of consecutive batches 
                    #& remove every other batch from these tuples to create gaps x[::2]
                    #& flatten sum(.,())
                    frgs += [sum(x[::2], ()) for x in batched(new_frgs[off_g:], n_frgs) if len(x) == n_frgs]         
            frgs += new_frgs
    return tuple(frgs)


def is_solitaire(seq: Tuple) -> bool:
    '''
    seq is supposed to be a tuple containing a index subset.
    This functions checks wether seq contains an isolated index.
    Example: is_solitaire((1,2,5)) -> True
             is_solitaire((1,2,3)) -> False
    '''
    seq = set(seq) #faster look-up
    for n in seq:
        if not ((n + 1 in seq) or (n - 1 in seq)):
            return True 
    return False


def total_fragments(seq: Tuple, max_frg_len: int, rm_solitaire: bool = True) -> Tuple:
    '''
    Takes a sequence of symbols seq and breaks it into fragments by building 
    all possible index sub-sets with a strictly positive cardiality of at most
    max_freg_len. 
    
    Output: A tuple of tuples containg the indices defining a fragment.
    ''' 
    assert len(seq) > 1, 'ERROR: Length of seq must be larger than 1.'
    assert 0 < max_frg_len < len(seq) and isinstance(max_frg_len, int), 'ERROR: max_gaps must be a integer sadisfying  0 < max_gaps < len(seq).'

    seq_ind = tuple(range(len(seq))) # create set of indices
    frgs = [] 
    for l in range(1, max_frg_len + 1):
        if rm_solitaire:
            frgs += [c for c in combinations(seq_ind, l) if not is_solitaire(c)]
        else:
            frgs += [c for c in combinations(seq_ind, l) ] 
    return tuple(frgs)


def template(seq: Tuple, frg: Tuple, slot_sym: str = '*') -> Tuple:
    '''
    Implements Eq. (2) of Andreas (2020).
    '''
    frg = set(frg) #optimize look up
    tpl = []
    for i in range(len(seq)):
        #if index is not in frg add to tpl
        if i not in frg:
            tpl.append(i)
        #index is in frg and either tpl is empty or gap starts add slot_sym to frg
        elif len(tpl) == 0 or tpl[-1] != slot_sym:
            tpl.append(slot_sym)
    return tuple(tpl)


def environment(tpl: Tuple, window_size: int = 1, slot_sym: str = '*') -> Tuple:
    '''
    Implements Eq. (3) of Andreas (2020).
    environmet is basically a filtering operation, that creates a ball of size 
    window_size around slot symbols in tpl.
    '''
    assert slot_sym in tpl, 'ERROR: There are no slots in tpl.'
    
    #check if tpl is a tuple of tuple. if so flatten tpl
    if any(isinstance(e, tuple) for e in tpl): 
        tpl = sum(map(tuple, tpl),())
    '''    
    ball = set()
    for i in range(len(tpl)):
        if tpl[i] == slot_sym:
            ball = ball.union(set(range(max(0, i - window_size), 
                                        min(len(tpl), i + window_size + 1))))
    return tuple(tpl[i] for i in sorted(ball))
    '''        
    ball = []
    for i in range(len(tpl)):
        if tpl[i] == slot_sym:
            ball_i = range(max(0, i - window_size), min(len(tpl), i + window_size + 1))
            ball.extend(ball_i)
    return tuple(tpl[i] for i in sorted(set(ball)))

    


def get_syms(seq: Tuple, ind: Tuple, slot_sym: str = '*', value_type: str = 'tpl') -> Tuple:
    '''
    Returns the values of seq located at the indices stored in ind. Depending on
    value_type ('frg', 'tpl', 'env') the return value would be:
    a flat tuple ('env'), a tuple of tuple containing slot symbols ('tpl') or 
    a tuple of tuples without slot symbols ('frg').  
    '''
    assert value_type in {'tpl', 'frg', 'env'}, 'ERROR: Undefind value_type requested.'
    assert max([i for i in ind if not isinstance(i, str)]) < len(seq), 'ERROR: Index out of range.'
    assert len(set(i for i in ind if isinstance(i, str))) < 2, 'ERROR: More than 1 slot_sym in ind.'
    assert len(set(i for i in ind if isinstance(i, str) and i != slot_sym)) == 0, 'Error: Wrong slot_sym defined.'


    syms = list()
    chunk = list()
    match value_type:
        case 'env':
            assert any([isinstance(i, str) for i in ind]), 'ERROR: ind does not specify an environment.'
            for i in ind:
                if i == slot_sym:
                    syms.append(slot_sym)
                else:
                    syms.append(seq[i])
        case 'tpl':
            assert any([isinstance(i, str) for i in ind]), 'ERROR: ind does not specify a template.'
            for i in ind:
                if i == slot_sym:
                    #append syms by chunk if the latter is not empty
                    if len(chunk) != 0: syms.append(tuple(chunk))
                    #add slot_sym to sym
                    syms.append(slot_sym)
                    #reset chunk
                    chunk = list()
                else:
                    #append chunk
                    chunk.append(seq[i])      
        case 'frg':
            assert not any([isinstance(i, str) for i in ind]), 'ERROR: ind does not specify a fragment.'
            for i in ind:
                #append chunk if empty or index is consecutive,
                #else append syms by chunk and reset the latter
                if len(chunk) == 0 or (previous_index == i - 1):
                    chunk.append(seq[i])
                else:
                    syms.append(tuple(chunk))
                    chunk = [seq[i]]
                previous_index = i
    if len(chunk) != 0: syms.append(tuple(chunk))
    return tuple(syms)
    

#### geca_generator:
def synthesize(tpl: Tuple, frg: Tuple, slot_sym: str = '*', flatten: bool = True) -> Tuple:
    '''
    Implementation of tpl / frg operation in Andreas (2020)
    '''
    assert len([i for i in tpl if i == slot_sym]) == len(frg), 'Error: Slots do not match fragment.'
    
    slot=0
    syn_example = list()
    for i in tpl:
        if i == slot_sym:
            #add segment from fragment 
            if flatten:
                syn_example.extend(frg[slot])
            else:
                syn_example.append(frg[slot])
            slot +=1
        else:
            #add segment form template
            if flatten:
                syn_example.extend(i,)
            else:
                syn_example.append(i,)

    return tuple(syn_example)


def indexify(tpl, frg, symseq_to_igoldseq, slot_sym = '*'):
    '''
    tpl / frg ergibt eine symbolosieret gold seq.
    '''

    gs_id = symseq_to_igoldseq[synthesize(tpl, frg, slot_sym)]

    n_pos = sum([len(c) for c in tpl + frg if c != slot_sym])
    indices = iter(range(n_pos))
    tpl_index = []
    frg_index = []  

    slot = 0
    for t in tpl:
        if t == slot_sym:
            f = frg[slot]
            frg_index.append([next(indices) for p in f])
            tpl_index.append(slot_sym)
            slot += 1
        else:
            tpl_index.append([next(indices) for p in t]) 

    return gs_id, tpl_index, frg_index 


def syn_timeSeries(gs_c: int, tpl_c: list, gs_b: int, frg_b: list, goldseq, slot_sym = '*'):
    '''
    '''
    slot=0
    syn_example = list()
    for t in tpl_c:
        if t == slot_sym:
            # add segment from fragment 
            f = frg_b[slot]
            syn_example.append(goldseq[gs_b,f,:])
            slot +=1
        else:
            # add segment form template
            syn_example.append(goldseq[gs_c,t,:])

    syn_example = np.concatenate(syn_example, axis=0)
    syn_shape  = syn_example.shape
    syn_example = syn_example.reshape(syn_shape[0] * syn_shape[1], syn_shape[2])
    return syn_example


def build_caches(dataset: set, window_size: int, frg_fun: Callable, **kwargs) -> Dict:
    '''
    Creates the mappings (f2t, t2f and e2t) need in geca algorithm.
    '''
    frg_to_tpl = defaultdict(list)
    tpl_to_frg = defaultdict(list) 
    env_to_tpl = defaultdict(list) #maps templates with same environment
    for seq in dataset:
        for frg in frg_fun(seq, **kwargs):
            tpl = template(seq, frg) 
            env = environment(tpl, window_size)
            #fetch values 
            frg_s = get_syms(seq, frg, value_type='frg')
            tpl_s = get_syms(seq, tpl, value_type='tpl')
            env_s = get_syms(seq, env, value_type='env')
            #append maps
            frg_to_tpl[frg_s].append(tpl_s)
            tpl_to_frg[tpl_s].append(frg_s)
            env_to_tpl[env_s].append(tpl_s)
    
    def mutate(d):
        for k, v in d.items():
            d[k] = set(v)

    mutate(frg_to_tpl)
    mutate(tpl_to_frg)
    mutate(env_to_tpl)
    return frg_to_tpl, tpl_to_frg, env_to_tpl


def syn_seq_dev(frg_to_tpl, tpl_to_frg, env_to_tpl, symseq_to_igoldseq, goldseq, window_size,  slot_sym = '*') -> tuple:
    '''
    Implements the matching operation that maps templates to fragments to build 
    synthetic examples. Matching operation (Fig1, Andreas 2020): 
    (1)    Get a pair of sequences with identical fragment but different templates,
           called a = tpl_a / frg and c = tpl_c / frg. 
    (2)    Fetch a sequence b = tpl_b / frg_b where env(tpl_b == env(tpl_a)
    (3)    Retrieve all fragments for tpl_b and add map them to tpl_c if 
           frg != frg_b
    '''
   
    for frg in list(frg_to_tpl):
        for tpl_c in frg_to_tpl[frg]:
            for tpl_a in frg_to_tpl[frg]:
                if tpl_a == tpl_c: break
                #retrieve templates with same environment as tpl_a 
                for tpl_b in env_to_tpl[environment(tpl_a, window_size)]:
                    #retrieve all fragments for tpl_b
                    for frg_b in tpl_to_frg[tpl_b]:
                            if frg_b != frg:
                                gs_c_list, itpl_c, ifrg_c = indexify(tpl_c, frg, symseq_to_igoldseq, slot_sym)
                                gs_b_list, itpl_b, ifrg_b = indexify(tpl_b, frg_b, symseq_to_igoldseq, slot_sym)
                                for gs_c in gs_c_list:
                                    for gs_b in gs_b_list:
                                        yield syn_timeSeries(gs_c, itpl_c, gs_b, ifrg_b, goldseq, slot_sym)


def syn_seq(frg_to_tpl, tpl_to_frg, env_to_tpl, symseq_to_igoldseq, goldseq, window_size,  slot_sym = '*') -> tuple:
    '''
    Implements the matching operation that maps templates to fragments to build 
    synthetic examples. Matching operation (Fig1, Andreas 2020): 
    (1)    Get a pair of sequences with identical fragment but different templates,
           called a = tpl_a / frg and c = tpl_c / frg. 
    (2)    Fetch a sequence b = tpl_b / frg_b where env(tpl_b == env(tpl_a)
    (3)    Retrieve all fragments for tpl_b and add map them to tpl_c if 
           frg != frg_b

    This is done by an infinite generator 
    '''
    
    def match(tpl_c_list):
        for tpl_c in tpl_c_list:
            for tpl_a in tpl_c_list:
                if tpl_a == tpl_c: break
                #retrieve templates with same environment as tpl_a 
                for tpl_b in env_to_tpl[environment(tpl_a, window_size)]:
                    #retrieve all fragments for tpl_b
                    for frg_b in tpl_to_frg[tpl_b]:
                            if frg_b != frg:
                                gs_c_list, itpl_c, ifrg_c = indexify(tpl_c, frg, symseq_to_igoldseq, slot_sym)
                                gs_b_list, itpl_b, ifrg_b = indexify(tpl_b, frg_b, symseq_to_igoldseq, slot_sym)
                                for gs_c in gs_c_list:
                                    for gs_b in gs_b_list:
                                        return syn_timeSeries(gs_c, itpl_c, gs_b, ifrg_b, goldseq, slot_sym)
        return None
    
    frg_list = list(frg_to_tpl)
    while True:
        shuffle(frg_list)
        for frg in frg_list:
            tpl_c_list = list(frg_to_tpl[frg])
            shuffle(tpl_c_list)
            syn_ex = match(tpl_c_list)
            if isinstance(syn_ex, np.ndarray): 
                yield syn_ex 


if __name__ == '__main__':
    window_size = 1 

    dataset={('she', 'picks', 'the', 'wug', 'up', 'in', 'fresno'), 
             ('she', 'puts', 'the', 'wug', 'down', 'in', 'tempe'),
             ('pat', 'picks', 'cats', 'up')}

    f2t, t2f, e2t = build_caches(dataset, window_size, total_fragments, max_frg_len=2, rm_solitaire=False)
    
    for s in syn_seq_test(f2t, t2f, e2t, window_size, True):
        print(s)

    N = 10
    n = 0
    for s in syn_seq(f2t, t2f, e2t, window_size):
        print(s)
        n += 1
        if n == N: break
