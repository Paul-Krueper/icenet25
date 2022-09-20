# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import uproot
import awkward as ak
import re
import ray
import multiprocessing
import os
import copy
import gc

from tqdm import tqdm
from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import iceroot
from icenet.tools.icemap import icemap


#@ray.remote
def read_MC(process_func, process, root_path, param, class_id):

    print(__name__ + f'.read_multiple_MC: {process}')

    # --------------------
    
    datasets        = process['path']
    xs              = process['xs']
    model_param     = process['model_param']
    force_xs        = process['force_xs']
    maxevents_scale = process['maxevents_scale']

    # --------------------

    rootfile     = io.glob_expand_files(datasets=datasets, datapath=root_path)

    # Custom scale event statistics
    maxevents    = np.max([1, int(param['maxevents'] * maxevents_scale)])
    
    # Load file
    X_uncut, ids = iceroot.load_tree(rootfile=rootfile, tree=param['tree'],
                    entry_start=param['entry_start'], entry_stop=param['entry_stop'],
                    maxevents=maxevents, ids=param['load_ids'], library='ak')
    
    N_before = len(X_uncut)

    ## ** Here one could save X before any cuts **
    
    # Apply selections
    X,ids,stats = process_func(X=X_uncut, ids=ids, **param)
    N_after  = len(X)

    eff_acc  = N_after / N_before

    print(__name__ + f'.read_multiple_MC: efficiency x acceptance = {eff_acc:0.6f}')

    Y = class_id * ak.Array(np.ones(N_after, dtype=int))

    # -------------------------------------------------
    # Visible cross-section weights
    if not force_xs:
        # Sum over W yields --> (eff_acc * xs)
        W = (ak.Array(np.ones(N_after, dtype=float)) / N_after) * (xs * eff_acc)
    
    # 'force_xs' mode is useful e.g. in training with a mix of signals with different eff x acc, but one wants
    # to have equal proportions for e.g. theory conditional MVA training.
    # (then one uses the same input 'xs' value for each process in the steering .yaml file)
    else:
        # Sum over W yields --> xs
        W = (ak.Array(np.ones(N_after, dtype=float)) / N_after) * xs

    # Save statistics information
    info = {'yaml': process, 'cut_stats': stats, 'eff_acc': eff_acc}

    # -------------------------------------------------
    # Add conditional (theory param) variables
    
    print(__name__ + f'.read_multiple_MC: Adding conditional theory (model) parameters')
    
    for var in model_param.keys():
            
        value = model_param[var]
        
        # Pick random between [a,b]
        if type(value) == list:
            col = value[0]  + (value[1] - value[0]) * ak.Array(np.random.rand(len(X), 1))
        # Pick specific fixed value
        else:
            if value is None:
                value = np.nan
            col = value * ak.Array(np.ones((len(X), 1)).astype(float))         
        
        # Create new 'record' (column) to ak-array
        col_name    = f'MODEL_{var}'
        X[col_name] = col
        ids.append(col_name)

    return {'X':X, 'Y':Y, 'W': W, 'ids':ids, 'info':info}


def read_multiple_MC(process_func, processes, root_path, param, class_id):
    """
    Loop over different MC processes as defined in the yaml files
    
    Args:
        process_func:  data processing function
        processes:     MC processes dictionary (from yaml)
        root_path:     main path of files
        param:         parameters of 'process_func'
        class_id:      class identifier (integer), e.g. 0, 1, 2 ...
    
    Returns:
        X, Y, W, ids, info (awkward array format)
    """

    # Combine results
    X,Y,W,ids,info = None,None,None,None,{}

    for i,key in enumerate(processes):
        
        data = read_MC(process_func, processes[key], root_path, param, class_id)

        # Concatenate processes
        if i == 0:
            X = copy.deepcopy(data['X'])
            Y = copy.deepcopy(data['Y'])
            W = copy.deepcopy(data['W'])
        else:
            X = ak.concatenate((X, data['X']), axis=0)
            Y = ak.concatenate((Y, data['Y']), axis=0)
            W = ak.concatenate((W, data['W']), axis=0)

        ids       = copy.deepcopy(data['ids']) # Same for all processes
        info[key] = copy.deepcopy(data['info'])
        
        del data # free memory
        gc.collect()
        io.showmem()

    return X,Y,W,ids,info


def load_tree_stats(rootfile, tree, key=None, verbose=False):
    """
    Load the number of events in a list of rootfiles

    Args:
        rootfile: a list of rootfiles
        tree:     tree name to open
        key:      key (variable name) to use to get the number of events, if None then use the first one
        verbose:  verbose output print
    
    Returns:
        number of events
    """

    if type(rootfile) is not list:
        rootfile = [rootfile]

    num_events = np.zeros(len(rootfile), dtype=int)
    
    for i in range(len(rootfile)):
        with uproot.open(rootfile[i]) as file:

            events   = file[tree]
            key_name = events.keys()[0] if key is None else key    

            num_events[i] = len(events.arrays(key_name))
            
            if verbose:
                print(__name__ + f'.load_tree_stats: {rootfile[i]}')
                print(__name__ + f'.load_tree_stats: keys(): {events.keys()}')
                print(__name__ + f'.load_tree_stats: values(): {events.values()}')
            
            file.close()

    return num_events


def events_to_jagged_numpy(events, ids, entry_start=0, entry_stop=None, label=None):
    """
    Process uproot tree to a jagged numpy (object) array
    
    Args:
        events:      uproot tree
        ids:         names of the variables to pick
        entry_start: first event to consider
        entry_stop:  last event to consider
    
    Returns:
        X
    """

    N_all  = len(events.arrays(ids[0]))
    X_test = events.arrays(ids[0], entry_start=entry_start, entry_stop=entry_stop)
    N      = len(X_test)
    X      = np.empty((N, len(ids)), dtype=object) 
    
    if label is not None:
        cprint( __name__ + f'.events_to_jagged_numpy: Loading: {label}', 'yellow')
    cprint( __name__ + f'.events_to_jagged_numpy: Entry_start = {entry_start}, entry_stop = {entry_stop} | realized = {N} ({100*N/N_all:0.3f} % | available = {N_all})', 'green')
    
    for j in range(len(ids)):
        x = events.arrays(ids[j], entry_start=entry_start, entry_stop=entry_stop, library="np", how=list)
        X[:,j] = np.asarray(x)

    return X, ids


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, maxevents=None, ids=None, library='np'):
    """
    Load ROOT files

    Args:
        rootfile:      Name of root file paths (string or a list of strings)
        tree:          Tree to read out
        entry_start:   First event to read per file
        entry_stop:    Last event to read per file
        maxevents:     Maximum number of events in total (over all files)
        ids:           Names of the variables to read out from the root tree
        library:       Return type 'np' (jagged numpy) or 'ak' (awkward) of the array
    
    Returns:
        array of type 'library'
    """
    
    if type(rootfile) is not list:
        rootfile = [rootfile]

    cprint(__name__ + f'.load_tree: Opening rootfile {rootfile} with a tree key <{tree}>', 'yellow')

    # Add Tree handles here using the string-syntax of uproot {file:tree}
    files = [rootfile[i] + f':{tree}' for i in range(len(rootfile))]
    
    # ----------------------------------------------------------
    ### Select variables
    with uproot.open(files[0]) as events:
        all_ids = events.keys()
    
    load_ids = aux.process_regexp_ids(ids=ids, all_ids=all_ids)

    print(__name__ + f'.load_tree: Loading variables ({len(load_ids)}): \n{load_ids} \n')
    print(__name__ + f'.load_tree: Reading {len(files)} root files ...')
    
    if   library == 'np':
        for i in tqdm(range(len(files))):
            with uproot.open(files[i]) as events:
                
                param  = {'events': events, 'ids': load_ids, 'entry_start': entry_start, 'entry_stop': entry_stop, 'label': files[i]}
                output, ids = events_to_jagged_numpy(**param)
                
                # Concatenate with other file results
                X = copy.deepcopy(output) if (i == 0) else np.concatenate((X, output), axis=0)
                del output
                gc.collect()
                
                if (maxevents is not None) and (len(X) > maxevents):
                    X = X[0:maxevents]
                    cprint(__name__ + f'.load_tree: Maximum event count {maxevents} reached', 'red')
                    break

        return X, ids

    elif library == 'ak':

        """
        # Non-multiprocessed version
        
        for i in tqdm(range(len(files))):
            
            # Get the number of events
            num_events = get_num_events(rootfile=files[i])
            
            with uproot.open(files[i]) as events:
                
                output = events.arrays(load_ids, entry_start=entry_start, entry_stop=entry_stop, library='ak', how='zip')
                
                # Concatenate with other file results
                X = output if (i == 0) else ak.concatenate((X, output), axis=0)
                        
                if (maxevents is not None) and (len(X) > maxevents):
                    X = X[0:maxevents]
                    cprint(__name__ + f'.load_tree: Maximum event count {maxevents} reached', 'red')
                    break
        """
        # ======================================================
        # Multiprocessing version

        num_workers  = min(len(files), multiprocessing.cpu_count()) # min handles the case #files < #cpu
        ray.init(num_cpus=num_workers, _temp_dir=f'{os.getcwd()}/tmp/')

        chunk_ind    = aux.split_start_end(range(len(files)), num_workers)
        submaxevents = aux.split_size(range(maxevents), num_workers)
        futures      = []
        
        print(__name__ + f'.load_tree: submaxevents per ray process: {submaxevents}')

        for k in range(num_workers):    
            futures.append(read_file_ak.remote(files[chunk_ind[k][0]:chunk_ind[k][-1]], load_ids, entry_start, entry_stop, submaxevents[k]))

        results = ray.get(futures) # synchronous read-out
        ray.shutdown()
        
        # Combine future returned sub-arrays
        print(__name__ + f'.load_tree: Concatenating results from futures')

        for k in tqdm(range(len(results))):
            X = copy.deepcopy(results[k]) if (k == 0) else ak.concatenate((X, results[k]), axis=0)

            results[k] = None # free memory
            gc.collect()

        return X, ak.fields(X)
        
    else:
        raise Exception(__name__ + f'.load_tree: Unknown library support')


@ray.remote
def read_file_ak(files, ids, entry_start, entry_stop, maxevents):
    """
    Ray multiprocess read-wrapper-function per root file.
    
    Remark:
        Multiprocessing within one file did not yield good
        performance empirically, but performance scales linearly for separate files.
    
    Args:
        files:        list of root filenames
        ids:          variables to read out
        entry_start:  first entry to to read
        entry_stop:   last entry to read
        maxevents:    maximum number of events (in total over all files)
    
    Returns:
        awkward array
    """
    for i in range(len(files)):

        # Get the number of entries
        #num_entries = get_num_events(rootfile=files[i])
        #print(__name__ + f'.read_file_ak: Found {num_entries} entries from the file: {files[i]}')
        
        with uproot.open(files[i]) as events:
            
            output = events.arrays(ids, entry_start=entry_start, entry_stop=entry_stop, library='ak', how='zip')
            
            # Concatenate with other file results
            X = copy.deepcopy(output) if (i == 0) else ak.concatenate((X, output), axis=0)
            del output
            gc.collect()
            io.showmem()
                        
            if (maxevents is not None) and (len(X) > maxevents):
                X = X[0:maxevents]
                cprint(__name__ + f'.load_tree: Maximum event count {maxevents} reached', 'red')
                break
    return X


def get_num_events(rootfile, key_index=0):
    """
    Get the number of entries in a rootfile by reading a key
    
    Args:
        rootfile:   rootfile string (with possible Tree name appended with :)
        key_index:  which variable use as a dummy
    
    Returns:
        number of entries
    """
    with uproot.open(rootfile) as events:
        return len(events.arrays(events.keys()[key_index]))

@ray.remote
def events_to_jagged_ak(events, ids, entry_start=None, entry_stop=None, library='ak', how='zip'):
    """ Wrapper for Ray """
    return events.arrays(ids, entry_start=entry_start, entry_stop=entry_stop, library=library, how=how)

