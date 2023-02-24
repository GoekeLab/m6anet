import gzip
import multiprocessing
import os
import pandas
from functools import reduce


class EventalignFile:

    def __init__(self, fn):
        self._fn = fn
        self._open()

    def _open(self):
        fn = self._fn
        if os.path.splitext(fn)[1] == '.gz':
            self._handle = gzip.open(fn)
            self._decode_method = bytes.decode
        else:
            self._handle = open(fn)
            self._decode_method = str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self._handle.close()

    def readline(self):
        self._handle.readline()

    def __iter__(self):
        return self

    def __next__(self):
        return self._decode_method(next(self._handle))


def decor_message(text,opt='simple'):
    text = text.upper()
    if opt == 'header':
        return text
    else:
        return '--- ' + text + ' ---\n'

def end_queue(task_queue,n_processes):
    for _ in range(n_processes):
        task_queue.put(None)
    return task_queue

def get_ids(f_index,data_info): #todo
    df_list = []

    for condition_name, run_names in data_info.items():
        list_of_set_ids = []
        for run_name in run_names:
            list_of_set_ids += [set(f_index[run_name].keys())]
        # ids = reduce(lambda x,y: x.intersection(y), list_of_set_ids)
        ids = reduce(lambda x,y: x.union(y), list_of_set_ids)
        df_list += [pandas.DataFrame({'ids':list(ids),condition_name:[1]*len(ids)})]
    df_merged = reduce(lambda  left,right: pandas.merge(left,right,on=['ids'], how='outer'), df_list).fillna(0).set_index('ids')
    return sorted(list(df_merged[df_merged.sum(axis=1) >= 2].index)) # At least two conditions.


class Consumer(multiprocessing.Process):
    """ For parallelisation """

    def __init__(self,task_queue,task_function,locks=None,result_queue=None):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.locks = locks
        self.task_function = task_function
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task_args = self.task_queue.get()
            if next_task_args is None:
                self.task_queue.task_done()
                break
            result = self.task_function(*next_task_args,self.locks)
            self.task_queue.task_done()
            if self.result_queue is not None:
                self.result_queue.put(result)

def read_last_line(filepath): # https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/3346788
    if not os.path.exists(filepath):
        return
    with open(filepath, "rb") as f:
        first = f.readline()        # Read the first line.
        if first == b'':
            return
        f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
        while f.read(1) != b"\n":   # Until EOL is found...
            f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
        last = f.readline()         # Read last line.
    return last

def is_successful(filepath):
    return read_last_line(filepath) == b'--- SUCCESSFULLY FINISHED ---\n'
