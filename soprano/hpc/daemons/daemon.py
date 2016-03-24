"""
Definition of DaemonHPC class

Base class for all Daemons to inherit from.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import numpy as np
import subprocess as sp
import multiprocessing as mp

# Again, necessary for compatibility
try:
    from Queue import Empty as EmptyQueue
    from Queue import Full as FullQueue
except ImportError:
    from queue import Empty as EmptyQueue
    from queue import Full as FullQueue


def _daemon_runner_mainloop(daemon_loopid):

    # Run an infinite loop that keeps chewing through the arguments passed
    # in the daemon's Queue using proc - until something goes wrong, that is

    daemon, loop_id = daemon_loopid

    # These open their own logs!
    iter_i = 1
    try:   
        logfile = open('{0}_{1}.log'.format(daemon._id, loop_id), 'w')
    except IOError as e:
        return ('Execution of process {0} stopped '
                'due to file I/O error: {1}').format(os.getpid(), e)

    try:
        while True:
            try:
                qval = daemon.queue.get(timeout=daemon.timeout)
            except EmptyQueue:
                return ('Execution of process {0} '
                        'stopped due to empty queue').format(os.getpid())
            rval = daemon.run_process(**qval)
            daemon.on_complete(rval)
            logfile.write('Iteration {0} completed\n'.format(iter_i))
            iter_i += 1
    except Exception as e:
        return ('Execution of process {0} stopped '
                'due to unforeseen error: {1}').format(os.getpid(), e)
    
class DaemonHPC(object):
    """DaemonHPC object

    A class that serves as template for all Daemons for HPC use. It implements
    methods that should be mostly overridden by the child classes. It is
    passed to the DaemonRunner that handles the heavy weight parallelization
    stuff.
    Four methods define its core behaviour:

    1) run_process is the core function that gets parallelized. This could for
       example include a subprocess.Popen to call some external program;
    2) next_processes generates input data for a number of new processes. This
       should be overridden preferrably over start_processes to edit behaviour
       of the Daemon;
    3) start_processes queues up a number of new processes with input data
       gotten by next_processes in the Pool;
    4) on_complete is a callback that defines behaviour once a process is 
       finished. By default, it grabs a new process to replace it from
       next_processes and submits it to start_processes. Here data should be
       stored/saved and so on.

    In addition, the Daemon will store a log of its activities if required.
    """


    def __init__(self, daemon_manager,
                       daemon_id=None,
                       verbose=False,
                       timeout=0.1):

        if daemon_id is None:
            self._id = '{0}_{1}'.format(self.__class__.__name__.lower(),
                                        os.getpid())
        else:
            self._id = daemon_id

        if verbose:
            self._logfile = open(self._id + '.log', 'w')
        else:
            self._logfile = None

        self.queue = daemon_manager.Queue()
        self.timeout = timeout

    def log(self, msg):

        if self._logfile is not None:
            self._logfile.write(msg + '\n')

    def run_process(self, daemon_pid):

        stdout, stderr = sp.Popen(['sleep', str(self.timeout)],
                                  stdout=sp.PIPE,
                                  stderr=sp.PIPE).communicate()
        return "{0}: {1}".format(daemon_pid, stdout)

    def start_processes(self, n=1):
        """Launch n new processes."""

        # Get the input data
        proc_data = self.next_processes(n)
        try:
            for d in proc_data:
                self.queue.put(d, timeout=self.timeout)
        except FullQueue:
            return

    def next_processes(self, n=1):
        """Get input data for n new processes. Must be a list of lists,
        even if it's just one argument per process.

        """

        return [{'daemon_pid': self._id} for i in range(n)]

    def on_complete(self, rval):

        if np.random.random() < 0.8:
            self.start_processes()


class DaemonRunner(object):
    
    """DaemonRunner object

    The engine that runs DaemonHPC and derived classes. It does all the heavy
    lifting! At its core, it uses the multiprocessing module to create a Pool
    of processes, then runs endless loops in it that will only stop when the
    queue of the Daemon is empty.

    """

    def __init__(self, daemon_type=DaemonHPC,
                       daemon_args={},
                       daemon_id=None,
                       proc_num=None,
                       verbose=True):

        """
        Initialize the DaemonHPC.

        | Args:
        |   daemon_type (class): class of the Daemon. Should be derived by
        |                        DaemonHPC. 
        |   daemon_args (dict): additional arguments to initialize the Daemon.
        |                       Valid entries depend on the class.
        |   daemon_id (Optional[str]): id of the Daemon. If not assigned a
        |                              unique id will be generated.
        |   proc_num (Optional[int]): number of processes to assign to run
        |                             at the same time in this Daemon's Pool.
        |                             If left empty the number of cores will
        |                             be used.
        |   verbose (Optional[bool]): if set to True, store a log of the 
        |                             Daemon's activity.

        """

        if proc_num is None:
            self._pnum = mp.cpu_count()
        else:
            self._pnum = proc_num

        # Initialize the manager and the daemon
        self._manager = mp.Manager()
        self._daemon = daemon_type(daemon_manager=self._manager,
                                   daemon_id=daemon_id,
                                   verbose=verbose,
                                   **daemon_args)

        # Now start up the pool
        self._pool = mp.Pool(processes=self._pnum)
        self._daemon.log('Started running pool')

        self._map = self._pool.map_async(_daemon_runner_mainloop,
                                         zip([self._daemon]*self._pnum,
                                             range(1, self._pnum+1)),
                                         callback=self._on_close)
        # And start running!
        self._daemon.start_processes(n=self._pnum)

        self._pool.close()
        self._pool.join()

    def _on_close(self, close_msgs):
        """Callback for the completed processes. Includes all values returned
        by the functions of the given batch.

        """

        for msg in close_msgs:
            self._daemon.log(msg)

if __name__ == '__main__':

    drun = DaemonRunner()
    print("Daemon execution complete")