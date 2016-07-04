"""
Definition of Submitter class and script to start/stop it.

Base class for all Submitters to inherit from.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import tempfile
import numpy as np
import threading as thr
import subprocess as sp

from soprano.hpc.submitter import QueueInterface

class Submitter(object):

	"""Submitter object

	Template to derive all specialised Submitters. These are meant to generate,
	submit and post-process any number of jobs on a queueing system in the form
	of a background process running on a head node. It implements
    methods that should be mostly overridden by the child classes.
    Four methods define its core behaviour:

    1) next_job is the function that outputs the specification for each new job
       to submit. The specification should be a dict with two members, 'name'
       (a string) and 'args' (ideally a dict). If no more jobs are available
       it should return None;
    2) setup_job takes as arguments name, args and folder (a temporary one
       created independently) and is supposed to generate the input files
       for the job before submission. It returns nothing;
    3) check_job takes as arguments job ID, name, args and folder and should
       return a bool confirmation of whether the job has finished or not. By 
       default it simply checks whether the job is still listed in the queue,
       however other checks can be implemented in its place;
    4) finish_job takes as arguments name, args and folder and takes care of
       the post processing once a job is complete. Here meaningful data should
       be extracted and useful files copied to permament locations, as the
       temporary folder will be deleted immediately afterwards. It returns
       nothing.

    In addition, the Submitter takes a template launching script which can
    be tagged with keywords, mainly <name> for the job name or any other
    arguments present in args. These will be replaced with the appropriate
    values when the script is submitted.
	"""

	def __init__(self, queue, submit_script, max_jobs=4, check_time=10,
				 msg_pipe=None, max_time=3600):

		# Check type
		if not isinstance(queue, QueueInterface):
			raise TypeError('A QueueInterface must be passed to the '
							'Submitter')

		if type(submit_script) is not str:
			raise TypeError('submit_script must be a string')	

		self.queue = queue
		self.submit_script = submit_script
		self.max_jobs = max_jobs
		self.check_time = check_time
		self.max_time = max_time if max_time > 0 else np.inf

		# Message pipe. Used to stop the thread from the outside
		if msg_pipe is not None:
			self._fifo_path = msg_pipe
		else:
			self._fifo_path = None

	def start(self):

		self._jobs = {}
		if self._fifo_path is not None:
			self._fifo = os.open(self._fifo_path, os.O_RDONLY|os.O_NONBLOCK)

		self._running = True
		self._t0 = time.time() # Starting time. Second precision is fine


		while self._running and (time.time()-self._t0) < self.max_time:

			# Submit jobs
			while len(self._jobs) < self.max_jobs:

				njob = self.next_job()
				if njob is None:
					break
				# Create the temporary folder
				njob['folder'] = tempfile.mkdtmp()
				# Perform setup
				setup_job(**njob)
				# Create custom script
				job_script = submit_script.replace('<name>', njob['name'])
				# Replace the rest of the tags
				for tag in njob['args']:
					job_script = job_script.replace('<{0}>'.format(tag),
													njob['args'][tag])
				# And submit!
				job_id = self.queue.submit(job_script, cwd=njob['folder'])
				self._jobs[job_id] = njob

			# Now check for finished jobs
			completed = [job_id for job_id in self._jobs
						 if check_job(job_id, **self._jobs[job_id])]
			for job_id in completed:
				finish_job(self._jobs[job_id])
				del(self._jobs[job_id])

			# Finally, grab messages from the PIPE if present
			if self._fifo_path is not None:
				msg = os.read(self._fifo, 16)
				if 'STOP' in msg:
					self._running = False

			time.sleep(self.check_time)

	def next_job():
		return {'name': 'default_job', 'args': {}}

	def setup_job(name, args, folder):
		pass

	def check_job(id, name, args, folder):
		return True

	def finish_job(name, args, folder):
		pass

