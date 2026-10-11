"""Private, bounded, in-memory PDF cache. Opaque IDs; no user-supplied report data."""
import time
import json
from collections import OrderedDict
from threading import RLock


class ReportStore:
    def __init__(self,ttl=900,max_reports=6,max_bytes=32*1024*1024):
        self.ttl=ttl
        self.max_reports=max_reports
        self.max_bytes=max_bytes
        self._reports=OrderedDict()
        self._lock=RLock()

    def _prune(self):
        now=time.monotonic()
        for key in list(self._reports):
            if self._reports[key][0]<=now:del self._reports[key]

    def put(self,report_id,content,result=None):
        # Serialized server evidence is immutable and counted in the memory budget.
        snapshot=json.dumps(result,separators=(',',':')).encode() if result is not None else b''
        size=len(content)+len(snapshot)
        if size>self.max_bytes:raise RuntimeError('Report evidence exceeds report cache capacity')
        with self._lock:
            self._prune()
            self._reports[report_id]=(time.monotonic()+self.ttl,content,snapshot)
            while len(self._reports)>self.max_reports or sum(len(item[1])+len(item[2]) for item in self._reports.values())>self.max_bytes:
                self._reports.popitem(last=False)

    def get(self,report_id):
        with self._lock:
            self._prune()
            entry=self._reports.get(report_id)
            return entry[1] if entry else None

    def evidence(self,report_id):
        with self._lock:
            self._prune()
            entry=self._reports.get(report_id)
            snapshot=entry[2] if entry else b''
        return json.loads(snapshot) if snapshot else None


REPORTS=ReportStore()
