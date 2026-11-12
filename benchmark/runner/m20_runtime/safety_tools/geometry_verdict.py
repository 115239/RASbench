"""Three-valued predicates over proven bounds on the global minimum distance."""
import math


def distance_interval(measurement):
    if not measurement.get('complete'):return -math.inf,math.inf
    value=measurement.get('distance_m')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return -math.inf,math.inf
    kind=str(measurement.get('distance_kind',''))
    if kind=='certified_global_minimum_interval':
        lo=measurement.get('distance_lower_bound_m');hi=measurement.get('distance_upper_bound_m')
        return (float(lo),math.inf if hi is None else float(hi))
    if 'lower_bound' in kind:return float(value),math.inf
    if 'upper_bound' in kind:return 0.,float(value)
    if kind.startswith('exact_'):return float(value),float(value)
    # Missing measurement semantics is not an exact geometric measurement.
    return -math.inf,math.inf


def threshold_verdict(measurement,minimum,maximum=None,intersection=False):
    lo,hi=distance_interval(measurement)
    if hi<lo or math.isnan(lo) or math.isnan(hi):return None
    if (hi<=0 if intersection else hi<minimum):return True
    if maximum is not None and lo>maximum:return True
    lower_ok=lo>0 if intersection else lo>=minimum
    if lower_ok and (maximum is None or hi<=maximum):return False
    return None
