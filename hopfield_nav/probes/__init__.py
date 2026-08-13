"""Offline probes of the quantities training depends on.

Nothing here trains or evaluates a policy. These answer questions about the
*inputs* -- what information is present in the channels the policy reads --
which is what decides whether a training failure is a learning problem or an
observability one. A knob sweep cannot distinguish those two, and running one
against an unobservable target is the most expensive way to find out.
"""
