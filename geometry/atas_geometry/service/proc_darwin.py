"""macOS process ownership primitives (libproc/sysctl), mirroring the Linux /proc rules.

Semantics are those of worker_protocol's Linux functions: a lease is owned when its
boot session, process group, session, UID and monotonic start stamps all agree.
Anything that cannot be read is never certified as ours. Struct sizes are checked
against the kernel's answer at runtime; on a mismatch every call fails closed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os

PROC_PIDTBSDINFO = 3
PROC_PGRP_ONLY = 2
SZOMB = 5
MAXCOMLEN = 16
RUSAGE_INFO_V4 = 4
CTL_KERN, KERN_PROCARGS2 = 1, 49
KERN_PROC, KERN_PROC_PID = 14, 1
KINFO_PROC_SIZE = 648  # sizeof(struct kinfo_proc) on 64-bit Darwin
_P_STAT_OFFSET = 36    # kp_proc.p_stat (after p_un[16], p_vmspace, p_sigacts, p_flag)


class ProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32), ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * MAXCOMLEN), ("pbi_name", ctypes.c_char * (2 * MAXCOMLEN)),
        ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64),
    ]


PROC_PIDTBSDINFO_SIZE = 136  # sizeof(struct proc_bsdinfo) in <sys/proc_info.h>
# rusage_info_v4: 16-byte uuid followed by uint64 fields; index 28 is ri_lifetime_max_phys_footprint.
_RUSAGE_V4_U64 = 35
_LIFETIME_MAX_FOOTPRINT = 28
_PHYS_FOOTPRINT = 7

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
_libproc.proc_pidinfo.restype = ctypes.c_int
_libproc.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
_libproc.proc_listpids.restype = ctypes.c_int
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int
_libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
_libc.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]

LAYOUT_OK = ctypes.sizeof(ProcBsdInfo) == PROC_PIDTBSDINFO_SIZE


def _bsdinfo(pid: int) -> ProcBsdInfo | None:
    if not LAYOUT_OK or pid <= 0:
        return None
    info = ProcBsdInfo()
    size = _libproc.proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info):
        return None
    return info


def _start_stamp(info: ProcBsdInfo) -> int:
    return int(info.pbi_start_tvsec) * 1_000_000 + int(info.pbi_start_tvusec)


def _kinfo(pid: int):
    """(p_stat, start stamp) from sysctl KERN_PROC_PID. Unlike proc_pidinfo this
    still answers for exiting and zombie processes, as Linux /proc/<pid>/stat does."""
    if pid <= 0:
        return None
    mib = (ctypes.c_int * 4)(CTL_KERN, KERN_PROC, KERN_PROC_PID, int(pid))
    buffer = ctypes.create_string_buffer(KINFO_PROC_SIZE)
    size = ctypes.c_size_t(KINFO_PROC_SIZE)
    if _libc.sysctl(mib, 4, buffer, ctypes.byref(size), None, 0) != 0 or size.value != KINFO_PROC_SIZE:
        return None
    raw = buffer.raw
    start = int.from_bytes(raw[0:8], "little") * 1_000_000 + int.from_bytes(raw[8:12], "little")
    return raw[_P_STAT_OFFSET], start


def boot_id() -> str:
    size = ctypes.c_size_t(0)
    if _libc.sysctlbyname(b"kern.bootsessionuuid", None, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl kern.bootsessionuuid failed")
    buffer = ctypes.create_string_buffer(size.value)
    if _libc.sysctlbyname(b"kern.bootsessionuuid", buffer, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl kern.bootsessionuuid failed")
    return buffer.value.decode().strip()


def process_start(pid):
    if not LAYOUT_OK:
        return None
    info = _kinfo(int(pid))
    return None if info is None else str(info[1])


def _group_pids(pgid: int) -> list[int]:
    capacity = 4096
    while True:
        buffer = (ctypes.c_int * capacity)()
        used = _libproc.proc_listpids(PROC_PGRP_ONLY, pgid, buffer, ctypes.sizeof(buffer))
        if used <= 0:
            return []
        count = used // ctypes.sizeof(ctypes.c_int)
        if count < capacity:
            return [pid for pid in buffer[:count] if pid > 0]
        capacity *= 2


def command_line(pid: int) -> bytes:
    mib = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, int(pid))
    size = ctypes.c_size_t(0)
    if _libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl KERN_PROCARGS2 failed")
    buffer = ctypes.create_string_buffer(size.value)
    if _libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl KERN_PROCARGS2 failed")
    return buffer.raw[4:size.value]  # skip argc; NUL-separated exec path and argv follow


def owned_group_members(lease):
    """Same contract as the Linux implementation: None = unprovable, else live member PIDs."""
    pid = int(lease.get("pid", 0))
    if pid <= 1 or not LAYOUT_OK:
        return None
    modern = lease.get("owner_contract") == "isolated_geometry_session_v1"
    if modern:
        if lease.get("boot_id") != boot_id():
            return []
        if (lease.get("pgid") != pid
                or lease.get("session_id") != pid or not lease.get("process_start")):
            return None
        current_start = process_start(pid)
        if current_start is not None and current_start != lease["process_start"]:
            return []
    else:
        if process_start(pid) != lease.get("process_start"):
            return None
        try:
            if (b"atas_geometry.service.isolated_worker" not in command_line(pid)
                    or os.getsid(pid) != pid or os.getpgid(pid) != pid):
                return None
        except OSError:
            return None
    members = []
    # The kernel lists exactly the members of process group `pid`.
    for member in _group_pids(pid):
        state = _kinfo(member)
        if state is None:
            continue
        status, start = state
        info = _bsdinfo(member)  # unreadable while exiting; uid is then unprovable but irrelevant
        if info is not None and int(info.pbi_pgid) != pid:
            continue
        try:
            session = os.getsid(member)
        except OSError:
            session = pid if status == SZOMB or info is None else None
        if (session != pid or start < int(lease["process_start"])
                or (modern and info is not None and int(info.pbi_uid) != lease.get("uid"))):
            return None
        if status != SZOMB:
            members.append(member)
    return members


def lease_running(lease):
    if lease.get("owner_contract") == "isolated_geometry_session_v1":
        members = owned_group_members(lease)
        return members is None or bool(members)
    pid = int(lease.get("pid", 0))
    if process_start(pid) != lease.get("process_start"):
        return False
    state = _kinfo(pid)
    return state is not None and state[0] != SZOMB


def _rusage(pid: int):
    buffer = (ctypes.c_uint64 * (_RUSAGE_V4_U64 + 2 + 16))()
    if _libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, buffer) != 0:
        return None
    return buffer[2:2 + _RUSAGE_V4_U64]  # the first 16 bytes are ri_uuid


def lifetime_peak_footprint(pid: int):
    values = _rusage(pid)
    return None if values is None else int(values[_LIFETIME_MAX_FOOTPRINT])


def phys_footprint(pid: int):
    values = _rusage(pid)
    return None if values is None else int(values[_PHYS_FOOTPRINT])
