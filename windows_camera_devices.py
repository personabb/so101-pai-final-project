"""Windows camera identities in OpenCV's backend-specific enumeration order.

Metadata only: no capture filter/media source is activated, no frames are read,
and no camera settings or serial ports are touched. Windows COM and Media
Foundation are accessed through ctypes, without additional dependencies.

API/GUID references:
https://learn.microsoft.com/windows/win32/directshow/selecting-a-capture-device
https://learn.microsoft.com/windows/win32/medfound/enumerating-video-capture-devices
https://github.com/microsoft/win32metadata/blob/main/generation/WinSDK/RecompiledIdlHeaders/um/mfidl.h
"""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
import json
import re
import sys
import uuid

HRESULT = C.c_int32
UINT32 = C.c_uint32
POINTER = C.c_void_p


class GUID(C.Structure):
    _fields_ = [("Data1", C.c_uint32), ("Data2", C.c_uint16),
                ("Data3", C.c_uint16), ("Data4", C.c_ubyte * 8)]

    @classmethod
    def parse(cls, text):
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


class VariantData(C.Union):
    # VARIANT's BRECORD union member is two pointers, including on x64.
    _fields_ = [("bstrVal", POINTER), ("llVal", C.c_int64),
                ("record", POINTER * 2)]


class VARIANT(C.Structure):
    _fields_ = [("vt", C.c_uint16), ("reserved1", C.c_uint16),
                ("reserved2", C.c_uint16), ("reserved3", C.c_uint16),
                ("data", VariantData)]


SOURCE_TYPE = GUID.parse("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
VIDEO_CAPTURE = GUID.parse("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
FRIENDLY_NAME = GUID.parse("60d0e559-52f8-4fa2-bbce-acdb34a8ec01")
SYMBOLIC_LINK = GUID.parse("58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f")
BACKENDS = {"dshow": 700, "msmf": 1400}


def _check(hr, operation):
    if hr < 0:
        raise RuntimeError(f"{operation}: HRESULT 0x{hr & 0xffffffff:08X}")
    return hr


class _WindowsAPI:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("カメラの機器識別にはWindowsが必要です。")
        self.ole32 = C.WinDLL("ole32")
        self.oleaut32 = C.WinDLL("oleaut32")
        self.mfplat = C.WinDLL("mfplat")
        self.mf = C.WinDLL("mf")
        self._function(self.ole32, "CoInitializeEx", HRESULT, POINTER, UINT32)
        self._function(self.ole32, "CoUninitialize", None)
        self._function(self.ole32, "CoCreateInstance", HRESULT, C.POINTER(GUID),
                       POINTER, UINT32, C.POINTER(GUID), C.POINTER(POINTER))
        self._function(self.ole32, "CreateBindCtx", HRESULT, UINT32, C.POINTER(POINTER))
        self._function(self.ole32, "CoTaskMemFree", None, POINTER)
        self._function(self.oleaut32, "VariantClear", HRESULT, C.POINTER(VARIANT))
        self._function(self.mfplat, "MFStartup", HRESULT, UINT32, UINT32)
        self._function(self.mfplat, "MFShutdown", HRESULT)
        self._function(self.mfplat, "MFCreateAttributes", HRESULT, C.POINTER(POINTER), UINT32)
        self._function(self.mf, "MFEnumDeviceSources", HRESULT, POINTER,
                       C.POINTER(C.POINTER(POINTER)), C.POINTER(UINT32))

    @staticmethod
    def _function(library, name, restype, *argtypes):
        function = getattr(library, name)
        function.restype, function.argtypes = restype, list(argtypes)

    @staticmethod
    def call(obj, slot, restype, argtypes, *args):
        vtable = C.cast(obj, C.POINTER(C.POINTER(POINTER))).contents
        return C.WINFUNCTYPE(restype, POINTER, *argtypes)(vtable[slot])(obj, *args)

    def release(self, obj):
        if obj and obj.value:
            self.call(obj, 2, UINT32, [])

    @contextmanager
    def com(self):
        hr = self.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        # A caller with an existing STA can enumerate in that apartment. It owns
        # its initialization, so this call must not uninitialize it.
        initialized = hr >= 0
        if not initialized and (hr & 0xffffffff) != 0x80010106:  # RPC_E_CHANGED_MODE
            _check(hr, "CoInitializeEx")
        try:
            yield
        finally:
            if initialized:
                self.ole32.CoUninitialize()

    def property_text(self, bag, name):
        value = VARIANT()
        try:
            hr = self.call(bag, 3, HRESULT,
                           [C.c_wchar_p, C.POINTER(VARIANT), POINTER], name, C.byref(value), None)
            if hr < 0:
                return None, f"0x{hr & 0xffffffff:08X}"
            if value.vt != 8:
                return None, f"Unexpected VARIANT type {value.vt}"
            return C.wstring_at(value.data.bstrVal) if value.data.bstrVal else "", None
        finally:
            _check(self.oleaut32.VariantClear(C.byref(value)), "VariantClear")

    def attribute_text(self, obj, key):
        text, length = POINTER(), UINT32()
        try:
            hr = self.call(obj, 13, HRESULT,
                           [C.POINTER(GUID), C.POINTER(POINTER), C.POINTER(UINT32)],
                           C.byref(key), C.byref(text), C.byref(length))
            if hr < 0:
                return None, f"0x{hr & 0xffffffff:08X}"
            return C.wstring_at(text.value, length.value) if text.value else "", None
        finally:
            if text.value:
                self.ole32.CoTaskMemFree(text)


def _device(backend, index, name, path):
    return {"backend": backend, "backend_id": BACKENDS[backend], "index": index,
            "name": name, "friendly_name": name, "device_path": path}


def enumerate_directshow_devices():
    """Read video monikers in their native order, matching CAP_DSHOW indices."""
    api = _WindowsAPI()
    system_enum = GUID.parse("62be5d10-60eb-11d0-bd3b-00a0c911ce86")
    create_enum = GUID.parse("29840822-5b84-11d0-bd3b-00a0c911ce86")
    category = GUID.parse("860bb310-5d01-11d0-bd3b-00a0c911ce86")
    property_bag = GUID.parse("55272a00-42cb-11ce-8135-00aa004bb851")
    devices, enumerator, bind_context = POINTER(), POINTER(), POINTER()
    rows = []
    with api.com():
        try:
            _check(api.ole32.CoCreateInstance(C.byref(system_enum), None, 1,
                                             C.byref(create_enum), C.byref(devices)), "CoCreateInstance")
            hr = api.call(devices, 3, HRESULT,
                          [C.POINTER(GUID), C.POINTER(POINTER), UINT32],
                          C.byref(category), C.byref(enumerator), 0)
            _check(hr, "CreateClassEnumerator")
            if hr == 1:
                return rows
            if not enumerator.value:
                raise RuntimeError("CreateClassEnumerator returned no enumerator")
            _check(api.ole32.CreateBindCtx(0, C.byref(bind_context)), "CreateBindCtx")
            while True:
                moniker, bag = POINTER(), POINTER()
                fetched = UINT32()
                try:
                    hr = api.call(enumerator, 3, HRESULT,
                                  [UINT32, C.POINTER(POINTER), C.POINTER(UINT32)],
                                  1, C.byref(moniker), C.byref(fetched))
                    _check(hr, "IEnumMoniker.Next")
                    if hr == 1:
                        break
                    if fetched.value != 1 or not moniker.value:
                        raise RuntimeError("IEnumMoniker.Next returned an invalid item")
                    row = _device("dshow", len(rows), None, None)
                    hr = api.call(moniker, 9, HRESULT,
                                  [POINTER, POINTER, C.POINTER(GUID), C.POINTER(POINTER)],
                                  bind_context, None, C.byref(property_bag), C.byref(bag))
                    if hr < 0:
                        row["property_bag_error"] = f"0x{hr & 0xffffffff:08X}"
                    else:
                        if not bag.value:
                            raise RuntimeError("BindToStorage returned no property bag")
                        for key, attribute in (("name", "FriendlyName"), ("device_path", "DevicePath")):
                            row[key], error = api.property_text(bag, attribute)
                            if error:
                                row[key + "_error"] = error
                        row["friendly_name"] = row["name"]
                    # Keep unreadable devices in the list: removing one shifts
                    # every later OpenCV index away from native COM order.
                    rows.append(row)
                finally:
                    api.release(bag)
                    api.release(moniker)
            return rows
        finally:
            api.release(bind_context)
            api.release(enumerator)
            api.release(devices)


def enumerate_msmf_devices():
    """Read MF activation attributes in MFEnumDeviceSources order; never activate."""
    api = _WindowsAPI()
    attributes = POINTER()
    activates = C.POINTER(POINTER)()
    count = UINT32()
    started = False
    rows = []
    with api.com():
        try:
            _check(api.mfplat.MFStartup(0x00020070, 0), "MFStartup")
            started = True
            _check(api.mfplat.MFCreateAttributes(C.byref(attributes), 1), "MFCreateAttributes")
            _check(api.call(attributes, 24, HRESULT, [C.POINTER(GUID), C.POINTER(GUID)],
                            C.byref(SOURCE_TYPE), C.byref(VIDEO_CAPTURE)), "IMFAttributes.SetGUID")
            _check(api.mf.MFEnumDeviceSources(attributes, C.byref(activates), C.byref(count)),
                   "MFEnumDeviceSources")
            if count.value and not activates:
                raise RuntimeError("MFEnumDeviceSources returned no activation array")
            for index in range(count.value):
                activate = POINTER(activates[index])
                if not activate.value:
                    raise RuntimeError("MFEnumDeviceSources returned a null activation object")
                row = _device("msmf", index, None, None)
                for key, attribute in (("name", FRIENDLY_NAME), ("device_path", SYMBOLIC_LINK)):
                    row[key], error = api.attribute_text(activate, attribute)
                    if error:
                        row[key + "_error"] = error
                row["friendly_name"] = row["name"]
                rows.append(row)
            return rows
        finally:
            try:
                if activates:
                    for index in range(count.value):
                        api.release(POINTER(activates[index]))
                    api.ole32.CoTaskMemFree(activates)
                api.release(attributes)
            finally:
                if started:
                    _check(api.mfplat.MFShutdown(), "MFShutdown")


def enumerate_windows_camera_devices():
    return {"msmf": enumerate_msmf_devices(), "dshow": enumerate_directshow_devices()}


def _inventory(devices):
    rows = [f"{backend}[{row['index']}]: {row.get('name') or row.get('friendly_name') or '(名前取得不可)'}"
            f" ({row.get('device_path') or 'パス取得不可'})"
            for backend, items in devices.items() for row in items]
    return "\n".join(rows) or "カメラは見つかりませんでした。"


def _is_camo(row):
    name = (row.get("name") or row.get("friendly_name") or "").strip().casefold()
    return re.match(r"^camo(?:\b|[ _-])", name) is not None


def _is_arm_usb(row):
    path = (row.get("device_path") or "").casefold()
    name = (row.get("name") or row.get("friendly_name") or "").strip().casefold()
    # The installed arm camera has these confirmed USB identifiers. An unrelated
    # USB camera (including Hy-UXGA built into the PC) is not a fallback.
    if "vid_" in path or "pid_" in path:
        return "vid_1e45" in path and "pid_8022" in path
    return name == "usb hd camera"


def resolve_iphone_camera_devices(indices=None):
    """Resolve stable camera roles from names/USB identity, correcting old numbers.

    indices are previous requested values (iPhone/MSMF, arm/DSHOW). They never
    override identity or resolve multiple matching devices. Equal numbers across
    distinct backends are allowed.
    """
    if indices is not None:
        if len(indices) != 2 or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
                                    for index in indices):
            raise ValueError("カメラ番号は0以上の整数を2つ指定してください。")
    devices = enumerate_windows_camera_devices()
    selected = {}
    for position, (role, backend, predicate, label) in enumerate((
            ("camera0", "msmf", _is_camo, "iPhoneのCamoカメラ (MSMF)"),
            ("camera1", "dshow", _is_arm_usb, "USBアームカメラ (USB HD Camera / VID_1E45 PID_8022, DirectShow)"))):
        candidates = [row for row in devices[backend] if predicate(row)]
        if len(candidates) != 1:
            reason = "見つかりません" if not candidates else "複数あり特定できません"
            raise RuntimeError(f"{label}が{reason}。\n検出した機器:\n{_inventory(devices)}")
        row = dict(candidates[0])
        requested = indices[position] if indices is not None else None
        row.update(role=role, requested_index=requested,
                   index_corrected=requested is not None and requested != row["index"])
        selected[role] = row
    first_path = selected["camera0"].get("device_path")
    second_path = selected["camera1"].get("device_path")
    if first_path and second_path and first_path.casefold() == second_path.casefold():
        raise RuntimeError(f"iPhoneとアームカメラが同じ機器として検出されました。\n検出した機器:\n{_inventory(devices)}")
    return selected


if __name__ == "__main__":
    try:
        print(json.dumps({"metadata_only": True, "selected": resolve_iphone_camera_devices()},
                         ensure_ascii=False, indent=2))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

