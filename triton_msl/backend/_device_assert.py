"""Host result boundary for launch-local, device-recorded assertion failures."""


def check_failure(code, descriptor):
    if type(code) is not int or not 0 <= code <= len(descriptor["messages"]):
        raise RuntimeError("invalid device assertion status; results are not valid")
    if code:
        from triton_msl.errors import MetalDeviceAssertionError

        raise MetalDeviceAssertionError("device assertion failed: " + descriptor["messages"][code - 1])


def check_binding(arguments, descriptor):
    if len(arguments) != descriptor["buffer_index"]:
        raise RuntimeError("device assertion buffer binding differs from the sealed launch descriptor")
