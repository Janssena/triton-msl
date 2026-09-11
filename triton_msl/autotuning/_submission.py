"""A shared, monotonic boundary between eligibility and caller-workload invocation."""

from triton_msl.errors import PostSubmitError


class SubmissionState:
    """Share one instance across the driver and every nested dispatch alternative.

    Mark before invoking the runtime, never reset, and never infer non-submission
    from an exception. Sharing also protects exceptions at a helper's return boundary.
    """

    def __init__(self):
        self.attempted = False

    def begin(self):
        self.attempted = True

    def reraise_if_attempted(self, error):
        if self.attempted:
            if isinstance(error, PostSubmitError):
                raise error
            raise PostSubmitError(
                "Caller-workload dispatch was attempted; submission/completion state is "
                "uncertain. Refusing fallback or replay after this failure."
            ) from error
