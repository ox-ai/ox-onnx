


class ModelInstallationError(Exception):
    """Exception raised for errors in the model installation process."""

    def __init__(
        self,
        message="Model installation failed due to an integrity check failure or missing files.",
    ):
        self.message = message
        super().__init__(self.message)


class ModelNotInterfacedError(Exception):
    """Exception raised for errors in the model interface process."""

    def __init__(
        self,
        message="Model id given is not interfaced use only interfased model ids",
    ):
        self.message = message
        super().__init__(self.message)
