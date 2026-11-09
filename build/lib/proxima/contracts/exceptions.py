class ContractValidationError(Exception):
    """Raised when contract input or output is structurally invalid."""


class DomainError(Exception):
    """Raised when parameters are outside the allowed physical domain."""


class NumericalStabilityError(Exception):
    """Raised when numeric stability constraints are violated."""


class IntegrationError(Exception):
    """Raised when geodesic integration fails irrecoverably."""


class PhysicalConstraintViolation(Exception):
    """Raised when hard physical constraints are violated."""


class DataProviderUnavailable(Exception):
    """Raised when an external or local data source is unavailable."""


class DataSchemaMismatch(Exception):
    """Raised when a provider response does not match expected schema."""


class RateLimitExceeded(Exception):
    """Raised when the upstream data source rate-limit is exceeded."""
