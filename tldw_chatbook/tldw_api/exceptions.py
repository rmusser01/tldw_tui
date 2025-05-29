# tldw_chatbook/tldw_api/exceptions.py
#
#
#######################################################################################################################
#
# Functions:

class TLDWAPIError(Exception):
    """Base exception for tldw_api errors."""
    pass

class APIConnectionError(TLDWAPIError):
    """Raised for network or connection issues."""
    pass

class APIRequestError(TLDWAPIError):
    """Raised for errors in constructing or sending the request (e.g., bad data)."""
    pass

class APIResponseError(TLDWAPIError):
    """Raised for non-2xx responses or issues parsing the response."""
    def __init__(self, status_code: int, message: str, response_data: dict = None):
        super().__init__(f"API Error {status_code}: {message}")
        self.status_code = status_code
        self.response_data = response_data or {}

class AuthenticationError(TLDWAPIError):
    """Raised for authentication failures."""
    pass

#
# End of tldw_chatbook/tldw_api/exceptions.py
########################################################################################################################
