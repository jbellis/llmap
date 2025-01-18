class AIException(Exception):
    """Base exception class for AI-related errors"""
    
    def __init__(self, message: str, filename: str = None, original_exception: Exception = None):
        """
        Initialize the AIException
        
        Args:
            message: Description of the error
            original_exception: The original exception that caused this error, if any
            filename: Name of the file being processed when error occurred, if applicable
        """
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message
        self.filename = filename
        
    def __str__(self):
        error_str = self.message
        if self.filename:
            error_str = f"{error_str} [File: {self.filename}]"
        if self.original_exception:
            error_str = f"{error_str} (Original exception: {str(self.original_exception)})"
        return error_str
