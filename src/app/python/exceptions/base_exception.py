"""
File : base_exception.py
Description : Class is responsible for custom exception handling
Created on : 
Author :
E-mail :
"""

from src.app.python.utils.logger import Logger #pylint: disable = import-error

class AIPreProcessingException(Exception):
    """
    This class is responsible for custom exception handling
    """

    message = ''

    def __init__(self, message):
        self.message = message
        (Logger.get_logger()).error(self.message)

    def __str__(self):
        (Logger.get_logger()).error(self.message)
        return repr(self.message)


class FileNotFoundException(AIPreProcessingException):
    def __init__(self, message):
        super(FileNotFoundException,self).__init__(message)

class NoSuchProcessRunning(AIPreProcessingException):
    def __init__(self, message):
        super(NoSuchProcessRunning,self).__init__(message)

class BrokenExecutorError(AIPreProcessingException):
    def __init__(self, message):
        super(BrokenExecutorError,self).__init__(message)

class MessagingServiceException(AIPreProcessingException):
    def __init__(self, message):
        super(MessagingServiceException,self).__init__(message)

class ServiceBusTopicSenderException(AIPreProcessingException):
    def __init__(self, message):
        super(ServiceBusTopicSenderException,self).__init__(message)

class ResourceExistError(AIPreProcessingException):
    def __init__(self, message):
        super(ResourceExistError,self).__init__(message)

class ImageConvertExceptionError(AIPreProcessingException):
    def __init__(self, message):
        super(ImageConvertExceptionError,self).__init__(message)

class GetPutFrameExceptionError(AIPreProcessingException):
    def __init__(self, message):
        super(GetPutFrameExceptionError,self).__init__(message)

class InternalServerException(AIPreProcessingException):
    def __init__(self, message):
        super(InternalServerException,self).__init__(message)

class UnauthenticationException(AIPreProcessingException):
    def __init__(self, message):
        super(UnauthenticationException,self).__init__(message)

class ParameterMissingException(AIPreProcessingException):
    def __init__(self, message):
        super(ParameterMissingException,self).__init__(message)

class UnauthorizedException(AIPreProcessingException):
    def __init__(self, message):
        super(UnauthorizedException,self).__init__(message)

class InvalidTokenException(AIPreProcessingException):
    def __init__(self, message):
        super(InvalidTokenException,self).__init__(message)

class SchemaValidationException(AIPreProcessingException):
    def __init__(self, message):
        super(SchemaValidationException,self).__init__(message)

class AlreadyExistsException(AIPreProcessingException):
    def __init__(self, message):
        super(AlreadyExistsException,self).__init__(message)

class DataGenrationError(AIPreProcessingException):
    def __init__(self, message):
        super(DataGenrationError,self).__init__(message)

class ConnectionError(AIPreProcessingException):
    def __init__(self, message):
        super(ConnectionError,self).__init__(message)

class MetaDataUpdationError(AIPreProcessingException):
    def __init__(self, message):
        super(MetaDataUpdationError,self).__init__(message)

class AzureMessagingServiceManagerException(AIPreProcessingException):
    def __init__(self, message):
        super(AzureMessagingServiceManagerException,self).__init__(message)

class SensorInitiationException(AIPreProcessingException):
    def __init__(self, message):
        super(SensorInitiationException,self).__init__(message)

class SensorProcessingHandlerException(AIPreProcessingException):
    def __init__(self, message):
        super(SensorProcessingHandlerException,self).__init__(message)

class DatabaseConnectorException(AIPreProcessingException):
    def __init__(self, message):
        super(DatabaseConnectorException,self).__init__(message)

class SensorMetadataHandlerException(AIPreProcessingException):
    def __init__(self, message):
        super(SensorMetadataHandlerException,self).__init__(message)


class SensorClusterdataHandlerException(AIPreProcessingException):
    def __init__(self, message):
        super(SensorClusterdataHandlerException,self).__init__(message)

class ClusterDataException(AIPreProcessingException):
    def __init__(self, message):
        super(ClusterDataException,self).__init__(message)

class ClusterListEmptyException(AIPreProcessingException):
    def __init__(self, message):
        super(ClusterListEmptyException,self).__init__(message)


class DLLInternalError(AIPreProcessingException):
    def __init__(self, message):
        super(DLLInternalError,self).__init__(message)

class DLLLoadErrorException(AIPreProcessingException):
    def __init__(self, message):
        super(DLLLoadErrorException,self).__init__(message)

class AIPreprocFiltersException(AIPreProcessingException):
    def __init__(self, message):
        super(AIPreprocFiltersException,self).__init__(message)

class SchemaValidationException(AIPreProcessingException):
    def __init__(self, message):
        super(SchemaValidationException, self).__init__(message)