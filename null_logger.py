"""
A dummy logger module that provides the same interface as the original
guided_diffusion.logger but does nothing. This is used to suppress all
logging output from the diffusion library globally.
"""

def configure(*args, **kwargs):
    """Does nothing."""
    pass

def log(*args, **kwargs):
    """Does nothing."""
    pass

def dump(*args, **kwargs):
    """Does nothing."""
    pass 