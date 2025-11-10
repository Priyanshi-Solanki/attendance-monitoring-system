"""
Dlib-based face recognition module for attendance system.
"""

from .train_dlib import train_dlib_model
from .predict_dlib import predict_dlib_faces

__all__ = ['train_dlib_model', 'predict_dlib_faces']

