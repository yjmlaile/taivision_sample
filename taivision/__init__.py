#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""taivision: a deep learning vision toolkit powered by pytorch."""
from __future__ import absolute_import

pytorch_version = '1.7.1'

__version__ = '0.0.0'
import warnings
import os  
import torch 

# from . import data
# from . import model_zoo
# from . import nn
# from . import utils
# from . import loss

from taivision import data 
from taivision import model_zoo
from taivision import nn
from taivision import utils
# from taivision import loss
from taivision import io 


