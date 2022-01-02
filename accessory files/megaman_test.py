#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:59:03 2019

@author: adamreidsmith
"""
import numpy as np
from megaman.geometry import Geometry
from megaman.embedding import Isomap

X = np.random.randn(100, 10)
radius = 5
adjacency_method = 'cyflann'
adjacency_kwds = {'radius':radius} # ignore distances above this radius

geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds)

isomap = Isomap(n_components=50, eigen_solver='arpack', geom=geom)
embed_isomap = isomap.fit_transform(X)