# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:18:57 2026

@author: Administrator
"""

def log_decision(msg):
    print("[DECISION]", msg)

def log_event(msg):
    print("[EVENT]", msg)
    
def print_grid(grid):

    for row in grid:
        print(" ".join(str(x) for x in row))