# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:35:50 2019

@author: Gad
"""

def replace(string,target,rep):
    if len(string) == 0:
        return ""
    else:
        if string[:len(target)] == target:
            return rep+replace(string[len(target):],target,rep)
        else:
            return string[0] + replace(string[1:],target,rep) 
        

        