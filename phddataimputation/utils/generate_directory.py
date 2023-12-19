#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:17:16 2023

@author: goharshoukat
"""
__all__ = ["generateDirectory"]
import os


def generateDirectory(path: str) -> None:

    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
