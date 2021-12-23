#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast 

class Grid:
    def __init__(self):
        self._grid = {}
        self._position = {}
        self._finished = False
        self._started = False
        self._len = 1
       
    def __iter__(self):
        if not self._grid:
            raise ValueError("Cannot iterate over empty grid")

        self._started = True
        return self

    def add(self, var_name, values):
        if self._started:
            raise ValueError("Cannot modify Grid after start of iteration")

        self._grid[var_name] = values
        self._position[var_name] = 0
        self._len *= len(values)

    def __next__(self):
        if self._finished:
            raise StopIteration

        pack = {}
        keys = self._grid.keys()
        for k in keys:
            values = self._grid[k]
            pack[k] = values[self._position[k]]

        for i, k in enumerate(keys):
            self._position[k] += 1
            if self._position[k] < len(self._grid[k]):
                break
            else:
                if i == len(self._grid) - 1:
                    self._finished = True
                self._position[k] = 0

        return pack

    def reset(self):
        self._finished = False

    def __len__(self):
        return self._len

    def __str__(self):
        if not self._grid:
            return ""
        s = ""
        for key, values in self._grid.items():
            s += str(key) + ":\t" + str(values) + "\n"
        return s[:-1]

    def dimension(self):
        return len(self._grid)

    def contains(self, name):
        return name in self._grid.keys()

    def values(self):
        return self._grid.values()

    def var_names(self):
        return self._grid.keys()


    @staticmethod
    def fromstring(string):
        string = string.strip()
        grid = Grid()
        lines = string.split("\n")
        for line in lines:
            name, lst = line.split(":")
            values = ast.literal_eval(lst[lst.find("["):])
            grid.add(name, values)
        return grid

    @staticmethod
    def generate_filename(pack):
        filename = ""
        for key in sorted(pack.keys()):
            filename += str(key)[:4] + "_" + str(pack[key]) + "_"
        return "run_" + str(abs(hash(filename)))[:6]

