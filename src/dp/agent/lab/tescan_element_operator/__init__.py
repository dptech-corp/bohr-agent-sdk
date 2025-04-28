# encoding: utf-8

from pywinauto_recorder.player import *

def start_electron_beam():
    with UIPath(u"TESCAN Essence||Window"):
        with UIPath(u"||ToolBar->||Custom->||Group->||Custom"):
            click(u"||CheckBox#[0,0]")

def start_ion_beam():
    with UIPath(u"TESCAN Essence||Window"):
        with UIPath(u"||ToolBar->||Custom->||Group->||Custom"):
            click(u"||CheckBox#[0,1]")



def toggle_electron_scanning():
    with UIPath(u"TESCAN Essence||Window"):
        with UIPath(u"||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Group->||Custom"):
            click(u"||CheckBox#[0,1]")


def toggle_ion_scanning():
    with UIPath(u"TESCAN Essence||Window"):
        with UIPath(u"||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Group->||Custom"):
            click(u"||CheckBox#[0,4]")
