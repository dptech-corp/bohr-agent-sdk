# encoding: utf-8

from pywinauto_recorder.player import *


with UIPath(u"TESCAN Essence||Window"):
	with UIPath(u"||ToolBar->||Custom->||Group->||Custom"):
		click(u"||CheckBox#[0,1]")

