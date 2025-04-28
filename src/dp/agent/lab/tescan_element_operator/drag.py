# encoding: utf-8

from pywinauto_recorder.player import *


with UIPath(u"TESCAN Essence||Window"):
	with UIPath(u"||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Group->||Custom->||Group->||Group->||Group->||Custom->||Group->||Group"):
		click(u"||Custom->||Group->||Table->I-Etching||DataItem")
	with UIPath(u"||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Group->||Custom->||Group->||Group->||Group->||Custom->||Group->||Group->Layers||Tab"):
		drag_and_drop(u"Objects||TabItem", u"Objects||TabItem")
	with UIPath(u"||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Custom->||Group->||Custom"):
		click(u"||Group->||Group->||Group->||Custom->||Group->||Group->Layers||Tab->Objects||TabItem")
		drag_and_drop(u"||Group#[1,1]", u"||Group#[1,1]")
		click(u"||Group->||Group->||Group->||Custom->||Group->||Group")
		drag_and_drop(u"||Group#[1,1]", u"||Group#[1,1]")
		drag_and_drop(u"||Group#[1,1]", u"||Group#[1,1]")
		drag_and_drop(u"||Group#[1,1]", u"||Group#[1,1]")

