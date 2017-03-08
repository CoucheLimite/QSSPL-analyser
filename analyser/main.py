#! py -2


from gui.heapofcrap import Analyser
import wx


def main():

    # False stands for?
    app = wx.App(False)
    Analyser(False)
    app.MainLoop()

if __name__ == "__main__":
    main()
