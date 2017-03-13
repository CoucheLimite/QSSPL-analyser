import unittest
import os
try:
    from analyser.utils import importexport as io
    from analyser.analysis import calculate_quantities as CQ
except:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),
                                 '..'))
    from analyser.utils import importexport as io


class Test_loading(unittest.TestCase):

    def test_loader_labview(self):

        fname = os.path.join(os.path.dirname(__file__),
                             'test_files', '500_1e2_1e8_Raw Data.dat')

        ld = io.LoadData(fname)
        data, inf = ld.load()

        assert data.dtype.names == ('time', 'PC', 'gen', 'PL')

    def test_loader_python0(self):
        fname = os.path.join(os.path.dirname(__file__),
                             r'test_files', 'F-AL_0.1Hz_625_1e3_1e7.Raw Data.dat')
        ld = io.LoadData(fname)
        data, inf = ld.load()

        assert data.dtype.names == ('time', 'gen', 'PC', 'PL')

    # def test_votlage(self):
    #
    #     nxc = CQ.nxc_from_photoconductance(some values)
    #     assert nxc == somenumber


# if __name__ == "__main__":
#
#     TestSomething().test_loader_labview()
#     TestSomething().test_loader_python0()
