#!/usr/bin/python
import matlab.engine
eng = matlab.engine.start_matlab()
eng.Palm_reader_My(nargout=0)
eng.quit()

