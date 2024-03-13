class LinearScaler:
    def __init__(self,slope=1.0,offset=0.0):
        self.slope = slope
        self.offset = offset
    def rescale_range(self,low_in,high_in,low_out,high_out):
        self.slope = (high_out-low_out)/(high_in-low_in)
        self.offset = low_in - low_out/self.slope
    def scale(self,x):
        return self.slope * (x-self.offset)
