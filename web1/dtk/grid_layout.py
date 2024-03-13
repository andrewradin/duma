
class Box:
    node_width=60
    node_height=60
    def __init__(self,start=None,mid=None,end=None,margin=0):
        self.start=start or []
        self.mid=mid or []
        self.end=end or []
        self.margin=margin
    def _iwidth(self,item):
        if isinstance(item,Box):
            return item.width()
        return self.node_width
    def _iheight(self,item):
        if isinstance(item,Box):
            return item.height()
        return self.node_height
    def _hlist(self):
        return [self._iheight(x) for x in self.start+self.mid+self.end]
    def _wlist(self):
        return [self._iwidth(x) for x in self.start+self.mid+self.end]
    def _runlist(self,positions,l,x,cy):
        for item in l:
            iw=self._along(item)
            ih=self._across(self)
            if isinstance(item,Box):
                item.layout(positions,*self._convert(x,cy-ih/2,iw,ih))
            else:
                ox,oy,_,_ = self._convert(x,cy-ih/2,iw,ih)
                positions.append((item,ox,oy))
            x += iw
    def layout(self,positions,x,y,w=None,h=None):
        if w is None:
            w = self.width()
        if h is None:
            h = self.height()
        self.x = x
        self.y = y
        x0,y0,w,h = self._convert(x,y,w,h)
        cy = y0+h/2
        if self.start:
            self._runlist(positions,self.start,x0+self.margin/2,cy)
        if self.mid:
            self._runlist(positions,self.mid,
                    x0 + w/2 - sum(self._along(x) for x in self.mid)/2,
                    cy,
                    )
        if self.end:
            self._runlist(positions,self.end,
                    x0 + w - sum(self._along(x) for x in self.end)-self.margin/2,
                    cy,
                    )

class VBox(Box):
    def width(self): return max([0]+self._wlist())
    def height(self): return sum(self._hlist())+self.margin
    def _convert(self,x,y,w,h): return (y,x,h,w)
    def _along(self,item): return self._iheight(item)
    def _across(self,item): return self._iwidth(item)

class HBox(Box):
    def width(self): return sum(self._wlist())+self.margin
    def height(self): return max([0]+self._hlist())
    def _convert(self,x,y,w,h): return (x,y,w,h)
    def _along(self,item): return self._iwidth(item)
    def _across(self,item): return self._iheight(item)

