from unittest import TestCase
from tools import Enum,ProgressReader,ProgressWriter

class ToolTest(TestCase):
    def test_enum(self):
        enum1 = Enum([],
                    [ ('RED',)
                    , ('BLUE','Azul')
                    , ('YELLOW',None,False)
                    , ('GREEN',)
                    ])
        self.assertEqual(enum1.RED,0)
        self.assertEqual(enum1.GREEN,3)
        self.assertEqual(enum1.get('label',enum1.BLUE),'Azul')
        self.assertEqual(enum1.get('label',enum1.GREEN),'Green')
        self.assertEqual(enum1.find('label','Green'),enum1.GREEN)
        self.assertEqual(enum1.choices(),[(0,'Red'),(1,'Azul'),(3,'Green')])
        enum2 = Enum(['hovertext'],
                    [ ('UNCLASSIFIED',)
                    , ('FDA_TREATMENT','FDA Approved Treatment')
                    , ('KNOWN_TREATMENT',None,None
                            ,"This is hovertext for KNOWN_TREATMENT")
                    , ('INITIAL_PREDICTION',)
                    , ('KNOWN_CAUSE',)
                    , ('CANDIDATE_CAUSE',)
                    , ('FDA_CAUSE','FDA Documented Cause')
                    ])
        self.assertEqual(enum2.UNCLASSIFIED,0)
        self.assertEqual(enum2.FDA_CAUSE,6)
        self.assertEqual(enum2.get('label',enum2.KNOWN_TREATMENT)
                    ,'Known Treatment')
        self.assertEqual(enum2.get('label',enum2.FDA_TREATMENT)
                    ,'FDA Approved Treatment')
        self.assertEqual(enum2.get('hovertext',enum2.KNOWN_TREATMENT)
                    ,'This is hovertext for KNOWN_TREATMENT')
        self.assertEqual(enum2.get('hovertext',enum2.UNCLASSIFIED)
                    ,None)
        self.assertEqual(enum2.get('hovertext',enum2.UNCLASSIFIED,"")
                    ,"")
        with self.assertRaises(AttributeError):
            enum2.get('garbage',0)
        with self.assertRaises(AttributeError):
            enum2.find('garbage','anything')
        with self.assertRaises(ValueError):
            enum2.find('label','garbage')
    def test_progress(self):
        phases = ["phase 1","phase 2","phase 3"]
        statuses = ["status 1","another status","won't apostrophes cause a problem?"]
        wr = ProgressWriter("/tmp/progress",phases)
        rd = ProgressReader("/tmp/progress")
        rpt = rd.get()
        self.assertEqual(rpt[0],[])
        self.assertEqual(rpt[1],[list(x) for x in zip(phases,['']*3)])
        wr.put(phases[0],statuses[0])
        rpt = rd.get()
        self.assertEqual(rpt[0],[list(x) for x in zip(phases,statuses[0:1])])
        self.assertEqual(rpt[1],[list(x) for x in zip(phases[1:],['']*2)])
        wr.put(phases[1],statuses[1])
        rpt = rd.get()
        self.assertEqual(rpt[0],[list(x) for x in zip(phases,statuses[0:2])])
        self.assertEqual(rpt[1],[list(x) for x in zip(phases[2:],['']*1)])
        with self.assertRaises(Exception):
            wr.put(phases[1],"something else")
        rpt = rd.get()
        self.assertEqual(rpt[0],[list(x) for x in zip(phases,statuses[0:2])])
        self.assertEqual(rpt[1],[list(x) for x in zip(phases[2:],['']*1)])
        wr.put(phases[2],statuses[2])
        rpt = rd.get()
        self.assertEqual(rpt[0],[list(x) for x in zip(phases,statuses[0:3])])
        self.assertEqual(rpt[1],[])

