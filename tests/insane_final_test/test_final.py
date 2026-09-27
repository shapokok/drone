from pathlib import Path
import sys,unittest,ast,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_final_test.protocol import select_specs

class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'history_s':20.,'recovery_s':10.,'duration_s':[10,30,60],'validation_onset_stride_s':60.}
    def test_selector_exact_original_algorithm(self):
        # Execute only the original time-only selection block, never old preparation main.
        text=(ROOT/'scripts/insane/prepare.py').read_text()
        start=text.index('        sensor,ref=loaded[seq];agg=aggs[seq];number=0')
        end=text.index('    for epoch in range(1,cfg[',start)
        import textwrap
        block=textwrap.dedent(text[start:end])
        for segments in [[(0.,175.)],[(.2471468449,92.24706435)],[(0.,62.)],[(0.,9.),(20.,140.)]]:
            t=np.arange(.05,200,.05);seq='sample'
            scope={'np':np,'loaded':{seq:(None,None)},'seq':seq,'aggs':{seq:{'t':t}},'inventory':{seq:{'segments_s':segments}},'cfg':self.cfg,'plans':[],'unsupported':[]}
            exec(compile(block,'frozen_original_selector','exec'),scope)
            actual,unsupported=select_specs(seq,segments,t,self.cfg,split='validation')
            self.assertEqual(actual,scope['plans']);self.assertEqual(unsupported,scope['unsupported'])
    def test_time_only_inputs_and_no_flight_stitch(self):
        a,missing=select_specs('mars_6',[(0.,35.),(40.,75.)],np.arange(.05,76,.05),self.cfg)
        self.assertEqual([s['duration_s'] for s in a],[0]);self.assertEqual(len(missing),3)
    def test_native_grid_equal_onsets_and_recovery(self):
        plans,_=select_specs('mars_7',[(0.,100.)],np.arange(.037,101,.05),self.cfg)
        outages=[s for s in plans if s['duration_s']]
        self.assertEqual(len(outages),3);self.assertEqual(len({s['onset_s'] for s in outages}),1)
        for s in outages:self.assertAlmostEqual(s['end_s']-s['onset_s'],s['duration_s']+10)
    def test_no_optimizer_backward_or_training_runner(self):
        text=(ROOT/'src/insane_final_test/run.py').read_text();tree=ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n,ast.Attribute):self.assertNotIn(n.attr,['backward','fit_scaler','batch_loss','choose_tau','step'])
        self.assertNotIn('torch.optim',text);self.assertNotIn('insane_adaptive.run',text)
    def test_explicit_test_gate(self):
        from insane_final_test.run import execute
        with self.assertRaisesRegex(RuntimeError,'Explicit final test authorization'):
            execute(None,None,None,allow_test=False)

if __name__=='__main__':unittest.main()
