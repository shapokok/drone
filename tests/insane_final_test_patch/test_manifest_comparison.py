from pathlib import Path
import sys,json,copy,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from insane_final_test_patch.comparison import compare,compare_and_record

class ManifestComparisonTests(unittest.TestCase):
    def setUp(self):self.expected=json.loads((ROOT/'configs/insane_final_test/test_manifest.json').read_text())
    def test_identical_passes(self):self.assertEqual(compare(self.expected,self.expected)['status'],'pass')
    def test_projection_roundoff_only_passes(self):
        a=copy.deepcopy(self.expected);a['inventory']['mars_6']['projection_error_m']+=5e-10
        r=compare(self.expected,a);self.assertEqual(r['status'],'pass');self.assertEqual(r['tolerated_inventory_difference_count'],1)
    def test_large_projection_difference_fails(self):
        a=copy.deepcopy(self.expected);a['inventory']['mars_6']['projection_error_m']+=1e-6
        self.assertEqual(compare(self.expected,a)['status'],'fail')
    def test_each_scientific_field_is_strict(self):
        for key in ['token_mask_sha256','native_gt_timestamp_sha256','id','n_native_gt_outage','onset_s','start_s','end_s']:
            with self.subTest(key=key):
                a=copy.deepcopy(self.expected);v=a['scenarios'][0][key];a['scenarios'][0][key]=v+'x' if isinstance(v,str) else v+1e-12
                self.assertEqual(compare(self.expected,a)['status'],'fail')
    def test_count_order_and_inventory_timestamp_are_strict(self):
        for change in ['count','order','epoch','origin','segments','sample_count','unknown_field']:
            with self.subTest(change=change):
                a=copy.deepcopy(self.expected)
                if change=='count':a['scenarios'].pop()
                elif change=='order':a['scenarios'].reverse()
                elif change=='epoch':a['inventory']['mars_6']['epoch_unix_s']+=1e-6
                elif change=='origin':a['inventory']['mars_6']['origin_enu_m'][0]+=1e-10
                elif change=='segments':a['inventory']['mars_6']['segments_s'][0][0]+=1e-12
                elif change=='sample_count':a['inventory']['mars_6']['imu_n']+=1
                else:a['inventory']['mars_6']['new_float']=1e-12
                self.assertEqual(compare(self.expected,a)['status'],'fail')
    def test_failure_records_actual_and_field_diff_before_raising(self):
        a=copy.deepcopy(self.expected);a['scenarios'][0]['token_mask_sha256']='changed'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'STOP before inference'):compare_and_record(self.expected,a,tmp)
            self.assertEqual(json.loads((Path(tmp)/'actual_regenerated_manifest.json').read_text()),a)
            d=json.loads((Path(tmp)/'manifest_field_diff.json').read_text());self.assertEqual(d['fields_differing'][0]['path'],'/scenarios/0/token_mask_sha256')
    def test_source_hash_mismatch_is_fatal_even_with_tolerated_float(self):
        a=copy.deepcopy(self.expected);a['inventory']['mars_7']['projection_error_m']+=1e-10
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):compare_and_record(self.expected,a,tmp,[{'path':'source','matched':False,'expected_sha256':'a','actual_sha256':'b'}])
            self.assertEqual(json.loads((Path(tmp)/'manifest_field_diff.json').read_text())['status'],'fail')
    def test_nonfinite_fails_and_is_recordable(self):
        a=copy.deepcopy(self.expected);a['inventory']['mars_7']['projection_error_m']=float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):compare_and_record(self.expected,a,tmp)
            self.assertTrue((Path(tmp)/'actual_regenerated_manifest.json').is_file())

if __name__=='__main__':unittest.main()
