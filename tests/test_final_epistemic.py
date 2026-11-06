import copy
import unittest
import numpy as np
from helixwm_final.epistemic import BootstrapDynamics, FrozenBootstrap, image_features, tree_digest

BINDING={'source_sha256':'a'*64,'protocol_sha256':'b'*64}

class EpistemicTests(unittest.TestCase):
    def create(self): return BootstrapDynamics(412,BINDING)
    def transition(self,m,i):
        r=np.random.default_rng(i)
        x=r.integers(0,256,(64,64,3),dtype=np.uint8)
        return m.update(x,np.zeros(6),x,1.,1.,transition_id=i,
                        previous_available=True,next_available=True)
    def test_features_axes_and_units(self):
        x=np.zeros((64,64,3),np.uint8);x[:16,:16]=255
        f=image_features(x);self.assertEqual(f.shape,(16,));self.assertEqual(f[0],1.);self.assertEqual(f.sum(),1.)
        self.assertEqual(image_features(x[None]).shape,(1,16))
    def test_separate_random_functions_and_fits(self):
        m=self.create()
        for i in range(40):self.transition(m,i)
        f=m.pin();self.assertTrue(f.ready)
        self.assertFalse(np.array_equal(m.projection[0],m.projection[1]))
        self.assertFalse(np.array_equal(m.gram[0],m.gram[1]))
        self.assertFalse(np.array_equal(m.theta[0],m.theta[1]))
        s=f.initialize(np.zeros((64,64,3),np.uint8),2)
        p,d=f.advance(s,np.zeros((2,6)))
        self.assertEqual(p.shape,(3,2,16));self.assertTrue(np.all(d['epistemic_disagreement']>0))
        self.assertEqual(m.additional_trainable_parameters,3456)
    def test_missing_images_never_touched(self):
        class Poison:
            def __array__(self,*a,**kw): raise AssertionError('Hidden pixels accessed')
        m=self.create();m.update(Poison(),None,Poison(),None,None,transition_id=0,
                                previous_available=False,next_available=False)
        self.assertEqual(m.accepted_pairs,0);self.assertEqual(m.skipped_pairs,1)
    def test_no_holdout_updates_or_replayed_transitions(self):
        m=self.create();before=tree_digest(m.snapshot())
        with self.assertRaises(ValueError):m.update(None,None,None,None,None,transition_id=0,
            previous_available=True,next_available=True,partition='validation')
        self.assertEqual(before,tree_digest(m.snapshot()))
        self.transition(m,0)
        with self.assertRaises(ValueError):self.transition(m,0)
    def test_frozen_copy_survives_learner_updates(self):
        m=self.create();f=m.pin();s=f.initialize(np.zeros((64,64,3),np.uint8));a=np.ones((1,6))*.1
        before=f.advance(s,a)
        for i in range(5):self.transition(m,i)
        m.pin();after=f.advance(s,a)
        np.testing.assert_array_equal(before[0],after[0]);self.assertFalse(f.ready)
    def test_restore_reproduces_next_bootstrap_update_and_prediction(self):
        m=self.create()
        for i in range(34):self.transition(m,i)
        m.pin();snapshot=m.snapshot();n=self.create();n.load(snapshot)
        self.transition(m,34);self.transition(n,34)
        self.assertEqual(tree_digest(m.snapshot()),tree_digest(n.snapshot()))
        self.assertEqual(m.pin().sha256,n.pin().sha256)
    def test_corruption_rejected_before_mutation(self):
        m=self.create();before=tree_digest(m.snapshot());bad=m.snapshot();bad['state']['rhs'][0,0,0]=1
        with self.assertRaises(ValueError):m.load(bad)
        self.assertEqual(before,tree_digest(m.snapshot()))
        bad=m.snapshot();bad['binding']['source_sha256']='c'*64
        bad['sha256']=tree_digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises(ValueError):m.load(bad)
    def test_frozen_hash_and_recursive_prediction(self):
        f=self.create().pin();g=FrozenBootstrap(f.snapshot());self.assertEqual(f.sha256,g.sha256)
        s=f.initialize(np.zeros((64,64,3),np.uint8))
        for _ in range(100):s,d=f.advance(s,np.zeros((1,6)))
        self.assertTrue(np.isfinite(s).all());self.assertEqual(f.observed_error(s,np.zeros((64,64,3),np.uint8)).shape,(1,))
        self.assertFalse(d['trained'])

    def test_tree_hash_is_structure_and_type_separated(self):
        # These different object topologies collided under unframed concatenation.
        left={'a':{'b':1},'c':2};right={'a':{},'b':1,'c':2}
        self.assertNotEqual(tree_digest(left),tree_digest(right))
        self.assertNotEqual(tree_digest([1,2]),tree_digest((1,2)))
        self.assertNotEqual(tree_digest({'x':[]}),tree_digest({'x':{}}))

    def test_frozen_schema_rejects_invalid_readiness_and_budget_metadata(self):
        record=self.create().pin().snapshot()
        edits=[('accepted_pairs',True),('accepted_pairs',-1),
            ('independent_bootstrap_counts',np.ones(2,np.int64)),
            ('independent_bootstrap_counts',np.ones(3,np.float64)),
            ('independent_bootstrap_counts',np.asarray([1,-1,1],np.int64)),
            ('additional_trainable_parameters',0),('calibrated_uncertainty',True)]
        for key,value in edits:
            bad=copy.deepcopy(record);bad[key]=value
            with self.subTest(key=key,value=str(value)):
                with self.assertRaises(ValueError):FrozenBootstrap(bad)

    def test_restore_rejects_foreign_or_future_pinned_head(self):
        m=self.create()
        for i in range(40):self.transition(m,i)
        m.pin();before=tree_digest(m.snapshot())
        for mutation in ('binding','version','accepted_pairs','counts','coefficients'):
            record=m.snapshot();pinned=record['state']['pinned']
            if mutation=='binding':pinned['binding']['source_sha256']='f'*64
            elif mutation=='version':pinned['version']+=1
            elif mutation=='accepted_pairs':pinned['accepted_pairs']+=1
            elif mutation=='counts':pinned['independent_bootstrap_counts']+=1000
            else:pinned['coefficients'][0,0,0]+=.5
            record['sha256']=tree_digest({k:v for k,v in record.items() if k!='sha256'})
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):m.load(record)
                self.assertEqual(before,tree_digest(m.snapshot()))

    def test_negative_bootstrap_counts_fail_even_with_recomputed_transport_digest(self):
        m=self.create();before=tree_digest(m.snapshot());record=m.snapshot()
        record['state']['bootstrap_counts'][0]=-1
        record['sha256']=tree_digest({k:v for k,v in record.items() if k!='sha256'})
        with self.assertRaises(ValueError):m.load(record)
        self.assertEqual(before,tree_digest(m.snapshot()))

    def test_invalid_transition_is_atomic_for_all_three_random_streams(self):
        m=self.create();before=tree_digest(m.snapshot());frame=np.zeros((64,64,3),np.uint8)
        cases=[(np.full(6,np.nan),1.,1.),(np.zeros(6),3.,1.),(np.zeros(6),1.,.5)]
        for action,reward,continuation in cases:
            with self.assertRaises(ValueError):m.update(frame,action,frame,reward,continuation,
                transition_id=0,previous_available=True,next_available=True)
            self.assertEqual(before,tree_digest(m.snapshot()))

if __name__=='__main__':unittest.main()
