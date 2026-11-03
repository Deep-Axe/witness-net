from dataclasses import replace
import random
import unittest
from coopx.assess import Assessor
from coopx.domain import ObjectClass
from coopx.coverage import CoverageMap
from coopx.envelope import EvidenceStore,evidence_ref
from coopx.identity import NodeKey,TrustAnchor,TrustClass
from coopx.eval.fusion import fuse,FUSION_MODES
from test_coverage import Fixture

class FusionTests(unittest.TestCase):
    def setup_fusion(self):
        f=Fixture();map=CoverageMap(f.spec,store=f.store);return f,map,Assessor(f.store,map)
    def call(self,mode,f,map,a,ego=(),remote=(),now=1_000_000):
        return fuse(mode,ego_observations=ego,remote_admitted=remote,store=f.store,coverage_map=map,assessor=a,now_us=now)

    def test_ego_union_and_two_independent_reports(self):
        for mode in FUSION_MODES:
            f,map,a=self.setup_fusion();local=f.observation(source='ego')
            pole=f.observation(source='pole',xyz=(3.6,1.5,1.))
            beliefs=self.call(mode,f,map,a,[local],[pole])
            self.assertEqual(len(beliefs),1 if mode=='ego' else 2)
            self.assertTrue(beliefs[0].local)
            if mode in ('geo','witness'):self.assertTrue(all(b.state=='confirmed' for b in beliefs))
            if mode=='union':self.assertEqual(beliefs[1].state,'unverified')

    def test_ghost_in_free_space_is_hazard_in_union_but_contradicted_by_witness(self):
        for mode in ('union','geo','witness'):
            f,map,a=self.setup_fusion();cov=f.coverage(free={(3,1)});map.add(cov,1_000_000)
            ghost=f.observation(source='attacker')
            beliefs=self.call(mode,f,map,a,remote=[ghost])
            self.assertEqual(beliefs[0].state,'unverified' if mode=='union' else 'contradicted')
            self.assertIn(evidence_ref(ghost),beliefs[0].refs)

    def test_remote_clear_never_removes_or_downgrades_local_witness_beliefs(self):
        f,map,a=self.setup_fusion();local=f.observation(source='ego')
        remote=f.observation(source='attacker',cls=ObjectClass.UNKNOWN)
        # Use actual qualified range free space (stronger than an unsupported clear).
        map.add(f.coverage(free={(3,1)}),1_000_000)
        before=self.call('ego',f,map,a,[local])
        after=self.call('witness',f,map,a,[local],[remote])
        self.assertEqual(before[0],after[0])

    def test_unanchored_sybil_echoes_do_not_corroborate_a_real_witness(self):
        f,map,a=self.setup_fusion();anchors=dict(f.store.anchors)
        for i in range(5):
            s=f'sybil{i}';key=NodeKey.generate();f.keys[s]=key;f.sequences[s]=0
            anchors[s]=TrustAnchor(key.public_key,s+'-key',TrustClass.B,physical_observer=None,anchored=False,
                                   permitted_region=(-100.,-100.,-10.,100.,100.,10.))
        f.store=EvidenceStore(anchors);map=CoverageMap(f.spec,store=f.store);a=Assessor(f.store,map)
        remote=[f.observation(source='pole')]+[f.observation(source=f'sybil{i}') for i in range(5)]
        beliefs=self.call('witness',f,map,a,remote=remote)
        self.assertTrue(all(b.state=='unverified' for b in beliefs))
        self.assertEqual(f.store.independent_witnesses(remote[0],remote),1)

    def test_contested_witness_preserves_disagreement_and_geo_is_three_state(self):
        for mode in ('geo','witness'):
            f,map,a=self.setup_fusion();map.add(f.coverage(free={(3,1)}),1_000_000)
            remote=[f.observation(source='pole'),f.observation(source='car')]
            beliefs=self.call(mode,f,map,a,remote=remote)
            self.assertEqual({b.state for b in beliefs},{'contested' if mode=='witness' else 'contradicted'})

    def test_stale_packets_and_latest_updates_are_not_extra_votes(self):
        f,map,a=self.setup_fusion()
        reports=[f.observation(capture=0),f.observation(capture=900_000),f.observation(capture=1_000_000)]
        beliefs=self.call('union',f,map,a,remote=reports)
        self.assertEqual(len(beliefs),1);self.assertEqual(beliefs[0].refs,(evidence_ref(reports[-1]),))
        with self.assertRaises(ValueError):self.call('invented',f,map,a)


class FusionWiringTests(unittest.TestCase):
    setup_fusion = FusionTests.setup_fusion
    def test_optional_assessor_and_mismatched_map(self):
        f,grid,a=self.setup_fusion();remote=[f.observation(source='pole')]
        result=fuse('witness',ego_observations=[],remote_admitted=remote,store=f.store,
                    coverage_map=grid,assessor=None,now_us=1_000_000)
        self.assertEqual(result[0].state,'unverified')
        with self.assertRaises(ValueError):
            fuse('witness',ego_observations=[],remote_admitted=remote,store=f.store,
                 coverage_map=CoverageMap(f.spec),assessor=a,now_us=1_000_000)


class StickyRefutationTests(unittest.TestCase):
    def test_memory_keeps_a_refuted_remote_claim_contradicted(self):
        from types import SimpleNamespace
        import coopx.eval.fusion as fusion
        # Exercised end to end in tests/test_eval.py (ghost_free caught on a mined clip);
        # here only the contract: memory is optional and keyword-only.
        import inspect
        sig = inspect.signature(fusion.fuse)
        self.assertIn("memory", sig.parameters)
        self.assertIsNone(sig.parameters["memory"].default)
