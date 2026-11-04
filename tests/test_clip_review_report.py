"""Visual approval is scoped to reviewed identities and cannot change metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from coopx.eval.report import render_summary, _scene_review

class ClipReviewReportTests(unittest.TestCase):
    def test_approved_selection_does_not_approve_timing_or_an_unreviewed_clip(self):
        runs=[{'split':'report','scene':'clip1','metrics':{}}]
        review={'date':'2026-11-04','approved_scenes':['clip1']}
        text=render_summary(runs,scene_review=review)
        self.assertIn('approved these report clips',text)
        self.assertIn('not warning timing',text)
        self.assertIn('`insufficient_evidence`',text)
        self.assertIn('0.2 s roadside support',text)
        unreviewed=render_summary(runs+[{'split':'report','scene':'clip2','metrics':{}}],scene_review=review)
        self.assertIn('approval has not been supplied',unreviewed)
        self.assertIn('approval has not been supplied',render_summary(runs))
        # Review text never changes generated numeric rows.
        self.assertEqual([line for line in text.splitlines() if line.startswith('|')],
                         [line for line in render_summary(runs).splitlines() if line.startswith('|')])

    def test_reused_clip_name_cannot_reuse_another_targets_review(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=root/'scenes.json'
            scene={'name':'clip1','scenario':'scene1','ego_id':'1','targets':[{'gt_id':'81'}],
                   'hidden_interval':{'81':['000029',None]},'observers':[['rsu','-1','cam1']]}
            item={'scene':'clip1','scenario':'scene1','ego_id':'1','gt_id':'81','frame_id':'000029',
                  'rsu_agent':'-1','rsu_camera':'cam1','approved_for_report':True}
            manifest.write_text(json.dumps({'scenes':[scene]}))
            self.assertIsNone(_scene_review(manifest))
            (root/'human-review.json').write_text(json.dumps({'reviewed_on':'2026-11-04','clips':[item]}))
            self.assertEqual(_scene_review(manifest)['approved_scenes'],['clip1'])
            scene['scenario']='another-scene'
            manifest.write_text(json.dumps({'scenes':[scene]}))
            self.assertEqual(_scene_review(manifest)['approved_scenes'],[])


class TuningVisibilityReviewTests(unittest.TestCase):
    def test_frame_96_keeps_insufficient_evidence_tuning_label(self):
        from coopx.eval.runner import load_eval_scenes
        with tempfile.TemporaryDirectory() as td:
            entries=load_eval_scenes('frame-000096',path=str(Path(td)/'absent.json'))
        scene,split=entries[0]
        self.assertEqual(split,'tune')
        self.assertEqual(scene.visibility[('53','2','000096')].label,'insufficient_evidence')
