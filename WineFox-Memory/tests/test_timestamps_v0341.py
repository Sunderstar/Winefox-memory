"""v0.3b.4.1 timestamps in injected memories, no network dependencies."""
import json
import tempfile
import unittest
from pathlib import Path

from maidcore import MaidCore
from maidcore.retrieval import MemoryRetriever


class TestMemoryTimestampInjection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.core=MaidCore(Path(self.temp.name)/'data')
        self.core.write_persona('测试用虚构角色')
        self.r=MemoryRetriever(self.core)
        self.r.update_settings({'semantic_backend':'lexical',
                                'spontaneous_recall_probability':0.,
                                'episode_tool_extra_cards':0})

    def episode(self,*,start=None,end=None,status='approved'):
        eid='ep_timestamp_case'
        with self.r._db().connect() as con:
            con.execute('''INSERT INTO extracted_episodes
                (id,run_id,local_id,title,summary,detail,state,evidence_ids,
                 review_status,event_status,time_start_utc,time_end_utc)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (eid,'run_test','one','夜晚去海边','聊到在海边散步和拾贝壳',
                 '两位虚构人物谈过海边散步的约定和贝壳。','completed','[]',
                 status,'occurred',start,end))
        return eid

    def cand(self,mid='memdb_timestamps',*,event=None,created='2026-11-06T12:00:00+00:00',
             origin=None, evidence='[]',status='active'):
        with self.r._db().connect() as con:
            con.execute('''INSERT INTO memories
                (id,content,category,subject,fact_key,importance,confidence,evidence_ids,
                 evidence_class,platform,event_time,origin_candidate_id,status,created_at,updated_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (mid,'喜欢海边散步','preference','shared','',.9,.9,evidence,
                 'user_statement','terminal',event,origin,status,created,created))
        return mid

    def test_episode_beijing_span_in_card_detail_and_model_tool(self):
        eid=self.episode(start='2026-11-02T16:45:00+00:00',end='2026-11-03T00:15:30+00:00')
        result=self.r.prepare('还记得海边散步和贝壳吗',fixed_seed=1)
        stamp='2026-11-03T00:45:00+08:00 ～ 2026-11-03T08:15:30+08:00'
        self.assertIn('来源对话时间（北京时间）：'+stamp,result['memory_text'])
        self.assertIn('选中事件的详细经过',result['memory_text'])
        self.assertIn(stamp,result['cards'][0]['time_label'])
        self.assertIn(stamp,result['details'][0]['time_label'])
        tool=self.r.expand_episode_tool([eid],result)
        self.assertEqual(tool['expanded_count'],1)
        self.assertIn(stamp,tool['episodes'][0]['time_label'])
        self.assertIn('贝壳',tool['episodes'][0]['detail'])
        messages,prepared=self.core.build_messages('还记得海边散步和贝壳吗',
                                                     retriever=self.r,return_retrieval=True)
        self.assertIn(stamp,messages[0]['content'])
        self.assertIn('来源对话时间是证据消息的记录时间',messages[0]['content'])
        self.assertIn(stamp,prepared['memory_text'])

    def test_cand_event_time_prefers_source_over_archive_time(self):
        mid=self.cand(event='2026-11-02T16:02:03+00:00')
        p=self.r.prepare('海边散步',fixed_seed=2)
        self.assertEqual(len(p['cand']),1)
        t=p['cand'][0]['time_label']
        self.assertIn('2026-11-03T00:02:03+08:00',t)
        self.assertNotIn('2026-11-06',t)
        self.assertIn(t,p['memory_text'])
        self.assertEqual(p['selected'][0]['time_label'],t)

    def test_cand_multiple_source_timestamps_recovered_from_saved_evidence(self):
        mid=self.cand(event=None,origin='cand_real',evidence=json.dumps(['message_a','message_b']))
        payload={'messages':[
            {'id':'message_a','role':'user','event_time':'2026-11-01T19:00:00+00:00'},
            {'id':'message_b','role':'assistant','event_time':'2026-11-04T05:00:00+00:00'},
            {'id':'message_non_evidence','role':'user','event_time':'2022-01-01T01:00:00+00:00'}]}
        with self.r._db().connect() as con:
            con.execute('''INSERT INTO extraction_runs
                (id,batch_key,prompt_version,prompt_sha,source,model,mode,status,
                 input_payload,system_prompt,created_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                ('run_source','batch','v1','hash','chat','mock','two-pass','succeeded',
                 json.dumps(payload),'prompt','2026-11-05T10:00:00+00:00'))
            con.execute('INSERT INTO candidate_audit(candidate_id,warnings,run_id) VALUES (?,?,?)',
                        ('cand_real','[]','run_source'))
        p=self.r.prepare('海边散步',fixed_seed=2)
        label=p['cand'][0]['time_label']
        self.assertIn('2026-11-02T03:00:00+08:00',label)
        self.assertIn('2026-11-04T13:00:00+08:00',label)
        self.assertNotIn('2022-01-01',label)
        self.assertNotIn('仅知收录时间',label)
        self.assertIn(label,p['memory_text'])

    def test_unknown_and_recorded_time_explicitly_not_event_time(self):
        self.episode(start='2026-11-03T09:00:00',end=None)
        self.cand(event='2026-11-02T10:00:00',created='2026-11-06T12:00:00+00:00')
        p=self.r.prepare('海边散步',fixed_seed=2)
        self.assertEqual(p['cards'][0]['time_label'],'时间未知')
        self.assertIn('仅知收录时间（非事件发生时间，北京时间）：2026-11-06T20:00:00+08:00',
                      p['cand'][0]['time_label'])
        self.assertNotIn('2026-11-02T10:00:00',p['memory_text'])

    def test_revoked_memory_stays_out_and_times_not_added_to_embedding_text(self):
        self.episode(start='2026-11-02T00:00:00+00:00',status='pending')
        mid=self.cand(status='revoked')
        self.assertEqual(self.r._entries(),[])
        p=self.r.prepare('海边散步',fixed_seed=1)
        self.assertFalse(p['cards'])
        self.assertFalse(p['cand'])
        with self.r._db().connect() as con:
            con.execute("UPDATE memories SET status='active' WHERE id=?",(mid,))
        entries=self.r._entries()
        self.assertEqual(entries[0]['search'],'喜欢海边散步 ')
        self.assertNotIn('2026-',entries[0]['search'])


if __name__=='__main__':
    unittest.main()
