"""Synthetic public regression tests; no developer persona, chat or real API key."""
import json
import threading
from urllib.request import urlopen
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maidcore import MaidCore
from maidcore.retrieval import MemoryRetriever, RetrievalSettings
from maidcore.llm import APIConfig, ChatCompletionsClient, ConfigError, WineFoxChatSession
from maidcore.auto_archive import AutoArchiveManager
from maidcore.improved_extraction import build_batches, run_extraction
from maidcore.memory_verification import VerifiedMemoryDatabase


class DemoModel:
    config=SimpleNamespace(model='offline-test',reasoning_effort='none',max_tokens=8192,timeout=30)
    last_usage=None

    def complete(self, messages):
        payload=json.loads(messages[-1]['content'].split('\n',1)[1])
        ids=[r['id'] for r in payload['messages']]
        if 'episodes' not in payload:
            return json.dumps({'episodes':[{'id':'e1','title':'参观自然博物馆',
              'summary':'用户与助手讨论博物馆之行。',
              'detail':'用户去了自然博物馆，看到恐龙化石，双方谈到展览。',
              'event_status':'occurred','plans':[],'evidence_ids':ids[:2]}]},ensure_ascii=False)
        return json.dumps({'memories':[{'content':'用户曾参观自然博物馆并喜欢恐龙化石',
          'category':'event','subject':'user','fact_key':'',
          'importance':0.7,'confidence':0.9,'evidence_ids':ids[:2],
          'episode_ids':['e1']}]},ensure_ascii=False)


class TestPublicBasics(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.core=MaidCore(self.root/'data')

    def test_empty_project_creates_only_generic_persona_and_no_history(self):
        self.assertTrue((self.core.root/'persona.md').exists())
        self.assertEqual(self.core.chat_history(),[])
        self.assertEqual(self.core.list_memories(),[])
        self.assertNotIn('API_KEY=',self.core.read_persona())

    def test_persona_write_and_invalid_blank(self):
        self.core.write_persona('喜欢书本的虚构角色')
        self.assertEqual(self.core.read_persona(),'喜欢书本的虚构角色')
        with self.assertRaises(ValueError):self.core.write_persona('   ')

    def test_explicit_platform_labels_do_not_integrate_external_services(self):
        row=self.core.append_chat('user','我想读天文学',platform='qq')
        self.assertEqual(row['platform'],'qq')
        self.assertEqual(len(self.core.chat_history()),1)

    def test_chat_exchange_saves_full_turn(self):
        rows=self.core.append_exchange('那颗星星是什么？','可能是一颗行星。',platform='terminal',model='demo')
        self.assertEqual([x['role'] for x in rows],['user','assistant'])
        self.assertEqual(len({x['id'] for x in rows}),2)
        self.assertEqual(len(self.core.chat_history()),2)

    def test_legacy_memory_create_edit_recall_and_forget(self):
        row=self.core.remember('喜欢在周末阅读图书',category='preference')
        self.assertEqual(self.core.search('周末')[0]['content'],row['content'])
        self.core.update_memory(row['id'],'喜欢阅读科普读物')
        self.assertIn('科普',self.core.list_memories()[0]['content'])
        self.core.forget(row['id'])
        self.assertEqual(self.core.list_memories(),[])

    def test_retrieval_defaults_and_override(self):
        settings=RetrievalSettings.load(self.core.root)
        self.assertGreater(settings.max_input_tokens,settings.memory_budget_tokens)
        r=MemoryRetriever(self.core)
        updated=r.update_settings({'semantic_backend':'lexical','cand_limit':5})
        self.assertEqual(r.get_settings().cand_limit,5)
        self.assertEqual(r.prepare('阅读科学')['backend'],'lexical only')

    def test_context_query_reflects_recent_both_sides(self):
        self.core.append_chat('user','我参观了博物馆',platform='terminal')
        self.core.append_chat('assistant','你看见恐龙化石了吗？',platform='terminal')
        r=MemoryRetriever(self.core)
        r.update_settings({'semantic_backend':'lexical'})
        _,result=self.core.build_messages('之前我们也聊过类似话题吗？',retriever=r,return_retrieval=True)
        self.assertIn('博物馆',result['query_context'])
        self.assertIn('恐龙化石',result['query_context'])
        self.assertEqual(result['context_weight'],.70)

    def test_archive_defaults_off_to_avoid_unexpected_bills(self):
        manager=AutoArchiveManager(self.core)
        self.assertFalse(manager.settings().enabled)
        self.assertFalse(manager.plan()['ready'])

    def test_api_config_missing_key_fails_without_network(self):
        env=self.root/'settings.env'
        env.write_text('API_KEY=\nMODEL=deepseek-chat\nBASE_URL=https://api.deepseek.com\n')
        with patch.dict('os.environ',{'API_KEY':''}):
            with self.assertRaises(ConfigError):APIConfig.load(env)

    def test_approved_episode_can_be_summarized_and_read_by_tool(self):
        r=MemoryRetriever(self.core)
        r.update_settings({'semantic_backend':'lexical'})
        with r._db().connect() as con:
            con.execute('''INSERT INTO extracted_episodes
                (id,run_id,local_id,title,summary,detail,state,evidence_ids,review_status,event_status)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                ('ep_demo','run_demo','e1','参观自然博物馆','参观馆内的恐龙化石',
                 '用户在自然博物馆看到了完整的恐龙骨架，后来讨论了展品。',
                 'completed','[]','auto_active','occurred'))
        prepared=r.prepare('你还记得我们参观自然博物馆吗？',fixed_seed=10)
        self.assertIn('ep_demo',prepared['tool_available_episode_ids'])
        res=r.expand_episode_tool(['ep_demo','ep_fake'],prepared)
        self.assertEqual(res['expanded_count'],1)
        self.assertIn('恐龙骨架',res['episodes'][0]['detail'])
        with r._db().connect() as con:
            con.execute("UPDATE extracted_episodes SET review_status='rejected' WHERE id='ep_demo'")
        self.assertEqual(r.expand_episode_tool(['ep_demo'],prepared)['expanded_count'],0)

    def test_regular_extraction_can_publish_without_human_review(self):
        for i in range(3):
            self.core.append_chat('user',f'自然博物馆里我看到化石 {i}')
            self.core.append_chat('assistant',f'恐龙展品真有意思 {i}')
        db=VerifiedMemoryDatabase(self.core.root/'memory.db')
        batch=build_batches(self.core,'chat',max_messages=40,max_chars=8000)[0]
        result=run_extraction(db,batch,DemoModel(),mode='two-pass')
        self.assertIn('run_id',result)
        mgr=AutoArchiveManager(self.core,db=db)
        preview=mgr.publish_pending(dry_run=True)
        self.assertEqual(preview['auto_candidates'],1)
        self.assertEqual(len(db.all_memories()),0)
        applied=mgr.publish_pending(dry_run=False)
        self.assertEqual(applied['auto_episodes'],1)
        self.assertEqual(len(db.all_memories()),1)
        self.assertEqual(len(self.core.chat_history()),6)

    def test_http_workbench_bootstrap_without_api_key(self):
        from maidcore.review_ui import MemoryWorkbench, create_server
        workbench=MemoryWorkbench(self.core.root,self.root/'missing.env')
        server=create_server(workbench)
        server_thread=threading.Thread(target=server.serve_forever,daemon=True)
        server_thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_address[1]}'
            with urlopen(url+'/') as response:
                page=response.read().decode('utf-8')
            self.assertIn('自动',page)
            with urlopen(url+'/api/retrieval/settings') as response:
                setting=json.load(response)
            self.assertIn('memory_budget_tokens',setting)
        finally:
            server.shutdown();server.server_close();server_thread.join(3)

    def test_new_project_memory_preview_with_no_personal_data(self):
        from maidcore.retrieval import MemoryRetriever
        r=MemoryRetriever(self.core)
        settings=r.get_settings()
        result=r.prepare('你好，今天读了什么书？',settings=settings)
        self.assertEqual(result['cards'],[])
        self.assertEqual(result['cand'],[])
        self.assertEqual(result['details'],[])

    def test_chat_model_without_tool_call_stays_single_request(self):
        config=APIConfig(api_key='dummy',base_url='http://localhost:8765',model='test')
        client=ChatCompletionsClient(config)
        r=MemoryRetriever(self.core)
        with patch.object(client,'_chat_request',return_value={'role':'assistant','content':'这是模拟回复。'}) as stub:
            reply=client.complete_with_episode_tool([{'role':'user','content':'你好'}],prepared=r.prepare('你好'),retriever=r)
        self.assertEqual(reply,'这是模拟回复。')
        self.assertEqual(stub.call_count,1)


if __name__=='__main__':unittest.main()
