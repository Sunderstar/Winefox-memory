"""Shareable diagnostics must not expose even deliberately planted fake secrets."""
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from maidcore import MaidCore
from maidcore.diagnostics import build_diagnostic_report, write_diagnostic_report
from maidcore.review_ui import MemoryWorkbench, create_server


class DiagnosticPrivacyTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.data=self.root/'data'
        self.core=MaidCore(self.data)
        self.secret='CANARY_PRIVATE_TEXT_34567890_NOT_TO_SHARE'
        self.path_marker='C:/Users/PRIVATE_PERSON_FOLDER'
        (self.root/'.env').write_text('API_KEY='+'sk-'+'TESTONLY012345678901234567890123'+'\n'+self.secret,encoding='utf-8')
        self.core.write_persona('A private persona says: '+self.secret)
        self.core.append_chat('user','A personal conversation is: '+self.secret)
        (self.data/'core_memory.md').write_text('Old Core: '+self.secret,encoding='utf-8')
        (self.data/'memory_retrieval.json').write_text(json.dumps({
            'enabled':True,'recent_chat_messages':200,'episode_tool_max_items':4,
            'query_instruction':self.secret,'embedding_model':self.secret,
            'context_weight_followup':self.secret}),encoding='utf-8')
        (self.data/'auto_archive_settings.json').write_text(json.dumps({
            'enabled':True,'trigger_messages':24,'keep_recent_messages':5,
            'exception_message':self.secret,'model_name':self.secret}),encoding='utf-8')
        logdir=self.data/'logs';logdir.mkdir(exist_ok=True)
        (logdir/'memory_recall.jsonl').write_text(json.dumps({
            'question':'User asks: '+self.secret,'query_context':self.secret,
            'estimated_total_input_tokens':200,'core_truncated':True,
            'tool_expansions':[{'id':self.secret,'tokens':300}]})+'\n',encoding='utf-8')
        (logdir/'memory_workbench_errors.jsonl').write_text(json.dumps({
            'type':'ValueError','message':self.secret,'traceback':self.path_marker})+'\n',encoding='utf-8')
        with sqlite3.connect(self.data/'memory.db') as con:
            con.execute('CREATE TABLE candidates(id TEXT,content TEXT,status TEXT)')
            con.execute('CREATE TABLE extracted_episodes(id TEXT,title TEXT,detail TEXT,review_status TEXT)')
            con.execute('CREATE TABLE memories(id TEXT,content TEXT,status TEXT)')
            con.execute('CREATE TABLE extraction_runs(id TEXT,status TEXT,input_payload TEXT)')
            con.execute('INSERT INTO candidates VALUES(?,?,?)',('secret_id',self.secret,'pending'))
            con.execute('INSERT INTO candidates VALUES(?,?,?)',('other_id',self.secret,self.secret))
            con.execute('INSERT INTO extracted_episodes VALUES(?,?,?,?)',('secret_id',self.secret,self.secret,'auto_active'))
            con.execute('INSERT INTO memories VALUES(?,?,?)',('secret_id',self.secret,'active'))
            con.execute('INSERT INTO extraction_runs VALUES(?,?,?)',('secret_id','succeeded',self.secret))

    def test_allowlist_excludes_private_values_and_paths(self):
        report=build_diagnostic_report(self.data,self.root)
        text=json.dumps(report,ensure_ascii=False)
        for forbidden in [self.secret,self.path_marker,'sk-'+'TESTONLY012345678901234567890123',str(self.root),
                          'secret_id','\"query_instruction\"','\"embedding_model\"','\"exception_message\"']:
            self.assertNotIn(forbidden,text)
        self.assertEqual(report['storage']['chat_message_lines'],1)
        self.assertEqual(report['database']['tables']['candidates']['count'],2)
        self.assertEqual(report['database']['tables']['candidates']['states']['other'],1)
        self.assertEqual(report['recent_recall_metrics']['core_truncated_turns'],1)
        self.assertEqual(report['recent_recall_metrics']['episode_expansion_turns'],1)
        self.assertTrue(report['storage']['model_env_file_present'])
        self.assertEqual(report['retrieval_settings']['recent_chat_messages'],200)
        self.assertEqual(report['auto_archive_settings']['trigger_messages'],24)
        self.assertNotIn('error_log',report)

    def test_diagnose_export_never_overwrites(self):
        path=write_diagnostic_report(self.data,self.root)
        self.assertTrue(path.is_file())
        self.assertNotIn(self.secret,path.read_text(encoding='utf-8'))
        with self.assertRaises(FileExistsError):
            write_diagnostic_report(self.data,self.root,path)

    def test_empty_data_without_db_is_reportable(self):
        with tempfile.TemporaryDirectory() as empty:
            report=build_diagnostic_report(Path(empty)/'empty',Path(empty))
        self.assertEqual(report['database'],{'present':False,'tables':{}})
        self.assertEqual(report['storage']['chat_message_lines'],0)

    def test_read_only_check_install_does_not_create_data(self):
        import subprocess
        import sys
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'nonexistent'
            output=subprocess.run([sys.executable,str(project/'main.py'),
                '--data',str(data),'check-install'],capture_output=True,text=True,check=True)
            self.assertFalse(data.exists())
            self.assertTrue(json.loads(output.stdout)['ready'])

    def test_non_newline_terminated_record_count(self):
        from maidcore.diagnostics import _count_lines
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'test.jsonl'
            file.write_bytes(b'first\nsecond')
            self.assertEqual(_count_lines(file),2)
            file.write_bytes(b'first\nsecond\n')
            self.assertEqual(_count_lines(file),2)

    def test_web_download_is_cleansed(self):
        # Workbench will perform its normal schema migrations; isolate a fresh
        # project from our hand-built minimal malicious DB.
        with tempfile.TemporaryDirectory() as t:
            data=Path(t)/'data';MaidCore(data).append_chat('user',self.secret)
            workbench=MemoryWorkbench(data,self.root/'.env')
            server=create_server(workbench,port=0)
            th=threading.Thread(target=server.serve_forever,daemon=True)
            th.start()
            try:
                with urlopen(f'http://127.0.0.1:{server.server_port}/api/diagnostics/download',timeout=4) as response:
                    report=response.read().decode('utf-8')
                    self.assertIn('attachment',response.headers.get('Content-Disposition',''))
                self.assertNotIn(self.secret,report)
                self.assertNotIn(str(self.root),report)
                self.assertEqual(json.loads(report)['report_format'],'winefox-diagnostic-v1')
            finally:
                server.shutdown();server.server_close();th.join(3)

if __name__=='__main__':unittest.main()
