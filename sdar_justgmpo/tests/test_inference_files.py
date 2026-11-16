import json
from pathlib import Path
import tempfile
import unittest

from inference_files import inference_model_path, generation_config_path, LEGACY_GENERATION_EXPORT


class InferenceFilesTests(unittest.TestCase):
    def test_qualified_local_origin_is_rewritten_only_in_temporary_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, model = root/'original', root/'checkpoint'
            original.mkdir(); model.mkdir()
            config = {'tokenizer_class': 'Qwen2Tokenizer', 'auto_map': {
                'AutoTokenizer': ['tokenization_qwen2.Qwen2Tokenizer', None]}}
            (original/'tokenizer_config.json').write_text(json.dumps(config))
            (original/'tokenization_qwen2.py').write_text('# original tokenizer')
            config['auto_map']['AutoTokenizer'][0] = str(original)+'--tokenization_qwen2.Qwen2Tokenizer'
            for path in (original, model):
                (path/'vocab.json').write_text('{}')
                (path/'merges.txt').write_text('same')
            # save_pretrained serializes the same vocabulary differently.
            (original/'vocab.json').write_text('{"a": 0, "b": 1}')
            (model/'vocab.json').write_text('{\n  "b": 1,\n  "a": 0\n}')
            (model/'tokenizer_config.json').write_text(json.dumps(config))
            (model/'generation_config.json').write_text('{}')
            (model/'resume.json').write_text(json.dumps({'config': {'model': str(original)}}))
            (model/'model.safetensors').write_bytes(b'weights')
            before = {p.name: p.read_bytes() for p in model.iterdir()}
            with inference_model_path(model) as view:
                restored = json.loads((view/'tokenizer_config.json').read_text())
                self.assertEqual(restored['auto_map']['AutoTokenizer'],
                                 ['tokenization_qwen2.Qwen2Tokenizer', None])
                self.assertFalse((view/'tokenizer_config.json').is_symlink())
                self.assertEqual((view/'tokenization_qwen2.py').resolve(), original/'tokenization_qwen2.py')
                self.assertEqual((view/'model.safetensors').resolve(), model/'model.safetensors')
            self.assertFalse(view.exists())
            self.assertEqual({p.name: p.read_bytes() for p in model.iterdir()}, before)

            for reference, error in (
                (str(root/'other')+'--tokenization_qwen2.Qwen2Tokenizer', 'unexpected origin'),
                ('remote/repo--tokenization_qwen2.Qwen2Tokenizer', 'unexpected origin'),
                (str(original)+'--tokenization_qwen2.OtherTokenizer', 'auto_map differs'),
            ):
                config['auto_map']['AutoTokenizer'][0] = reference
                (model/'tokenizer_config.json').write_text(json.dumps(config))
                with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, error):
                    with inference_model_path(model):
                        pass

    def test_legacy_export_uses_temporary_view_and_rejects_different_vocab(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, model = root/'original', root/'checkpoint'
            original.mkdir(); model.mkdir()
            for path in (original,model):
                (path/'generation_config.json').write_text('{"eos_token_id": [151645, 151643]}')
                (path/'tokenizer_config.json').write_text(json.dumps(
                    {'auto_map': {'AutoTokenizer': ['tokenization_qwen2.Qwen2Tokenizer',None]}}))
                (path/'vocab.json').write_text('{}')
                (path/'merges.txt').write_text('same')
            (original/'tokenization_qwen2.py').write_text('# original tokenizer')
            (model/'resume.json').write_text(json.dumps({'config':{'model':str(original)}}))
            (model/'model.safetensors').write_bytes(b'weights')
            with inference_model_path(model) as view:
                self.assertNotEqual(view,model)
                self.assertEqual((view/'model.safetensors').resolve(),model/'model.safetensors')
                self.assertEqual((view/'tokenization_qwen2.py').read_text(),'# original tokenizer')
            self.assertFalse(view.exists())
            self.assertFalse((model/'tokenization_qwen2.py').exists())
            (model/'vocab.json').write_text('{"changed":1}')
            with self.assertRaisesRegex(ValueError,'vocab.json differs'):
                with inference_model_path(model):
                    pass

    def test_complete_model_is_used_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model/'tokenizer_config.json').write_text('{}')
            (model/'generation_config.json').write_text('{"eos_token_id": 151643}')
            with inference_model_path(model) as view:
                self.assertEqual(view,model)

    def test_legacy_generation_config_comes_from_recorded_resume_without_editing_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, model = root/'final', root/'checkpoint-75'
            parent.mkdir(); model.mkdir()
            original = {'eos_token_id': [151645, 151643], 'temperature': 0.6}
            derived = {'_from_model_config': True, 'eos_token_id': 151643}
            (parent/'generation_config.json').write_text(json.dumps(original))
            (model/'generation_config.json').write_text(json.dumps(derived))
            (model/'tokenizer_config.json').write_text('{}')
            metadata = {'config': {'resume': str(parent)},
                        'fingerprint': {'source': LEGACY_GENERATION_EXPORT}}
            (model/'resume.json').write_text(json.dumps(metadata))
            (model/'model.safetensors').write_bytes(b'weights')
            before = {p.name: p.read_bytes() for p in model.iterdir()}
            self.assertEqual(generation_config_path(model), parent/'generation_config.json')
            with inference_model_path(model) as view:
                self.assertEqual(json.loads((view/'generation_config.json').read_text()), original)
                self.assertEqual((view/'model.safetensors').resolve(), model/'model.safetensors')
            self.assertFalse(view.exists())
            self.assertEqual({p.name: p.read_bytes() for p in model.iterdir()}, before)

            # Even a known legacy trainer must not override an explicit config.
            (model/'generation_config.json').write_text(json.dumps(original))
            self.assertEqual(generation_config_path(model), model/'generation_config.json')
            # Unknown exporters can legitimately choose a single EOS.
            (model/'generation_config.json').write_text(json.dumps(derived))
            metadata['fingerprint']['source'] = {'train.py': 'different implementation'}
            (model/'resume.json').write_text(json.dumps(metadata))
            self.assertEqual(generation_config_path(model), model/'generation_config.json')

    def test_missing_or_cyclic_generation_lineage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model/'generation_config.json').write_text(json.dumps(
                {'_from_model_config': True, 'eos_token_id': 151643}))
            for parent, error in ((model/'missing', FileNotFoundError), (model, ValueError)):
                (model/'resume.json').write_text(json.dumps({
                    'config': {'resume': str(parent)},
                    'fingerprint': {'source': LEGACY_GENERATION_EXPORT}}))
                with self.assertRaises(error):
                    generation_config_path(model)
