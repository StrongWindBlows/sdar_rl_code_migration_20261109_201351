"""Recover legacy inference metadata in a temporary view; never edit checkpoints."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile


# This trainer reconstructed generation_config from config.json on resume.
# Restrict recovery to the audited exporter; do not override other models' EOS.
LEGACY_GENERATION_EXPORT = {
    'train.py': 'ad54600fc464624a88c687a3324e001a48101da6791bd3a571fe38b041ec5b8b',
    'runtime.py': 'b4d87a41dd646a04ba52f1df7ea3ce16e1879b9cc790b40c33c5787b5df0e3ac',
}


def generation_config_path(model_path):
    """Resolve the config actually used by the legacy run's rollout engine."""
    model = Path(model_path).resolve()
    seen = set()
    while True:
        if model in seen:
            raise ValueError(f'Cycle in generation-config resume lineage: {model}')
        seen.add(model)
        path = model/'generation_config.json'
        config = json.loads(path.read_text())
        metadata_path = model/'resume.json'
        if not config.get('_from_model_config') or not metadata_path.is_file():
            return path
        metadata = json.loads(metadata_path.read_text())
        source = metadata.get('fingerprint', {}).get('source', {})
        parent = metadata.get('config', {}).get('resume')
        if not parent or any(source.get(k) != v for k, v in LEGACY_GENERATION_EXPORT.items()):
            return path
        # The legacy Rollout loaded this directory once at startup and only
        # updated weights afterwards. Follow that recorded source, not the base
        # model or a hard-coded EOS list. Missing ancestors must fail visibly.
        model = Path(parent).resolve()


@contextmanager
def inference_model_path(model_path):
    model = Path(model_path).resolve()
    generation = generation_config_path(model)
    config = json.loads((model/'tokenizer_config.json').read_text())
    refs = config.get('auto_map', {}).get('AutoTokenizer', [])
    refs = [refs] if isinstance(refs, str) else refs
    qualified = any('--' in ref for ref in refs if ref)
    modules = {ref.rsplit('--', 1)[-1].split('.')[0]+'.py' for ref in refs if ref}
    missing = [name for name in modules if not (model/name).is_file()]
    if not missing and not qualified and generation == model/'generation_config.json':
        yield model
        return
    tokenizer_files = []
    if missing or qualified:
        # These checkpoints record the exact original model used for training.
        metadata = json.loads((model/'resume.json').read_text())
        original = Path(metadata['config']['model']).resolve()
        original_config = json.loads((original/'tokenizer_config.json').read_text())
        def local_auto_map(tokenizer_config):
            mapping = dict(tokenizer_config.get('auto_map', {}))
            references = mapping.get('AutoTokenizer', [])
            is_string = isinstance(references, str)
            references = [references] if is_string else references
            local = []
            for ref in references:
                if ref and '--' in ref:
                    origin, ref = ref.rsplit('--', 1)
                    if not Path(origin).is_absolute() or Path(origin).resolve() != original:
                        raise ValueError(f'Cannot complete tokenizer: unexpected origin {origin!r}')
                local.append(ref)
            if 'AutoTokenizer' in mapping:
                mapping['AutoTokenizer'] = local[0] if is_string else local
            return mapping

        local_mapping = local_auto_map(config)
        if local_mapping != local_auto_map(original_config):
            raise ValueError('Cannot complete tokenizer: original auto_map differs')
        if config.get('tokenizer_class') != original_config.get('tokenizer_class'):
            raise ValueError('Cannot complete tokenizer: original tokenizer_class differs')
        for name in ('vocab.json', 'merges.txt'):
            saved, source = (model/name).read_bytes(), (original/name).read_bytes()
            # Tokenizer export can change JSON indentation/key ordering while
            # retaining the exact token -> ID mapping.
            if name == 'vocab.json':
                saved, source = json.loads(saved), json.loads(source)
            if saved != source:
                raise ValueError(f'Cannot complete tokenizer: original {name} differs')
        for name in missing:
            if not (original/name).is_file():
                raise ValueError(f'Original tokenizer module is missing: {original/name}')
        tokenizer_files = list(original.glob('tokenization_*.py'))
        # save_pretrained may qualify the class with its original local path.
        # LMDeploy needs a self-contained view with unqualified local references.
        config['auto_map'] = local_mapping
    # Symlinks avoid copying multi-GB weights or altering resume.json fingerprints.
    with tempfile.TemporaryDirectory(prefix='sdar_inference_') as directory:
        view = Path(directory)
        for file in model.iterdir():
            if file.is_file():
                if qualified and file.name == 'tokenizer_config.json':
                    (view/file.name).write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
                    continue
                target = generation if file.name == 'generation_config.json' else file
                (view/file.name).symlink_to(target)
        for file in tokenizer_files:
            if not (view/file.name).exists():
                (view/file.name).symlink_to(file)
        yield view
