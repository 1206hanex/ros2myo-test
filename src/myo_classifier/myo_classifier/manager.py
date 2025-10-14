#!/usr/bin/env python3
import os
import json
import inspect
import traceback
import importlib
import subprocess
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from myo_msgs.srv import AddGesture, TrainModel

class Manager(Node):
    def __init__(self):
        super().__init__('manager')

        # ---------- Parameters you can set via YAML/CLI ----------
        # Default locations if the request doesn't provide them
        self.declare_parameter('data_root', 'data')
        self.declare_parameter('out_root', 'runs')
        self.declare_parameter('python_exe', 'python3')

        self.data_root: str = self.get_parameter('data_root').value
        self.out_root: str = self.get_parameter('out_root').value
        self.python_exe: str = self.get_parameter('python_exe').value

        # Map model keys -> importable module path
        self.model_modules = {
            'rf': 'myo_classifier.train.rf',
            'svm': 'myo_classifier.train.svm',
            'cnn_lstm': 'myo_classifier.train.cnn_lstm',
        }

        # Optional: map model keys -> script path (used if import fails)
        # Update these if you prefer calling scripts.
        self.model_scripts = {
            'rf': None,         # e.g., '/home/hanex/ros2_ws/src/myo_classifier/scripts/train_rf.py'
            'svm': None,
            'cnn_lstm': None,
        }

        # Reentrant group for services
        self.cb_srv = ReentrantCallbackGroup()

        # Manager’s AddGesture — forwards to recorder
        self.srv_add = self.create_service(
            AddGesture,
            'myo_classifier/add_gesture',
            self._srv_add_gesture,
            callback_group=self.cb_srv
        )

        # TrainModel service (stub/your impl)
        self.srv_train = self.create_service(
            TrainModel,
            'myo_classifier/train_model',
            self._srv_train_model,
            callback_group=self.cb_srv
        )

        # Client to the recorder service — make sure the name matches the recorder node!
        self.recorder_cli = self.create_client(AddGesture, '/myo_classifier/recorder_add_gesture')

        self.get_logger().info('Manager ready (forwarding to recorder)')

    def _srv_add_gesture(self, req, resp):
        return None

    def _srv_train_model(self, req, resp):
        # implement training reading CSVs from req or your cfg defaults
        """
        TrainModel handler:
        - Reads the model key from the request ("rf" | "svm" | "cnn_lstm")
        - Collects optional kwargs (data_dir, out_dir, epochs, etc.)
        - Calls an importable module's train()/main(), or a script via subprocess
        - Returns success flag + details (+ optional metrics/model_path if your srv has those)
        """
        model_key = self._get_req_model_key(req)
        if not model_key:
            self.get_logger().error('TrainModel: no model specified on request')
            self._safe_set_resp(resp, success=False, detail='no model specified')
            return resp

        if model_key not in self.model_modules and model_key not in self.model_scripts:
            self.get_logger().error(f'TrainModel: unsupported model "{model_key}"')
            self._safe_set_resp(resp, success=False, detail=f'unsupported model "{model_key}"')
            return resp

        kw = self._collect_train_kwargs(req)
        self.get_logger().info(f'TrainModel: model={model_key}, kwargs={kw}')

        try:
            # Try importable module first
            if model_key in self.model_modules and self.model_modules[model_key]:
                out = self._call_module_train(self.model_modules[model_key], kw)
            # Else try script fallback
            elif model_key in self.model_scripts and self.model_scripts[model_key]:
                out = self._call_script_train(self.model_scripts[model_key], kw)
            else:
                raise RuntimeError(f'No module or script configured for model "{model_key}"')

            # Fill response safely (only if fields exist in your .srv)
            self._safe_set_resp(resp, success=bool(out.get('ok', False)))
            self._safe_set_resp(resp, detail=str(out.get('detail', '')))
            if 'metrics' in out:
                self._safe_set_resp(resp, metrics_json=json.dumps(out['metrics']))
            if 'model_path' in out:
                self._safe_set_resp(resp, model_path=str(out['model_path']))

            return resp

        except Exception as e:
            err = f'TrainModel failed: {e}'
            self.get_logger().error(err)
            self.get_logger().debug(traceback.format_exc())
            self._safe_set_resp(resp, success=False, detail=err)
            return resp
        
# ---------------------- Utilities ----------------------

    def _get_req_model_key(self, req) -> str:
        """Extract model name from request across common field names, normalized."""
        candidates = []
        for field in ('model', 'model_name', 'type', 'algo', 'algorithm'):
            if hasattr(req, field):
                val = getattr(req, field)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip().lower())
        if not candidates:
            return ''
        key = candidates[0]
        # simple normalization
        synonyms = {
            'random_forest': 'rf',
            'forest': 'rf',
            'svm_linear': 'svm',
            'svm_rbf': 'svm',
            'cnn': 'cnn_lstm',
            'lstm': 'cnn_lstm',
            'cnn-lstm': 'cnn_lstm',
            'cnn_lstm': 'cnn_lstm',
        }
        return synonyms.get(key, key)

    def _collect_train_kwargs(self, req) -> Dict[str, Any]:
        """Collect optional training kwargs from the request; fall back to node params."""
        kw: Dict[str, Any] = {}

        # Common path-ish fields
        for name in ('data_dir', 'dataset_dir', 'dataset_path', 'csv_dir'):
            if hasattr(req, name):
                v = getattr(req, name)
                if isinstance(v, str) and v.strip():
                    kw['data_dir'] = v
                    break
        if 'data_dir' not in kw:
            kw['data_dir'] = self.data_root

        for name in ('out_dir', 'output_dir', 'run_dir'):
            if hasattr(req, name):
                v = getattr(req, name)
                if isinstance(v, str) and v.strip():
                    kw['out_dir'] = v
                    break
        if 'out_dir' not in kw:
            # put each run in a per-model subfolder
            kw['out_dir'] = self.out_root

        # Optional hyperparams commonly passed
        for name in ('epochs', 'batch_size', 'seed', 'folds', 'lr', 'patience'):
            if hasattr(req, name):
                kw[name] = getattr(req, name)

        # Optional switches
        for name in ('save_model', 'overwrite', 'use_imu', 'use_emg'):
            if hasattr(req, name):
                kw[name] = bool(getattr(req, name))

        return kw

    def _safe_set_resp(self, resp, **fields):
        """Set response fields only if they exist in the service definition."""
        for k, v in fields.items():
            if hasattr(resp, k):
                setattr(resp, k, v)

    def _call_module_train(self, module_path: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a module and call train(**kwargs) or main().
        Returns a dict with 'ok', 'detail', and optional 'metrics'/'model_path'.
        """
        out: Dict[str, Any] = {}
        mod = importlib.import_module(module_path)

        # Prefer train(**kwargs); fall back to main()
        train_fn = getattr(mod, 'train', None)
        if train_fn is not None and callable(train_fn):
            sig = inspect.signature(train_fn)
            allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result = train_fn(**allowed)  # result can be dict / tuple / None
        else:
            main_fn = getattr(mod, 'main', None)
            if main_fn is None or not callable(main_fn):
                raise RuntimeError(f'Module "{module_path}" exposes neither train() nor main()')
            result = main_fn()

        # Normalize results
        out['ok'] = True
        if isinstance(result, dict):
            out['metrics'] = result.get('metrics', result)
            if 'model_path' in result:
                out['model_path'] = result['model_path']
            out['detail'] = 'Training completed (module).'
        elif isinstance(result, (list, tuple)) and len(result) >= 1:
            out['metrics'] = result[0]
            if len(result) >= 2:
                out['model_path'] = result[1]
            out['detail'] = 'Training completed (module tuple).'
        else:
            out['detail'] = 'Training completed.'
        return out

    def _call_script_train(self, script_path: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a standalone training script via subprocess. We pass common args if present.
        The script should print JSON with metrics/model_path to stdout (optional).
        """
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f'Script not found: {script_path}')

        cmd = [self.python_exe, script_path]
        # Add a few conventional CLI flags if available
        if 'data_dir' in kwargs:
            cmd += ['--data', kwargs['data_dir']]
        if 'out_dir' in kwargs:
            cmd += ['--out', kwargs['out_dir']]
        for key in ('epochs', 'batch_size', 'seed', 'folds', 'lr', 'patience'):
            if key in kwargs and kwargs[key] is not None:
                cmd += [f'--{key.replace("_","-")}', str(kwargs[key])]
        if kwargs.get('save_model', None) is True:
            cmd += ['--save-model']
        if kwargs.get('overwrite', None) is True:
            cmd += ['--overwrite']

        self.get_logger().info(f'Running: {" ".join(cmd)}')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'Training script failed ({proc.returncode}): {proc.stderr.strip()}')

        # Try to parse JSON from stdout; if it fails, return raw text
        detail = proc.stdout.strip()
        out: Dict[str, Any] = {'ok': True, 'detail': 'Training completed (script).'}
        try:
            payload = json.loads(detail)
            if isinstance(payload, dict):
                if 'metrics' in payload:
                    out['metrics'] = payload['metrics']
                if 'model_path' in payload:
                    out['model_path'] = payload['model_path']
                if 'detail' in payload:
                    out['detail'] = str(payload['detail'])
        except Exception:
            out['detail'] += ' (non-JSON output captured)'
        return out


def main():
    rclpy.init()
    node = Manager()
    try:
        exec = MultiThreadedExecutor(num_threads=2)
        exec.add_node(node)
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
