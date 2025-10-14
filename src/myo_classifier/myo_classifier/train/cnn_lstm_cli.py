#!/usr/bin/env python3
import os, json, argparse
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cnn_lstm as mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sampling-rate', type=float, default=200.0)
    ap.add_argument('--window-sec', type=float, default=0.2)
    ap.add_argument('--overlap', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lstm-units', type=int, default=64)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--random-state', type=int, default=42)
    args = ap.parse_args()

    # Use the module’s CSV loader + trainer; it returns a dict
    out = mod.train(
        data_dir=os.path.expanduser(args.data),
        out_dir=os.path.expanduser(args.out),
        sampling_rate=args.sampling_rate,
        window_sec=args.window_sec,
        overlap=args.overlap,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        logger=print,
    )
    print(json.dumps(out))

if __name__ == "__main__":
    main()
